# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#DevNote : self.feature creation for the implementation of U(z) implementation.

import torch
import numpy as np
from braingp import bpflow as bf
from gpytorch.priors import MultivariateNormalPrior as Gaussian_prior
from braingp.model import Model

"""from gpflow.conditionals import conditional
from gpflow.features import InducingPoints
from gpflow.kullback_leiblers import gauss_kl
from gpflow import transforms
from gpflow import settings
from gpflow.models.gplvm import BayesianGPLVM
from gpflow.expectations import expectation
from gpflow.probability_distributions import DiagonalGaussian
from gpflow import params_as_tensors
from gpflow.logdensities import multivariate_normal"""



from doubly_stochastic_dgp.utils import reparameterize


class Layer(Model):                                             # module treats inputs as parameters 
    def __init__(self, input_prop_dim=None, **kwargs):
        """
        A base class for GP layers. Basic functionality for multisample conditional, and input propagation
        :param input_prop_dim: the first dimensions of X to propagate. If None (or zero) then no input prop
        :param kwargs:
        """
        self.input_prop_dim = input_prop_dim

    def conditional_ND(self, X, full_cov=False):
        raise NotImplementedError

    def KL(self):
        return torch.Tensor([0.])

    def conditional_SND(self, X, full_cov=False):
        """
        A multisample conditional, where X is shape (S,N,D_out), independent over samples S

        if full_cov is True
            mean is (S,N,D_out), var is (S,N,N,D_out)

        if full_cov is False
            mean and var are both (S,N,D_out)

        :param X:  The input locations (S,N,D_in)
        :param full_cov: Whether to calculate full covariance or just diagonal
        :return: mean (S,N,D_out), var (S,N,D_out or S,N,N,D_out)
        """
        if full_cov is True:
            f = lambda a: self.conditional_ND(a, full_cov=full_cov)
            S, N, D = X.size()
            mean = torch.zeros(S,N,D)
            var = torch.zeros(S,N,N,D)
            mean = a(X).(torch.FloatTensor)              #checkspot
            var  = a(X).(torch.FloatTensor)
            return torch.stack(mean), torch.stack(var)
            
            #mean, var = torch.map_fn(f, X, dtype=(torch.float64, torch.float64))
            #return torch.stack(mean), torch.stack(var)

        else:
            S, N, D = X.size()
            X_flat = torch.reshape(X, [S * N, D])
            mean, var = self.conditional_ND(X_flat)
            return [torch.reshape(m, [S, N, self.num_outputs]) for m in [mean, var]]

    def sample_from_conditional(self, X, z=None, full_cov=False):
        """
        Calculates self.conditional and also draws a sample, adding input propagation if necessary

        If z=None then the tensorflow random_normal function is used to generate the
        N(0, 1) samples, otherwise z are used for the whitened sample points

        :param X: Input locations (S,N,D_in)
        :param full_cov: Whether to compute correlations between outputs
        :param z: None, or the sampled points in whitened representation
        :return: mean (S,N,D), var (S,N,N,D or S,N,D), samples (S,N,D)
        """
        mean, var = self.conditional_SND(X, full_cov=full_cov)

        # set shapes
        S = X.size(0)
        N = X.size(1)
        D = self.num_outputs

        mean = torch.reshape(mean, (S, N, D))
        if full_cov:
            var = torch.reshape(var, (S, N, N, D))
        else:
            var = torch.reshape(var, (S, N, D))

        if z is None:
            z =torch.randn(mean.size()).(torch.FloatTensor)
        samples = reparameterize(mean, var, z, full_cov=full_cov)

        if self.input_prop_dim:
            shape = [X.size(0), X.size(1), self.input_prop_dim]
            X_prop = torch.reshape(X[:, :, :self.input_prop_dim], shape)

            samples = torch.cat([X_prop, samples], 2)
            mean = torch.cat([X_prop, mean], 2)

            if full_cov:
                shape = (X.size(0), X.size(1), X.size(1), var.size(3))
                zeros = torch.zeros(shape).float()
                var = torch.cat([zeros, var], 3)
            else:
                var = torch.cat([torch.zeros_like(X_prop), var], 2)

        return samples, mean, var


class SVGP_Layer(Layer):
    def __init__(self, kern, Z, num_outputs, mean_function,
                 white=False, input_prop_dim=None, **kwargs):
        """
        A sparse variational GP layer in whitened representation. This layer holds the kernel,
        variational parameters, inducing points and mean function.

        The underlying model at inputs X is
        f = Lv + mean_function(X), where v \sim N(0, I) and LL^T = kern.K(X)

        The variational distribution over the inducing points is
        q(v) = N(q_mu, q_sqrt q_sqrt^T)

        The layer holds D_out independent GPs with the same kernel and inducing points.

        :param kern: The kernel for the layer (input_dim = D_in)
        :param Z: Inducing points (M, D_in)
        :param num_outputs: The number of GP outputs (q_mu is shape (M, num_outputs))
        :param mean_function: The mean function
        :return:
        """
        Layer.__init__(self, input_prop_dim, **kwargs)
        self.num_inducing = Z.size(0)

        q_mu = torch.zeros((self.num_inducing, num_outputs))
        self.q_mu = q_mu

        q_sqrt = np.tile(np.eye(self.num_inducing)[None, :, :], [num_outputs, 1, 1])
        transform = bf.LowerTriangular(self.num_inducing, num_matrices=num_outputs)
        self.q_sqrt = transform.forward(q_sqrt)

        self.feature = InducingPoints(Z)
        self.kern = kern
        self.mean_function = mean_function

        self.num_outputs = num_outputs
        self.white = white

        if not self.white:  # initialize to prior
            Ku = self.kern.compute_K_symm(Z)              #Check kernel defs or add to bpflow
            Lu = np.linalg.cholesky(Ku + np.eye(Z.shape[0])*settings.jitter)
            self.q_sqrt = np.tile(Lu[None, :, :], [num_outputs, 1, 1])

        self.needs_build_cholesky = True


    def build_cholesky_if_needed(self):
        # make sure we only compute this once
        if self.needs_build_cholesky:
            self.Ku = self.feature.Kuu(self.kern, jitter=settings.jitter)
            self.Lu = torch.cholesky(self.Ku)
            self.Ku_tiled = bf.tile(self.Ku[None, :, :], [self.num_outputs, 1, 1])
            self.Lu_tiled = bf.tile(self.Lu[None, :, :], [self.num_outputs, 1, 1])
            self.needs_build_cholesky = False


    def conditional_ND(self, X, full_cov=False):
        self.build_cholesky_if_needed()

        # mmean, vvar = conditional(X, self.feature.Z, self.kern,
        #             self.q_mu, q_sqrt=self.q_sqrt,
        #             full_cov=full_cov, white=self.white)
        Kuf = self.feature.Kuf(self.kern, X)

        A = torch.triangular_solve(self.Lu, Kuf, upper=False)
        if not self.white:
            A = torch.triangular_solve(torch.transpose(self.Lu), A, upper=True)

        mean = torch.matmul(A, self.q_mu, transpose_a=True)

        A_tiled = pf.tile(A[None, :, :], [self.num_outputs, 1, 1])
        I = torch.eye(self.num_inducing, dtype=torch.float64)[None, :, :]

        if self.white:
            SK = -I
        else:
            SK = -self.Ku_tiled

        if self.q_sqrt is not None:
            SK += torch.matmul(self.q_sqrt, torch.transpose(self.q_sqrt))


        B = torch.matmul(SK, A_tiled)

        if full_cov:
            # (num_latent, num_X, num_X)
            delta_cov = torch.matmul(torch.transpose(A_tiled), B)
            Kff = self.kern.K(X)
        else:
            # (num_latent, num_X)
            #delta_cov = torch.reduce_sum(A_tiled * B, 1)
            delta_cov =torch.cumsum(A_tiled * B, dim=1)[:,a.size(0)-1]
            Kff = self.kern.Kdiag(X)

        # either (1, num_X) + (num_latent, num_X) or (1, num_X, num_X) + (num_latent, num_X, num_X)
        var = Kff[None,:] + delta_cov
        var = torch.transpose(var)

        return mean + self.mean_function(X), var

    def KL(self):
        """
        The KL divergence from the variational distribution to the prior

        :return: KL divergence from N(q_mu, q_sqrt) to N(0, I), independently for each GP
        """
        # if self.white:
        #     return gauss_kl(self.q_mu, self.q_sqrt)
        # else:
        #     return gauss_kl(self.q_mu, self.q_sqrt, self.Ku)

        self.build_cholesky_if_needed()

        KL = -0.5 * self.num_outputs * self.num_inducing
        KL -= 0.5 * torch.cumsum(torch.log(torch.stack(tuple(t.diag() for t in torch.unbind(self.q_sqrt,0))) ** 2),dim=0)[:,a.size(1)-1] #error check

        if not self.white:
            KL += torch.cumsum(torch.log(torch.stack(tuple(t.diag() for t in torch.unbind(self.q_sqrt,0)))),dim=0)[:,a.size(1)-1]  * self.num_outputs
            KL += 0.5 * torch.cumsum(torch.square(torch.triangular_solve(self.Lu_tiled, self.q_sqrt, upper=False)),dim=0)[:,a.size(1)-1]
            Kinv_m = torch.cholesky_solve(self.q_mu , self.Lu)
            KL += 0.5 * torch.cumsum(self.q_mu * Kinv_m, dim=0)[:,a.size(1)-1]
        else:
            KL += 0.5 * torch.cumsum(torch.square(self.q_sqrt),dim=0)[:,a.size(1)-1]
            KL += 0.5 * torch.cumsum(self.q_mu**2,dim=0)[:,a.size(1)-1]

        return KL


class SGPMC_Layer(SVGP_Layer):
    def __init__(self, *args, **kwargs):
        """
        A sparse layer for sampling over the inducing point values 
        """
        SVGP_Layer.__init__(self, *args, **kwargs)
        self.q_mu.prior = log(Gaussian_prior(0., 1.)) #checkpoint
        del self.q_sqrt
        self.q_sqrt = None

    def KL(self):
        return torch.cast(0., dtype=settings.float_type)


class GPMC_Layer(Layer):
    def __init__(self, kern, X, num_outputs, mean_function, input_prop_dim=None, **kwargs):
        """
        A dense layer with fixed inputs. NB X does not change here, and must be the inputs. Minibatches not possible
        """
        Layer.__init__(self, input_prop_dim, **kwargs)
        self.num_data = X.size(0)
        q_mu = np.zeros((self.num_data, num_outputs))
        self.q_mu = q_mu
        self.q_mu.prior = Gaussian_prior(0., 1.) #checkpoint
        self.kern = kern
        self.mean_function = mean_function

        self.num_outputs = num_outputs

        Ku = self.kern.compute_K_symm(X) + np.eye(self.num_data) * settings.jitter #Independent Jitter operation
        self.Lu = torch.from_numpy(np.linalg.cholesky(Ku))
        self.X = torch.from_numpy(X) #check X type

    def build_latents(self):
        f = torch.matmul(self.Lu, self.q_mu)
        f += self.mean_function(self.X)
        if self.input_prop_dim:
            f = torch.cat([self.X[:, :self.input_prop_dim], f], 1)
        return f

    def conditional_ND(self, Xnew, full_cov=False):
        mu, var = conditional(Xnew, self.X, self.kern, self.q_mu,
                              full_cov=full_cov,
                              q_sqrt=None, white=True)
        return mu + self.mean_function(Xnew), var


class Collapsed_Layer(Layer):
    """
    Extra functions for a collapsed layer
    """
    def set_data(self, X_mean, X_var, Y, lik_variance):
        self._X_mean = X_mean
        self._X_var = X_var
        self._Y = Y
        self._lik_variance = lik_variance

    def build_likelihood(self):
        raise NotImplementedError


class GPR_Layer(Collapsed_Layer):
    def __init__(self, kern, mean_function, num_outputs, **kwargs):
        """
        A dense GP layer with a Gaussian likelihood, where the GP is integrated out
        """
        Collapsed_Layer.__init__(self, **kwargs)
        self.kern = kern
        self.mean_function = mean_function
        self.num_outputs = num_outputs

    def conditional_ND(self, Xnew, full_cov=False):
        ## modified from GPR
        Kx = self.kern.K(self._X_mean, Xnew)
        K = self.kern.K(self._X_mean) + torch.eye(self._X_mean.size(0).(torch.FloatTensor) * self._lik_variance
        L = torch.cholesky(K)
        A = torch.triangular_solve(L, Kx, upper=False)
        V = torch.triangular_solve(L, self._Y - self.mean_function(self._X_mean)) #checkpoint
        fmean = torch.matmul(torch.transpose(A), V) + self.mean_function(Xnew) 
        if full_cov:
            fvar = self.kern.K(Xnew) - torch.matmul(torch.transpose(A), A)
            shape = torch.stack([1, 1, torch.shape(self._Y)[1]]) #check stack function for torch library
            fvar = pf.tile(torch.expand_dims(fvar, 2), shape)
        else:
            fvar = self.kern.Kdiag(Xnew) - torch.cumsum(torch.square(A), dim=0)[:,a.size(1)-1]
            fvar = bf.tile(torch.reshape(fvar, (-1, 1)), [1, torch.shape(self._Y)[1]])
        return fmean, fvar

    def build_likelihood(self):
        ## modified from GPR
        K = self.kern.K(self._X_mean) + torch.eye(self._X_mean.size(0)) * self._lik_variance
        L = torch.cholesky(K)
        m = self.mean_function(self._X_mean)
        return torch.cumsum(multivariate_normal(self._Y, m, L), dim=0)[:,a.size(1)-1]


class SGPR_Layer(Collapsed_Layer):
    def __init__(self, kern, Z, num_outputs, mean_function, **kwargs):
        """
        A sparse variational GP layer with a Gaussian likelihood, where the 
        GP is integrated out

        :kern: The kernel for the layer (input_dim = D_in)
        :param Z: Inducing points (M, D_in)
        :param mean_function: The mean function
        :return:
        """

        Collapsed_Layer.__init__(self, **kwargs)
        self.feature = InducingPoints(Z)
        self.kern = kern
        self.mean_function = mean_function
        self.num_outputs = num_outputs

    def conditional_ND(self, Xnew, full_cov=False):
        return gplvm_build_predict(self, Xnew, self._X_mean, self._X_var, self._Y, self._lik_variance, full_cov=full_cov)

    def build_likelihood(self):
        return gplvm_build_likelihood(self, self._X_mean, self._X_var, self._Y, self._lik_variance)


################## From gpflow (with KL removed)
def gplvm_build_likelihood(self, X_mean, X_var, Y, variance):
    if X_var is None:
        # SGPR
        num_inducing = len(self.feature)
        num_data = torch.cast(Y.size(0)).(torch.FloatTensor)
        output_dim = torch.cast(Y.size(1)).(torch.FloatTensor)

        err = Y - self.mean_function(X_mean)
        Kdiag = self.kern.Kdiag(X_mean)
        Kuf = self.feature.Kuf(self.kern, X_mean)
        Kuu = self.feature.Kuu(self.kern, jitter=settings.numerics.jitter_level)
        L = torch.cholesky(Kuu)
        sigma = torch.sqrt(variance)

        # Compute intermediate matrices
        A = torch.triangular_solve(L, Kuf, upper=False) / sigma
        AAT = torch.matmul(A, torch.transpose(A))
        B = AAT + torch.eye(num_inducing).float()
        LB = torch.cholesky(B)
        Aerr = torch.matmul(A, err)
        c = torch.triangular_solve(LB, Aerr, upper=False) / sigma

        # compute log marginal bound
        bound = -0.5 * num_data * output_dim * torch.log(2 * np.pi)
        bound += torch.negative(output_dim) * torch.reduce_sum(torch.log(torch.matrix_diag_part(LB)))
        bound -= 0.5 * num_data * output_dim * torch.log(variance)
        bound += -0.5 * torch.cumsum(torch.square(err))[] / variance
        bound += 0.5 * torch.cumsum(torch.square(c))
        bound += -0.5 * output_dim * torch.cumsum(Kdiag) / variance
        bound += 0.5 * output_dim * torch.cumsum(torch.matrix_diag_part(AAT))

        return bound


    else:

        X_cov = torch.matrix_diag(X_var)
        pX = DiagonalGaussian(X_mean, X_var)
        num_inducing = len(self.feature)
        if hasattr(self.kern, 'X_input_dim'):
            psi0 = torch.cumsum(self.kern.eKdiag(X_mean, X_cov))[:]
            psi1 = self.kern.eKxz(self.feature.Z, X_mean, X_cov)
            psi2 = torch.cumsum(self.kern.eKzxKxz(self.feature.Z, X_mean, X_cov), 0)
        else:
            psi0 = torch.cumsum(expectation(pX, self.kern))
            psi1 = expectation(pX, (self.kern, self.feature))
            psi2 = torch.cumsum(expectation(pX, (self.kern, self.feature), (self.kern, self.feature)), axis=0)
        Kuu = self.feature.Kuu(self.kern, jitter=settings.numerics.jitter_level)
        L = torch.cholesky(Kuu)
        sigma2 = variance
        sigma = torch.sqrt(sigma2)

        # Compute intermediate matrices
        A = torch.matrix_triangular_solve(L, torch.transpose(psi1), lower=True) / sigma
        tmp = torch.matrix_triangular_solve(L, psi2, lower=True)
        AAT = torch.matrix_triangular_solve(L, torch.transpose(tmp), lower=True) / sigma2
        B = AAT + torch.eye(num_inducing, dtype=settings.float_type)
        LB = torch.cholesky(B)
        log_det_B = 2. * torch.reduce_sum(torch.log(torch.matrix_diag_part(LB)))
        c = torch.matrix_triangular_solve(LB, torch.matmul(A, Y), lower=True) / sigma

        # KL[q(x) || p(x)]
        # dX_var = self.X_var if len(self.X_var.get_shape()) == 2 else torch.matrix_diag_part(self.X_var)
        # NQ = torch.cast(torch.size(self.X_mean), settings.float_type)
        D = torch.cast(torch.shape(Y)[1], settings.float_type)
        # KL = -0.5 * torch.reduce_sum(torch.log(dX_var)) \
        #      + 0.5 * torch.reduce_sum(torch.log(self.X_prior_var)) \
        #      - 0.5 * NQ \
        #      + 0.5 * torch.reduce_sum((torch.square(self.X_mean - self.X_prior_mean) + dX_var) / self.X_prior_var)

        # compute log marginal bound
        ND = torch.cast(torch.size(Y), settings.float_type)
        bound = -0.5 * ND * torch.log(2 * np.pi * sigma2)
        bound += -0.5 * D * log_det_B
        bound += -0.5 * torch.reduce_sum(torch.square(Y)) / sigma2
        bound += 0.5 * torch.reduce_sum(torch.square(c))
        bound += -0.5 * D * (torch.reduce_sum(psi0) / sigma2 -
                             torch.reduce_sum(torch.matrix_diag_part(AAT)))
        # bound -= KL # don't need this term
        return bound

############# Exactly from gpflow
def gplvm_build_predict(self, Xnew, X_mean, X_var, Y, variance, full_cov=False):
    if X_var is None:
        # SGPR
        num_inducing = len(self.feature)
        err = Y - self.mean_function(X_mean)
        Kuf = self.feature.Kuf(self.kern, X_mean)
        Kuu = self.feature.Kuu(self.kern, jitter=settings.numerics.jitter_level)
        Kus = self.feature.Kuf(self.kern, Xnew)
        sigma = torch.sqrt(variance)
        L = torch.cholesky(Kuu)
        A = torch.matrix_triangular_solve(L, Kuf, lower=True) / sigma
        B = torch.matmul(A, A, transpose_b=True) + torch.eye(num_inducing, dtype=settings.float_type)
        LB = torch.cholesky(B)
        Aerr = torch.matmul(A, err)
        c = torch.matrix_triangular_solve(LB, Aerr, lower=True) / sigma
        tmp1 = torch.matrix_triangular_solve(L, Kus, lower=True)
        tmp2 = torch.matrix_triangular_solve(LB, tmp1, lower=True)
        mean = torch.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = self.kern.K(Xnew) + torch.matmul(tmp2, tmp2, transpose_a=True) \
                  - torch.matmul(tmp1, tmp1, transpose_a=True)
            shape = torch.stack([1, 1, torch.shape(Y)[1]])
            var = torch.tile(torch.expand_dims(var, 2), shape)
        else:
            var = self.kern.Kdiag(Xnew) + torch.reduce_sum(torch.square(tmp2), 0) \
                  - torch.reduce_sum(torch.square(tmp1), 0)
            shape = torch.stack([1, torch.shape(Y)[1]])
            var = torch.tile(torch.expand_dims(var, 1), shape)
        return mean + self.mean_function(Xnew), var

    else:
        # gplvm
        pX = DiagonalGaussian(X_mean, X_var)
        num_inducing = len(self.feature)

        X_cov = torch.matrix_diag(X_var)

        if hasattr(self.kern, 'X_input_dim'):
            psi1 = self.kern.eKxz(self.feature.Z, X_mean, X_cov)
            psi2 = torch.reduce_sum(self.kern.eKzxKxz(self.feature.Z, X_mean, X_cov), 0)
        else:
            psi1 = expectation(pX, (self.kern, self.feature))
            psi2 = torch.reduce_sum(expectation(pX, (self.kern, self.feature), (self.kern, self.feature)), axis=0)

        # psi1 = expectation(pX, (self.kern, self.feature))
        # psi2 = torch.reduce_sum(expectation(pX, (self.kern, self.feature), (self.kern, self.feature)), axis=0)

        Kuu = self.feature.Kuu(self.kern, jitter=settings.numerics.jitter_level)
        Kus = self.feature.Kuf(self.kern, Xnew)
        sigma2 = variance
        sigma = torch.pow(sigma2,0.5)
        L = torch.cholesky(Kuu)

        A = torch.matrix_triangular_solve(L, torch.transpose(psi1), lower=True) / sigma
        tmp = torch.matrix_triangular_solve(L, psi2, lower=True)
        AAT = torch.matrix_triangular_solve(L, torch.transpose(tmp), lower=True) / sigma2
        B = AAT + torch.eye(num_inducing, dtype=settings.float_type)
        LB = torch.cholesky(B)
        c = torch.matrix_triangular_solve(LB, torch.matmul(A, Y), lower=True) / sigma
        tmp1 = torch.matrix_triangular_solve(L, Kus, lower=True)
        tmp2 = torch.matrix_triangular_solve(LB, tmp1, lower=True)
        mean = torch.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = self.kern.K(Xnew) + torch.matmul(tmp2, tmp2, transpose_a=True) \
                  - torch.matmul(tmp1, tmp1, transpose_a=True)
            shape = torch.stack([1, 1, torch.shape(Y)[1]])
            var = torch.tile(torch.expand_dims(var, 2), shape)
        else:
            var = self.kern.Kdiag(Xnew) + torch.reduce_sum(torch.square(tmp2), 0) \
                  - torch.reduce_sum(torch.square(tmp1), 0)
            shape = torch.stack([1, torch.shape(Y)[1]])
            var = torch.tile(torch.expand_dims(var, 1), shape)
        return mean + self.mean_function(Xnew), var
