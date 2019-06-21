import torch

from gpytorch import settings
from gpytorch.likelihoods import GaussianLikelihoodBase
from braingp import Pytfunct as pf


def reparameterize(mean, var, z, full_cov=False):
    """
    Implements the 'reparameterization trick' for the Gaussian, either full rank or diagonal

    If the z is a sample from N(0, 1), the output is a sample from N(mean, var)

    If full_cov=True then var must be of shape S,N,N,D and the full covariance is used. Otherwise
    var must be S,N,D and the operation is elementwise

    :param mean: mean of shape S,N,D , (Passes in as torch tensor)
    :param var: covariance of shape S,N,D or S,N,N,D
    :param z: samples form unit Gaussian of shape S,N,D
    :param full_cov: bool to indicate whether var is of shape S,N,N,D or S,N,D
    :return sample from N(mean, var) of shape S,N,D
    """
    if var is None:
        return mean

    if full_cov is False:
        return mean + z * (var + settings.tridiagonal_jitter) ** 0.5 # if error tridiagonal_jitter

    else:
        S, N, D = mean.size() # var is SNND
        mean = mean.transpose(2, 1)  # SND -> SDN
        var = var.permute(0,3,1,2)  # SNND -> SDNN
        I = settings.tridiagonal_jitter * torch.eye(N)[None, None, :, :].float() # 11NN
        chol = torch.cholesky(var + I)  # SDNN
        z_SDN1 = z.permute(0, 2, 1)[:, :, :, None]  # SND->SDN1
        f = mean + torch.matmul(chol, z_SDN1)[:, :, :, 0]  # SDN(1)
        return f.transpose(2, 1) # SND


class BroadcastingLikelihood():
    """
    A wrapper for the likelihood to broadcast over the samples dimension. The Gaussian doesn't
    need this, but for the others we can apply reshaping and tiling.

    With this wrapper all likelihood functions behave correctly with inputs of shape S,N,D,
    but with Y still of shape N,D
    """
    def __init__(self, likelihood):
        self.likelihood = likelihood

        if isinstance(likelihood, GaussianLikelihoodBase):
            self.needs_broadcasting = False
        else:
            self.needs_broadcasting = True

    def _broadcast(self, f, vars_SND, vars_ND):
        if self.needs_broadcasting is False:
            return f(vars_SND, [v[None,:] for v in vars_ND])

        else:
            S, N, D = vars_SND.size()
            vars_tiled = [pf.tile(x[None, :, :],S, 1, 1) for x in vars_ND]  #Error benchmark

            flattened_SND = [torch.reshape(x, [S*N, D]) for x in vars_SND]
            flattened_tiled = [torch.reshape(x, [S*N, -1]) for x in vars_tiled]

            flattened_result = f(flattened_SND, flattened_tiled)
            if isinstance(flattened_result, tuple):
                return [torch.reshape(x, [S, N, -1]) for x in flattened_result]
            else:
                return torch.reshape(flattened_result, [S, N, -1])

    
    # @param_as_tensors converts the dataholders and parameterized to computable unconstrained Tensors. 
    # Therefore, prepassing objets as torch tensors
    
    def variational_expectations(self, Fmu, Fvar, Y):
        f = lambda vars_SND, vars_ND: self.likelihood.variational_expectations(vars_SND[0],
                                                                                vars_SND[1],
                                                                                vars_ND[0])
        return self._broadcast(f,[Fmu, Fvar], [Y])

    
    def logp(self, F, Y):
        f = lambda vars_SND, vars_ND: self.likelihood.logp(vars_SND[0], vars_ND[0])
        return self._broadcast(f, [F], [Y])

    
    def conditional_mean(self, F):
        f = lambda vars_SND, vars_ND: self.likelihood.conditional_mean(vars_SND[0])
        return self._broadcast(f,[F], [])

    
    def conditional_variance(self, F):
        f = lambda vars_SND, vars_ND: self.likelihood.conditional_variance(vars_SND[0])
        return self._broadcast(f,[F], [])

    
    def predict_mean_and_var(self, Fmu, Fvar):
        f = lambda vars_SND, vars_ND: self.likelihood.predict_mean_and_var(vars_SND[0],
                                                                             vars_SND[1])
        return self._broadcast(f,[Fmu, Fvar], [])


    def predict_density(self, Fmu, Fvar, Y):
        f = lambda vars_SND, vars_ND: self.likelihood.predict_density(vars_SND[0],
                                                                       vars_SND[1],
                                                                       vars_ND[0])
        return self._broadcast(f,[Fmu, Fvar], [Y])