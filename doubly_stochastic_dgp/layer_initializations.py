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

import torch as th
import numpy as np

from braingp.model import Model, Param
from braingp.mean_function import Identity, Linear, Zero
from gpytorch import settings

from braingp import kernel

from doubly_stochastic_dgp.layers import SVGP_Layer

def init_layers_linear(X, Y, Z, kernels,                         #XYZ are pytorch tensors
                       num_outputs=None,
                       mean_function=Zero(),
                       Layer=SVGP_Layer,
                       white=False):
    num_outputs = num_outputs or Y.size(1)

    layers = []

    X_running, Z_running = torch.tensor(X.numpy), torch.tensor(Z.numpy)  # gpflow.kernel object , kernel-gpytorch with ARD
    for kern_in, kern_out in zip(kernels[:-1], kernels[1:]):   #kernels is a list
        dim_in = kern_in.input_dim          #using gptorch kernel object type
        dim_out = kern_out.input_dim
        print(dim_in, dim_out)
        if dim_in == dim_out:
            mf = Identity()

        else:
            if dim_in > dim_out:  # stepping down, use the pca projection
                _, _, V = np.linalg.svd(X_running, full_matrices=False)
                W = V[:dim_out, :].T

            else: # stepping up, use identity + padding
                W = np.concatenate([np.eye(dim_in), np.zeros((dim_in, dim_out - dim_in))], 1)

            mf = Linear(W)
            mf = th.from_numpy(mf)
            mf.set_trainable(False)   #check parameterized.py gpflow no alt. in torch.parameter.nn

        layers.append(Layer(kern_in, Z_running, dim_out, mf, white=white))

        if dim_in != dim_out:
            Z_running = th.matmul(Z_running,W)
            X_running = th.matmul(X_running,W)

    # final layer
    layers.append(Layer(kernels[-1], Z_running, num_outputs, mean_function, white=white))
    return layers


def init_layers_input_prop(X, Y, Z, kernels,
                           num_outputs=None,
                           mean_function=Zero(),
                           Layer=SVGP_Layer,
                           white=False):
    num_outputs = num_outputs or Y.shape[1]
    D = X.size(1)
    M = Z.size(0)

    layers = []

    for kern_in, kern_out in zip(kernels[:-1], kernels[1:]):
        dim_in = kern_in.input_dim
        dim_out = kern_out.input_dim - D
        std_in = kern_in.variance.read_value()**0.5
        pad = np.random.randn(M, dim_in - D) * 2. * std_in
        Z_padded = np.concatenate([Z, pad], 1)
        layers.append(Layer(kern_in, Z_padded, dim_out, Zero(), white=white, input_prop_dim=D))

    dim_in = kernels[-1].input_dim
    std_in = kernels[-2].variance.read_value()**0.5 if dim_in > D else 1.
    pad = np.random.randn(M, dim_in - D) * 2. * std_in
    Z_padded = np.concatenate([Z, pad], 1)
    Z_padded = th.from_numpy(Z_padded)
    layers.append(Layer(kernels[-1], Z_padded, num_outputs, mean_function, white=white))
    return layers
