""" Contains more functions for the processing of data in the relevant files """

import torch

def tile(a, *arg):
    """the funcion performs tiling similar to tf.tile"""
    dim = len(arg)-1
    for k in range(0, dim+1):
        i = dim-k
        repeat_idx = [1] * a.dim()
        repeat_idx[i] = arg[i]
        a = a.repeat(*(repeat_idx))
    return a

class LowerTriangular():
    """
    A transform of the form
       y = vec_to_tri(x)
    x is the 'packed' version of shape num_matrices x (N**2 + N)/2
    y is the 'unpacked' version of shape num_matrices x N x N.
    :param N: the size of the final lower triangular matrices.
    :param num_matrices: Number of matrices to be stored.
    :param squeeze: If num_matrices == 1, drop the redundant axis.
    :raises ValueError: squeezing is impossible when num_matrices > 1.
    """

    def __init__(self, N, num_matrices=1, squeeze=False):
        """
        Create an instance of LowerTriangular transform.
        """
        self.N = N
        self.num_matrices = num_matrices  # We need to store this for reconstruction.
        self.squeeze = squeeze

        if self.squeeze and (num_matrices != 1):
            raise ValueError("cannot squeeze matrices unless num_matrices is 1.")

    def forward(self, x):
        """
        Transforms from the packed to unpacked representations (numpy)
        
        :param x: packed numpy array. Must have shape `self.num_matrices x triangular_number
        :return: Reconstructed numpy array y of shape self.num_matrices x N x N
        """
        fwd = np.zeros((self.num_matrices, self.N, self.N), settings.float_type)
        indices = np.tril_indices(self.N, 0)
        z = np.zeros(len(indices[0])).astype(int)
        for i in range(self.num_matrices):
            fwd[(z + i,) + indices] = x[i, :]
        return fwd.squeeze(axis=0) if self.squeeze else fwd

    def backward(self, y):
        """
        Transforms a series of triangular matrices y to the packed representation x (numpy)
        
        :param y: unpacked numpy array y, shape self.num_matrices x N x N
        :return: packed numpy array, x, shape self.num_matrices x triangular number
        """
        if self.squeeze:
            y = y[None, :, :]
        ind = np.tril_indices(self.N)
        return np.vstack([y_i[ind] for y_i in y])

    def log_jacobian_tensor(self, x):
        """
        This function has a jacobian of one, since it is simply an identity mapping (with some packing/unpacking)
        """
        return tf.zeros((1,), settings.float_type)

    def __str__(self):
        return "LoTri->vec"
