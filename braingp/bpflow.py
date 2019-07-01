""" Contains general functions for the processing of data in the relevant files 
may be contributed to the main repo """

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


"""Class kernel : def K(self, X, X2=None):
    X, X2 = self._slice(X, X2)
    X = tf.cast(X[:, 0], tf.int32)
    if X2 is None:
        X2 = X
    else:
        X2 = tf.cast(X2[:, 0], tf.int32)
    B = tf.matmul(self.W, self.W, transpose_b=True) + tf.matrix_diag(self.kappa)
    return tf.gather(tf.transpose(tf.gather(B, X2)), X)

@autoflow((settings.float_type, [None, None]))
    def compute_K_symm(self, X):
        return self.K(X) """

def tensordot(a, b, axes=2):
    # code adapted from numpy
    try:
        iter(axes)
    except Exception:
        axes_a = list(range(-axes, 0))
        axes_b = list(range(0, axes))
    else:
        axes_a, axes_b = axes
    try:
        na = len(axes_a)
        axes_a = list(axes_a)
    except TypeError:
        axes_a = [axes_a]
        na = 1
    try:
        nb = len(axes_b)
        axes_b = list(axes_b)
    except TypeError:
        axes_b = [axes_b]
        nb = 1
    
    # uncomment in pytorch >= 0.5
    # a, b = torch.as_tensor(a), torch.as_tensor(b)
    as_ = a.shape
    nda = a.dim()
    bs = b.shape
    ndb = b.dim()
    equal = True
    if na != nb:
        equal = False
    else:
        for k in range(na):
            if as_[axes_a[k]] != bs[axes_b[k]]:
                equal = False
                break
            if axes_a[k] < 0:
                axes_a[k] += nda
            if axes_b[k] < 0:
                axes_b[k] += ndb
    if not equal:
        raise ValueError("shape-mismatch for sum")

    # Move the axes to sum over to the end of "a"
    # and to the front of "b"
    notin = [k for k in range(nda) if k not in axes_a]
    newaxes_a = notin + axes_a
    N2 = 1
    for axis in axes_a:
        N2 *= as_[axis]
    newshape_a = (int(np.multiply.reduce([as_[ax] for ax in notin])), N2)
    olda = [as_[axis] for axis in notin]

    notin = [k for k in range(ndb) if k not in axes_b]
    newaxes_b = axes_b + notin
    N2 = 1
    for axis in axes_b:
        N2 *= bs[axis]
    newshape_b = (N2, int(np.multiply.reduce([bs[ax] for ax in notin])))
    oldb = [bs[axis] for axis in notin]

    at = a.permute(newaxes_a).reshape(newshape_a)
    bt = b.permute(newaxes_b).reshape(newshape_b)

    res = at.matmul(bt)
    return res.reshape(olda + oldb)

