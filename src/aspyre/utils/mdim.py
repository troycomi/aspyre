import numpy as np
from scipy.linalg import eigh

from aspyre.utils.matlab_compat import m_reshape


def eigs(A, k):
    """
    Multidimensional partial eigendecomposition
    :param A: An array of size `sig_sz`-by-`sig_sz`, where `sig_sz` is a size containing d dimensions.
        The array represents a matrix with d indices for its rows and columns.
    :param k: The number of eigenvalues and eigenvectors to calculate (default 6).
    :return: A 2-tuple of values
        V: An array of eigenvectors of size `sig_sz`-by-k.
        D: A matrix of size k-by-k containing the corresponding eigenvalues in the diagonals.
    """
    sig_sz = A.shape[:int(A.ndim/2)]
    sig_len = np.prod(sig_sz)
    A = m_reshape(A, (sig_len, sig_len))

    dtype = A.dtype
    w, v = eigh(A.astype('float64'), eigvals=(sig_len-1-k+1, sig_len-1))

    # Arrange in descending order (flip column order in eigenvector matrix) and typecast to proper type
    w = w[::-1].astype(dtype)
    v = np.fliplr(v)

    v = m_reshape(v, sig_sz + (k,)).astype(dtype)

    return v, np.diag(w)
