import math
import numpy as np
from numpy.linalg import qr, solve
from scipy.special import erfinv
from scipy.fftpack import ifftshift, ifft2, fftshift, fft2

from aspyre.utils.matlab_compat import m_reshape


# A list of random states, used as a stack
random_states = []

SQRT2 = np.sqrt(2)
SQRT2_R = 1/SQRT2


def randi(i_max, size, seed=None):
    """
    A MATLAB compatible randi implementation that returns numbers from a discrete uniform distribution.
    While a direct use of np.random.choice would be convenient, this doesn't seem to return results
    identical to MATLAB.

    :param iMax: TODO
    :param size: size of the resulting np array
    :param seed: Random seed to use (None to apply no seed)
    :return: A np array
    """
    with Random(seed):
        return np.ceil(i_max * np.random.random(size=size)).astype('int')


def randn(*args, **kwargs):
    """
    Calls rand and applies inverse transform sampling to the output.
    """
    seed = None
    if 'seed' in kwargs:
        seed = kwargs.pop('seed')

    with Random(seed):
        uniform = np.random.rand(*args, **kwargs)
        result = SQRT2 * erfinv(2 * uniform - 1)
        # TODO: Rearranging elements to get consistent behavior with MATLAB 'randn2'
        result = m_reshape(result.flatten(), args)
        return result


def rand(size, seed=None):
    with Random(seed):
        return m_reshape(np.random.random(np.prod(size)), size)


class Random:
    def __init__(self, seed=None):
        self.seed = seed

    def __enter__(self):
        if self.seed is not None:
            # Push current state on stack
            random_states.append(np.random.get_state())

            seed = self.seed
            # 5489 is the default seed used by MATLAB for seed 0 !
            if seed == 0:
                seed = 5489

            new_state = np.random.RandomState(seed)
            np.random.set_state(new_state.get_state())

    def __exit__(self, *args):
        if self.seed is not None:
            np.random.set_state(random_states.pop())


def voltage_to_wavelength(voltage):
    """
    Convert from electron voltage to wavelength.
    :param voltage: float, The electron voltage in kV.
    :return: float, The electron wavelength in nm.
    """
    return 12.2643247 / math.sqrt(voltage*1e3 + 0.978466*voltage**2)


def wavelength_to_voltage(wavelength):
    """
    Convert from electron voltage to wavelength.
    :param wavelength: float, The electron wavelength in nm.
    :return: float, The electron voltage in kV.
    """
    return (-1e3 + math.sqrt(1e6 + 4 * 12.2643247**2 * 0.978466 / wavelength**2)) / (2 * 0.978466)


def cart2pol(x, y):
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return theta, r


def cart2sph(x, y, z):
    """
    Transform cartesian coordinates to spherical. All input arguments must be the same shape.

    :param x: X-values of input co-ordinates.
    :param y: Y-values of input co-ordinates.
    :param z: Z-values of input co-ordinates.
    :return: A 3-tuple of values, all of the same shape as the inputs.
        (<azimuth>, <elevation>, <radius>)
    azimuth and elevation are returned in radians.

    This function is equivalent to MATLAB's cart2sph function.
    """
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r


def grid_2d(n):
    grid_1d = np.ceil(np.arange(-n/2, n/2)) / (n/2)
    x, y = np.meshgrid(grid_1d, grid_1d, indexing='ij')
    phi, r = cart2pol(x, y)

    return {
        'x': x,
        'y': y,
        'phi': phi,
        'r': r
    }


def grid_3d(n):
    grid_1d = np.ceil(np.arange(-n/2, n/2)) / (n/2)
    x, y, z = np.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')
    phi, theta, r = cart2sph(x, y, z)

    # TODO: Should this theta adjustment be moved inside cart2sph?
    theta = np.pi/2 - theta

    return {
        'x': x,
        'y': y,
        'z': z,
        'phi': phi,
        'theta': theta,
        'r': r
    }


def angles_to_rots(angles):
    n_angles = angles.shape[-1]
    rots = np.zeros(shape=(3, 3, n_angles))

    for i in range(n_angles):
        rots[:, :, i] = erot(angles[:, i])
    return rots


def erot(angles):
    return zrot(angles[0]) @ yrot(angles[1]) @ zrot(angles[2])


def zrot(theta):
    sin, cos = np.sin(theta), np.cos(theta)
    return np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])


def yrot(theta):
    sin, cos = np.sin(theta), np.cos(theta)
    return np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])


def unroll_dim(X, dim):
    # TODO: dim is still 1-indexed like in MATLAB to reduce headaches for now
    # TODO: unroll/roll are great candidates for a context manager since they're always used in conjunction.
    dim = dim - 1
    old_shape = X.shape
    new_shape = old_shape[:dim]
    if dim < len(old_shape):
        new_shape += (-1, )
    if old_shape != new_shape:
        Y = m_reshape(X, new_shape)
    else:
        Y = X
    removed_dims = old_shape[dim:]

    return Y, removed_dims


def roll_dim(X, dim):
    # TODO: dim is still 1-indexed like in MATLAB to reduce headaches for now
    if len(dim) > 0:
        old_shape = X.shape
        new_shape = old_shape[:-1] + dim
        Y = m_reshape(X, new_shape)
        return Y
    else:
        return X


def centered_ifft2(x):
    """
    Calculate a centered, two-dimensional inverse FFT
    :param x: The two-dimensional signal to be transformed.
        The inverse FFT is only applied along the first two dimensions.
    :return: The centered inverse Fourier transform of x.
    """
    x = ifftshift(ifftshift(x, 0), 1)
    x = ifft2(x, axes=(0, 1))
    x = fftshift(fftshift(x, 0), 1)
    return x


def centered_fft2(x):
    x = ifftshift(ifftshift(x, 0), 1)
    x = fft2(x, axes=(0, 1))
    x = fftshift(fftshift(x, 0), 1)
    return x


def mdim_ifftshift(x, dims=None):
    """
    Multi-dimensional FFT unshift
    :param x: The array to be unshifted.
    :param dims: An array of dimension indices along which the unshift should occur.
        If None, the unshift is performed along all dimensions.
    :return: The x array unshifted along the desired dimensions.
    """
    if dims is None:
        dims = range(0, x.ndim)
    for dim in dims:
        x = ifftshift(x, dim)
    return x


def mdim_fftshift(x, dims=None):
    """
    Multi-dimensional FFT shift

    :param x: The array to be shifted.
    :param dims: An array of dimension indices along which the shift should occur.
        If None, the shift is performed along all dimensions.
    :return: The x array shifted along the desired dimensions.
    """
    if dims is None:
        dims = range(0, x.ndim)
    for dim in dims:
        x = fftshift(x, dim)
    return x


def im_to_vec(im):
    """
    Roll up images into vectors
    :param im: An N-by-N-by-... array.
    :return: An N^2-by-... array.
    """
    shape = im.shape
    ensure(im.ndim >= 2, "Array should have at least 2 dimensions")
    ensure(shape[0] == shape[1], "Array should have first 2 dimensions identical")

    return m_reshape(im, (shape[0]**2,) + (shape[2:]))


def vol_to_vec(X):
    """
    Roll up volumes into vectors
    :param X: N-by-N-by-N-by-... array.
    :return: An N^3-by-... array.
    """
    shape = X.shape
    ensure(X.ndim >= 3, "Array should have at least 3 dimensions")
    ensure(shape[0] == shape[1] == shape[2], "Array should have first 3 dimensions identical")

    return m_reshape(X, (shape[0]**3,) + (shape[3:]))


def vec_to_im(X):
    """
    Unroll vectors to images
    :param X: N^2-by-... array.
    :return: An N-by-N-by-... array.
    """
    shape = X.shape
    N = round(shape[0]**(1/2))
    ensure(N**2 == shape[0], "First dimension of X must be square")

    return m_reshape(X, (N, N) + (shape[1:]))


def vec_to_vol(X):
    """
    Unroll vectors to volumes
    :param X: N^3-by-... array.
    :return: An N-by-N-by-N-by-... array.
    """
    shape = X.shape
    N = round(shape[0]**(1/3))
    ensure(N**3 == shape[0], "First dimension of X must be cubic")

    return m_reshape(X, (N, N, N) + (shape[1:]))


def vecmat_to_volmat(X):
    """
    Roll up vector matrices into volume matrices
    :param X: A vector matrix of size L1^3-by-L2^3-by-...
    :return: A volume "matrix" of size L1-by-L1-by-L1-by-L2-by-L2-by-L2-by-...
    """
    # TODO: Use context manager?
    shape = X.shape
    ensure(X.ndim >= 2, "Array should have at least 2 dimensions")

    L1 = round(shape[0]**(1/3))
    L2 = round(shape[1]**(1/3))

    ensure(L1**3 == shape[0], "First dimension of X must be cubic")
    ensure(L2**3 == shape[1], "Second dimension of X must be cubic")

    return m_reshape(X, (L1, L1, L1, L2, L2, L2) + (shape[2:]))


def volmat_to_vecmat(X):
    """
    Unroll volume matrices to vector matrices
    :param X: A volume "matrix" of size L1-by-L1-by-L1-by-L2-by-L2-by-L2-by-...
    :return: A vector matrix of size L1^3-by-L2^3-by-...
    """
    # TODO: Use context manager?
    shape = X.shape
    ensure(X.ndim >= 6, "Array should have at least 6 dimensions")
    ensure(shape[0] == shape[1] == shape[2], "Dimensions 1-3 should be identical")
    ensure(shape[3] == shape[4] == shape[5], "Dimensions 4-6 should be identical")

    l1 = shape[0]
    l2 = shape[3]

    return m_reshape(X, (l1**3, l2**3) + (shape[6:]))


def mdim_mat_fun_conj(X, d1, d2, f):
    """
    Conjugate a multidimensional matrix using a linear mapping
    :param X: An N_1-by-...-by-N_d1-by-N_1...-by-N_d1-by-... array, with the first 2*d1 dimensions corresponding to
        matrices with columns and rows of dimension d1.
    :param d1: The dimension of the input matrix X
    :param d2: The dimension of the output matrix Y
    :param f: A function handle of a linear map that takes an array of size N_1-by-...-by-N_d1-by-... and returns an
        array of size M_1-by-...-by-M_d2-by-... .
    :return: An array of size M_1-by-...-by-M_d2-by-M_1-by-...-by-M_d2-by-... resulting from applying fun to the rows
        and columns of the multidimensional matrix X.

    TODO: Very complicated to wrap head around this one!
    """
    X, sz_roll = unroll_dim(X, 2*d1 + 1)
    X = f(X)

    # Swap the first d2 axes block of X with the next d1 axes block
    X = np.moveaxis(X, list(range(d1+d2)), list(range(d1, d1+d2)) + list(range(d1)))

    X = np.conj(X)
    X = f(X)

    # Swap the first d2 axes block of X with the next d2 axes block
    X = np.moveaxis(X, list(range(2*d2)), list(range(d2, 2*d2)) + list(range(d2)))

    X = np.conj(X)
    X = roll_dim(X, sz_roll)

    return X


def symmat_to_vec_iso(mat):
    """
    Isometrically maps a symmetric matrix to a packed vector
    :param mat: An array of size N-by-N-by-... where the first two dimensions constitute symmetric or Hermitian
        matrices.
    :return: A vector of size N*(N+1)/2-by-... consisting of the lower triangular part of each matrix, reweighted so
        that the Frobenius inner product is mapped to the Euclidean inner product.
    """
    mat, sz_roll = unroll_dim(mat, 3)
    N = mat.shape[0]
    mat = mat_to_vec(mat)
    mat[np.arange(0, N ** 2, N + 1)] *= SQRT2_R
    mat *= SQRT2
    mat = vec_to_mat(mat)
    mat = roll_dim(mat, sz_roll)
    vec = symmat_to_vec(mat)

    return vec


def vec_to_symmat_iso(vec):
    """
    Isometrically map packed vector to symmetric matrix
    :param vec: A vector of size N*(N+1)/2-by-... describing a symmetric (or Hermitian) matrix.
    :return: An array of size N-by-N-by-... which indexes symmetric/Hermitian matrices that occupy the first two
        dimensions. The lower triangular parts of these matrices consists of the corresponding vectors in vec,
        reweighted so that the Euclidean inner product maps to the Frobenius inner product.
    """
    mat = vec_to_symmat(vec)
    mat, sz_roll = unroll_dim(mat, 3)
    N = mat.shape[0]
    mat = mat_to_vec(mat)
    mat[np.arange(0, N ** 2, N + 1)] *= SQRT2
    mat *= SQRT2_R
    mat = vec_to_mat(mat)
    mat = roll_dim(mat, sz_roll)
    return mat


def symmat_to_vec(mat):
    """
    Packs a symmetric matrix into a lower triangular vector
    :param mat: An array of size N-by-N-by-... where the first two dimensions constitute symmetric or
        Hermitian matrices.
    :return: A vector of size N*(N+1)/2-by-... consisting of the lower triangular part of each matrix.

    Note that a lot of acrobatics happening here (swapaxes/triu instead of tril etc.) are so that we can get
    column-major ordering of elements (to get behavior consistent with MATLAB), since masking in numpy only returns
    data in row-major order.
    """
    N = mat.shape[0]
    ensure(mat.shape[1] == N, "Matrix must be square")

    mat, sz_roll = unroll_dim(mat, 3)
    triu_indices = np.triu_indices(N)
    vec = mat.swapaxes(0, 1)[triu_indices]
    vec = roll_dim(vec, sz_roll)

    return vec


def vec_to_symmat(vec):
    """
    Convert packed lower triangular vector to symmetric matrix
    :param vec: A vector of size N*(N+1)/2-by-... describing a symmetric (or Hermitian) matrix.
    :return: An array of size N-by-N-by-... which indexes symmetric/Hermitian matrices that occupy the first two
        dimensions. The lower triangular parts of these matrices consists of the corresponding vectors in vec.
    """
    # TODO: Handle complex values in vec
    if np.iscomplex(vec).any():
        raise NotImplementedError('Coming soon')

    # M represents N(N+1)/2
    M = vec.shape[0]
    N = int(round(np.sqrt(2 * M + 0.25) - 0.5))
    ensure((M == 0.5*N*(N+1)) and N != 0, "Vector must be of size N*(N+1)/2 for some N>0.")

    vec, sz_roll = unroll_dim(vec, 2)
    I = np.empty((N, N))
    i_upper = np.triu_indices_from(I)
    I[i_upper] = np.arange(M)    # Incrementally populate upper triangle in row major order
    I.T[i_upper] = I[i_upper]  # Copy to lower triangle

    mat = vec[I.flatten('F').astype('int')]
    mat = m_reshape(mat, (N, N) + mat.shape[1:])
    mat = roll_dim(mat, sz_roll)

    return mat


def mat_to_vec(mat, is_symmat=False):
    """
    Converts a matrix into vectorized form
    :param mat: An array of size N-by-N-by-... containing the matrices to be vectorized.
    :param is_symmat: Specifies whether the matrices are symmetric/Hermitian, in which case they are stored in packed
        form using symmat_to_vec (default False).
    :return: The vectorized form of the matrices, with dimension N^2-by-... or N*(N+1)/2-by-... depending on the value
        of is_symmat.
    """
    if not is_symmat:
        sz = mat.shape
        N = sz[0]
        ensure(sz[1] == N, "Matrix must be square")
        return m_reshape(mat, (N**2,) + sz[2:])
    else:
        return symmat_to_vec(mat)


def vec_to_mat(vec, is_symmat=False):
    """
    Converts a vectorized matrix into a matrix
    :param vec: The vectorized representations. If the matrix is non-symmetric, this array has the dimensions
        N^2-by-..., but if the matrix is symmetric, the dimensions are N*(N+1)/2-by-... .
    :param is_symmat: True if the vectors represent symmetric matrices (default False)
    :return: The array of size N-by-N-by-... representing the matrices.
    """
    if not is_symmat:
        sz = vec.shape
        N = int(round(np.sqrt(sz[0])))
        ensure(sz[0] == N**2, "Vector must represent square matrix.")
        return m_reshape(vec, (N, N) + sz[1:])
    else:
        return vec_to_symmat(vec)


def make_symmat(A):
    """
    Symmetrize a matrix
    :param A: A matrix.
    :return: The Hermitian matrix (A+A')/2.
    """
    return 0.5 * (A + A.T)


def src_wiener_coords(sim, mean_vol, eig_vols, lambdas=None, noise_var=0, batch_size=512):
    """
    Calculate coordinates using Wiener filter
    :param sim: A simulation object containing the images whose coordinates we want.
    :param mean_vol: The mean volume of the source in an L-by-L-by-L array.
    :param eig_vols: The eigenvolumes of the source in an L-by-L-by-L-by-K array.
    :param lambdas: The eigenvalues in a K-by-K diagonal matrix (default `eye(K)`).
    :param noise_var: The variance of the noise in the images (default 0).
    :param batch_size: The size of the batches in which to compute the coordinates (default 512).
    :return: A K-by-`src.n` array of coordinates corresponding to the Wiener filter coordinates of each image in sim.

    The coordinates are obtained by the formula
        alpha_s = eig_vols^T H_s ( y_s - P_s mean_vol ) ,

    where P_s is the forward image mapping and y_s is the sth image,
        H_s = Sigma * P_s^T ( P_s Sigma P_s^T + noise_var I )^(-1) ,

    and Sigma is the covariance matrix eig_vols * lambdas * eig_vols^T.
    Note that when noise_var is zero, this reduces to the projecting y_s onto the span of P_s eig_vols.

    # TODO: Find a better place for this functionality other than in utils
    """
    k = eig_vols.shape[-1]
    if lambdas is None:
        lambdas = np.eye(k)

    coords = np.zeros((k, sim.n))
    covar_noise = noise_var * np.eye(k)

    for i in range(0, sim.n, batch_size):
        ims = sim.images(i, batch_size)
        batch_n = ims.shape[-1]
        ims -= sim.vol_forward(mean_vol, i, batch_n)

        Qs, Rs = qr_vols_forward(sim, i, batch_n, eig_vols, k)

        Q_vecs = mat_to_vec(Qs)
        im_vecs = mat_to_vec(ims)

        for j in range(batch_n):
            im_coords = Q_vecs[:, :, j].T @ im_vecs[:, j]
            covar_im = (Rs[:, :, j] @ lambdas @ Rs[:, :, j].T) + covar_noise
            xx = solve(covar_im, im_coords)
            im_coords = lambdas @ Rs[:, :, j].T @ xx
            coords[:, i+j] = im_coords

    return coords


def qr_vols_forward(sim, s, n, vols, k):
    """
    TODO: Write docstring
    TODO: Find a better place for this!
    :param sim:
    :param s:
    :param n:
    :param vols:
    :param k:
    :return:
    """
    ims = np.zeros((sim.L, sim.L, n, k), dtype=vols.dtype)
    for ell in range(k):
        ims[:, :, :, ell] = sim.vol_forward(vols[:, :, :, ell], s, n)

    ims = np.swapaxes(ims, 2, 3)
    Q_vecs = np.zeros((sim.L**2, k, n), dtype=vols.dtype)
    Rs = np.zeros((k, k, n), dtype=vols.dtype)

    im_vecs = mat_to_vec(ims)
    for i in range(n):
        Q_vecs[:, :, i], Rs[:, :, i] = qr(im_vecs[:, :, i])
    Qs = vec_to_mat(Q_vecs)

    return Qs, Rs


def anorm(x, axes=None):
    """
    Calculate array norm along given axes
    :param x: An array of arbitrary size and shape.
    :param axes: The axis along which to compute the norm. If None, the norm is calculated along all axes.
    :return: The Euclidean (l^2) norm of x along specified axes.
    """
    if axes is None:
        axes = range(x.ndim)
    return np.sqrt(ainner(x, x, axes))


def acorr(x, y, axes=None):
    """
    Calculate array correlation along given axes
    :param x: An array of arbitrary shape
    :param y: An array of same shape as x
    :param axes: The axis along which to compute the correlation. If None, the correlation is calculated along all axes.
    :return: The correlation of x along specified axes.
    """
    ensure(x.shape == y.shape, "The shapes of the inputs have to match")

    if axes is None:
        axes = range(x.ndim)
    return ainner(x, y, axes) / (anorm(x, axes) * anorm(y, axes))


def ainner(x, y, axes=None):
    """
    Calculate array inner product along given axes
    :param x: An array of arbitrary shape
    :param y: An array of same shape as x
    :param axes: The axis along which to compute the inner product. If None, the product is calculated along all axes.
    :return:
    """
    ensure(x.shape == y.shape, "The shapes of the inputs have to match")

    if axes is None:
        axes = range(x.ndim)
    return np.tensordot(x, y, axes=(axes, axes))


def ensure(cond, error_message=None):
    """
    assert statements in Python are sometimes optimized away by the compiler, and are for internal testing purposes.
    For user-facing assertions, we use this simple wrapper to ensure conditions are met at relevant parts of the code.

    :param cond: Condition to be ensured
    :param error_message: An optional error message if condition is not met
    :return: If condition is met, returns nothing, otherwise raises AssertionError
    """
    if not cond:
        raise AssertionError(error_message)
