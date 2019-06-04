import numpy as np
import numpy.linalg.norm as norm

from numpy.linalg import solve
from scipy.linalg import block_diag

def blk_diag_zeros(blk_partition, precision=None):
    if precision is None:
        precision = 'double'
    blk_diag = []
    for blk_sz in blk_partition:
        blk_diag.append(np.zeros(blk_sz,dtype=precision))
    return blk_diag

def blk_diag_eye(blk_partition, precision=None):
    if precision is None:
        precision = 'double'
    blk_diag = []
    for blk_sz in blk_partition:
        blk_diag.append(np.ones(blk_sz,dtype=precision))
    return blk_diag

def blk_diag_partition(blk_diag):
    # blk_diag = blk_diag[:]
    blk_partition = []
    for mat in blk_diag:
        blk_partition.append(np.shape(mat))
    return blk_partition

def blk_diag_add(blk_diag_a, blk_diag_b):
    #blk_diag_a = blk_diag_a[:]
    #blk_diag_b = blk_diag_b[:]
    if len(blk_diag_a) != len(blk_diag_b):
        raise RuntimeError('Dimensions of two block diagonal matrices are not equal!')
    elif np.allclose([np.size(mat_a) for mat_a in blk_diag_a], [np.size(mat_b) for mat_b in blk_diag_b]):
    blk_diag_c = []
    for i in range(0, len(blk_diag_a)):
        mat_c = blk_diag_a[i] + blk_diag_b[i]
        blk_diag_c.append(mat_c)
    else:
        raise RuntimeError('Elements of two block diagonal matrices have not the same size!')
    return blk_diag_c

def blk_diag_minus(blk_diag_a, blk_diag_b):
    #blk_diag_a = blk_diag_a[:]
    #blk_diag_b = blk_diag_b[:]
    if len(blk_diag_a) != len(blk_diag_b):
        raise RuntimeError('Dimensions of two block diagonal matrices are not equal!')
    elif np.allclose([np.size(mat_a) for mat_a in blk_diag_a], [np.size(mat_b) for mat_b in blk_diag_b]):
        blk_diag_c = []
        for i in range(0,len(blk_diag_a)):
            mat_c = blk_diag_c[i] - blk_diag_c[i]
            blk_diag_c.append(mat_c)
    else:
        raise RuntimeError('Elements of two block diagonal matrices have not the same size!')
    return blk_diag_c

def blk_diag_apply(blk_diag, x):
    #blk_diag = blk_diag[:]
    cols=np.array([np.size(mat, 1) for mat in blk_diag])

    if np.sum(cols) != np.size(x, 0) :
        raise RuntimeError('Sizes of matrix `blk_diag` and `x` are not compatible.')
    rows=np.array([np.size(x,1)])
    cellarray=Cell2D(cols, rows, dtype=x.dtype)
    x = cellarray.mat2cell(x, cols, rows)
    y = []
    for i in range(0, len(blk_diag)):
        mat=blk_diag[i] @ x[i]
        y.append(mat)
    y=np.concatenate(y,axis=0)
    return y

def blk_diag_mult(blk_diag_a, blk_diag_b):
    #blk_diag_a = blk_diag_a[:]
    #blk_diag_b = blk_diag_b[:]

    if blk_diag_isnumeric(blk_diag_a):
        blk_diag_c = blk_diag_mult(blk_diag_b, blk_diag_a)
        return blk_diag_c

    if blk_diag_isnumeric(blk_diag_a):
        blk_diag_c = []
        for i in range(0, len(blk_diag_b)):
            mat = blk_diag_a * blk_diag_b[i]
            blkc_diag_c.append(mat)
        return blk_diag_c

    if len(blk_diag_a) != len(blk_diag_b):
        raise RuntimeError('Dimensions of two block diagonal matrices are not equal!')
    elif np.allclose([np.size(mat_a, 1) for mat_a in blk_diag_a], [np.size(mat_b, 0) for mat_b in blk_diag_b]):
        blk_diag_c = []
        for i in range(0, len(blk_diag_a)):
            mat_c=blk_diag_a[i] @ blk_diag_b[i]
            blk_diag_c.append(mat_c)
            return blk_diag_c
    else:
        raise RuntimeError('Block diagonal matrix sizes are incompatible.!')

def blk_diag_norm(blk_diag):
    maxvalues=[]
    for i in range(0, len(blk_diag)):
        maxvalues.append(norm(blk_diag[i]))
    max_results = max(maxvalues)
    return max_results

def blk_diag_solve(blk_diag, y):
    #blk_diag = blk_diag[:]
    rows=np.array([np.size(mat_a, 0) for mat_a in blk_diag])
    if sum(rows) != np.size(y, 0):
        raise RuntimeError('Sizes of matrix `blk_diag` and `y` are not compatible.')
    cols=np.array([np.size(y, 1)])
    cellarray=Cell2D(rows, cols, dtype=y.dtype)
    y = cellarray.mat2cell(y, rows, cols)
    x = []
    for i in range(0,len(blk_diag)):
        x.append(solve(blk_diag[i], y[i]))
    x = np.concatenate(x, axis=0)
    return x

def blk_diag_to_mat(blk_diag):
    mat = block_diag(blk_diag)
    return mat

def mat_to_blk_diag(mat, blk_partition):
    rows = blk_partition[:, 0]
    cols = blk_partition[:, 1]
    cellarray=Cell2D(rows, cols, dtype=mat.dtype)
    blk_diag = cellarray.mat2blk_diag(mat, rows, cols)
    return blk_diag

def blk_diag_transpose(blk_diag):
    blk_diag_t = []
    for i in range(0, len(blk_diag)):
        blk_diag_t.append(blk_diag[i].T)
    return blk_diag_t

def blk_diag_compatible(blk_diag_a, blk_diag_b):
    if len(blk_diag_a) != len(blk_diag_b):
        return False
    elif np.allclose([np.size(mat_a, 1) for mat_a in blk_diag_a], [np.size(mat_b, 0) for mat_b in blk_diag_b]):
        return True
    else:
        return False

def blk_diag_isnumeric(x):
    try:
        return 0 == x*0
    except:
        return False