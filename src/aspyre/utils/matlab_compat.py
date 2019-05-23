"""
Functions for compatibility with MATLAB behavior.
At some point when the package is full validated against MatLab, the 'order' arguments in the functions here
can be changed to 'C', and subsequently, this package deprecated altogether (i.e. the reshape/flatten methods used
directly by the caller).
"""


def m_reshape(x, new_shape):
    # This is a somewhat round-about way of saying:
    #   return x.reshape(new_shape, order='F')
    # We follow this approach since numba/cupy don't support the 'order'
    # argument, and we may want to use those decorators in the future
    # Note that flattening is required before reshaping, because
    if isinstance(new_shape, tuple):
        return m_flatten(x).reshape(new_shape[::-1]).T
    else:
        return x


def m_flatten(x):
    # This is a somewhat round-about way of saying:
    #   return x.flatten(order='F')
    # We follow this approach since numba/cupy don't support the 'order'
    # argument, and we may want to use those decorators in the future
    return x.T.flatten()
