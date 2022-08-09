import numpy as np
import cython
import array

@cython.ccall
def sid(p:array.array , q:array.array) -> cython.float:
    p = p + np.spacing(1)
    q = q + np.spacing(1)
    return np.sum(p * np.log(p / q) + q * np.log(q / p))


