"""
This is an implementation of Algorithm 3 in the paper:
Dong, S. & Sebag, M. (2022). From graphs to DAGs: a low-complexity model and a scalable algorithm.

Contact: shuyu.dong@m4x.org
"""

import numpy as np

DTYPE = np.double 

cimport cython
@cython.boundscheck(False)  # 
@cython.wraparound(False)   # 

def spmaskmatmul(double [:, :] L, double [:, :] R, unsigned int[:] I, unsigned int[:] J):

    cdef Py_ssize_t dim_y = L.shape[1]
    cdef Py_ssize_t dim_res = I.shape[0]

    assert tuple(L.shape) == tuple(R.shape)

    res         = np.zeros((dim_res,), dtype=DTYPE)
    cdef double[:] res_view = res 

    cdef Py_ssize_t k

    for s in range(dim_res):
        res_view[s] = 0 
        for k in range(dim_y):
            res_view[s] += L[I[s],k] * R[J[s], k] 
    return res

def spmaskmatmul_odot(double [:, :] L, double [:, :] R, unsigned int[:] I, unsigned int[:] J):

    cdef Py_ssize_t dim_y = L.shape[1]
    cdef Py_ssize_t dim_res = I.shape[0]

    assert tuple(L.shape) == tuple(R.shape)

    res         = np.zeros((dim_res,), dtype=DTYPE)
    cdef double[:] res_view = res 

    cdef Py_ssize_t k

    for s in range(dim_res):
        res_view[s] = 0 
        for k in range(dim_y):
            res_view[s] += L[I[s],k] * R[J[s], k] 
    return res, res * res, 2*res 

def spmaskmatmul_abs(double [:, :] L, double [:, :] R, unsigned int[:] I, unsigned int[:] J):

    cdef Py_ssize_t dim_y = L.shape[1]
    cdef Py_ssize_t dim_res = I.shape[0]

    assert tuple(L.shape) == tuple(R.shape)

    res         = np.zeros((dim_res,), dtype=DTYPE)
    cdef double[:] res_view = res 

    cdef Py_ssize_t k

    for s in range(dim_res):
        res_view[s] = 0 
        for k in range(dim_y):
            res_view[s] += L[I[s],k] * R[J[s], k] 
    return res, np.absolute(res), np.sign(res)


def spmaskmatmul_f_sktch(double [:, :] M, double [:, :] ZM, unsigned int[:] I, unsigned int[:] J, unsigned int[:] SUBS):
    # ! DEPRECATED ! 
    # SUBS is an index set comprised of index samples from [1,2,.., N] 
    # The computation of MMt(i,j) is realized by the sketching M(i,SUBS) * M(j,SUBS)'
    cdef Py_ssize_t N = M.shape[1]
    cdef Py_ssize_t m = I.shape[0]

    assert M.shape[0] == ZM.shape[0]

    mmt         = np.zeros((m,), dtype=DTYPE)
    zmmt         = np.zeros((m,), dtype=DTYPE)
    cdef double[:] mmt_view = mmt 
    cdef double[:] zmmt_view = zmmt  

    cdef Py_ssize_t k

    for s in range(m):
        mmt_view[s] = 0 
        zmmt_view[s] = 0 
        for k in SUBS:
            mmt_view[s] += M[I[s],k] * M[k,J[s]] 
            zmmt_view[s] += ZM[I[s],k] * M[k,J[s]] 
    return mmt - zmmt, mmt, zmmt  


def spmaskmatmul_f(double [:, :] M, double [:, :] ZM, unsigned int[:] I, unsigned int[:] J):

    cdef Py_ssize_t N = M.shape[1]
    cdef Py_ssize_t m = I.shape[0]

    assert M.shape[0] == ZM.shape[0]

    MMt         = np.zeros((m,), dtype=DTYPE)
    ZMMt         = np.zeros((m,), dtype=DTYPE)
    cdef double[:] mmt_view = MMt 
    cdef double[:] zmmt_view = ZMMt  

    cdef Py_ssize_t k

    for s in range(m):
        mmt_view[s] = 0 
        zmmt_view[s] = 0 
        for k in range(N):
            mmt_view[s] += M[I[s],k] * M[J[s],k] 
            zmmt_view[s] += ZM[I[s],k] * M[J[s],k] 
    return ZMMt - MMt, MMt, ZMMt  

