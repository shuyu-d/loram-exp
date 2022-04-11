"""Compute the action of a Hadamard product matrix of a sparsified matrix
exponential and a DAG candidate matrix.

This function is adapted from scipy.linalg.expm_multiply, which is based on the
algorithm 3.2 of Al-Mohy and Higham (2011).

This is an implementation of algorithms in the paper:
Dong, S. & Sebag, M. (2022). From graphs to DAGs: a low-complexity model and a scalable algorithm.

Contact: shuyu.dong@m4x.org
"""

import numpy as np

import scipy.linalg
import scipy.sparse.linalg
from scipy.sparse.linalg import aslinearoperator
# from scipy.sparse.sputils import is_pydata_spmatrix
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
import spmaskmatmul
from scipy.sparse.linalg import expm

__all__ = ['splr_expmv']


def _exact_inf_norm(A):
    # A compatibility function which should eventually disappear.
    if scipy.sparse.isspmatrix(A):
        return max(abs(A).sum(axis=1).flat)
    # elif is_pydata_spmatrix(A):
    #     return max(abs(A).sum(axis=1))
    else:
        return np.linalg.norm(A, np.inf)


def _exact_1_norm(A):
    # A compatibility function which should eventually disappear.
    if scipy.sparse.isspmatrix(A):
        return max(abs(A).sum(axis=0).flat)
    # elif is_pydata_spmatrix(A):
    #     return max(abs(A).sum(axis=0))
    else:
        return np.linalg.norm(A, 1)


def _trace(A):
    # # A compatibility function which should eventually disappear.
    # if scipy.sparse.isspmatrix(A):
    #     return A.diagonal().sum()
    # else:
    #     return np.trace(A)
    return A.diagonal().sum()

def _ident_like(A):
    # A compatibility function which should eventually disappear.
    if scipy.sparse.isspmatrix(A):
        return scipy.sparse.construct.eye(A.shape[0], A.shape[1],
                dtype=A.dtype, format=A.format)
    # elif is_pydata_spmatrix(A):
    #     import sparse
    #     return sparse.eye(A.shape[0], A.shape[1], dtype=A.dtype)
    else:
        return np.eye(A.shape[0], A.shape[1], dtype=A.dtype)




def splr_expmv_simple(L, R, I, J, B, t=1.0, balance=False):
    """
    Compute the action of the sparsified matrix exponential of a low-rank
    matrix product, using numpy notation:
        F = expm(Ps((LR')*(LR'))).T * Ps(LR')) B,
    where Ps denotes the sparsification projection depending on the subsampling index set (I, J), * denotes the Hadamard product (elementwise product) of two matrices and the blank space between two matrices denotes the matrix product.

    Parameters
    ----------
    L, R : transposable low-rank factor matrices
        The sparsified product Ps((LR' odot LR')) whose exponential is of interest.
    B : ndarray
        The matrix to be multiplied by the matrix exponential of A.
    t : float
        A time point.
    balance : bool
        Indicates whether or not to apply balancing.

    Returns
    -------
    F : ndarray

    Notes
    -----
    This algorithm is adapted from algorithm 3.2 in Al-Mohy and Higham (2011).

    """
    [spLRt_, A_] = spmaskmatmul.spmaskmatmul(L, R, I, J)
    A = csc_matrix((A_, (I, J)), shape=(R.shape[0], L.shape[0]))
    spLRt = csc_matrix((spLRt_, (I, J)), shape=(L.shape[0], R.shape[0]))
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')
    if A.shape[1] != B.shape[0]:
        raise ValueError('shapes of matrices A {} and B {} are incompatible'
                         .format(A.shape, B.shape))
    ident = _ident_like(A)
    n = A.shape[0]
    if len(B.shape) == 1:
        n0 = 1
    elif len(B.shape) == 2:
        n0 = B.shape[1]
    else:
        raise ValueError('expected B to be like a matrix or a vector')
    tol = 1e-24
    mu = _trace(A) / float(n)
    A_1_norm = _exact_1_norm(A - mu * ident)
    print('A_norm_1 = %.3e'% A_1_norm)
    if t*A_1_norm == 0:
        m_star, s = 0, 1
    else:
        ell = 2
        norm_info = LazyOperatorNormInfo(A, A_1_norm= A_1_norm, ell=ell)
        m_star, s = _fragment_3_1(norm_info, n0, tol, ell=ell)
    print('(m-star, s) = (%i,%i)'% (m_star, s))
    return _splr_expmv_core_b(A, spLRt, B, ident, mu, m_star, s, tol, balance)

def _splr_expmv_core_b(A, spLRt, B, ident, mu, m_star, s, tol=None, balance=False):
    """
    The core iterations
    """
    # F = B  # this line is the major bug that induces the error! Instead F should be (I * spLRt) B, where * denotes the Hadamard product
    Bj = (spLRt.multiply(ident)).dot(B)
    F = Bj
    Aj = ident
    for i in range(s):
        c1 = _exact_inf_norm(Bj)
        for j in range(2*m_star):
            coeff = 1.0 / float(j+1)
            # compute Ps(A (Aj_old))
            Aj = A @ Aj #A.dot(Aj)
            Bj = coeff * (spLRt.multiply(Aj)).dot(B)
            c2 = _exact_inf_norm(Bj)
            F = F + Bj
            if c1 + c2 <= tol * _exact_inf_norm(F):
                break
            c1 = c2
    return F

def splr_expmv_core_d(L,R,I,J, B, t=1.0, tol=None, balance=False):
    """
    The core iterations
    """
    [spLRt_, A_] = spmaskmatmul.spmaskmatmul(R, L, J, I)
    A = csc_matrix((A_, (J, I)), shape=(R.shape[0], L.shape[0]))
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')
    if A.shape[1] != B.shape[0]:
        raise ValueError('shapes of matrices A {} and B {} are incompatible'
                         .format(A.shape, B.shape))
    ident = _ident_like(A)
    n = A.shape[0]
    if len(B.shape) == 1:
        n0 = 1
    elif len(B.shape) == 2:
        n0 = B.shape[1]
    else:
        raise ValueError('expected B to be like a matrix or a vector')
    tol = 1e-24
    mu = _trace(A) / float(n)
    A_1_norm = _exact_1_norm(A - mu * ident)
    print('A_norm_1 = %.3e'% A_1_norm)
    if t*A_1_norm == 0:
        m_star, s = 0, 1
    else:
        ell = 2
        norm_info = LazyOperatorNormInfo(A, A_1_norm= A_1_norm, ell=ell)
        m_star, s = _fragment_3_1(norm_info, n0, tol, ell=ell)
    print('(m-star, s) = (%i,%i)'% (m_star, s))
    [bjo_, _] = spmaskmatmul.spmaskmatmul(B[:,0].reshape((B.shape[0],1)),
            np.ones(B.shape[0]).reshape((B.shape[0],1)), J, I)
    bjoct_ = bjo_ * spLRt_
    Bj = csc_matrix((bjoct_, (J,I)), shape=(R.shape[0],L.shape[0]))
    F = Bj
    for i in range(s):
        c1 = _exact_inf_norm(Bj)
        for j in range(2*m_star):
            coeff = 1.0 / float(j+1)
            Bj = coeff * (A.dot(Bj))
            c2 = _exact_inf_norm(Bj)
            F = F + Bj
            # print(scipy.sparse.isspmatrix(F))
            print('iter j = %i |nf = %.3e |c1 = %.3e | c2 = %.3e'
                    %(j, _exact_inf_norm(F), c1,c2))
            if c1 + c2 <= tol * _exact_inf_norm(F):
                break
            c1 = c2
            Bj = F
    return F.diagonal()


def splr_func_odot(L,R, I,J):
    [_, A_, dA_] = spmaskmatmul.spmaskmatmul_odot(L, R, I, J)
    return A_, dA_


def splr_func_abs(L,R, I,J):
    [_, A_, dA_] = spmaskmatmul.spmaskmatmul_abs(L, R, I, J)
    return A_, dA_

def splr_sigma_odot(L,R, I,J):
    [sp_zvec, A_, dA_] = spmaskmatmul.spmaskmatmul_odot(L, R, I, J)
    return A_, dA_, sp_zvec

def splr_sigma_abs(L,R, I,J):
    [sp_zvec, A_, dA_] = spmaskmatmul.spmaskmatmul_abs(L, R, I, J)
    return A_, dA_, sp_zvec

def splr_expmv_inexact_1b(A, C, B, t=1.0):
    """
    Parameters
    ----------
    A, C: transposable sparse matrix
    B   : ndarray
        The matrix to be multiplied by the matrix exponential of A.

    Returns
    -------
    F : ndarray

    Notes
    -----
    This algorithm is adapted from algorithm 3.2 in Al-Mohy and Higham (2011).

    """
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')
    if A.shape[1] != B.shape[0]:
        raise ValueError('shapes of matrices A {} and B {} are incompatible'
                         .format(A.shape, B.shape))
    ident = _ident_like(A)
    n = A.shape[0]
    if len(B.shape) == 1:
        n0 = 1
    elif len(B.shape) == 2:
        n0 = B.shape[1]
    else:
        raise ValueError('expected B to be like a matrix or a vector')
    u_d = 2**-53
    tol = u_d
    mu = _trace(A) / float(n)
    A = A - mu * ident
    A_1_norm = _exact_1_norm(A)
    # print('A_norm_1 = %.3e'% A_1_norm)
    if t*A_1_norm == 0:
        m_star, s = 0, 1
    else:
        ell = 2
        norm_info = LazyOperatorNormInfo(A, A_1_norm= A_1_norm, ell=ell)
        m_star, s = _fragment_3_1(norm_info, n0, tol, ell=ell)
    # print('(m-star, s) = (%i,%i)'% (m_star, s))
    # Core iterations
    F = (C.multiply(ident)).dot(B)
    B = (C.multiply(A)).dot(B)
    F = F + B
    eta = np.exp(mu / float(s))
    for i in range(s):
        c1 = _exact_inf_norm(B)
        for j in range(m_star):
            coeff = 1.0 / float(s*(j+1))
            B = coeff * A.dot(B)
            c2 = _exact_inf_norm(B)
            F = F + B
            # print('eta = %.2e, iter j = %i |nf = %.3e |c1 = %.3e | c2 = %.3e'
            #         %(eta, j, _exact_inf_norm(F), c1,c2))
            if c1 + c2 <= tol * _exact_inf_norm(F):
                break
            c1 = c2
        F = eta * F
        B = F
    return F

def splr_expmv_inexact1_paired(A, C, B, Bb, t=1.0):
    """
    Compute the action of the sparsified matrix exponential of a low-rank
    matrix product, using numpy notation:
        F = expm(Ps((LR')*(LR'))).T * Ps(LR')) B,
    where Ps denotes the sparsification projection depending on the subsampling index set (I, J), * denotes the Hadamard product (elementwise product) of two matrices and the blank space between two matrices denotes the matrix product.

    Parameters
    ----------
    A, C: transposable sparse matrix
    B   : ndarray
        The matrix to be multiplied by the matrix exponential of A.

    Returns
    -------
    F : ndarray

    Notes
    -----
    This algorithm is adapted from algorithm 3.2 in Al-Mohy and Higham (2011).

    """
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')
    if A.shape[1] != B.shape[0]:
        raise ValueError('shapes of matrices A {} and B {} are incompatible'
                         .format(A.shape, B.shape))
    ident = _ident_like(A)
    n = A.shape[0]
    if len(B.shape) == 1:
        n0 = 1
    elif len(B.shape) == 2:
        n0 = B.shape[1]
    else:
        raise ValueError('expected B to be like a matrix or a vector')
    u_d = 2**-53
    tol = u_d
    mu = _trace(A) / float(n)
    A = A - mu * ident
    A_1_norm = _exact_1_norm(A)
    # print('A_norm_1 = %.3e'% A_1_norm)
    if t*A_1_norm == 0:
        m_star, s = 0, 1
    else:
        ell = 2
        norm_info = LazyOperatorNormInfo(A, A_1_norm= A_1_norm, ell=ell)
        m_star, s = _fragment_3_1(norm_info, n0, tol, ell=ell)
    # print('(m-star, s) = (%i,%i)'% (m_star, s))
    # Core iterations
    F = (C.multiply(ident)).dot(B)
    Fb = (C.multiply(ident)).dot(Bb)

    S = C.multiply(A)
    St = S.T

    B = S.dot(B)
    Bb = (St).dot(Bb)
    F = F + B
    Fb = Fb + Bb

    eta = np.exp(mu / float(s))
    for i in range(s):
        c1 = _exact_inf_norm(B)
        for j in range(m_star):
            coeff = 1.0 / float(s*(j+1))
            B = coeff * S.dot(B)
            Bb = coeff * (St).dot(Bb)
            c2 = _exact_inf_norm(B)
            F = F + B
            Fb = Fb + Bb
            # print('eta = %.2e, iter j = %i |nf = %.3e |c1 = %.3e | c2 = %.3e'
            #         %(eta, j, _exact_inf_norm(F), c1,c2))
            if c1 + c2 <= tol * _exact_inf_norm(F):
                break
            c1 = c2
        F = eta * F
        Fb = eta * Fb
        B = F
        Bb = Fb
    return {'X':F, 'Y':Fb}


def splr_expmv_inexact_1(A, C, B, t=1.0):
    """
    Compute the action of the sparsified matrix exponential of a low-rank
    matrix product, using numpy notation:
        F = expm(Ps((LR')*(LR'))).T * Ps(LR')) B,
    where Ps denotes the sparsification projection depending on the subsampling index set (I, J), * denotes the Hadamard product (elementwise product) of two matrices and the blank space between two matrices denotes the matrix product.

    Parameters
    ----------
    A, C: transposable sparse matrix
    B   : ndarray
        The matrix to be multiplied by the matrix exponential of A.

    Returns
    -------
    F : ndarray

    Notes
    -----
    This algorithm is adapted from algorithm 3.2 in Al-Mohy and Higham (2011).

    """
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')
    if A.shape[1] != B.shape[0]:
        raise ValueError('shapes of matrices A {} and B {} are incompatible'
                         .format(A.shape, B.shape))
    ident = _ident_like(A)
    n = A.shape[0]
    if len(B.shape) == 1:
        n0 = 1
    elif len(B.shape) == 2:
        n0 = B.shape[1]
    else:
        raise ValueError('expected B to be like a matrix or a vector')
    u_d = 2**-53
    tol = u_d
    mu = _trace(A) / float(n)
    A = A - mu * ident
    A_1_norm = _exact_1_norm(A)
    # print('A_norm_1 = %.3e'% A_1_norm)
    if t*A_1_norm == 0:
        m_star, s = 0, 1
    else:
        ell = 2
        norm_info = LazyOperatorNormInfo(A, A_1_norm= A_1_norm, ell=ell)
        m_star, s = _fragment_3_1(norm_info, n0, tol, ell=ell)
    # print('(m-star, s) = (%i,%i)'% (m_star, s))
    # Core iterations
    F = (C.multiply(ident)).dot(B)
    B = (C.multiply(A)).dot(B)
    F = F + B
    eta = np.exp(mu / float(s))
    for i in range(s):
        c1 = _exact_inf_norm(B)
        for j in range(m_star):
            coeff = 1.0 / float(s*(j+1))
            B = coeff * (C.multiply(A)).dot(B)
            c2 = _exact_inf_norm(B)
            F = F + B
            # print('eta = %.2e, iter j = %i |nf = %.3e |c1 = %.3e | c2 = %.3e'
            #         %(eta, j, _exact_inf_norm(F), c1,c2))
            if c1 + c2 <= tol * _exact_inf_norm(F):
                break
            c1 = c2
        F = eta * F
        B = F
    return F


def splr_expmv_inexact_2(L, R, C, B):
    """
    Compute the action of the sparsified matrix exponential of a low-rank
    matrix product, using numpy notation:
        F = expm(Ps((LR')*(LR'))).T * Ps(LR')) B,
    where Ps denotes the sparsification projection depending on the subsampling index set (I, J), * denotes the Hadamard product (elementwise product) of two matrices and the blank space between two matrices denotes the matrix product.

    Parameters
    ----------
    L, R : transposable low-rank factor matrices
        The sparsified product Ps((LR' odot LR')) whose exponential is of interest.
    B : ndarray
        The matrix to be multiplied by the matrix exponential of A.

    Returns
    -------
    F : ndarray

    Notes
    -----
    This algorithm is adapted from algorithm 3.2 in Al-Mohy and Higham (2011).

    """
    r = L.shape[1]
    Q = (L.T) @ R
    id_r = np.eye(r, r, dtype=L.dtype)
    id_d = np.eye(L.shape[0], L.shape[0], dtype=L.dtype)

    if len(B.shape) == 1:
        n0 = 1
    elif len(B.shape) == 2:
        n0 = B.shape[1]
    else:
        raise ValueError('expected B to be like a matrix or a vector')
    # Core computations
    Et = id_d + R @ ( (expm(Q)-id_r) @ np.linalg.inv(Q)) @ L.T
    return (C.multiply(Et)).dot(B)


## EOF splr-main



# This table helps to compute bounds.
# They seem to have been difficult to calculate, involving symbolic
# manipulation of equations, followed by numerical root finding.
_theta = {
        # The first 30 values are from table A.3 of Computing Matrix Functions.
        1: 2.29e-16,
        2: 2.58e-8,
        3: 1.39e-5,
        4: 3.40e-4,
        5: 2.40e-3,
        6: 9.07e-3,
        7: 2.38e-2,
        8: 5.00e-2,
        9: 8.96e-2,
        10: 1.44e-1,
        # 11
        11: 2.14e-1,
        12: 3.00e-1,
        13: 4.00e-1,
        14: 5.14e-1,
        15: 6.41e-1,
        16: 7.81e-1,
        17: 9.31e-1,
        18: 1.09,
        19: 1.26,
        20: 1.44,
        # 21
        21: 1.62,
        22: 1.82,
        23: 2.01,
        24: 2.22,
        25: 2.43,
        26: 2.64,
        27: 2.86,
        28: 3.08,
        29: 3.31,
        30: 3.54,
        # The rest are from table 3.1 of
        # Computing the Action of the Matrix Exponential.
        35: 4.7,
        40: 6.0,
        45: 7.2,
        50: 8.5,
        55: 9.9,
        }


def _onenormest_matrix_power(A, p,
        t=2, itmax=5, compute_v=False, compute_w=False):
    """
    Efficiently estimate the 1-norm of A^p.

    Parameters
    ----------
    A : ndarray
        Matrix whose 1-norm of a power is to be computed.
    p : int
        Non-negative integer power.
    t : int, optional
        A positive parameter controlling the tradeoff between
        accuracy versus time and memory usage.
        Larger values take longer and use more memory
        but give more accurate output.
    itmax : int, optional
        Use at most this many iterations.
    compute_v : bool, optional
        Request a norm-maximizing linear operator input vector if True.
    compute_w : bool, optional
        Request a norm-maximizing linear operator output vector if True.

    Returns
    -------
    est : float
        An underestimate of the 1-norm of the sparse matrix.
    v : ndarray, optional
        The vector such that ||Av||_1 == est*||v||_1.
        It can be thought of as an input to the linear operator
        that gives an output with particularly large norm.
    w : ndarray, optional
        The vector Av which has relatively large 1-norm.
        It can be thought of as an output of the linear operator
        that is relatively large in norm compared to the input.

    """
    #XXX Eventually turn this into an API function in the  _onenormest module,
    #XXX and remove its underscore,
    #XXX but wait until expm_multiply goes into scipy.
    return scipy.sparse.linalg.onenormest(aslinearoperator(A) ** p)

class LazyOperatorNormInfo:
    """
    Information about an operator is lazily computed.

    The information includes the exact 1-norm of the operator,
    in addition to estimates of 1-norms of powers of the operator.
    This uses the notation of Computing the Action (2011).
    This class is specialized enough to probably not be of general interest
    outside of this module.

    """
    def __init__(self, A, A_1_norm=None, ell=2, scale=1):
        """
        Provide the operator and some norm-related information.

        Parameters
        ----------
        A : linear operator
            The operator of interest.
        A_1_norm : float, optional
            The exact 1-norm of A.
        ell : int, optional
            A technical parameter controlling norm estimation quality.
        scale : int, optional
            If specified, return the norms of scale*A instead of A.

        """
        self._A = A
        self._A_1_norm = A_1_norm
        self._ell = ell
        self._d = {}
        self._scale = scale

    def set_scale(self,scale):
        """
        Set the scale parameter.
        """
        self._scale = scale

    def onenorm(self):
        """
        Compute the exact 1-norm.
        """
        if self._A_1_norm is None:
            self._A_1_norm = _exact_1_norm(self._A)
        return self._scale*self._A_1_norm

    def d(self, p):
        """
        Lazily estimate d_p(A) ~= || A^p ||^(1/p) where ||.|| is the 1-norm.
        """
        if p not in self._d:
            est = _onenormest_matrix_power(self._A, p, self._ell)
            self._d[p] = est ** (1.0 / p)
        return self._scale*self._d[p]

    def alpha(self, p):
        """
        Lazily compute max(d(p), d(p+1)).
        """
        return max(self.d(p), self.d(p+1))

def _compute_cost_div_m(m, p, norm_info):
    """
    A helper function for computing bounds.

    This is equation (3.10).
    It measures cost in terms of the number of required matrix products.

    Parameters
    ----------
    m : int
        A valid key of _theta.
    p : int
        A matrix power.
    norm_info : LazyOperatorNormInfo
        Information about 1-norms of related operators.

    Returns
    -------
    cost_div_m : int
        Required number of matrix products divided by m.

    """
    return int(np.ceil(norm_info.alpha(p) / _theta[m]))


def _compute_p_max(m_max):
    """
    Compute the largest positive integer p such that p*(p-1) <= m_max + 1.

    Do this in a slightly dumb way, but safe and not too slow.

    Parameters
    ----------
    m_max : int
        A count related to bounds.

    """
    sqrt_m_max = np.sqrt(m_max)
    p_low = int(np.floor(sqrt_m_max))
    p_high = int(np.ceil(sqrt_m_max + 1))
    return max(p for p in range(p_low, p_high+1) if p*(p-1) <= m_max + 1)


def _fragment_3_1(norm_info, n0, tol, m_max=55, ell=2):
    """
    A helper function for the _expm_multiply_* functions.

    Parameters
    ----------
    norm_info : LazyOperatorNormInfo
        Information about norms of certain linear operators of interest.
    n0 : int
        Number of columns in the _expm_multiply_* B matrix.
    tol : float
        Expected to be
        :math:`2^{-24}` for single precision or
        :math:`2^{-53}` for double precision.
    m_max : int
        A value related to a bound.
    ell : int
        The number of columns used in the 1-norm approximation.
        This is usually taken to be small, maybe between 1 and 5.

    Returns
    -------
    best_m : int
        Related to bounds for error control.
    best_s : int
        Amount of scaling.

    Notes
    -----
    This is code fragment (3.1) in Al-Mohy and Higham (2011).
    The discussion of default values for m_max and ell
    is given between the definitions of equation (3.11)
    and the definition of equation (3.12).

    """
    if ell < 1:
        raise ValueError('expected ell to be a positive integer')
    best_m = None
    best_s = None
    if _condition_3_13(norm_info.onenorm(), n0, m_max, ell):
        for m, theta in _theta.items():
            s = int(np.ceil(norm_info.onenorm() / theta))
            if best_m is None or m * s < best_m * best_s:
                best_m = m
                best_s = s
    else:
        # Equation (3.11).
        for p in range(2, _compute_p_max(m_max) + 1):
            for m in range(p*(p-1)-1, m_max+1):
                if m in _theta:
                    s = _compute_cost_div_m(m, p, norm_info)
                    if best_m is None or m * s < best_m * best_s:
                        best_m = m
                        best_s = s
        best_s = max(best_s, 1)
    return best_m, best_s


def _condition_3_13(A_1_norm, n0, m_max, ell):
    """
    A helper function for the _expm_multiply_* functions.

    Parameters
    ----------
    A_1_norm : float
        The precomputed 1-norm of A.
    n0 : int
        Number of columns in the _expm_multiply_* B matrix.
    m_max : int
        A value related to a bound.
    ell : int
        The number of columns used in the 1-norm approximation.
        This is usually taken to be small, maybe between 1 and 5.

    Returns
    -------
    value : bool
        Indicates whether or not the condition has been met.

    Notes
    -----
    This is condition (3.13) in Al-Mohy and Higham (2011).

    """

    # This is the rhs of equation (3.12).
    p_max = _compute_p_max(m_max)
    a = 2 * ell * p_max * (p_max + 3)

    # Evaluate the condition (3.13).
    b = _theta[m_max] / float(n0 * m_max)
    return A_1_norm <= a * b



