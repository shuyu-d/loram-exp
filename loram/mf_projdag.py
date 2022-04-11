"""
Optimization of the LoRAM-based DAG learning problem.
The h function is based on splr representation:
       lr:          two term low-rank matrix product
       po:          mask
       sigma_abs:   produce the nonnegative surrogate matrix

This is an implementation of algorithms in the paper:
Dong, S. & Sebag, M. (2022). From graphs to DAGs: a low-complexity model and a scalable algorithm.

Contact: shuyu.dong@m4x.org
"""
import numpy as np
import random
from timeit import default_timer as timer

from loram import spmaskmatmul
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import expm, expm_multiply
from loram.splr_expmv import splr_expmv_inexact1_paired, splr_expmv_inexact_1, splr_expmv_inexact_1b, splr_sigma_abs, splr_func_abs


def lin_comb(scalar_1, dir_1, scalar_2=None, dir_2=None):
    x_ = scalar_1 * dir_1['X'] + scalar_2 * dir_2['X']
    y_ = scalar_1 * dir_1['Y'] + scalar_2 * dir_2['Y']
    return {'X':x_, 'Y': y_}

def elem_inner_prod(mat_1, mat_2):
    return np.sum(mat_1 * mat_2)

def lin_inner_prod(dir_1, dir_2):
    """ Inner product in the euclidean space of R(dxk) x R(dxk)
    """
    return elem_inner_prod(dir_1['X'], dir_2['X']) + \
            elem_inner_prod(dir_1['Y'], dir_2['Y'])

def lin_tspace_norm(dir_):
    return np.sqrt(elem_inner_prod(dir_['X'], dir_['X']) + \
            elem_inner_prod(dir_['Y'], dir_['Y']))

def lin_tspace_odot(dir_):
    return {'X': dir_['X'] * dir_['X'], \
            'Y': dir_['Y'] * dir_['Y']}

def lin_tspace_divide(dir_1, dir_2):
    return {'X': dir_1['X'] / dir_2['X'], \
            'Y': dir_1['Y'] / dir_2['Y']}

class Projdag():

    def __init__(self, Zref, k, I=None, J=None, maxiter=500, Zinit=None,\
                        M=None):
        """
        Arguments
        - k (int)           : number of latent dimensions
        - maxiter (float)   : limit of iteration number
        - Zref (float)      : Observed DAG matrix
        """
        self.Zref = Zref
        self.d    = Zref.shape[0]
        self.k = k
        if I is None:
            self._gen_I_J(9e-1)
        else:
            self.I = I
            self.J = J
        self.z = {'X':np.ones((self.d, self.k)), 'Y': np.ones((self.d, self.k)) }
        self.iterdb = []
        self.iterdb_old = []
        # Problem and opt hyper parameters
        self.maxiter = maxiter
        self.Zinit = Zinit
        self.M     = M

    def get_adjmatrix_dense(self, z):
        loram_vec, loram_abs, _= spmaskmatmul.spmaskmatmul_abs(z['X'], z['Y'], \
                                                      self.I, self.J)
        return np.asarray(csc_matrix((loram_vec, (self.I, self.J)), \
                            shape=(self.d, self.d)).todense()), max(loram_abs)

    def get_scale_loram(self, z):
        """computes the infinity norm of the loram variable """
        _, loram_abs, _ = spmaskmatmul.spmaskmatmul_abs(z['X'], z['Y'], \
                                                      self.I, self.J)
        return max(loram_abs)

    def get_adj_matrix(self, z):
        splr_vec, _, _= spmaskmatmul.spmaskmatmul_abs(z['X'], z['Y'], \
                                                      self.I, self.J)
        return csc_matrix((splr_vec, (self.I, self.J)), shape=(self.d, self.d))

    def _gen_I_J_projdag(self, zref, rho=5e-2):
        """ Indices of the edges in the observed (non-DAG) graph
        """
        z_abs  = abs(zref)
        ind_2d = np.nonzero(z_abs)
        return {'I': ind_2d[0].astype(np.uintc),
                'J': ind_2d[1].astype(np.uintc)}

    def _get_I_J_zref(self, zref):
        """ Get the index set of all edges in the reference (non-DAG) graph
        """
        ind_2d = np.nonzero(zref)
        return {'I': ind_2d[0].astype(np.uintc),
                'J': ind_2d[1].astype(np.uintc)}

    def _gen_I_J_max_cov(self, rho=5e-2):
        """ Take the k largest pairs among the entries of the covariance matrix
        """
        C       = np.cov(self.M)
        print(C.shape)
        print(self.d)
        iu      = np.triu_indices(self.d, 1)
        cvech   = C[iu]
        n_k     = int(np.ceil(rho * self.d**2 /2 ))
        inds    = np.argsort(cvech)[-n_k:]
        Ipre = iu[0][inds].astype(np.uintc)
        Jpre = iu[1][inds].astype(np.uintc)
        return {'I': np.concatenate((Ipre, Jpre), axis = None),
                'J': np.concatenate((Jpre, Ipre), axis = None)}

    def _gen_I_J(self, rho):
        m = np.ceil(rho * self.d **2)
        supp = random.sample(range(self.d**2), \
                             min(np.inf, np.ceil(m).astype(np.int)))
        # Convert 1d index supp into 2d index of matrices
        x, y = np.unravel_index(supp, (self.d,self.d))
        self.I = np.array(x).astype(np.uintc)
        self.J = np.array(y).astype(np.uintc)

    def _comp_func_h(self):
        """ The exponential trace of po(XY') minus d """
        return np.trace(expm(self.iterdb['At']).todense()) - self.d

    def _comp_naive_func_h(self, z):
        """ The exponential trace of po(XY') minus d """
        _, sigmaZ, _= spmaskmatmul.spmaskmatmul_abs(z['X'], z['Y'], self.I, self.J)
        sca = 1
        mat_z = csc_matrix((sigmaZ/sca, (self.I, self.J)), shape=(self.d, self.d))
        val =  np.trace(expm(mat_z).todense()) - self.d
        return val

    def _comp_func_h_thres(self, z):
        """ The exponential trace of T(po(XY')) minus d, where T is a simple
            hard threshold function
        """
        _, sigmaZ, _ = spmaskmatmul.spmaskmatmul_abs(z['X'], z['Y'], \
                                                      self.I, self.J)
        mm = max(sigmaZ)
        sigmaZ[sigmaZ < 5e-2*mm] = 0
        mat_z = csc_matrix((sigmaZ, (self.I, self.J)), shape=(self.d, self.d))
        val =  np.trace(expm(mat_z).todense()) - self.d
        return val

    def _comp_func_h_norma(self, z):
        # This func is deprecated
        _, sigmaZ, _= spmaskmatmul.spmaskmatmul_abs(z['X'], z['Y'], \
                                                    self.I, self.J)
        sca = max(sigmaZ)
        mat_z = csc_matrix((sigmaZ/sca, (self.I, self.J)), \
                                            shape=(self.d, self.d))
        val =  np.trace(expm(mat_z).todense()) - self.d
        return val

    """ ELEMENTARY FUNCTIONS """
    def _comp_grad_h_exact(self):
        A_, dA_, sp_zvec = splr_sigma_abs(self.z['X'], self.z['Y'], self.I, self.J)
        # Form A' (transpose of A)
        At = csc_matrix((A_, (self.J, self.I)), shape=(self.d, self.d))
        # Form the matrix of mask(Z)
        dA = csc_matrix((dA_, (self.I, self.J)), shape=(self.d, self.d))
        # Record the matrices to iterdb
        S = dA.multiply(expm(At))
        gh_x = S.dot(self.z['Y'])
        gh_y = (S.T).dot(self.z['X'])
        return {'X': gh_x, 'Y': gh_y}, At, dA, sp_zvec

    def _comp_grad_h_inexact1(self):
        A_, dA_, sp_zvec = splr_sigma_abs(self.z['X'], self.z['Y'], self.I, self.J)
        """ Form A' (transpose of A) """
        At = csc_matrix((A_, (self.J, self.I)), shape=(self.d, self.d))
        """ Form the matrix of mask(Z) """
        dA = csc_matrix((dA_, (self.I, self.J)), shape=(self.d, self.d))
        """ Record the matrices to iterdb """
        return {'X': splr_expmv_inexact_1(At, dA, self.z['Y']), \
                 'Y': splr_expmv_inexact_1(At.T, dA.T, self.z['X'])}, At, dA, sp_zvec

    def _comp_grad_h_inexact1b(self):
        A_, dA_, sp_zvec = splr_sigma_abs(self.z['X'], self.z['Y'], self.I, self.J)
        # Form A' (transpose of A)
        At = csc_matrix((A_, (self.J, self.I)), shape=(self.d, self.d))
        # Form the matrix of mask(Z)
        dA = csc_matrix((dA_, (self.I, self.J)), shape=(self.d, self.d))
        # Record the matrices to iterdb
        return {'X': splr_expmv_inexact_1b(At, dA, self.z['Y']), \
                 'Y': splr_expmv_inexact_1b(At.T, dA.T, self.z['X'])}, At, dA, sp_zvec

    def _comp_grad_zdiff_precomp(self, sp_zvec):
        """
        Compute the gradient with the residual = po(Z - Zref)
        """
        res_vec = sp_zvec - self.Zref[self.I, self.J]
        Mat = csc_matrix((res_vec, (self.I, self.J)), shape=(self.d, self.d))
        df_x =  Mat.dot(self.z['Y'])
        df_y = (Mat.T).dot(self.z['X'])
        return {'X': df_x, 'Y': df_y}, res_vec


    """ GRADIENT COMPUTATION """
    def _compute_grad_projdag(self):
        grad_h, At, dA, sp_zvec = self._comp_grad_h_inexact1()
        grad_f, res_vec = self._comp_grad_zdiff_precomp(sp_zvec)
        return grad_f, grad_h, sp_zvec, At, dA, res_vec


    """ GRADIENT DESCENT UPDATE RULES """
    def _comp_stepsize_bb(self):
        """ dir_desc is a pair of matrices in R(dxk) x R(dxk) """
        z   = lin_comb(1, self.iterdb['z'], \
                       -1, self.iterdb_old['z'])
        y   = lin_comb(-1, self.iterdb['dir_desc'], \
                        1, self.iterdb_old['dir_desc'])
        sbb = min(1e1, max(1e-15, lin_inner_prod(z, z) / lin_inner_prod(z, y))) # sbb1
        # sbb = min(1e3, max(1e-15, lin_inner_prod(z, y) / lin_inner_prod(y, y))) # sbb2
        return sbb

    def _gd_fixedstep(self, alpha, stepsize):
        """
        Perform gradient descent with respect to the primal function
                F = f + alpha * h
        """
        grad_f, grad_h, sp_zvec, At, dA, res_vec \
                    = self._compute_grad_projdag()
        _dir_desc   = lin_comb(-1/alpha, grad_f, -1, grad_h)
        self.iterdb = {'z': self.z, 'grad_f':grad_f, \
                       'grad_h': grad_h, 'dir_desc': _dir_desc, \
                       'sp_zvec': sp_zvec, 'At':At, 'dA':dA, \
                       'res_vec': res_vec}
        self.z      = lin_comb(1, self.z, stepsize , _dir_desc)
        return stepsize

    def solver_primal_accGD(self, z_init, alpha=1.0, maxiter=150, s_init=1e-3,\
                            h_tol=1e-10):
        """ Accelrated gradient descent for solving the primal problem
            alpha     :  dual parameter
        """

        # Initialize the factor matrices
        self.init_factors(z_init)
        x_old = self.z
        iterhist = []
        self.stat = {'f_residual': np.nan, \
                     'hval':self._comp_naive_func_h(x_old),\
                     'h_thres':self._comp_func_h_thres(x_old), \
                     'time': 1e-5, 'gradnorm': np.nan, \
                     'Znorm': np.nan, 'stepsize':np.nan, \
                     'Fval': np.nan}
        iterhist.append([0, self.stat['f_residual'], \
                           self.stat['hval'], self.stat['h_thres'], \
                           self.stat['time'], self.stat['gradnorm'], \
                           self.stat['Znorm'], self.stat['stepsize'],\
                           self.stat['Fval']])

        # 1st iteration
        self._gd_fixedstep(alpha, s_init)
        x_new = self.z

        for i in range(maxiter):
            t0 = timer()
            # The auxilary point y is maintained by self.z
            self.iterdb_old = self.iterdb
            x_old = x_new
            # Compute the BB stepsize:
            grad_f, grad_h, sp_zvec, At, dA, res_vec \
                        = self._compute_grad_projdag()
            _dir_desc   = lin_comb(-1/alpha, grad_f, -1, grad_h)
            self.iterdb = {'z': self.z, 'grad_f':grad_f, \
                           'grad_h': grad_h, 'dir_desc': _dir_desc, \
                           'sp_zvec': sp_zvec, 'At':At, 'dA':dA, \
                           'res_vec': res_vec}
            stepsize    = max(self._comp_stepsize_bb(), 1e-20)
            # Gradient descent with Nesterov's acceleration:
            x_new       = lin_comb(1, self.z, stepsize , _dir_desc)
            self.z      = lin_comb(1, x_new, i/(i+3), \
                                    lin_comb(1, x_new, -1, x_old))
            ti = timer() - t0
            # Get iter stats
            if (i+1) % 50 == 0:
                self._comp_iterhist_accGD(x_new, ti, stepsize, alpha)
                iterhist.append([i+1, self.stat['f_residual'], \
                                   self.stat['hval'], self.stat['h_thres'], \
                                   self.stat['time'], self.stat['gradnorm'], \
                                   self.stat['Znorm'], self.stat['stepsize'],\
                                   self.stat['Fval']])
            else:
                tt = self.stat['time'] + ti
                self.stat['time'] =  tt
                iterhist.append([i+1, np.nan, \
                                   np.nan, np.nan, \
                                   self.stat['time'], np.nan, \
                                   np.nan, stepsize,\
                                   np.nan])
            if (i+1) % 100 == 0:
                print("%d| f: %.2e | alpha: %.2e | h: %.7e | gradn: %.3e | \
                        s: %.4e| %.3e (sec)" % (i+1, self.stat['f_residual'], \
                        alpha, self.stat['hval'], self.stat['gradnorm'], \
                        stepsize, ti))
            if self.stat['gradnorm'] <= 1e-6:
                break
            # if self.stat['hval'] <= h_tol:
            #     break
        return iterhist, stepsize, x_new

    def run_projdag(self, alpha=1, h_tol=1e-10, maxiter=200):
        _omega  = self._gen_I_J_projdag(self.Zref)
        self.I = _omega['I']
        self.J = _omega['J']

        """ Initialize the factor matrices"""
        t0  = timer()
        z_ = self.init_factors_gaussian(sca=1e-2)
        ti  = timer() - t0
        self.Zinit = np.asarray(self.get_adj_matrix(z_).todense())
        """ Start iterations"""
        iterhist, _, x_sol = self.solver_primal_accGD(z_, alpha=alpha,\
                                                      h_tol=h_tol, \
                                                      maxiter=maxiter)
        return iterhist, x_sol


    """ OTHER AUXILARY FUNCTIONS
    """
    def init_factors(self, z_init):
        print('Initialize z with a given point')
        self.z = z_init

    def init_factors_gaussian(self, sca = 1e-2):
        print('Default initialization method: Gaussian matrices')
        z = {'X': np.random.normal(scale=sca, size=[self.d,self.k]),
             'Y': np.random.normal(scale=sca, size=[self.d,self.k])}
        self.z = z
        return z

    def init_factors_zero(self):
        return {'X': np.zeros([self.d,self.k]),
                'Y': np.zeros([self.d,self.k])}

    def _comp_iterhist(self, ti, stepsize, alpha):
        """
        A function to compute the total mean square error
        """
        _z = self.z
        tt = self.stat['time'] + ti
        self.stat['time'] =  tt
        #
        self.stat['f_residual'] = np.linalg.norm(self.iterdb['res_vec'])
        self.stat['hval']       = self._comp_naive_func_h(_z)
        self.stat['h_thres']   = self._comp_func_h_thres(_z)
        self.stat['gradnorm']   = lin_tspace_norm(self.iterdb['dir_desc'])
        self.stat['stepsize']   = stepsize
        self.stat['Znorm']      = np.linalg.norm(self.iterdb['At'].todense())
        self.stat['Fval']       = .5*self.stat['f_residual'] **2 / alpha + \
                                    self.stat['hval']


    def _comp_iterhist_accGD(self, xnew, ti, stepsize, alpha):
        """
        A function to compute the total mean square error
        """
        tt = self.stat['time'] + ti
        self.stat['time'] =  tt
        self.stat['f_residual'] = np.linalg.norm(self.iterdb['res_vec'])
        self.stat['hval']       = self._comp_naive_func_h(xnew)
        self.stat['h_thres']   = self._comp_func_h_thres(xnew)
        self.stat['gradnorm']   = lin_tspace_norm(self.iterdb['dir_desc'])
        self.stat['stepsize']   = stepsize
        self.stat['Znorm']      = np.linalg.norm(self.iterdb['At'].todense())
        self.stat['Fval']       = .5*self.stat['f_residual'] **2 / alpha + \
                                    self.stat['hval']

    def verif_grad_h(self):
        A_, dA_, sp_zvec = splr_sigma_abs(self.z['X'], self.z['Y'], self.I, self.J)
        # Form A' (transpose of A)
        At = csc_matrix((A_, (self.J, self.I)), shape=(self.d, self.d))
        # Form the matrix of mask(Z)
        dA = csc_matrix((dA_, (self.I, self.J)), shape=(self.d, self.d))
        # Compute grad-h
        grad_hin = {'X': splr_expmv_inexact_1(At, dA, self.z['Y']),  \
                    'Y': splr_expmv_inexact_1(At.T, dA.T, self.z['X'])}
        ## Verification
        S = dA.multiply(expm(At))
        gh_x = S.dot(self.z['Y'])
        gh_y = (S.T).dot(self.z['X'])
        grad_h = {'X': gh_x, 'Y': gh_y}
        relerr = lin_tspace_norm(lincomb(-1, grad_h, 1, grad_hin)) / (lin_tspace_norm(grad_h) * lin_tspace_norm(grad_hin))
        ang = self.lin_inner_prod(grad_h, grad_hin)  /  np.sqrt(self.lin_inner_prod(grad_h, grad_h) * self.lin_inner_prod(grad_hin, grad_hin))
        print('grad_inexa vs grad relerr:%.3e' % relerr)
        print('grad_inexa vs grad cos-angle:%.3e' % ang)
        return ang, rel_err

