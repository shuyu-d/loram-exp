"""Generate noisy random graphs from a given DAG matrix"""

import random
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csc_matrix

import matplotlib.pyplot as plt
from numpy import asarray
from loram import utils
import time, os
timestr = time.strftime("%H%M%S%m%d")


def gen_noise_gaussian(d, sparsity=1e-2, sigma=1e-1):
    m = np.ceil(sparsity * d **2).astype(np.uintc)
    supp = random.sample(range(d**2), m)
    # Convert 1d index supp into 2d index of matrices
    x, y = np.unravel_index(supp, (d,d))
    Nv = np.random.normal(scale=sigma, size=[m,])
    N = np.asarray(csc_matrix((Nv, (x, y)), shape=(d, d)).todense())
    return N, x, y, Nv


if __name__ == '__main__':
    utils.set_random_seed(1)
    """ List of test settings
    MAXITER     : Iteration budget
    """
    indir    = 'aux'
    oudir    = 'outputs/exp1c_zstar'

    list_   = pd.read_csv('%s/gen_list_noisygr.csv' % indir)
    n_tests = list_.shape[0]

    """ Set up the problem parameters"""
    for i in range(n_tests):
        gtype,d,rho_ref,rho,noisel = list_['gtype'][i], \
                             list_['d'][i].astype(np.uintc), \
                             list_['rho_ref'][i], \
                             list_['rho'][i], \
                             list_['noisel'][i]
        print("======[%d/%d] | d: %d |rho: %.2e | noisel: %.2e" % (i+1, n_tests, \
                                                              d, rho, noisel))
        n   = np.ceil(1.2 * d).astype(np.uint)
        s0  = np.ceil(rho_ref * d **2).astype(np.uint)

        B_true = utils.simulate_dag(d, s0, gtype)
        W_pre = utils.simulate_parameter(B_true)

        N, _,_,_= gen_noise_gaussian(d, sparsity=rho, sigma=noisel)
        Zstar = W_pre + N

        if not os.path.exists(oudir):
            os.makedirs(oudir)
        np.savetxt('%s/t%d_Z1c_in.csv' % (oudir,i), Zstar, delimiter=',')

