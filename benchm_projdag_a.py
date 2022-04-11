"""Generate a random noisy graph in addition to a given DAG matrix and test projection performances """

import random
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csc_matrix

import matplotlib.pyplot as plt
from numpy import asarray
from loram import utils
import time, os
from timeit import default_timer as timer
timestr = time.strftime("%H%M%S%m%d")
import shutil

from external.test_projdag_notears import notears_projdag
from loram.mf_projdag import Projdag


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
    k           : Rank parameter of the LoRAM matrix model
    rho         : Sparsity of the LoRAM candidate set
    ALPHA       : Parameter in the proximal mapping computation
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

        """ RUN SOLVERS  """
        Zstar = Zstar / ((1+noisel) * 5e2)
        sca_true = max(abs(Zstar.ravel()))
        pb = Projdag(Zstar, 40)

        t0 = timer()
        iterhist, x_sol = pb.run_projdag(alpha=10.0, maxiter=500)
        tt = timer() - t0
        print('Total run time LoRAM (d=%i) is: %.4e' %(pb.d, tt) )

        mat = np.asarray(pb.get_adj_matrix(x_sol).todense())
        mat_init = pb.Zinit

        """ Post process mat """
        nmat = max(abs(mat.ravel()))
        nmat0 = max(abs(mat_init.ravel()))

        wmat = sca_true * mat / nmat
        wmat_init = sca_true * mat_init / nmat0

        acc = utils.count_accuracy(W_pre != 0, wmat != 0)
        print('LoRAM-----results:')
        print(acc)
        """ running notears """
        Zstar = Zstar * ((1+noisel) * 5e2)
        West_no, iterhist_no = notears_projdag(Zstar, lambda1=0.1, \
                                              loss_type='l2', Wtrue=W_pre)
        acc_no = utils.count_accuracy(W_pre != 0, West_no != 0)
        print('NOTEARS-----results:')
        print(acc_no)

        """ Save to files """
        for j in range(1): #range(2):
            if j == 0:
                fdir = 'outputs/benchm_projdag_n'
                if not os.path.exists(fdir):
                    os.makedirs(fdir)
                np.savetxt('%s/t%d_Wdag_orig.csv' % (fdir,i), W_pre, delimiter=',')
                np.savetxt('%s/t%d_W_est.csv'    % (fdir,i), wmat, delimiter=',')
                np.savetxt('%s/t%d_West_notears.csv' % (fdir,i), West_no, delimiter=',')
            else:
                fdir = 'outputs/benchm_projdag_n%s' % timestr
            if not os.path.exists(fdir):
                os.makedirs(fdir)
            df_iterhist = pd.DataFrame(iterhist, columns=["iter", "f_residual",\
                                    "hval", "h_thres", \
                                    "time", "gradnorm", \
                                    "Znorm", "stepsize",\
                                    "Fval"])#.to_csv('%s/t%d_iterhist.csv' % (fdir,i))
            df_iterh_no = pd.DataFrame(iterhist_no, columns=\
                          ["gap", "gap_eq", "f", "time", \
                          "relerr_fro", "splevel",\
                           "niters_primal"])#.to_csv('%s/t%d_iterh_notears.csv' % (fdir,i))

            shutil.copy2('%s/gen_list_noisygr.csv' % indir, '%s/' % fdir)

            """RECORD RESULTS AND APPEND"""
            list_.loc[i, "FDR"] = acc['fdr']
            list_.loc[i, "TPR"] = acc['tpr']
            list_.loc[i, "FPR"] = acc['fpr']
            list_.loc[i, "SHD"] = acc['shd']

            list_.loc[i, "FDR_no"] = acc_no['fdr']
            list_.loc[i, "TPR_no"] = acc_no['tpr']
            list_.loc[i, "FPR_no"] = acc_no['fpr']
            list_.loc[i, "SHD_no"] = acc_no['shd']

            # add runtime etc
            list_.loc[i, "runtime"] = df_iterhist.iloc[-1]['time']
            list_.loc[i, "avetime"] = df_iterhist.iloc[-1]['time'] / \
                                            df_iterhist.iloc[-1]['iter']

            list_.loc[i, "runtime_no"] = df_iterh_no.iloc[-1]['time']
            list_.loc[i, "avetime_no"] = df_iterh_no.iloc[-1]['time'] / \
                                                    df_iterh_no.shape[0]

            df_iterhist.to_csv('%s/t%d_iterhist.csv' % (fdir,i))
            df_iterh_no.to_csv('%s/t%d_iterhist.csv' % (fdir,i))

""" END OF BENCHMARK """

print('BENCHMARK FINISHED!')

print(list_)
fdir = 'outputs/benchm_projdag_n'
_ = list_.to_csv('%s/benchm_result_2meth_table.csv' % fdir)



