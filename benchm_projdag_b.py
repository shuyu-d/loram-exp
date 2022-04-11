""" Generate cause-effection confusions in addition to a given DAG matrix and
test projection performances using LORAM (ours) and NoTEARS.
"""

import numpy as np
from timeit import default_timer as timer
import pandas as pd
import matplotlib.pyplot as plt
from numpy import asarray
from loram import utils
import time, os
timestr = time.strftime("%H%M%S%m%d")
import shutil

from external.test_projdag_notears import notears_projdag
from loram.mf_projdag import Projdag

utils.set_random_seed(1)
""" List of test settings
k           : Rank parameter of the LoRAM matrix model
rho         : Sparsity of the LoRAM candidate set
ALPHA       : Parameter in the proximal mapping computation
MAXITER     : Iteration budget
"""
indir    = 'aux'
list_   = pd.read_csv('%s/benchm_list_projdag_all.csv' % indir)
n_tests = list_.shape[0]
""" Set up the problem parameters"""
for i in range(n_tests):
    gtype,semtype,d,k,rho_ref,rho,ALPHA,MAXITER = list_['gtype'][i], \
                                                  list_['semtype'][i], \
                                                  list_['d'][i].astype(np.uint), \
                                                  list_['k'][i].astype(np.uint), \
                                                  list_['rho_ref'][i], \
                                                  list_['rho'][i], \
                                                  list_['ALPHA'][i], \
                                                  list_['MAXITER'][i]
    print("======[%d/%d] | d,k: (%d,%d) | alpha: %.2e" % (i+1, n_tests, \
                                                          d, k, ALPHA))
    n   = np.ceil(1.2 * d).astype(np.uint)
    s0  = np.ceil(rho_ref * d **2).astype(np.uint)

    B_true = utils.simulate_dag(d, s0, gtype)
    W_pre = utils.simulate_parameter(B_true)
    X = utils.simulate_linear_sem(W_pre, n, semtype)
    Zstar = (W_pre + 0.4*W_pre.T) / (1.4 * 5e2)
    sca_true = max(abs(Zstar.ravel()))
    #
    print('scale of zstar is: %.3e' % sca_true)

    """" Solve the DAG learning problem """
    pb = Projdag(Zstar, k)

    t0 = timer()
    iterhist, x_sol = pb.run_projdag(alpha=ALPHA, maxiter=MAXITER)
    tt = timer() - t0
    print('Total run time LoRAM (d=%i) is: %.4e' %(pb.d, tt) )

    mat = np.asarray(pb.get_adj_matrix(x_sol).todense())
    mat_init = pb.Zinit

    """ Post process mat """
    nmat = max(abs(mat.ravel()))
    nmat0 = max(abs(mat_init.ravel()))

    wmat = sca_true * mat / nmat
    wmat_init = sca_true * mat_init / nmat0

    """ running notears """
    West_no, iterhist_no = notears_projdag(5e2*Zstar, lambda1=0.1, \
                                          loss_type='l2', Wtrue=W_pre)

    """ Save to files """
    for j in range(1): #range(2):
        if j == 0:
            fdir = 'outputs/benchm_projdag_2'
            if not os.path.exists(fdir):
                os.makedirs(fdir)
            np.savetxt('%s/t%d_Wdag_orig.csv' % (fdir,i), W_pre, delimiter=',')
            np.savetxt('%s/t%d_Zstar.csv'     % (fdir,i), Zstar, delimiter=',')
            np.savetxt('%s/t%d_W_init.csv'    % (fdir,i), wmat_init, delimiter=',')
            np.savetxt('%s/t%d_W_est.csv'    % (fdir,i), wmat, delimiter=',')
            np.savetxt('%s/t%d_West_notears.csv' % (fdir,i), West_no, delimiter=',')
        else:
            fdir = 'outputs/benchm_projdag_%s' % timestr
        if not os.path.exists(fdir):
            os.makedirs(fdir)
        df_iterhist = pd.DataFrame(iterhist, columns=["iter", "f_residual",\
                                "hval", "h_thres", \
                                "time", "gradnorm", \
                                "Znorm", "stepsize",\
                                "Fval"]).to_csv('%s/t%d_iterhist.csv' % (fdir,i))
        df_iterh_no = pd.DataFrame(iterhist_no, columns=\
                      ["gap", "gap_eq", "f", "time", \
                      "relerr_fro", "splevel",\
                       "niters_primal"]).to_csv('%s/t%d_iterh_notears.csv' \
                                                 % (fdir,i))

        shutil.copy2('%s/benchm_list_projdag.csv' % indir, '%s/' % fdir)

""" END OF BENCHMARK """

print('BENCHMARK FINISHED!')

""" PRODUCE RESULTS """
from numpy import genfromtxt

fdir = 'outputs/benchm_projdag_2'
for i in range(n_tests):
    gtype,semtype,d,k,rho_ref,rho,ALPHA,MAXITER = list_['gtype'][i], \
                                                  list_['semtype'][i], \
                                                  list_['d'][i], \
                                                  list_['k'][i], \
                                                  list_['rho_ref'][i], \
                                                  list_['rho'][i], \
                                                  list_['ALPHA'][i], \
                                                  list_['MAXITER'][i]
    print("======[%d/%d] | d,k: (%d,%d) | alpha: %.2e" % (i+1, n_tests, \
                                                          d, k, ALPHA))
    """ Load results """
    iterh = pd.read_csv('%s/t%d_iterhist.csv' % (fdir,i))
    iterh_no = pd.read_csv('%s/t%d_iterh_notears.csv' % (fdir,i))

    Zstar = genfromtxt('%s/t%d_Zstar.csv'   % (fdir,i), delimiter=',')
    Wini = genfromtxt('%s/t%d_W_init.csv'   % (fdir,i), delimiter=',')
    Wdag = genfromtxt('%s/t%d_Wdag_orig.csv' % (fdir,i), delimiter=',')

    Wsol = genfromtxt('%s/t%d_W_est.csv'   % (fdir,i), delimiter=',')

    Wsol_no = genfromtxt('%s/t%d_West_notears.csv'   % (fdir,i), delimiter=',')

    mm = max(abs(Wsol.ravel()))
    Wdag = mm * Wdag / max(abs(Wdag.ravel()))

    """ Compute accuracy """
    mm = max(abs(Wini.ravel()))
    Wini[abs(Wini) < 5e-2*mm] = 0
    mm = max(abs(Wsol.ravel()))
    Wsol[abs(Wsol) < 5e-2*mm] = 0

    acc_ini = utils.count_accuracy(Wdag != 0, Wini != 0)
    acc = utils.count_accuracy(Wdag != 0, Wsol != 0)
    acc_no = utils.count_accuracy(Wdag != 0, Wsol_no != 0)
    print(acc)
    # add scores to list (fdr': 0.0, 'tpr': 1.0, 'fpr': 0.0, 'shd': 0)
    list_.loc[i, "FDR"] = acc['fdr']
    list_.loc[i, "TPR"] = acc['tpr']
    list_.loc[i, "FPR"] = acc['fpr']
    list_.loc[i, "SHD"] = acc['shd']

    list_.loc[i, "FDR_no"] = acc_no['fdr']
    list_.loc[i, "TPR_no"] = acc_no['tpr']
    list_.loc[i, "FPR_no"] = acc_no['fpr']
    list_.loc[i, "SHD_no"] = acc_no['shd']

    # add runtime etc
    list_.loc[i, "runtime"] = iterh.iloc[-1]['time']
    list_.loc[i, "avetime"] = iterh.iloc[-1]['time'] / iterh.iloc[-1]['iter']

    list_.loc[i, "runtime_no"] = iterh_no.iloc[-1]['time']
    list_.loc[i, "avetime_no"] = iterh_no.iloc[-1]['time'] / iterh.iloc[-1]['iter']

print(list_)
_ = list_.to_csv('%s/benchm_result_2meth_table.csv' % fdir)




