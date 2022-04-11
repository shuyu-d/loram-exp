import numpy as np
from timeit import default_timer as timer
import pandas as pd
import matplotlib.pyplot as plt
from numpy import asarray
from notears import utils
import time, os
import shutil
from test_projdag_notears import notears_projdag

timestr = time.strftime("%H%M%S%m%d")
utils.set_random_seed(1)
""" List of test settings
k           : Maximal rank of the sparse low-rank (splr) matrix model
rho         : Sparsity of the splr matrix model
ALPHA       : Parameter of the h (exponential trace) function
MAXITER     : Iteration budget
"""
indir    = 'outputs'
list_   = pd.read_csv('%s/benchm_list_projdag.csv' % indir)
n_tests = list_.shape[0]
""" Set up the problem parameters"""
for i in range(n_tests):
    gtype,semtype,d,rho_ref = list_['gtype'][i], \
                              list_['semtype'][i], \
                              list_['d'][i].astype(np.uint), \
                              list_['rho_ref'][i]
    print("======[%d/%d] | d: %d | rho: %.2e" % (i+1, n_tests, \
                                                 d, rho_ref))
    s0  = np.ceil(rho_ref * d **2).astype(np.uint)

    B_true = utils.simulate_dag(d, s0, gtype)
    W_true = utils.simulate_parameter(B_true)
    Zstar = (W_true + 0.4*W_true.T) / 1.4

    """" Solve the DAG learning problem """
    W_est, iterhist = notears_projdag(Zstar, lambda1=0.1, \
                                      loss_type='l2', Wtrue=W_true)
    assert utils.is_dag(W_est)

    acc = utils.count_accuracy(B_true, W_est != 0)
    print(acc)

    """ Save to files """
    for j in range(2):
        if j == 0:
            fdir = 'outputs/ben_notears_projdag_'
            if not os.path.exists(fdir):
                os.makedirs(fdir)
            np.savetxt('%s/t%d_W_true.csv' % (fdir,i), W_true, delimiter=',')
            np.savetxt('%s/t%d_Zstar.csv'     % (fdir,i), Zstar, delimiter=',')
            np.savetxt('%s/t%d_W_est.csv'    % (fdir,i), W_est, delimiter=',')
        else:
            fdir = 'outputs/benchm_projdag_%s' % timestr
        if not os.path.exists(fdir):
            os.makedirs(fdir)
        df_iterhist = pd.DataFrame(iterhist, columns= \
                         ["gap", "gap_eq", "f", "time", \
                         "relerr_fro", "splevel",\
                         "niters_primal"]).to_csv('%s/t%d_iterhist.csv' % (fdir,i))
        shutil.copy2('%s/benchm_list_projdag.csv' % indir, '%s/' % fdir)



