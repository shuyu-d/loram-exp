import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
from timeit import default_timer as timer
import pandas as pd

def notears_linear(X, lambda1, loss_type, Wtrue, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func_primal(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, _ = _loss(W)
        fprimal = loss + lambda1 * w.sum()
        return fprimal

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    t0 = timer()
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    tinit = timer() - t0
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    # initialize iter hist
    gap = 1
    gap_eq = _h(_adj(w_est))[0]
    fprimal = _func_primal(w_est)
    time = tinit
    relerr_fro = np.linalg.norm(_adj(w_est)-Wtrue) / np.linalg.norm(Wtrue)
    splevel = np.count_nonzero(_adj(w_est))/(d*d)
    niters_primal = np.nan
    iterhist = []
    iterhist.append([gap, gap_eq, fprimal, time, relerr_fro, \
                        splevel, niters_primal])
    # start iterations
    print('Iter |  Duality gap | Equality violation | niters-primal ')
    for it in range(max_iter):
        t0 = timer()
        w_new, h_new = None, None
        niters_primal = 1
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
                niters_primal += 1
            else:
                break
        w_est, h = w_new, h_new
        # Dual ascent
        alpha += rho * h
        # Compute and gather iter stats
        time = iterhist[it][3] + timer() - t0
        ftemp = _func_primal(w_new)
        wtemp = _adj(w_est)
        gap = ftemp - iterhist[it][2] - alpha* h_new
        iterhist.append([gap, \
                        h_new, \
                        ftemp, \
                        time, \
                        np.linalg.norm(wtemp-Wtrue) / np.linalg.norm(Wtrue), \
                        np.count_nonzero(wtemp) / (d*d), \
                        niters_primal])
        print('%i | %.5e | %.3e | %i' % (it, gap, h_new, niters_primal))
        if h_new <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est, iterhist

if __name__ == '__main__':
    from notears import utils
    from numpy import asarray
    # import utils
    utils.set_random_seed(1)

    n = 1500
    d = 500
    spr = 1e-3
    s0, graph_type, sem_type = int(np.ceil(spr*d**2)), 'ER', 'gauss'
    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true)
    np.savetxt('Wtrue_100.csv', W_true, delimiter=',')

    X = utils.simulate_linear_sem(W_true, n, sem_type)
    np.savetxt('X_100.csv', X, delimiter=',')

    W_est, iterhist = notears_linear(X, lambda1=0.1, \
                                    h_tol=1e-2, loss_type='l2', \
                                    Wtrue=W_true)
    assert utils.is_dag(W_est)

    np.savetxt('West_100.csv', W_est, delimiter=',')
    acc = utils.count_accuracy(B_true, W_est != 0)
    print(acc)

    # Record iter hist
    df_iterhist = pd.DataFrame(iterhist, columns= \
                                ["gap", "gap_eq", \
                                "f", "time", \
                                "relerr_fro", "splevel",\
                                "niters_primal"]).to_csv('iterhist_.csv')
    # df_iterhist.to_csv('iterhist_.csv')
    # np.savetxt('gaps.csv', asarray(gaps), delimiter=',')
    # np.savetxt('gaps_eq.csv', asarray(gaps_eq), delimiter=',')
    # np.savetxt('fvals.csv', asarray(fvals), delimiter=',')
    # np.savetxt('times.csv', asarray(times), delimiter=',')


