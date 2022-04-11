from notears import utils
import numpy as np
utils.set_random_seed(1)

n, d, s0, graph_type, sem_type = 500, 200, 60, 'ER', 'gauss'
B_true = utils.simulate_dag(d, s0, graph_type)
W_true = utils.simulate_parameter(B_true)
np.savetxt('W_true_.csv', W_true, delimiter=',')

X = utils.simulate_linear_sem(W_true, n, sem_type)
np.savetxt('X_.csv', X, delimiter=',')

print("Dataset saved.")

