import numba
import numpy as np
import torch

class iLQR:

    def __init__(self, dynamics, cost):
        '''
           iterative Linear Quadratic Regulator
           Args:
             dynamics: dynamics container
             cost: cost container
        '''
        self.cost = cost
        self.dynamics = dynamics
        self.params = {'alphas'  : 0.5**np.arange(8), #line search candidates
                       'regu_init': 20,    #initial regularization factor
                       'max_regu' : 10000,
                       'min_regu' : 0.001}

    def fit(self, x0, us_init, maxiters = 50, early_stop = True):
        '''
        Args:
          x0: initial state
          us_init: initial guess for control input trajectory
          maxiter: maximum number of iterations
          early_stop: stop early if improvement in cost is low.

        Returns:
          xs: optimal states
          us: optimal control inputs
          cost_trace: cost trace of the iterations
        '''
        return run_ilqr(self.dynamics.f, self.dynamics.f_prime, self.cost.L,
                        self.cost.Lf, self.cost.L_prime, self.cost.Lf_prime,
                        x0, us_init, maxiters, early_stop, **self.params)

    def rollout(self, x0, us):
        '''
        Args:
          x0: initial state
          us: control input trajectory

        Returns:
          xs: rolled out states
          cost: cost of trajectory
        '''
        return rollout(self.dynamics.f, self.cost.L, self.cost.Lf, x0, us)


class MPC:

    def __init__(self, controller, control_horizon = 1):
        '''
        Initialize MPC
        '''
        self.ch = control_horizon
        self.controller = controller
        self.us_init = None

    def set_initial(self, us_init):
        '''
        Set initial guess of actions
        '''
        if us_init.shape[0] <= self.ch:
            raise Exception('prediction horizon must be greater than control horizon')
        self.us_init = us_init

    def control(self, x0, maxiters = 50, early_stop = True):
        '''
        Returns optimal actions
        Supposed to be called Sequentially with observed state
        '''
        if self.us_init is None:
            raise Exception('initial guess has not been set')
        xs, us, cost_trace = self.controller.fit(x0, self.us_init, maxiters, early_stop)
        self.us_init[:-self.ch] = self.us_init[self.ch:]
        return us[:self.ch]


# @numba.njit
def run_ilqr(f, f_prime, L, Lf, L_prime, Lf_prime, x0, u_init, max_iters, early_stop,
             alphas, regu_init = 20, max_regu = 10000, min_regu = 0.001):
    '''
       iLQR main loop
    '''
    us = u_init
    regu = regu_init
    # First forward rollout
    xs, J_old = rollout(f, L, Lf, x0, us)
    # cost trace
    cost_trace = [J_old]

    # Run main loop
    for it in range(max_iters):
        ks, Ks, exp_cost_redu = backward_pass(f_prime, L_prime, Lf_prime, xs, us, regu)

        # Early termination if improvement is small
        if it > 3 and early_stop and np.abs(exp_cost_redu) < 1e-5: break

        # Backtracking line search
        for alpha in alphas:
          xs_new, us_new, J_new = forward_pass(f, L, Lf, xs, us, ks, Ks, alpha)
          if J_old - J_new > 0:
              # Accept new trajectories and lower regularization
              J_old = J_new
              xs = xs_new
              us = us_new
              regu *= 0.7
              break
        else:
            # Reject new trajectories and increase regularization
            regu *= 2.0

        cost_trace.append(J_old)
        regu = min(max(regu, min_regu), max_regu)

    return xs, us, cost_trace


# @numba.njit
def rollout(f, L, Lf, x0, us):
    '''
      Rollout with initial state and control trajectory
    '''
    xs = np.empty((us.shape[0] + 1, x0.shape[0]))
    xs[0] = x0
    cost = 0
    for n in range(us.shape[0]):
      xs[n+1] = f(xs[n], us[n])
      cost += L(xs[n], us[n])
    cost += Lf(xs[-1])
    return xs, cost


# @numba.njit
def forward_pass(f, L, Lf, xs, us, ks, Ks, alpha):
    '''
       Forward Pass
    '''
    xs_new = np.empty(xs.shape)

    cost_new = 0.0
    xs_new[0] = xs[0]
    us_new = us + alpha*ks

    for n in range(us.shape[0]):
        us_new[n] += Ks[n].dot(xs_new[n] - xs[n])
        xs_new[n + 1] = f(xs_new[n], us_new[n])
        cost_new += L(xs_new[n], us_new[n])

    cost_new += Lf(xs_new[-1])

    return xs_new, us_new, cost_new


# @numba.njit
def backward_pass(f_prime, L_prime, Lf_prime, xs, us, regu):
    '''
       Backward Pass
    '''
    ks = np.empty(us.shape)
    Ks = np.empty((us.shape[0], us.shape[1], xs.shape[1]))

    delta_V = 0
    V_x, V_xx = Lf_prime(xs[-1])
    regu_I = regu*np.eye(V_xx.shape[0])
    for n in range(us.shape[0] - 1, -1, -1):

        f_x, f_u = f_prime(xs[n], us[n])
        l_x, l_u, l_xx, l_ux, l_uu  = L_prime(xs[n], us[n])

        # Q_terms
        Q_x  = l_x  + f_x.T@V_x
        Q_u  = l_u  + f_u.T@V_x
        Q_xx = l_xx + f_x.T@V_xx@f_x
        Q_ux = l_ux + f_u.T@V_xx@f_x
        Q_uu = l_uu + f_u.T@V_xx@f_u

        # gains
        f_u_dot_regu = f_u.T@regu_I
        Q_ux_regu = Q_ux + f_u_dot_regu@f_x
        Q_uu_regu = Q_uu + f_u_dot_regu@f_u
        Q_uu_inv = np.linalg.inv(Q_uu_regu)

        k = -Q_uu_inv@Q_u
        K = -Q_uu_inv@Q_ux_regu
        ks[n], Ks[n] = k, K

        # V_terms
        V_x  = Q_x + K.T@Q_u + Q_ux.T@k + K.T@Q_uu@k
        V_xx = Q_xx + 2*K.T@Q_ux + K.T@Q_uu@K
        #expected cost reduction
        delta_V += Q_u.T@k + 0.5*k.T@Q_uu@k

    return ks, Ks, delta_V


# @numba.njit
def simulate(x0, U, f, L, Lf):
    T = U.shape[0]
    x = x0.copy()
    cost = 0.0
    for t in range(T):
        u = U[t]
        cost += L(x, u)
        x = f(x, u)
    cost += Lf(x)
    return cost

def simulate_torch(x0, h0, U, f, L, Lf):
    T = U.shape[0]
    x = x0.copy()
    cost = 0.0
    for t in range(T):
        u = U[t]
        cost += L(x, u)
        x = f(x, u)
    cost += Lf(x)
    return cost

def cem_planner(dynamics, cost, x0, horizon, u_dim, num_samples=1000, num_elite=100, max_iters=5, init_mean=None, init_var=None, alpha=0.5, noise_factor=1.0):
    """
    CEM-based planner for Model-Based RL.

    Parameters:
        dynamics: Dynamics object
        cost: Cost object
        x0: Initial state (numpy array)
        horizon: Planning horizon (T)
        u_dim: Dimension of control input
        num_samples: Number of samples per iteration
        num_elite: Number of elite samples to use for updating
        max_iters: Number of CEM iterations
        init_mean: Initial mean of control sequence (T, u_dim)
        init_var: Initial variance of control sequence (T, u_dim)
        alpha: Smoothing factor for updating mean and variance
        noise_factor: Multiplicative factor for initial variance if not provided

    Returns:
        Optimal control sequence (T, u_dim)
    """
    # Initialize mean and variance
    if init_mean is None:
        init_mean = np.zeros((horizon, u_dim))
    else:
        init_mean = init_mean.copy()
    
    if init_var is None:
        init_var = noise_factor * np.ones((horizon, u_dim))
    else:
        init_var = init_var.copy()
    
    mean = init_mean
    var = init_var

    best_u = None
    best_cost = np.inf

    for _ in range(max_iters):
        # Sample control sequences
        samples = np.random.normal(mean, np.sqrt(var), size=(num_samples, horizon, u_dim))
        
        # Evaluate all samples
        costs = np.zeros(num_samples)
        for i in range(num_samples):
            costs[i] = simulate(x0, samples[i], dynamics.f, cost.L, cost.Lf)
        
        # Select elites
        elite_indices = np.argsort(costs)[:num_elite]
        elites = samples[elite_indices]
        
        # Update distribution
        new_mean = np.mean(elites, axis=0)
        new_var = np.var(elites, axis=0)
        
        # Apply smoothing
        mean = alpha * new_mean + (1 - alpha) * mean
        var = alpha * new_var + (1 - alpha) * var
        
        # Track best solution
        current_best_idx = np.argmin(costs)
        current_best_cost = costs[current_best_idx]
        if current_best_cost < best_cost:
            best_cost = current_best_cost
            best_u = samples[current_best_idx]

    return best_u