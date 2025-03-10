import numba
import numpy as np
import torch
from ilqr.utils import kl_divergence
from numba import njit


class CEMPlanner:
    """
    A persistent Cross-Entropy Method (CEM) planner that updates its internal
    distribution parameters (mean and variance) every time its iterate() method is called.
    """
    def __init__(self,
                 dynamics, 
                 cost, 
                 horizon, 
                 x_dim, 
                 u_dim, 
                 num_samples=1000, 
                 num_elite=100, 
                 max_iters=100, 
                 alpha=0.5, 
                 noise_factor=1.0, 
                 init_mean=None, 
                 init_cov=None,
                 init_err_mu=None,
                 init_err_cov=None,
                 init_u_disturbance_mu=None,
                 init_u_disturbance_cov=None,
                 debugger=None):
        self.dynamics = dynamics
        self.cost = cost
        self.horizon = horizon
        self.u_dim = u_dim
        self.x_dim = x_dim
        self.num_samples = num_samples
        self.num_elite = num_elite
        self.alpha = alpha
        # Initialize mean and variance for the control trajectory.
        self.u_dim = u_dim
        self.dim = self.horizon * u_dim
        self.mean = np.zeros(self.dim) if init_mean is None else init_mean
        self.cov = noise_factor * np.eye(self.dim) if init_cov is None else init_cov
        self.best_u = None
        self.best_cost = np.inf
        self.max_iters = max_iters
        self.h = (torch.zeros(self.dynamics.num_layers, num_samples, self.dynamics.hidden_size),
                  torch.zeros(self.dynamics.num_layers, num_samples, self.dynamics.hidden_size))
        
        # The distributions for the second "shooting" stage
        self.err_mu = np.zeros(x_dim) if init_err_mu is None else init_err_mu
        self.err_cov = np.eye(x_dim) if init_err_cov is None else init_err_cov
        self.S = np.zeros((x_dim, x_dim))
        self.u_disturbance_mu = np.zeros(u_dim) if init_u_disturbance_mu is None else init_u_disturbance_mu
        self.u_disturbance_cov = np.eye(u_dim) if init_u_disturbance_cov is None else init_u_disturbance_cov
        self.window = 1000
        self.num_steps = 0
        self.debugger = debugger

    def iterate(self, x0, xgoal):
        """
        Perform one iteration (i.e. one timestep) of the CEM update.
        Samples control trajectories from the current Gaussian distribution,
        evaluates their cost, selects the elite samples, and then updates the
        distribution parameters.
        """
        # Sample control trajectories as vectors of length dim.
        samples_vector = np.random.multivariate_normal(self.mean, self.cov, size=self.num_samples)
        # Reshape each sample vector to shape (horizon, u_dim)
        samples = samples_vector.reshape(self.num_samples, self.horizon, self.u_dim)
        
        for _ in range(self.max_iters):
            costs = np.zeros((self.num_samples, self.horizon))
            h = self.h[0].clone(), self.h[1].clone()
            # Evaluate cost for each sample trajectory.
            # Optimize using batch processing.
            x = x0.clone()
            for t in range(self.horizon):
                x, h, costs_time_step, = simulate_torch(x, h, samples[:, t], self.dynamics.f, self.cost.L, xgoal)
                
                costs[:, t] = costs_time_step
            # Compute total cost for each sample trajectory.
            costs = np.sum(costs, axis=1)
            # Select elite samples (lowest costs)
            elite_indices = njit(lambda a: np.argsort(a))(costs)[:self.num_elite]
            elites = samples[elite_indices]
            
            # Optionally, update best found trajectory.
            current_best_idx = njit(lambda a: np.argmin(a))(costs)
            if costs[current_best_idx] < self.best_cost:
                self.best_cost = costs[current_best_idx]
                self.best_u = samples[current_best_idx]
                self.h = h[0][:, current_best_idx, :], h[1][:, current_best_idx, :]
                
            new_mean = njit(lambda a, b: np.mean(a, axis=b))(elites, 0)
            new_cov = njit(lambda a, b: np.cov(a, rowvar=b))(elites, False)
            # Smoothly update the distribution parameters.
            updated_mean = self.alpha * new_mean + (1 - self.alpha) * self.mean
            updated_cov = self.alpha * new_cov + (1 - self.alpha) * self.cov
            
            if kl_divergence(self.mean, self.cov, updated_mean, updated_cov) < 1e-3:
                break
            self.mean = self.alpha * new_mean + (1 - self.alpha) * self.mean
            self.cov = self.alpha * new_cov + (1 - self.alpha) * self.cov
        return self.best_u
    
    def calc_diff(self, x_real):
        x_pred, _ = self.dynamics.f(x_real, self.best_u, self.h, grad=False)
        diff = x_real - x_pred
        self.update_error_distribution(x_real, x_pred)
    
    def update_error_distribution(self, diff):
        self.num_steps += 1
        delta = diff - self.err_mu
        self.err_mu = self.err_mu + delta / self.num_steps
        self.S = self.S + delta * (diff - self.err_mu).T
        self.err_cov = self.S / (self.num_steps - 1) if self.num_steps > 1 else self.S / self.num_steps
        self.err_cov = self.err_cov + 1e-6 * np.eye(self.x_dim)
    
    def update_u_disturbance_distribution(self, x0):
        disturbance = njit(np.random.multivariate_normal)(self.u_disturbance_mu, self.u_disturbance_cov, size=self.num_samples)
        us = self.best_u + disturbance
        x = x0.copy()
        # Rollout one step with the disturbance
        h = self.h[0].clone().repeat(self.num_samples, 1, 1), self.h[1].clone().repeat(self.num_samples, 1, 1)
        x_disturb, _ = self.dynamics.f(x, us, h, grad=False)
        x_pred, _ = self.dynamics.f(x, self.best_u, self.h, grad=False)
        diff = x_disturb - x_pred
        delta = diff - self.u_disturbance_mu
        delta = delta.to_numpy()
        # spectral decomp error covariance   
        U, S, V = njit(lambda a: np.linalg.svd(a))(self.u_disturbance_cov)
        rotated_delta = njit(lambda a, b: np.dot(a, b))(V.T, delta)
        # Check if each dimension of rotated delta is within 3 standard deviations
        # of the corresponding dimension of the error covariance
        good_us = us[np.abs(rotated_delta) < 3 * np.sqrt(S)]
        self.u_disturbance_mu = njit(lambda a, b: np.mean(a, axis=b))(good_us, 0)
        self.u_disturbance_cov = njit(lambda a, b: np.cov(a, rowvar=b))(good_us, False)
            
        
        
        
        
        
        

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

def simulate_torch(x0, h0, U, f, L, xgoal):
    T = U.shape[0]
    x = x0.clone()
    h = (h0[0].clone(), h0[1].clone())
    x_next, hs = f(x, U, h, grad=False)
    cost = L(x, U, xgoal)
    return x_next, hs, cost


