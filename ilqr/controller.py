import numpy as np
import torch
from ilqr.utils import kl_divergence, mean, covar
from ilqr.containers import Dynamics
import time

class CEMBase:
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
                 noise_factor=0.25, 
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
        self.mean = np.zeros(self.u_dim) if init_mean is None else init_mean
        self.cov = 5 * np.eye(self.u_dim) if init_cov is None else init_cov
        self.best_u = None
        self.best_cost = np.inf
        self.max_iters = max_iters
        
        # The distributions for the second "shooting" stage
        self.err_mu = np.zeros(x_dim) if init_err_mu is None else init_err_mu
        self.err_cov = np.eye(x_dim) if init_err_cov is None else init_err_cov
        self.S = np.eye(x_dim) * noise_factor
        self.u_disturbance_mu = np.zeros(u_dim) if init_u_disturbance_mu is None else init_u_disturbance_mu
        self.u_disturbance_cov = np.eye(u_dim)*noise_factor if init_u_disturbance_cov is None else init_u_disturbance_cov
        self.next_best_state = None 
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
        samples_vector = np.random.multivariate_normal(self.mean, self.cov, size=self.num_samples*self.horizon)
        samples_vector = np.clip(samples_vector, -10, 10) # prevent large control inputs
        # Reshape each sample vector to shape (horizon, u_dim)
        samples = samples_vector.reshape(self.num_samples, self.horizon, self.u_dim)
        self.best_u = np.zeros(self.u_dim)
        self.best_cost = np.inf
        
        for _ in range(self.max_iters):
            elites = self.rollout(x0, xgoal, samples)
            for t in range(self.horizon):    
                new_mean = mean(elites[:, t])
                new_cov = covar(elites[:, t])
                # Smoothly update the distribution parameters.
                updated_mean = self.alpha * new_mean + (1 - self.alpha) * self.mean
                updated_cov = self.alpha * new_cov + (1 - self.alpha) * self.cov
                if kl_divergence(self.mean, self.cov, updated_mean, updated_cov) < 1e-3:
                    self.debugger.get_logger().info(f"Converged at iteration: {_}")
                    break
                self.mean = self.alpha * new_mean + (1 - self.alpha) * self.mean
                self.cov = self.alpha * new_cov + (1 - self.alpha) * self.cov
        # self.debugger.get_logger().info(f"Best u: {self.best_u}")
        
        return self.best_u
    
    def rollout(self, x0, xgoal, samples):
        costs = np.zeros(self.num_samples)
        x = x0.copy()
        x = np.tile(x, (self.num_samples, 1))
        
        for t in range(self.horizon):
            u = samples[:, t]
            x = self.dynamics.f(x, u)
            # self.debugger.get_logger().info(f"t: {x.shape}, u: {u.shape}")
            costs += self.cost.L(x, u, xgoal)
            # self.debugger.get_logger().info(f"t: {x.shape}, u: {u.shape}")
        
        # Compute total cost for each sample trajectory.
        # start_time = time.time()
        # costs_sum = np.sum(costs, axis=1)
        # Select elite samples (lowest costs)
        elite_indices = np.argsort(costs)[:self.num_elite]
        elites = samples[elite_indices]
        # end_time = time.time()
        # self.debugger.get_logger().info(f"Time taken for one roll out: {end_time - start_time}")
        # Optionally, update best found trajectory.
        # current_best_idx = njit(lambda a: np.argmin(a))(costs_sum)
        current_best_idx = elite_indices[0]
        
        self.debugger.get_logger().info(f"Current best idx: {current_best_idx}")
        if costs[current_best_idx] < self.best_cost:
            self.best_cost = costs[current_best_idx]
            self.best_u = samples[current_best_idx][0]
        return elites
        
    
    def calc_diff(self, x_real, x_prev, prev_action):
        x_pred = self.dynamics.f(x_prev, prev_action)
        diff = x_real - x_pred
        self.update_error_distribution(diff)
    
    def update_error_distribution(self, diff):
        # only care about error in theta space
        self.num_steps += 1
        delta = np.squeeze(diff)[:2] - self.err_mu
        self.err_mu = self.err_mu + delta / self.num_steps
        self.S = self.S + np.outer(delta, (np.squeeze(diff)[:2] - self.err_mu))
        self.err_cov = self.S / (self.num_steps - 1) if self.num_steps > 1 else self.S
        self.err_cov = self.err_cov + 1e-6 * np.eye(self.x_dim)
    
    def update_u_disturbance_distribution(self, x0):
        for _ in range(10):
            disturbance = np.random.multivariate_normal(self.u_disturbance_mu, self.u_disturbance_cov, size=self.num_samples)
            disturbance[0] = np.zeros(self.u_dim)
            us = self.best_u + disturbance
            # Rollout one step with the disturbance
            x_disturb = self.rollout_disturb(x0, us)
            diff = x_disturb[1:] - x_disturb[0]
            delta = diff.squeeze(1)[:, :2].detach().cpu().numpy() - self.err_mu
            # spectral decomp error covariance   
            U, S, V = np.linalg.svd(self.err_cov)
            # self.debugger.get_logger().info(f"U: {V@delta.T}, S: {S}, V: {V}, Sig: {self.err_cov}")
            rotated_delta = np.dot(V, delta.T).T
            # self.debugger.get_logger().info(f"r d: {rotated_delta.shape}")
            
            # Check if each dimension of rotated delta is within 3 standard deviations
            # of the corresponding dimension of the error covariance
            good_us = us[1:][np.all(np.abs(rotated_delta) < 3 * np.sqrt(S).repeat(self.num_samples - 1).reshape(self.num_samples - 1, 2), axis=1)]
            self.u_disturbance_mu = self.alpha* mean(good_us) + (1 - self.alpha) * self.u_disturbance_mu
            self.u_disturbance_cov = self.alpha* covar(good_us) + (1 - self.alpha) * self.u_disturbance_cov
        return self.best_u + np.random.multivariate_normal(self.u_disturbance_mu, self.u_disturbance_cov)
    
    def rollout_disturb(self, x, us):
        x = self.dynamics.f(x, us)
        return x
            

class CEMPlanner(CEMBase):
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
        super().__init__(dynamics, cost, horizon, x_dim, u_dim, num_samples, num_elite, max_iters, alpha, noise_factor, init_mean, init_cov, init_err_mu, init_err_cov, init_u_disturbance_mu, init_u_disturbance_cov, debugger)
        # Initialize the hidden state for the LSTM
        if type(self.dynamics) == Dynamics:
            self.h = (torch.zeros(self.dynamics.num_layers, 1, self.dynamics.hidden_size),
                      torch.zeros(self.dynamics.num_layers, 1, self.dynamics.hidden_size))
            self.last_h = (torch.zeros(self.dynamics.num_layers, 1, self.dynamics.hidden_size),
                            torch.zeros(self.dynamics.num_layers, 1, self.dynamics.hidden_size))

    
    def calc_diff(self, x_real, x_prev, prev_action):
        # x_real is real x t-1
        x_prev = torch.tensor(x_prev, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        prev_action = torch.tensor(prev_action, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # self.debugger.get_logger().info(f"x_prev: {x_prev.shape}")
        # self.debugger.get_logger().info(f"prev_action: {prev_action.shape}")
        # self.debugger.get_logger().info(f"h: {self.h[0].shape}")
        x_pred, _ = self.dynamics.f(x_prev, prev_action, self.last_h, grad=False)
        diff = x_real - x_pred.squeeze(1).squeeze(1).detach().cpu().numpy()
        self.update_error_distribution(diff)
        
    def rollout(self, x0, xgoal, samples):
        """
        Rollout the system with the given control samples. for T timesteps
        """
        costs = np.zeros((self.num_samples, self.horizon))
        # h = self.h[0].clone().repeat(1, self.num_samples, 1), self.h[1].clone().repeat(1, self.num_samples, 1)
        
        self.last_h = (self.h[0].clone(), self.h[1].clone())
        h = (self.h[0].clone().repeat(1, self.num_samples, 1), self.h[1].clone().repeat(1, self.num_samples, 1))
        # Evaluate cost for each sample trajectory.
        # Optimize using batch processing.
        x = torch.tensor(x0.copy(), dtype=torch.float32).unsqueeze(0)
        xgoal = torch.tensor(xgoal, dtype=torch.float32)
        x = x.repeat(self.num_samples, 1).unsqueeze(1)
        traj = torch.zeros(self.num_samples, self.horizon, 4)
        for t in range(self.horizon):
            # self.debugger.get_logger().info(f"t: {t}")
            U = torch.tensor(samples[:, t], dtype=torch.float32).unsqueeze(1)
            x, h, costs_time_step, = simulate_torch(x, h, U, self.dynamics.f, self.cost.L, xgoal, debugger=self.debugger)
            if t == 0:
                h_next = h
            traj[:, t] = x.squeeze(1).squeeze(1)
            costs[:, t] = costs_time_step
        # Compute total cost for each sample trajectory.
            costs_sum = np.sum(costs, axis=1)
            # Select elite samples (lowest costs)
            elite_indices = np.argsort(costs_sum)[:self.num_elite]
            elites = samples[elite_indices]
            # Optionally, update best found trajectory.
            current_best_idx = elite_indices[0]
            # self.debugger.get_logger().info(f"Current best idx: {current_best_idx}")
            if costs_sum[current_best_idx] < self.best_cost:
                self.best_cost = costs_sum[current_best_idx]
                self.best_u = samples[current_best_idx][0]
                self.next_best_state = traj[current_best_idx][0]
                self.h = h_next[0][:, current_best_idx:current_best_idx + 1, :], h_next[1][:, current_best_idx: current_best_idx+1, :]
        return elites
    
    def rollout_disturb(self, x, us):
        x = torch.tensor(x.copy(), dtype=torch.float32).unsqueeze(0)
        x = x.repeat(self.num_samples, 1).unsqueeze(1)
        h = self.last_h[0].clone().repeat(1, self.num_samples, 1), self.last_h[1].clone().repeat(1, self.num_samples, 1)
        us_tensors = torch.tensor(us, dtype=torch.float32).unsqueeze(1)
        x_disturb, _ = self.dynamics.f(x, us_tensors, h, grad=False)
        return x_disturb


class MPPIBase(CEMBase):
    def __init__(self,
                 dynamics, 
                 cost, 
                 horizon, 
                 x_dim, 
                 u_dim, 
                 num_samples=1000, 
                 lambda_=1.0,         # Temperature parameter for soft–max weighting.
                 noise_factor=1.0,    # Factor to scale the noise covariance.
                 init_u_nominal=None, # Initial nominal control trajectory.
                 init_cov=None,       # Initial noise covariance.
                 debugger=None,
                 init_err_mu=None,
                 init_err_cov=None,
                 init_u_disturbance_mu=None,
                 init_u_disturbance_cov=None,
                 ):
        super().__init__(dynamics, cost, horizon, x_dim, u_dim,
                         num_samples=num_samples,
                         noise_factor=noise_factor,
                         init_err_cov=init_err_cov,
                         init_err_mu=init_err_mu,
                         init_u_disturbance_mu=init_u_disturbance_mu,
                         init_u_disturbance_cov=init_u_disturbance_cov,
                         debugger=debugger)
        
        self.lambda_ = lambda_

        # Initialize the nominal control trajectory.
        # Shape: (horizon, u_dim)
        if init_u_nominal is None:
            self.u_nominal = np.zeros((self.horizon, self.u_dim))
        else:
            self.u_nominal = init_u_nominal

        # Set up the noise covariance used for sampling perturbations.
        if init_cov is None:
            self.noise_cov = noise_factor * np.eye(self.u_dim)
        else:
            self.noise_cov = init_cov
        
    def iterate(self, x0, xgoal, u_init=None):
        """
        Perform one MPPI iteration.
        1. Sample a set of noise trajectories (perturbations) to add to the current nominal trajectory.
        2. Roll out all candidate trajectories from the initial state x0.
        3. Compute the cumulative cost of each trajectory.
        4. Use an exponential weighting (soft–max) to compute a weighted average of the first-step noise.
        5. Update the nominal trajectory by shifting (dropping the first action) and inserting the new update.
        6. Return the updated first control action.
        """
        if u_init is not None:
            self.u_nominal = u_init
        # Sample noise trajectories of shape (num_samples, horizon, u_dim)
        noise_samples = np.random.multivariate_normal(
            np.zeros(self.u_dim), self.noise_cov, size=(self.num_samples, self.horizon)
        )
        # Create candidate control trajectories around the current nominal trajectory.
        # Each candidate is: u_nominal + noise, where u_nominal is broadcast to shape (num_samples, horizon, u_dim)
        candidate_controls = self.u_nominal[None, :, :] + noise_samples

        # Evaluate the cost for each candidate trajectory.
        costs = self.rollout(x0, candidate_controls, xgoal)

        # Compute weights using the soft–max (exponential) transformation.
        # Shift costs by the minimum cost to improve numerical stability.
        cost_min = np.min(costs)
        exp_cost = np.exp(-1.0 / self.lambda_ * (costs - cost_min))
        weights = exp_cost / np.sum(exp_cost)

        # Compute the weighted noise for the first time step.
        delta_u = np.sum(weights[:, None] * noise_samples[:, 0, :], axis=0)
        best_u = self.u_nominal[0] + delta_u

        if self.debugger is not None:
            self.debugger.get_logger().info(f"MPPI: min cost = {np.min(costs)}, best control update = {best_u}")

        # Shift the nominal control trajectory: drop the first action and append a zero (or any default).
        self.u_nominal[:-1] = self.u_nominal[1:]
        self.u_nominal[-1] = np.zeros(self.u_dim)
        # Update the first action with the computed best action.
        self.u_nominal[0] = best_u
        self.best_u = best_u
        return best_u

    def rollout(self, x0, candidate_controls, xgoal):
        """
        Evaluate the cost of each candidate control trajectory.
        Each candidate trajectory is rolled out from the initial state x0 using the system dynamics,
        and the cumulative cost is computed over the horizon.
        """
        costs = np.zeros(self.num_samples)
        # Start with the same initial state for all candidate trajectories.
        x = np.tile(x0, (self.num_samples, 1))
        for t in range(self.horizon):
            # candidate_controls[:, t, :] has shape (num_samples, u_dim)
            u = candidate_controls[:, t, :]
            x = self.dynamics.f(x, u)
            costs += self.cost.L(x, u, xgoal)
        return costs


class FiniteHorizonLQRController:
    def __init__(self, dynamics, cost, horizon, dt=0.075):
        """
        Parameters:
            dynamics: an instance of LimbDynamicsLQR (which provides attribute M and f)
            cost: an instance with attributes Q and R.
                  If cost.Q is not 4x4, it is assumed to be 2x2 and will be embedded into a 4x4 matrix.
            horizon: finite time horizon (an integer number of steps)
            Qf: terminal cost matrix. If None, Qf is taken to be Q (embedded in 4x4).
            dt: discretization time step (default 0.075)
        """
        self.dt = dt
        self.horizon = horizon
        self.dynamics = dynamics

        # Extract system matrices.
        # Our dynamics: f(x, u) = x + dt*( M @ [x; u] )
        # M is 4x6 so that:
        #   A = I + dt * M[:, :4]
        #   B = dt * M[:, 4:6]
        M = dynamics.M  # expected shape (4,6)
        self.A = np.eye(4) + dt * M[:, :4]
        self.B = dt * M[:, 4:6]
        
        # Process cost matrices.
        # Embed Q into 4x4 if necessary.
        if cost.Q.shape[0] != 4:
            Q2 = cost.Q.astype(np.float64)
            self.Q = np.zeros((4, 4), dtype=np.float64)
            self.Q[:2, :2] = Q2
        else:
            self.Q = cost.Q.astype(np.float64)
        self.R = cost.R.astype(np.float64)  # assume R is 2x2
        
        # # Terminal cost Qf.
        # if Qf is None:
        #     Qf = self.Q.copy()
        # else:
        #     Qf = Qf.astype(np.float64)
        
        # Backward recursion to compute time-varying gains.
        self.P = [None]*(horizon + 1)
        self.K = [None]*(horizon)
        
        # Terminal cost: No terminal cost for now.
        self.P[horizon] = np.zeros(self.Q.shape)
        
        # Recursion for t = horizon-1,...,0
        for t in range(horizon - 1, -1, -1):
            BtP = self.B.T @ self.P[t+1]
            inv_term = np.linalg.inv(self.R + BtP @ self.B)
            K_t = inv_term @ (BtP @ self.A)
            self.K[t] = K_t
            self.P[t] = self.Q + self.A.T @ self.P[t+1] @ self.A - self.A.T @ self.P[t+1] @ self.B @ K_t
        
        # Initialize time index.
        self.current_step = 0

    def control(self, x, x_goal):
        """
        Compute the control action at the current time step.
        
        Parameters:
            x: current state (4-dimensional numpy array)
            x_goal: desired goal state (4-dimensional numpy array)
                   (if your goal is only for a subset of states, extend it appropriately)
        
        Returns:
            u: control action (2-dimensional numpy array) clipped to [-1,1]
        """
        # Compute the error.
        error = x.astype(np.float64) - x_goal.astype(np.float64)
        
        # Use the gain corresponding to the current time step.
        # For steps beyond the horizon, use the last computed gain.
        idx = min(self.current_step, self.horizon - 1)
        u = -self.K[idx] @ error
        
        # Clip control to [-1, 1] element-wise.
        u = np.clip(u, -1, 1)
        
        # Increment time step.
        self.current_step += 1
        return u

    def reset(self):
        """
        Reset the controller’s time index (e.g., at the start of a new trajectory).
        """
        self.current_step = 0
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
def simulate(x0, U, f, L):
    T = U.shape[0]
    x = x0.copy()
    cost = 0.0
    for t in range(T):
        u = U[t]
        cost += L(x, u)
        x = f(x, u)
    return cost

def simulate_torch(x0, h0, U, f, L, xgoal, debugger=None):
    x = x0.clone()
    h = (h0[0].clone(), h0[1].clone())
    x_next, hs = f(x, U, h, grad=False)
    # cost = njit(lambda a, b, c: L(a, b, c))(x.detach().cpu().numpy(), U.detach().cpu().numpy(), xgoal.detach().cpu().numpy())
    x_cost = x.squeeze(1)
    U_cost = U.squeeze(1)
    cost = L(x_cost.detach().cpu().numpy(), U_cost.detach().cpu().numpy(), xgoal.detach().cpu().numpy(), debugger=debugger)
    return x_next, hs, cost
