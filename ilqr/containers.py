import sympy as sp
import numpy as np
from numba import njit
from .utils import *


class Dynamics:

    def __init__(self, f, f_x, f_u, torch_model = False, debugger=None):
        '''
           Dynamics container.
              f: Function approximating the dynamics.
              f_x: Partial derivative of 'f' with respect to state
              f_u: Partial derivative of 'f' with respect to action
              f_prime: returns f_x and f_u at once
        '''
        self.f = f
        self.f_x = f_x
        self.f_u = f_u
        self.debugger = debugger
        self.f.debugger = debugger
        if torch_model:
            self.f_prime = lambda x, u, h: (f_x(x, u, h), f_u(x, u, h))
            self.num_layers = f.num_layers
            self.hidden_size = f.hidden_size
        else:
            self.f_prime = njit(lambda x, u: (f_x(x,u), f_u(x,u)))


    @staticmethod
    def Discrete(f, x_eps = 1e-4, u_eps = 1e-4):
        '''
           Construct from a discrete time dynamics function
        '''
        f = njit(f, cache = True)
        f_x = njit(lambda x, u: FiniteDiff(f, x, u, 0, x_eps))
        f_u = njit(lambda x, u: FiniteDiff(f, x, u, 1, u_eps))
        return Dynamics(f, f_x, f_u)
    
    @staticmethod
    def Torch(f, debugger=None):
        '''
           Construct from a torch model
        '''
        f_x = lambda x, u, h: autograd_jacobian(f, x, u, h, 0)
        f_u = lambda x, u, h: autograd_jacobian(f, x, u, h, 1)
        return Dynamics(f, f_x, f_u, torch_model = True, debugger=debugger)


    @staticmethod
    def SymDiscrete(f, x, u):
        '''
           Construct from Symbolic discrete time dynamics
        '''
        f_x = f.jacobian(x)
        f_u = f.jacobian(u)

        f = sympy_to_numba(f, [x, u])
        f_x = sympy_to_numba(f_x, [x, u])
        f_u = sympy_to_numba(f_u, [x, u])

        return Dynamics(f, f_x, f_u)


    @staticmethod
    def Continuous(f, dt = 0.1, x_eps = 1e-4, u_eps = 1e-4):
        '''
           Construct from a continuous time dynamics function
        '''
        f = njit(f)
        f_d = lambda x, u: x + f(x, u)*dt
        return Dynamics.Discrete(f_d, x_eps, u_eps)


    @staticmethod
    def SymContinuous(f, x, u, dt = 0.1):
        '''
           Construct from Symbolic continuous time dynamics
        '''
        return Dynamics.SymDiscrete(x + f*dt, x, u)


class LimbDynamics:
    # Derived from Richard Desatnik's paper
    k, c, a1, a2, a3, a4 = 0.33, 3.72, 0.195, 0.189, 0.154, 0.224
    M = np.array([[ 1,  0,  0,  0,  0,   0,  0,   0],
                  [-k, -c,  0,  0, a1, a2,  0,   0],
                  [ 0,  0,  0,  1,  0,   0,  0,   0],
                  [ 0,  0, -k, -c,  0,   0, a3, a4]])
                           
        
    @staticmethod
    @njit
    def cont_dynamics(X, U, M):
        n = X.shape[0]
        # Compute PWM signals for each control input in a vectorized way
        p_x_pos = np.where(U[:, 0] > 0, np.abs(U[:, 0]), 0.0)
        p_x_neg = np.where(U[:, 0] < 0, np.abs(U[:, 0]), 0.0)
        p_y_pos = np.where(U[:, 1] > 0, np.abs(U[:, 1]), 0.0)
        p_y_neg = np.where(U[:, 1] < 0, np.abs(U[:, 1]), 0.0)
        
        # Construct the augmented state matrix: shape (n, 8)
        aug_state = np.empty((n, 8), dtype=np.float64)
        aug_state[:, 0:4] = X
        aug_state[:, 4] = p_x_pos
        aug_state[:, 5] = p_x_neg
        aug_state[:, 6] = p_y_pos
        aug_state[:, 7] = p_y_neg
        return aug_state @ M.T
    
    @staticmethod
    def f(x, u, dt=0.075):
        """
        Discretizes the dynamics using Euler integration:
        x_{t+1} = x_t + dt * f(x_t, u_t) works because it is linear
        """
        dx = LimbDynamics.cont_dynamics(x, u, LimbDynamics.M)
        return x + dt * dx
    
class LimbDynamicsLQR:
    k, c, a1, a3 = 0.33, 3.72, 0.189, 0.189
    M = np.array([[ 1,  0,  0,  0,  0,   0],
                  [-k, -c,  0,  0, a1,  0],
                  [ 0,  0,  0,  1,  0,  0],
                  [ 0,  0, -k, -c,  0,  a3]])
                           
        
    @staticmethod
    @njit
    def cont_dynamics(x, u, M):
        
        # Construct the augmented state matrix: shape (n, 8)
        aug_state = np.empty(8, dtype=np.float64)
        aug_state[:4] = x
        aug_state[4:] = u
        return M @ aug_state
    
    @staticmethod
    def f(x, u, dt=0.075):
        """
        Discretizes the dynamics using Euler integration:
        x_{t+1} = x_t + dt * f(x_t, u_t) works because it is linear
        """
        dx = LimbDynamics.cont_dynamics(x, u, LimbDynamics.M)
        return x + dt * dx
    
class Cost:

    def __init__(self, L, L_x, L_u, L_xx, L_ux, L_uu, Lf, Lf_x, Lf_xx):
        '''
           Container for Cost.
              L:  Running cost
              Lf: Terminal cost
        '''
        #Running cost and it's partial derivatives
        self.L = L
        self.L_x  = L_x
        self.L_u  = L_u
        self.L_xx = L_xx
        self.L_ux = L_ux
        self.L_uu = L_uu
        self.L_prime = njit(lambda x, u: (L_x(x, u), L_u(x, u), L_xx(x, u), L_ux(x, u), L_uu(x, u)))

        #Terminal cost and it's partial derivatives
        self.Lf = Lf
        self.Lf_x = Lf_x
        self.Lf_xx = Lf_xx
        self.Lf_prime = njit(lambda x: (Lf_x(x), Lf_xx(x)))


    @staticmethod
    def Symbolic(L, Lf, x, u):
        '''
           Construct Cost from Symbolic functions
        '''
        #convert costs to sympy matrices
        L_M  = sp.Matrix([L])
        Lf_M = sp.Matrix([Lf])

        #Partial derivatives of running cost
        L_x  = L_M.jacobian(x)
        L_u  = L_M.jacobian(u)
        L_xx = L_x.jacobian(x)
        L_ux = L_u.jacobian(x)
        L_uu = L_u.jacobian(u)

        #Partial derivatives of terminal cost
        Lf_x  = Lf_M.jacobian(x)
        Lf_xx = Lf_x.jacobian(x)

        #Convert all sympy objects to numba JIT functions
        funs = [L, L_x, L_u, L_xx, L_ux, L_uu, Lf, Lf_x, Lf_xx]
        for i in range(9):
          args = [x, u] if i < 6 else [x]
          redu = 0 if i in [3, 4, 5, 8] else 1
          funs[i] = sympy_to_numba(funs[i], args, redu)

        return Cost(*funs)

    @staticmethod
    def QR(Q, R, QT, x_goal, add_on = 0):
        '''
           Construct Quadratic cost
        '''
        x, u = GetSyms(Q.shape[0], R.shape[0])
        er = x - sp.Matrix(x_goal)
        L  = er.T@Q@er + u.T@R@u
        Lf = er.T@QT@er
        return Cost.Symbolic(L[0] + add_on, Lf[0], x, u)

class CEMCost:
    def __init__(self, Q, R):
        self.Q = Q.astype(np.float64)
        self.R = R.astype(np.float64)
    
    def L(self, x, u, x_goal, debugger=None):
        if x.shape[1] == 4:
            x = x.copy()[:, :2]
        x = x.astype(np.float64)
        u = u.astype(np.float64)
        x_goal = x_goal.astype(np.float64)
        return _L_numba(x, u, x_goal, self.Q, self.R)

@njit
def _L_numba(x, u, x_goal, Q, R):
    """
    A numba-compiled function that expects x, u, x_goal, Q, R as raw NumPy arrays.
    It returns the cost = diag((x - x_goal) Q (x - x_goal).T) + diag(u R u.T).
    """
    # If you need to handle x.shape[1] == 4 by slicing columns,
    # you can do that logic here or do it before calling this function.
    
    er = x - x_goal
    return np.diag(er @ Q @ er.T) + np.diag(u @ R @ u.T)