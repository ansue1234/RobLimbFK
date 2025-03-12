import sympy as sp
import numpy as np
from numba import njit
import torch


def GetSyms(n_x, n_u):
  '''
      Returns matrices with symbolic variables for states and actions
      n_x: state size
      n_u: action size
  '''

  x = sp.IndexedBase('x')
  u = sp.IndexedBase('u')
  xs = sp.Matrix([x[i] for i in range(n_x)])
  us = sp.Matrix([u[i] for i in range(n_u)])
  return xs, us


def Constrain(cs, eps = 1e-4):
    '''
    Constraint via logarithmic barrier function
    Limitation: Doesn't work with infeasible initial guess.
    cs: list of constraints of form g(x, u) >= 0
    eps : parameters
    '''
    cost = 0
    for i in range(len(cs)):
        cost -= sp.log(cs[i] + eps)
    return 0.1*cost


def Bounded(vars, high, low, *params):
    '''
    Logarithmic barrier function to constrain variables.
    Limitation: Doesn't work with infeasible initial guess.
    '''
    cs = []
    for i in range(len(vars)):
        diff = (high[i] - low[i])/2
        cs.append((high[i] - vars[i])/diff)
        cs.append((vars[i] - low[i])/diff)
    return Constrain(cs, *params)


def SoftConstrain(cs, alpha = 0.01, beta = 10):
    '''
    Constraint via exponential barrier function
    cs: list of constraints of form g(x, u) >= 0
    alpha, beta : parameters
    '''
    cost = 0
    for i in range(len(cs)):
        cost += alpha*sp.exp(-beta*cs[i])
    return cost


def Smooth_abs(x, alpha = 0.25):
    '''
    smooth absolute value
    '''
    return sp.sqrt(x**2 + alpha**2) - alpha


# @njit
def FiniteDiff(fun, x, u, i, eps):
  '''
     Finite difference approximation
  '''

  args = (x, u)
  fun0 = fun(x, u)

  m = x.size
  n = args[i].size

  Jac = np.zeros((m, n))
  for k in range(n):
    args[i][k] += eps
    Jac[:, k] = (fun(args[0], args[1]) - fun0)/eps
    args[i][k] -= eps

  return Jac

def autograd_jacobian(fun, x, u, hidden, i):
    """
    Compute the Jacobian of fun with respect to one of its arguments (x or u)
    using PyTorch's autograd.
    
    Parameters:
      fun: A callable representing the neural network’s forward pass.
      x, u: Input tensors.
      i: Index specifying which argument to differentiate with respect to.
         Use 0 for x and 1 for u.
    
    Returns:
      The Jacobian tensor with shape: output_shape + input_shape.
    """
    if i == 0:
        # Differentiate with respect to x
        x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        # Define a function of x with u fixed
        f = lambda x_: fun(x_, u, hidden)
        # Compute the Jacobian
        jac = torch.autograd.functional.jacobian(f, x)
    elif i == 1:
        # Differentiate with respect to u
        u = torch.tensor(u, dtype=torch.float32, requires_grad=True)
        # Define a function of u with x fixed
        f = lambda u_: fun(x, u_, hidden)
        # Compute the Jacobian
        jac = torch.autograd.functional.jacobian(f, u)
    else:
        raise ValueError("Invalid index: choose 0 (for x) or 1 (for u)")
    
    return jac




def sympy_to_numba(f, args, redu = True):
    '''
       Converts sympy matrix or expression to numba jitted function
    '''
    modules = [{'atan2':np.arctan2}, 'numpy']

    if isinstance(f, sp.Matrix):
        #To convert all elements to floats
        m, n = f.shape
        f += 1e-64*np.ones((m, n))

        #To eleminate extra dimension
        if (n == 1 or m == 1) and redu:
            if n == 1: f = f.T
            f = sp.Array(f)[0, :]
            f = njit(sp.lambdify(args, f, modules = modules))
            f_new = lambda *args: np.asarray(f(*args))
            return njit(f_new)

    f = sp.lambdify(args, f, modules = modules)
    return njit(f)

# Finds KL Divergence
@njit
def kl_divergence(mu0, sigma0, mu1, sigma1):
    """
    Compute the KL divergence between two multivariate Gaussians:
        P ~ N(mu0, sigma0)
        Q ~ N(mu1, sigma1)
    
    Parameters:
        mu0 (np.ndarray): Mean vector of P.
        sigma0 (np.ndarray): Covariance matrix of P.
        mu1 (np.ndarray): Mean vector of Q.
        sigma1 (np.ndarray): Covariance matrix of Q.
    
    Returns:
        float: The KL divergence D_KL(P || Q).
    """
    k = mu0.shape[0]
    sigma1_inv = np.linalg.inv(sigma1)
    det_sigma1 = np.linalg.det(sigma1)
    det_sigma0 = np.linalg.det(sigma0)
    
    term1 = np.log(det_sigma1 / det_sigma0)
    term2 = np.trace(np.dot(sigma1_inv, sigma0))
    
    diff = mu1 - mu0
    term3 = np.dot(diff.T, np.dot(sigma1_inv, diff))
    
    return 0.5 * (term1 - k + term2 + term3)


def mean(a):
    return np.mean(a, axis=0)

def covar(a):
    return np.cov(a, rowvar=False)