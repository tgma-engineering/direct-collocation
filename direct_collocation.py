import numpy as np

from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize

class TrapezoidOpt:
    def __init__(self, N, n, m, coll_grid, J, J_jac, J_hess, f, f_jac, f_hess):
        """
        There are N collocation spline segments (N+1 collocation points).
        n is the state dimension, m the actuation dimension.

        J(x, u, tf) -> Scalar. x and u are all states and actuations from all collocation points
        concatenated in a big list of sizes n*(N+1) and m*(N+1).
        J_jac(x, u, tf) -> List of len n*(N+1)+m*(N+1)+1. Just the gradient.
        J_hess(x, u, tf) -> Square matrix of size n*(N+1)+m*(N+1)+1. Hessian of augmented state, actuation
        and final time vector.

        f(x, u) -> List of len n. System dynamics of form x_dot = f(x, u).
        f_jac(x, u) -> Matrix of shape n x n+m. Jacobian of f with respect to x concatenated with u.
        f_hess(x, u) -> Tensor of shape n x n+m x n+m. Hessian of all n f constraints with respect to x concatenated with u.
        """
        self.N = N
        self.n = n
        self.m = m

        self.coll_grid = coll_grid
        
        self.J = J  # Cost function to be minimized
        self.J_jac = J_jac
        self.J_hess = J_hess

        self.f = f  # State dynamics of the form x_dot = f(x, u, t)
        self.f_jac = f_jac
        self.f_hess = f_hess

        # Collocation constraints
        self.constr = NonlinearConstraint(self._coll_constr, .0, .0, jac=self._coll_jac, hess=self._coll_hess)
    
    def _set_bounds(self, x_l, x_h, u_l, u_h, tf_l, tf_h):
        """
        x_l = [x_l_0, x_l_1, ... , x_l_N], Length: n*(N+1)
        Similar for others.
        tf_l and tf_h are scalars.
        """
        bounds_l = np.concatenate((x_l, u_l, tf_l))
        bounds_h = np.concatenate((x_h, u_h, tf_h))

        self.bounds = Bounds(bounds_l, bounds_h)
    
    def _coll_constr(self, s):
        N = self.N
        n = self.n
        m = self.m
        t = self.coll_grid

        h_r = t[1:N+1] - t[:N]
        h = np.atleast_2d(h_r).T

        x_r = s[:n*(N+1)]
        u_r = s[n*(N+1):-1]
        tf = s[-1]

        x = np.reshape(x_r, (N+1, n))
        u = np.reshpae(u_r, (N+1, m))

        f_res = np.array([self.f(x[i], u[i]) for i in range(N+1)])  # (N+1) x n

        constr = x[:N] - x[1:N+1] + h/2 * tf * (f_res[1:N+1] + f_res[:N])
        return np.reshape(constr, N*n)
    
    def _coll_jac(self, s):
        """
        Working this out was ugly.
        """
        N = self.N
        n = self.n
        m = self.m
        t = self.coll_grid

        h = t[1:N+1] - t[:N]

        x_r = s[:n*(N+1)]
        u_r = s[n*(N+1):-1]
        tf = s[-1]

        x = np.reshape(x_r, (N+1, n))
        u = np.reshpae(u_r, (N+1, m))

        I = np.eye(n)  # n x n

        f_res = np.array([self.f(x[i], u[i]) for i in range(N+1)])  # (N+1) x n
        
        f_jac_res = np.array([self.f_jac(x[i], u[i]) for i in range(N+1)])  # (N+1) x n x (n+m)
        f_jac_x_res = f_jac_res[:, :, :n]  # (N+1) x n x n
        f_jac_u_res = f_jac_res[:, :, n:]  # (N+1) x n x m

        c_jac = np.zeros((n*N, (n+m)*(N+1)+1))

        for i in range(N):
            c_jac_x_k = I + h[i]/2 * tf * f_jac_x_res[i]
            c_jac_x_kp = -I + h[i]/2 * tf * f_jac_x_res[i+1]
            c_jac_u_k = h[i]/2 * tf * f_jac_u_res[i]
            c_jac_u_kp = h[i]/2 * tf * f_jac_u_res[i+1]
            c_jac_tf = h[i]/2 * (f_res[i+1] + f_res[i])

            c_jac[n*i : n*(i+1), n*i : n*(i+2)] = np.concatenate((c_jac_x_k, c_jac_x_kp), axis=1)
            c_jac[n*i : n*(i+1), n*(N+1)+m*i : n*(N+1)+m*(i+2)] = np.concatenate((c_jac_u_k, c_jac_u_kp), axis=1)
            c_jac[n*i : n*(i+1), -1] = c_jac_tf
        
        return c_jac
    
    def _coll_hess(self, s, v):
        # TODO
        pass

    def _cost(self, s):
        N = self.N
        n = self.n

        x = s[:n*(N+1)]
        u = s[n*(N+1):-1]
        tf = s[-1]

        return self.J(x, u, tf)
    
    def _cost_jac(self, s):
        N = self.N
        n = self.n

        x = s[:n*(N+1)]
        u = s[n*(N+1):-1]
        tf = s[-1]

        return self.J_jac(x, u, tf)

    def _cost_hess(self, s):
        N = self.N
        n = self.n

        x = s[:n*(N+1)]
        u = s[n*(N+1):-1]
        tf = s[-1]

        return self.J_hess(x, u, tf)

    def optimize(self, x0, u0, tf0):
        N = self.N
        n = self.n
        m = self.m

        x0_r = np.reshape(x0, n * (N+1))
        u0_r = np.reshape(u0, m * (N+1))

        s0 = np.concatenate((x0, u0, tf0))

        s_res = minimize(self._cost, s0, method='trust-constr', jac=self._cost_jac,
                         hess=self._cost_hess, bounds=self.bounds, constraints=self.constr)
        
        x_res_r = s_res[:n*(N+1)]
        u_res_r = s_res[n*(N+1):-1]
        tf_res = s_res[-1]

        x_res = np.reshape(x_res_r, (N+1, n))
        u_res = np.reshape(u_res_r, (N+1, m))

        return x_res, u_res, tf_res