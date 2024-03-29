"""
Author: Marvin Ahlborn

Python implementation of Direct Trapezoidal Collocation for solving Nonlinear Optimal Control Problems.
It uses the scipy's NLP Solver to compute an open loop trajectory for a given dynamical system
subject to linear path constraints and minimizing a cost function.

References:
'An Introduction to Trajectory Optimization' by Matthew Kelly
'Practical Methods for Optimal Control Using Nonlinear Programming' by John T. Betts
"""

import numpy as np
from matplotlib import pyplot as plt

from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize

class TrapezoidOpt:
    def __init__(self, N, n, m, coll_grid, J, J_jac, J_hess, f, f_jac, f_hess):
        """
        There are N collocation spline segments (N+1 collocation points).
        n is the state dimension, m the actuation dimension.

        coll_grid is the time grid of collocation points. It's normalized by the final time which is an optimization variable.
        It has to start with 0 (beginning) and end with 1 (final time) and it has to have N+1 entries for the times
        at each collocation point. An example for an evenly spaced grid with five collocation points would be
        [0, 0.25, 0.5, 0.75, 1].

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

        self.f = f  # State dynamics of the form x_dot = f(x, u)
        self.f_jac = f_jac
        self.f_hess = f_hess

        # Collocation constraints
        self.coll_constr = NonlinearConstraint(self._coll_constr, .0, .0, jac=self._coll_jac, hess=self._coll_hess)
        self.lin_constr = []
    
    def set_bounds(self, x_l, x_h, u_l, u_h, tf_l, tf_h):
        """
        x_l = [x_l_0, x_l_1, ... , x_l_N], Length: n*(N+1)
        Similar for x_h, u_l and u_h.
        tf_l and tf_h are scalars.
        """
        bounds_l = np.concatenate((x_l, u_l, np.array([tf_l])))
        bounds_h = np.concatenate((x_h, u_h, np.array([tf_h])))

        self.bounds = Bounds(bounds_l, bounds_h)
    
    def add_constr(self, C, c_l, c_h):
        """
        Adds linear path constraints of the form: c_l <= C @ s <= c_h
        to the optimization problem. s is the augmented state vector of the form
        [x0, x1, ... , xN, u0, u1, ... , uN, tf] with size n*(N+1)+m*(N+1)+1.
        """
        constr = LinearConstraint(C, c_l, c_h)
        self.lin_constr.append(constr)
    
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
        u = np.reshape(u_r, (N+1, m))

        f_res = np.array([self.f(x[k], u[k]) for k in range(N+1)])  # (N+1) x n

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
        u = np.reshape(u_r, (N+1, m))

        I = np.eye(n)  # n x n

        f_res = np.array([self.f(x[k], u[k]) for k in range(N+1)])  # (N+1) x n
        
        f_jac_res = np.array([self.f_jac(x[k], u[k]) for k in range(N+1)])  # (N+1) x n x (n+m)
        f_jac_x_res = f_jac_res[:, :, :n]  # (N+1) x n x n
        f_jac_u_res = f_jac_res[:, :, n:]  # (N+1) x n x m

        c_jac = np.zeros((n*N, (n+m)*(N+1)+1))

        for k in range(N):
            c_jac_x_k = I + h[k]/2 * tf * f_jac_x_res[k]
            c_jac_x_kp = -I + h[k]/2 * tf * f_jac_x_res[k+1]
            c_jac_u_k = h[k]/2 * tf * f_jac_u_res[k]
            c_jac_u_kp = h[k]/2 * tf * f_jac_u_res[k+1]
            c_jac_tf = h[k]/2 * (f_res[k+1] + f_res[k])

            c_jac[n*k : n*(k+1), n*k : n*(k+2)] = np.concatenate((c_jac_x_k, c_jac_x_kp), axis=1)
            c_jac[n*k : n*(k+1), n*(N+1)+m*k : n*(N+1)+m*(k+2)] = np.concatenate((c_jac_u_k, c_jac_u_kp), axis=1)
            c_jac[n*k : n*(k+1), -1] = c_jac_tf
        
        return c_jac
    
    def _coll_hess(self, s, v):
        """
        This was not as hard to derive but very annoying to write out.
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
        u = np.reshape(u_r, (N+1, m))

        f_jac_res = np.array([self.f_jac(x[i], u[i]) for i in range(N+1)])  # (N+1) x n x (n+m)
        f_jac_x_res = f_jac_res[:, :, :n]  # (N+1) x n x n
        f_jac_u_res = f_jac_res[:, :, n:]  # (N+1) x n x m

        f_hess_res = np.array([self.f_hess(x[i], u[i]) for i in range(N+1)])  # (N+1) x n x (n+m) x (n+m)
        f_hess_xx_res = f_hess_res[:, :, :n, :n]  # (N+1) x n x n x n
        f_hess_uu_res = f_hess_res[:, :, n:, n:]  # (N+1) x n x m x m
        f_hess_xu_res = f_hess_res[:, :, :n, n:]  # (N+1) x n x n x m

        c_hess = np.zeros((n*(N+1)+m*(N+1)+1, n*(N+1)+m*(N+1)+1))

        # Iterate through all constraints
        for k in range(N):
            for i in range(n):
                # Helpfull indices
                id_v = n*k+i

                id_xa = n*k
                id_xb = n*(k+1)
                id_xc = n*(k+2)

                id_ua = n*(N+1)+m*k
                id_ub = n*(N+1)+m*(k+1)
                id_uc = n*(N+1)+m*(k+2)

                # d2f/dx2
                c_hess[id_xa:id_xb, id_xa:id_xb] += v[id_v] * h[k]/2 * tf * f_hess_xx_res[k, i]
                c_hess[id_xb:id_xc, id_xb:id_xc] += v[id_v] * h[k]/2 * tf * f_hess_xx_res[k+1, i]

                # d2f/du2
                c_hess[id_ua:id_ub, id_ua:id_ub] += v[id_v] * h[k]/2 * tf * f_hess_uu_res[k, i]
                c_hess[id_ub:id_uc, id_ub:id_uc] += v[id_v] * h[k]/2 * tf * f_hess_uu_res[k+1, i]

                # d2f/dxdu
                c_hess[id_xa:id_xb, id_ua:id_ub] += v[id_v] * h[k]/2 * tf * f_hess_xu_res[k, i]
                c_hess[id_xb:id_xc, id_ub:id_uc] += v[id_v] * h[k]/2 * tf * f_hess_xu_res[k+1, i]
                # Symmetry
                c_hess[id_ua:id_ub, id_xa:id_xb] += v[id_v] * h[k]/2 * tf * f_hess_xu_res[k, i].T
                c_hess[id_ub:id_uc, id_xb:id_xc] += v[id_v] * h[k]/2 * tf * f_hess_xu_res[k+1, i].T

                # d2f/dtfdx
                c_hess[id_xa:id_xb, -1] += v[id_v] * h[k]/2 * f_jac_x_res[k, i]
                c_hess[id_xb:id_xc, -1] += v[id_v] * h[k]/2 * f_jac_x_res[k+1, i]
                # Symmetry
                c_hess[-1, id_xa:id_xb] += v[id_v] * h[k]/2 * f_jac_x_res[k, i]
                c_hess[-1, id_xb:id_xc] += v[id_v] * h[k]/2 * f_jac_x_res[k+1, i]

                # d2f/dtfdu
                c_hess[id_ua:id_ub, -1] += v[id_v] * h[k]/2 * f_jac_u_res[k, i]
                c_hess[id_ub:id_uc, -1] += v[id_v] * h[k]/2 * f_jac_u_res[k+1, i]
                # Symmetry
                c_hess[-1, id_ua:id_ub] += v[id_v] * h[k]/2 * f_jac_u_res[k, i]
                c_hess[-1, id_ub:id_uc] += v[id_v] * h[k]/2 * f_jac_u_res[k+1, i]
        
        return c_hess

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

        s0 = np.concatenate((x0_r, u0_r, np.array([tf0])))

        constr = self.lin_constr.copy()
        constr.append(self.coll_constr)

        res = minimize(self._cost, s0, method='trust-constr', jac=self._cost_jac,
                         hess=self._cost_hess, bounds=self.bounds, constraints=constr)
        
        s_res = res.x

        x_res_r = s_res[:n*(N+1)]
        u_res_r = s_res[n*(N+1):-1]
        tf_res = s_res[-1]

        x_res = np.reshape(x_res_r, (N+1, n))
        u_res = np.reshape(u_res_r, (N+1, m))

        # Real functions of time with spline interpolation between collocation points
        func_x = lambda t : self._func_x(x_res, u_res, tf_res, t)
        func_u = lambda t : self._func_u(u_res, tf_res, t)

        return x_res, u_res, tf_res, func_x, func_u
    
    def _func_x(self, x, u, tf, t):
        """
        Computes a continuous function x(t) from the x values at discrete collocation points.
        """
        t_s = t / tf
        k = np.atleast_1d(np.argwhere(self.coll_grid <= t_s).squeeze())[-1]  # Get Collocation Point right before time t.

        # Edge case
        if k == len(self.coll_grid) - 1:
            return x[k]
        
        h_k = tf * (self.coll_grid[k+1] - self.coll_grid[k])
        tau = t - tf * self.coll_grid[k]

        f_ka = self.f(x[k], u[k])
        f_kb = self.f(x[k+1], u[k+1])

        return x[k] + f_ka * tau + tau**2 / (2 * h_k) * (f_kb - f_ka)  # Quadratic Spline Approximation
    
    def _func_u(self, u, tf, t):
        """
        Computes a continuous function u(t) from the u values at discrete collocation points.
        """
        t_s = t / tf
        k = np.atleast_1d(np.argwhere(self.coll_grid <= t_s).squeeze())[-1]  # Get Collocation Point right before time t.

        # Edge case
        if k == len(self.coll_grid) - 1:
            return u[k]

        h_k = tf * (self.coll_grid[k+1] - self.coll_grid[k])
        tau = t - tf * self.coll_grid[k]

        return u[k] + tau / h_k * (u[k+1] - u[k])  # Linear interpolation


"""
Classic Block Move Example
"""
if __name__ == '__main__':
    f = lambda x, u : np.array([x[1], u[0]], dtype=np.float64)  # x_dot_dot = u
    f_jac = lambda x, u : np.array([[0, 1, 0], [0, 0, 1]], dtype=np.float64)
    f_hess = lambda x, u : np.zeros((2, 3, 3), dtype=np.float64)

    J = lambda x, u, tf : (u**2).sum()
    J_jac = lambda x, u, tf : np.concatenate((np.zeros_like(x), 2*u, np.zeros(1)))
    J_hess = lambda x, u, tf : np.diag(np.concatenate((np.zeros_like(x), np.ones_like(u)*2, np.zeros(1))))

    N = 100
    n = 2
    m = 1
    coll_grid = np.linspace(0, 1, N+1)

    x_l = np.ones(n*(N+1)) * -np.inf
    x_l[:2] = np.array([0., 0.])
    x_l[-2:] = np.array([1., 0.])
    x_h = np.ones(n*(N+1)) * np.inf
    x_h[:2] = np.array([0., 0.])
    x_h[-2:] = np.array([1., 0.])
    u_l = np.ones(m*(N+1)) * -np.inf
    u_h = np.ones(m*(N+1)) * np.inf
    tf_l = 1.0
    tf_h = 1.0

    x0 = np.ones(n*(N+1))
    x0[::2] = np.linspace(0, 1, N+1)
    u0 = np.zeros(m*(N+1))
    tf0 = 1.0

    optimizer = TrapezoidOpt(N, n, m, coll_grid, J, J_jac, J_hess, f, f_jac, f_hess)
    optimizer.set_bounds(x_l, x_h, u_l, u_h, tf_l, tf_h)
    x_res, u_res, tf_res, func_x, func_u = optimizer.optimize(x0, u0, tf0)

    z = 100
    t = np.linspace(0, 1, z)
    x = np.empty(z)
    v = np.empty(z)
    u = np.empty(z)

    for i in range(z):
        x[i] = func_x(t[i])[0]
        v[i] = func_x(t[i])[1]
        u[i] = func_u(t[i])[0]

    plt.plot(t, x, 'k-')
    plt.plot(t, v, 'b-')
    plt.plot(t, u, 'r-')
    plt.grid(True)
    plt.savefig('block_move.png', dpi=400)