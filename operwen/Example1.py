import numpy as np 
from scipy.integrate import odeint

class nonlinearFrictionODE:
    def __init__(self, xi=0.3, gamma=1., eps=1.9):
        self.xi = xi
        self.gamma = gamma
        self.eps = eps
    def dYdt(self, Y, t=0):
        udot = Y[1] - 1
        vdot = -self.gamma**2*(Y[0] + (1/self.xi)*(Y[2] + np.log(Y[1])))
        fdot = -Y[1]*(Y[2] + (1+self.eps)*np.log(Y[1]))
        return np.array([udot, vdot, fdot])

    def solve(self, initCond, tt):
        sol = odeint(self.dYdt, initCond, tt)
        return sol

    def dXdt_Jac(self, X,t=0):
        return np.array([[0., 1.], [-self.gamma**2, -self.gamma**2/(self.xi*(X[1] + 1))]])

class nonlinearFrictionODE2:
    def __init__(self, xi=0.3, gamma=1., eps = 1.9):
        self.xi = xi
        self.gamma = gamma
        self.eps = eps

    m = nonlinearFrictionODE(xi, gamma, eps)

    Yscale = 0.1
    Tscale = 20.

    def dZdt(self, Z, t=0):
        return Yscale*Tscale*m.dXdt(Z/Yscale, Tscale*t)

    def solve(self, initCond, tt):
        sol = odeint(self.dZdt, initCond, tt)


