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


def setup_example(tt):
    initCond = np.array([0., 0.5, 0.8])

    Tscale = 10.
    Zscale = 0.1

    ode = nonlinearFrictionODE()

    def dZdt(Z, t):
        return Zscale*Tscale*ode.dYdt(Z/Zscale, t*Tscale)

    sol = odeint(dZdt, initCond*Zscale, tt)

    return sol


