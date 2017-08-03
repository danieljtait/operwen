import numpy as np
import scipy.integrate

class odeModel:
    def __init__(self, isLinear = True):
        self.isLinear = isLinear

    def set_initCond(self, initTime, initCond):
        self.initTime = initTime
        self.initCond = initCond

    def set_flow(dXdt, dXdt_Jac = None):
        self.dXdt = dXdt
        self.dXdt_Jac = dXdt_Jac
        
    def solve(tt):
        if tt[0] != self.initTime:
            return None
        else:
            return scipy.integrate.odeint(self.dXdt, self.initCond, tt)


