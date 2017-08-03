

class odeModel:
    def __init__(self, initTime, initCond, dXdt, dXdt_Jac):
        self.initTime = initTime
        self.initCond = initCond
        self.dXdt = dXdt
        self.dXdt_Jac = dXdt_Jac
