import numpy as np
import scipy.integrate
import scipy.linalg

class ODE:
    def __init__(self, initTime, initCond, dXdt, dXdt_xJac,
                 isLinear=True):
        
        self.initTime = initTime
        self.initCond = initCond
        self.dXdt = dXdt
        self.dXdt_xJac = dXdt_xJac

        self.isLinear = isLinear

    def solve(self,tt):
        i = 0
        if tt[0] != self.initTime:
            tt = np.concatenate(([self.initTime], tt))
            i = 1
        return scipy.integrate.odeint(self.dXdt, self.initCond, tt)[i:,]


class tvODE:
    def __init__(self, tknots, odeModelObj, dim):
        self.tknots = tknots
        self.odeModelObj = odeModelObj
        self.isSetup = False
        self.dim = dim

    def setup_time_vec(self, inputTimes):
        self.ttk, self.ttk_mid, self.ttk_full, self.ttk_inds = setup_time_vecs(inputTimes[0], inputTimes[-1], self.tknots)        

    ###
    # Returns a list of matrices [ A(t) t in self.ttk_mids ] where
    # A is the Jacobian of dXdt in the specified ODE model
    def get_Atk(self):
        if self.odeModelObj.isLinear:
            # Linear model the Jacobian will simply be A(t) and so independent
            # of the X argument
            return [self.odeModelObj.dXdt_xJac(None, t) for t in self.ttk_mid]
        else:
            sol = self.odeModelObj.solve(self.ttk_full)
            sol_tk = sol[self.ttk_inds,:]
        return [self.odeModelObj.dXdt_xJac(x, t) for x,t in zip(sol_tk, self.ttk_mid)]

    def get_transformation_matrix(self, inputTimes):
        return makeTransformation(inputTimes, self.ttk_mid, self.get_Atk(), self.dim)



def setup_time_vecs(tmin, tmax, tknots):
    if tmin < tknots[0] or tmax > tknots[-1]:
        print "input point outside the range of knot points"
        return None,_

    else:
        ttk = tknots[tknots < tmax]

        ttk_mid = ttk + 0.5*np.diff(tknots[:ttk.size + 1])
        
        ttk_full = np.concatenate(( ttk_mid, [] )) # Consider augmenting the size of tvec

        ttk_solve_sort_inds = np.argsort(ttk_full)
        ttk_inds = [np.where(ttk_solve_sort_inds == i)[0][0] for i in range(ttk.size) ]

        return ttk, ttk_mid, ttk_full[ttk_solve_sort_inds], ttk_inds


##
# 
def makeTransformation(tt, tk, Atk, dim):

    NG = sum(tk < tt[-1]) - 1

    result = np.zeros((tt.size*dim, NG*dim))
    Id = np.diag(np.ones(dim))

    tauSet = []
    M = 0
    for i in range(tt.size):
        if tt[i] > tk[1]:

            tauSet = np.concatenate(( tauSet, [s for s in tk[M+1:] if s < tt[i] ] ))
            M = len(tauSet)

            result[i*dim:(i+1)*dim, :M*dim] = np.column_stack((Id for nt in range(M) ))

            for k in range(M-1):
                eA = scipy.linalg.expm( Atk[k+1]*(tauSet[k+1] - tauSet[k]) )
                for j in range(k+1):
                    result[i*dim:(i+1)*dim, j*dim:(j+1)*dim] = np.dot(eA, result[i*dim:(i+1)*dim, j*dim:(j+1)*dim])

            eAi = scipy.linalg.expm(Atk[M]*(tt[i] - tauSet[-1]))
            for k in range(M):
                result[i*dim:(i+1)*dim, k*dim:(k+1)*dim] = np.dot(eAi, result[i*dim:(i+1)*dim, k*dim:(k+1)*dim])
            
    return result


from core import makeCovarMat_sqExpk_specDecomp_noSens_2 as makeCov

def get_cov(tt, Att,
            cScales, lScales, dim=2, s0=0., t0=0.):
    N = tt.size

    result = np.zeros((N*dim, N*dim))

    for i in range(N):
        Aeig, UA = np.linalg.eig(Att[i])
        UAinv = np.linalg.inv(UA)

        for j in range(i+1):
            Beig, UB = np.linalg.eig(Att[j].T)
            UBinv = np.linalg.inv(UB)

            k = makeCov(np.array(tt[i]), np.array(tt[j]),
                        (Aeig, UA, UAinv), (Beig, UB, UBinv),
                        lScales = lScales, cScales = cScales, dim, s0, t0)

            result[i:(i+1)*dim, j:(j+1)*dim] = k
            result[j:(j+1)*dim, i:(i+1)*dim] = k.T

    return result
