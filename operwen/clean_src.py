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

        # various calculated quantities can be stored
        # for future computations, including predictive
        # distributions
        self.CGG_is_stored = False
        self.mean_is_stored = False

    def setup_time_vec(self, inputTimes):
        self.ttk, self.ttk_mid, self.ttk_full, self.ttk_inds = setup_time_vecs(inputTimes[0], inputTimes[-1], self.tknots)        
        self.isSetup = True

    def mean(self, t):
        if t == self.odeModelObj.initTime:
            return self.odeModelObj.initCond
        else:
            if self.mean_is_stored:
                mInd = sum(self.ttk < t) - 1
                meanmInd = self.storedMeans[mInd]
                dt = t - self.ttk[mInd]
                eA = scipy.linalg.expm( self.Atk_mid[mInd]*dt )
                return np.dot(eA, meanmInd)
            else:
                self.storedMeans = self.odeModelObj.solve(self.ttk)
                self.mean_is_stored = True
                return self.mean(t)

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
        return makeTransformation(inputTimes, self.ttk, self.get_Atk(), self.dim)


    def get_cov(self, cScales, lScales, inputTimes, store_CGG=True):
        self.inputTimes = inputTimes

        Atk_mid = self.get_Atk()
        self.Atk_mid = Atk_mid
        
        ## sorts out the limits of integration for getting the covar
        tta = []
        ttb = []
        Att = []

        for i in range(inputTimes.size):

            n = sum(self.ttk < inputTimes[i])
            
            tta.append(self.ttk[n-1])
            ttb.append(inputTimes[i])

            Att.append( Atk_mid[n-1] )

        tta = np.concatenate(( tta, self.ttk[:-1] ))
        ttb = np.concatenate(( ttb, self.ttk[1: ] ))
        for a in Atk_mid[:-1]:
            Att.append( a )

        Covar = get_cov((tta, ttb), Att, cScales, lScales, self.dim)

        if store_CGG:
            self.CGG_is_stored = True
            self.CGG = Covar
            self.CGG_eval_ts = (tta, ttb) # points and matrices at which
            self.CGG_eval_As = Att        # CGG was evaluated


        TMat = self.get_transformation_matrix(inputTimes)
        self.TMat = TMat

        TMat_ = np.column_stack(( np.diag(np.ones(inputTimes.size*self.dim)),
                                  TMat))

        
        return np.dot(TMat_, np.dot(Covar, TMat_.T))

    def get_Trow(self, tpred):
        Tnew_row = makeTransformation(np.array([tpred]), self.ttk, self.Atk_mid, self.dim)
        if Tnew_row == []:
            Tnew_row = np.zeros((self.dim, self.TMat.shape[1]))
        else:
            Tnew_row = np.column_stack((Tnew_row,
                                        np.zeros((self.dim, self.TMat.shape[1] - Tnew_row.shape[1]))
                                        ))
        return Tnew_row
                                      

    def get_kvec(self, tpred, cScales, lScales):

        n = sum(self.ttk < tpred)
        ttk_ = self.ttk[n-1]
        A_ = self.get_Atk()[n-1].T

        if self.CGG_is_stored:
            tta, ttb = self.CGG_eval_ts
            Att = self.CGG_eval_As

        cc = get_cov_col((tta, ttb), Att,
                         ttk_, tpred, A_,
                         cScales, lScales, self.dim)

        cr = np.zeros(cc.shape).T
        for i in range(cc.shape[0]/self.dim):
            cr[:, i*self.dim:(i+1)*self.dim] = cc[i*self.dim:(i+1)*self.dim,:].T            

        CGG_new = np.row_stack(( cr[:,2:], self.CGG ))
        CGG_new = np.column_stack(( cc, CGG_new ))


        Trow_new = self.get_Trow(np.array([tpred]))
        Tnew = np.row_stack((Trow_new, self.TMat))
        TMat = np.column_stack(( np.diag(np.ones((self.inputTimes.size+1)*2)),
                                 Tnew))
        return np.dot(TMat[:self.dim,:], np.dot(CGG_new, TMat.T))

    def get_Gcov_row(self, tpred, cScales, lScales, inputTimes):

        n = sum(self.ttk < tpred)
        ttk_ = self.ttk[n-1]
        A_ = self.get_Atk()[n-1].T

        if self.CGG_is_stored:
            tta, ttb = self.CGG_eval_ts
            Att = self.CGG_eval_As
            
        cc = get_cov_col((tta, ttb), Att,
                         ttk_, tpred, A_,
                         cScales, lScales, self.dim)

        cr = np.zeros(cc.shape).T
        for i in range(cc.shape[0]/self.dim):
            cr[:, i*self.dim:(i+1)*self.dim] = cc[i*self.dim:(i+1)*self.dim,:].T

        return cr

        """ Horribly backwards way of doing this

        cr = np.zeros((self.dim, cc.shape[0] - self.dim))

        for i in range((cc.shape[0]/self.dim - 1)):
            cr[:2, i*self.dim:(i+1)*self.dim] = cc[(i+1)*self.dim:(i+2)*self.dim,:].T

        # ======= Store CGG ======= #
        Covar = get_cov((tta, ttb), Att, cScales, lScales, self.dim)
        CovarNew = np.row_stack((cr, Covar))
        CovarNew = np.column_stack((cc, CovarNew))

        return CovarNew
        """
        
        


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

def get_cov_col(tatb, Att,
                sa, sb, Bs,
                cScales, lScales, dim):

    tta, ttb = tatb
    N = tta.size

    result = np.zeros((N*dim, dim))

    Beig, UB = np.linalg.eig(Bs)
    UBinv = np.linalg.inv(UB)

    for i in range(N):
        Aeig, UA = np.linalg.eig(Att[i])
        UAinv = np.linalg.inv(UA)

        k = makeCov(np.array(ttb[i]), np.array(sb),
                   (Aeig, UA, UAinv), (Beig, UB, UBinv),
                   lScales, cScales, dim, tta[i], sa)

        result[i*dim:(i+1)*dim, :dim] = k
        

    var = makeCov(np.array(sb), np.array(sb),
                  (Beig, UBinv.T, UB.T),
                  (Beig, UB, UBinv),
                  lScales, cScales, dim, sa, sa)

    return np.row_stack((var, result))


def get_cov(tatb, Att,
            cScales, lScales, dim=2):

    tta, ttb = tatb
    N = tta.size

    result = np.zeros((N*dim, N*dim))

    for i in range(N):
        Aeig, UA = np.linalg.eig(Att[i])
        UAinv = np.linalg.inv(UA)

        for j in range(i+1):
            Beig, UB = np.linalg.eig(Att[j].T)
            UBinv = np.linalg.inv(UB)

            k = makeCov(np.array(ttb[i]), np.array(ttb[j]),
                        (Aeig, UA, UAinv), (Beig, UB, UBinv),
                        lScales, cScales, dim, tta[i], tta[j])

            result[i*dim:(i+1)*dim, j*dim:(j+1)*dim] = k
            result[j*dim:(j+1)*dim, i*dim:(i+1)*dim] = k.T

    return result
