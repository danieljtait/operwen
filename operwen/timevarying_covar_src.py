import numpy as np
import scipy.linalg
import core

def setup(tt, tknots):
    ttk = tknots[tknots < tt[-1]]
    ttk_mids = ttk + np.diff(tknots[:ttk.size+1])*0.5 # These are the points at which the ODE gets solved
    

    
def update_A_handler(tt, tknots):


    
    tti = tt.copy()
    ttk = tknots[tknots < tt[-1]]
    ttk_mids = ttk + np.diff(tknots[:ttk.size+1])*0.5  # Points at which the ode is solve
    
    ttSolve = np.concatenate(( tti, ttk_mids )) #
    tt_inds = np.argsort(ttSolve)               # 
    
    tti_ind = 

    

def getCov(tt, tk, At,
           cScales, lScales, dim, s0, t0,
           rescale=False):
    
    tt_ = np.concatenate((tt, tk[tk < tt[-1]][1:] ))
    Ati, Atk = setup_(tt, tk, At)

    rescaleVal = 1.
    if rescale:
        rescaleVal = tt[-1] - t0#

        cScales *= rescaleVal**2
        lScales /= rescaleVal
    
        tt_ /= rescaleVal

    Tmat = make_T_matrix( Ati, Atk, tt, tk, dim )
    
    A_ = Ati + Atk
    N = len(A_)

    #print cScales
    #print lScales
    #print tt_

    cov = np.zeros((N*dim, N*dim))
    for i in range(N):
    
        Aeig, UA = np.linalg.eig(A_[i])
        Aeig *= rescaleVal
        UAinv = np.linalg.inv(UA)

        for j in range(i+1):
            Beig, UB = np.linalg.eig(A_[j].T)
            Beig *= rescaleVal
            UBinv = np.linalg.inv(UB)
            
            #print Aeig, Beig

            kval = core.makeCovarMat_sqExpk_specDecomp_noSens_2(
                    np.array(tt_[i]), np.array(tt_[j]), 
                    (Aeig, UA, UAinv), (Beig, UB, UBinv),
                    lScales, cScales, dim, s0, t0)
            #print tt_[i], tt_[j]
            #print kval
            cov[i*dim:(i+1)*dim, j*dim:(j+1)*dim] = kval
            cov[j*dim:(j+1)*dim, i*dim:(i+1)*dim] = kval.T
    
    #print tt.size
    #print Tmat.shape
    #print cov.shape
    M = np.column_stack(( np.diag(np.ones(tt.size*dim)), Tmat ))
    #print M.shape
    #print cov.shape
    return np.dot(M, np.dot(cov, M.T))


def setup_( tt, tk, At, AtType='func'):
    if AtType == 'func':
        Ati = []
        Atk = []
        for i in range(tt.size):
            interval = sum(tk < tt[i]) # Interval tt[i] in (tk[interval-1], tk[interval])
            Ati.append( At(0.5*(tk[interval-1] + tk[interval])) )
        for i in range(tt.size):
            if tk[i+1] < tt[-1]:
                Atk.append( At(0.5*(tk[i]+tk[i+1])) )
            else:
                break

    tt_ = np.concatenate((tt, tk[tk < tt[-1]][1:] ))
    return Ati, Atk



def make_T_matrix( Ati, Atk, tt, tk, dim):
    NG = sum(tk < tt[-1]) - 1
    Id = np.diag(np.ones(dim))

    result = np.zeros((tt.size*dim, NG*dim))

    for i in range(tt.size):
        result[i*dim:(i+1)*dim, :] = np.column_stack((Id for nt in range(NG)))

        tauSet = tk[tk < tt[i]]
        tauSet = tauSet[1:]

        if len(tauSet > 0):
            for nt in range(tauSet.size - 1):
                eA = scipy.linalg.expm(Atk[nt+1]*(tauSet[nt+1]-tauSet[nt]))
                for k in range(nt+1):
                    result[i*dim:(i+1)*dim, k*dim:(k+1)*dim] = np.dot(eA, result[i*dim:(i+1)*dim, k*dim:(k+1)*dim] )

            eAi = scipy.linalg.expm(Ati[i]*(tt[i] - tauSet[-1]))
            for k in range(len(tauSet)):
                result[i*dim:(i+1)*dim, k*dim:(k+1)*dim] = np.dot( eAi, result[i*dim:(i+1)*dim, k*dim:(k+1)*dim] )

    return result


def make_T_matrix2(Atkm, tt, tk, dim):

    NG = sum(tk < ttk[-1]) - 1
    Id = np.diag(np.ones(dim))

    result = np.zeros((tt.size*dim, NG*dim))
