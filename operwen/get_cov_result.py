import numpy as np
import scipy.linalg
from timevarying_covar import time_discretisation_handler
from covariance_util import ksqexp


def result(evalt, tk, dim,
            tcond, cScales, lScales,
            odeModel, initCond, initTime):

    tta, ttm, ttb, Att = time_discretisation_handler(evalt, tk, lambda x: None, dim)
    

    ### 
    # Handle the sorting of time arguments for a 
    # given discretisation scheme
    sortInds = np.argsort(ttm)
    evalt_inds = [np.where(sortInds == i)[0][0] for i in range(evalt.size) ]
    tk_inds = [i for i in range(ttm.size) if not i in evalt_inds]

    tt_ode = np.concatenate(([initTime], ttm[sortInds]))

    sol = odeModel.solve(initCond, tt_ode)
    At_ = [odeModel.dXdt_Jac(x) for x in sol[1:,:dim] ]
    Ati = [At_[i] for i in evalt_inds]
    Atk = [At_[i] for i in tk_inds]


    ##### Ready to make covariance matrix

    Tmat = makeT( Ati, Atk, evalt, tk, dim)

    CGGcond = cond_wrapped_makeCov_sym( (tta, ttm, ttb), Ati + Atk,
                                        dim, tcond, cScales, lScales)

    print Tmat.shape
    print CGGcond.shape

    M = np.column_stack((np.diag(np.ones(evalt.size*dim)), Tmat))
    print M.shape
    return np.dot(M, np.dot(CGGcond, M.T))

# ================================= #
def makeT( Ati, Atk, tt, tk, dim):
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


def Kcond(tt, tcond, cScales, lScales, dim):
    result = np.zeros((tt.size*dim, tt.size*dim))
    tt_ = np.concatenate((tt, tcond))
    Sigma = ksqexp(tt_, tt_, cScales, lScales, S=None, returnType='ind')

    for k in range(dim):
        if cScales[k] != 0.:
            Sigma_k = Sigma[k]             # The kth component of the GP
            S11 = Sigma_k[:tt.size,:tt.size]
            S12 = Sigma_k[:tt.size,tt.size:]
            S22 = Sigma_k[tt.size:, tt.size:]
            L = np.linalg.cholesky(S22)

            covCond = S11 - np.dot(S12, np.linalg.solve(L.T, np.linalg.solve(L, S12.T)))

            # get the indices to put the conditioned covariance back into result
            rows = np.column_stack(( np.array([i*dim + k for i in range(tt.size)])[:,None] for j in range(tt.size)))
            result[rows, rows.T] = covCond

    return result

def cond_wrapped_makeCov_sym(S, A, dim, tcond, cScales, lScales):
    sa, sm, sb = S
    ss_full = np.concatenate((sa, sm, sb))
    K = Kcond(ss_full, tcond, cScales, lScales, dim)

    N = S[0].size

    K11 = K[:N*dim, :N*dim]
    K12 = K[:N*dim, N*dim:2*N*dim]
    K13 = K[:N*dim, 2*N*dim:3*N*dim]

    K21 = K12.T
    K22 = K[N*dim:2*N*dim, N*dim:2*N*dim]
    K23 = K[N*dim:2*N*dim, 2*N*dim:3*N*dim]

    K31 = K13.T
    K32 = K23.T
    K33 = K[2*N*dim:, 2*N*dim: ]

    result = np.zeros((N*dim, N*dim))

    for i in range(N):

        eAi1 = scipy.linalg.expm(A[i]*(sb[i]-sa[i]))
        eAi2 = scipy.linalg.expm(A[i]*(sb[i]-sm[i]))

        for j in range(i+1):

            eBj1 = scipy.linalg.expm(A[j].T*(sb[j]-sa[j]))
            eBj2 = scipy.linalg.expm(A[j].T*(sb[j]-sm[j]))

            I11 = np.dot(eAi1, np.dot(K11[i*dim:(i+1)*dim, j*dim:(j+1)*dim], eBj1))
            I12 = 4*np.dot(eAi1, np.dot(K12[i*dim:(i+1)*dim, j*dim:(j+1)*dim], eBj2))
            I13 = np.dot(eAi1, K13[i*dim:(i+1)*dim, j*dim:(j+1)*dim])

            I21 = np.dot(eAi2, np.dot(K21[i*dim:(i+1)*dim, j*dim:(j+1)*dim], eBj1))
            I22 = 4*np.dot(eAi2, np.dot(K22[i*dim:(i+1)*dim, j*dim:(j+1)*dim], eBj2))
            I23 = np.dot(eAi2, K23[i*dim:(i+1)*dim, j*dim:(j+1)*dim])

            I31 = np.dot(K31[i*dim:(i+1)*dim, j*dim:(j+1)*dim], eBj1)
            I32 = 4*np.dot(K32[i*dim:(i+1)*dim, j*dim:(j+1)*dim], eBj2)
            I33 = K33[i*dim:(i+1)*dim, j*dim:(j+1)*dim]

            expr1 = I11 + I12 + I13
            expr2 = I21 + I22 + I23
            expr3 = I31 + I32 + I33

            ival = (sb[i]-sa[i])*(sb[j]-sa[j])*(expr1 + 4*expr2 + expr3)/36.

            result[i*dim:(i+1)*dim, j*dim:(j+1)*dim] = ival
            result[j*dim:(j+1)*dim, i*dim:(i+1)*dim] = ival.T

    return result

