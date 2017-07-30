###
#
# 
#
###
import numpy as np
import scipy.linalg

###
# Handles the discretisation of the of the model
# 
def time_discretisation_handler(tt, tknots, At, dim, AtType='func'):
    tmax = tt[-1] # Assumption that time is discretized
    
    tta = []
    ttb = []
    ttm = []
    Att = []
    Btt = []

    for t in tt:
        tau = tknots[tknots < t][-1]
        tta.append(tau)
        ttb.append(t)
        ttm.append(0.5*(tau+t))
        Att.append(At(0.5*(tau+t)))

    for i in range(tknots.size-1):
        tb = tknots[i+1]
        if tb < tmax:
            tta.append(tknots[i])
            ttb.append(tb)
            ttm.append(0.5*(tknots[i]+tb))
            Att.append(At(0.5*(tknots[i]+tb)))
        else:
            break

    return np.array(tta), np.array(ttm), np.array(ttb), Att
        
#######
#
# Change in model parameterisation will require an update of the 
#
#
def update_funsol(ttm, AtNew):
    Att_new = [AtNew(t) for t in ttm]
    return Att_new


def makeCov_sym(S, A, kernel, dim):
    sa, sm, sb = S
    N = sa.size

    K11 = kernel(sa, sa)
    K12 = kernel(sa, sm)
    K13 = kernel(sa, sb)

    K21 = K12.T
    K22 = kernel(sm, sm)
    K23 = kernel(sm, sb) 

    K31 = K13.T
    K32 = K23.T
    K33 = kernel(sb, sb) 

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

            I31 = np.dot(K31[i*dim:(i+1)*dim, j*dim:(j+1)*dim], eBj2)
            I32 = 4*np.dot(K32[i*dim:(i+1)*dim, j*dim:(j+1)*dim], eBj2)
            I33 = K33[i*dim:(i+1)*dim, j*dim:(j+1)*dim]

            expr1 = I11 + I12 + I13
            expr2 = I21 + I22 + I23
            expr3 = I31 + I32 + I33

            ival = (sb[i]-sa[i])*(sb[j]-sa[j])*(expr1 + 4*expr2 + expr3)/36.

            result[i*dim:(i+1)*dim, j*dim:(j+1)*dim] = ival
            result[j*dim:(j+1)*dim, i*dim:(i+1)*dim] = ival.T

    return result

