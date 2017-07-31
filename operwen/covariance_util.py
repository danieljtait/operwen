import numpy as np
import scipy.linalg
from scipy.special import erf

def covarLinODEGP_diagLF():
    pass

##
# Covariance of the linear GP ODE model
#
#  dX/dt = AX(t) + S f
#
# where F is not necessariliy diagonal
def covarLinODEGP_SLF(tt, Adecomp, S, cScales, lScales):
    Bdecomp = Adecomp[0], Adecomp[2].T, Adecomp[1].T
    return cov(tt, tt, S, Adecomp, Bdecomp, cScales.size, cScales, lScales)

###
# Cleaner implementation of the covariance functions
# Calculates the covariance terms for UDU^{-1} K_ij PDP^{-1}
def cov_ij(ss, tt, i, j,
           S, Adecomp, Bdecomp, dim,
           cScales, lScales):

    Aeig, UA, UAinv = Adecomp
    Beig, UB, UBinv = Bdecomp

    N1 = ss.size
    N2 = tt.size

    # expand and ravel everything for vectorisation
    Aeig_aug = np.array([Aeig for n in range(N1)]).ravel()
    Beig_aug = np.array([Beig for n in range(N2)]).ravel()
    ss_aug = np.column_stack((ss for d in range(dim) )).ravel()
    tt_aug = np.column_stack((tt for d in range(dim) )).ravel()

    # meshgrid everything
    eigValB, eigValA = np.meshgrid(Beig_aug, Aeig_aug)
    T_, S_ = np.meshgrid(tt_aug, ss_aug)

    # Array to store the result, finished output will be real
    result = np.zeros((N1*dim, N2*dim), dtype='complex')

    # Matrices for pre and post multiplication
    m1 = np.dot(UA, np.diag(UAinv[:,i]))
    PreMat = m1
    for n in range(N1-1):
        PreMat = scipy.linalg.block_diag( PreMat, m1 )

    m2 = np.dot(np.diag(UB[j,:]), UBinv)
    PostMat = m2
    for n in range(N2-1):
        PostMat = scipy.linalg.block_diag( PostMat, m2 )

    # These points will return a nan under the NON_ZERO_ flag
    posnegPairs = eigValA.ravel() + eigValB.ravel() == 0

    if np.sum(posnegPairs) > 0:
        IsPosNeg = True
    else:
        IsPosNeg = False

    ### Actual result calculated now
    with np.errstate(divide='ignore', invalid='ignore'):
        for k in range(dim):
            if cScales[k] > 0:
                M = TwiceIntegratedSEKernel(S_.ravel(), T_.ravel(),
                                            eigValA.ravel(), eigValB.ravel(),
                                            lScales[k], cScales[k],
                                            0., 0., NON_ZERO_=True)
                if IsPosNeg:
                    M2 = TwiceIntegratedSEKernel(S_.ravel(), T_.ravel(),
                                                 eigValA.ravel(), eigValB.ravel(),
                                                 lScales[k], cScales[k],
                                                 0., 0., ZERO_ = True)
                    # Catch the nans
                    M[np.isnan(M)] = 0.
                    M = M*(1-posnegPairs) + M2*posnegPairs

                result += S[i,k]*S[j,k]*np.dot(PreMat, np.dot(M.reshape(result.shape), PostMat))

    return np.real(result)


def cov(ss, tt,
        S, Adecomp, Bdecomp, dim,
        cScales, lScales):

    result = np.zeros((ss.size*dim, tt.size*dim))

    for i in range(dim):
        for j in range(dim):
            result += cov_ij(ss, tt, i, j, S, Adecomp, Bdecomp, dim, cScales, lScales)

    return result


# ===============================

rootPi = np.sqrt(np.pi)
rootTwo = np.sqrt(2.)
rootTwoRecip = 1./rootTwo

################################################################
# Returns the intergral between lower and upper of the         #
# function                                                     #
#                                                              #    
#                  /                                           #
#                  |  a + bu                                   #
#                  | e      erf(u)du                           #
#                  /                                           #
#                                                              #
################################################################
def linexperfInt(a, b, lower, upper):
    if b==0:
        expr1 = upper*erf(upper) + np.exp(-upper**2)/rootPi
        expr2 = lower*erf(lower) + np.exp(-lower**2)/rootPi
        return np.exp(a)*(expr1 - expr2)
    else:
        expr1 = np.exp(0.25*b**2)*erf(0.5*b - upper) + np.exp(b*upper)*erf(upper)
        expr2 = np.exp(0.25*b**2)*erf(0.5*b - lower) + np.exp(b*lower)*erf(lower)
        return np.exp(a)*(expr1 - expr2)/b
##
# As above but with the assumption that b != 0 for easier handling
# of vector arguments
def linexperfInf_nzb(a, b, lower, upper):
    expr1 = np.exp(0.25*b**2)*erf(0.5*b - upper) + np.exp(b*upper)*erf(upper)
    expr2 = np.exp(0.25*b**2)*erf(0.5*b - lower) + np.exp(b*lower)*erf(lower)
    return np.exp(a)*(expr1 - expr2)/b

def linexperfInf_zb(a, lower, upper):
    expr1 = upper*erf(upper) + np.exp(-upper**2)/rootPi
    expr2 = lower*erf(lower) + np.exp(-lower**2)/rootPi
    return np.exp(a)*(expr1 - expr2)


################################################################
#                                                              #
# Requires checking [ ]                                        #
#                                                              #
#  /S  /T                                                      #
#  |   |    d1(S-s)        d2(T-t)                             #
#  |   |   e       k(s,t)  e      dsdt                         #
#  /s0 /t0                                                     #
#                                                              #
#  - where k(s,t) is the common parameterisation of the        #
#    squared exponential kernel                                #
#                                                              #
# Vectorised:                                                  #
# - equal length vector arguments S and T                      #
# - l_scalesPar, c_scalePar, d1, d2 (req. additional assump.   #
#                                                              #
################################################################
def TwiceIntegratedSEKernel(S, T,
                            d1, d2,
                            l_scalePar, c_scalePar,
                            s0, t0, NON_ZERO_=False, ZERO_=False):
    # expr1: int_t0^t exp(d2(T-t) * erf( (S+d1*l**2 - t)/rootTwo*l )
    a1 = -(d1+d2)*(S + d1*l_scalePar**2)
    b1 = (d1+d2)*rootTwo*l_scalePar
    lower1 = (S + d1*l_scalePar**2 - t0)*rootTwoRecip/l_scalePar
    upper1 = (S + d1*l_scalePar**2 - T)*rootTwoRecip/l_scalePar
    if NON_ZERO_:
        expr1 = -rootTwo*l_scalePar*linexperfInf_nzb(a1, b1, lower1, upper1)
    elif ZERO_:
        expr1 = -rootTwo*l_scalePar*linexperfInf_zb(a1, lower1, upper1)
    else:
        expr1 = -rootTwo*l_scalePar*linexperfInt(a1, b1, lower1, upper1)
    expr1 *= np.exp(d2*T)
    
    # expr2: int_t0*t exp(d2(T-t) (s0 + d1*l**2 - t)/rootTwo*l )
    a2 = -(d1+d2)*(s0 + d1*l_scalePar**2)
    b2 = (d1+d2)*rootTwo*l_scalePar
    lower2 = (s0 + d1*l_scalePar**2 - t0)*rootTwoRecip/l_scalePar
    upper2 = (s0 + d1*l_scalePar**2 - T)*rootTwoRecip/l_scalePar
    if NON_ZERO_:
        expr2 = -rootTwo*l_scalePar*linexperfInf_nzb(a2, b2, lower2, upper2)
    elif ZERO_:
        expr2 = -rootTwo*l_scalePar*linexperfInf_zb(a2, lower2, upper2)
    else:
        expr2 = -rootTwo*l_scalePar*linexperfInt(a2, b2, lower2, upper2)
    expr2 *= np.exp(d2*T)

    Const1 = rootPi*rootTwoRecip*l_scalePar
    Const2 = np.exp(0.5*d1*d1*l_scalePar*l_scalePar + d1*S)
    
    return c_scalePar*Const1*Const2*(expr1 - expr2)


###############################
#
# Squared Exponential Kernel 
#
##############################
def ksqexp(s, t, cScales, lScales, S=None, returnType='ind'):
    s = np.asarray(s) 
    t = np.asarray(t)
    S = np.asarray(S)

    dim = cScales.size
    N1 = s.size
    N2 = t.size

    Spre = []
    Spost = []
    if not (S.any() == None):
        returnType = 'matrix'
        Spre = S.copy()
        for l in range(N1-1):
            Spre = scipy.linalg.block_diag(Spre, S)
        if N1 != N2:
            Spost = S.T.copy()
            for l in range(N2-1):
                Spost = scipy.linalg.block_diag(Spost, S.T)
        else:
            Spost = Spre.T

    t_, s_ = np.meshgrid(t, s)
    
    if returnType == 'matrix':
        result = np.zeros((N1*dim, N2*dim))
        rows = np.row_stack(( (nt*dim)*np.ones(N2, dtype=np.intp) for nt in range(N1) ))
        cols = np.column_stack(( (nt*dim)*np.ones(N1, dtype=np.intp) for nt in range(N2) ))
        for i in range(dim):
            result[rows + i, cols + i] = cScales[i]*np.exp(-0.5*(s_.ravel()-t_.ravel())**2/lScales[i]**2).reshape((N1, N2))
        
        if(S.any() == None):
            return result
        else:
            return np.dot(Spre, np.dot(result, Spost))
    else:
        result = []
        for i in range(dim):
            result.append( cScales[i]*np.exp(-0.5*(s_.ravel()-t_.ravel())**2/lScales[i]**2).reshape((N1, N2)) )
        
        return result
