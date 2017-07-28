from core import TwiceIntegratedSEKernel
import numpy as np
import scipy.linalg
from scipy.integrate import dblquad

Dim = 3

A = np.random.uniform(size=9).reshape(3, 3)
Aeig, UA = np.linalg.eig(A)
UAinv = np.linalg.inv(UA)

B = np.random.uniform(size=9).reshape(3, 3)
Beig, UB = np.linalg.eig(B)
UBinv = np.linalg.inv(UB)

Sens = np.random.uniform(size=9).reshape(3, 3)

cScales = np.array([1.1, 1.0, 0.5])
lScales = np.array([0.5, 1.1, 0.8])

S = 0.59
T = 0.85


def k_(s,t,c,l):
    return c*np.exp(-0.5*(s-t)**2/l**2)

def Cov(s,t):
    kd = np.diag([k_(s,t,c,l) for c,l in zip(cScales, lScales)])
    return np.dot(Sens, np.dot(kd, Sens.T))

def kern_ij(s,t,i,j):
    val = 0.
    for l in range(Dim):
        val += Sens[i,l]*Sens[j,l]*k_(s,t,cScales[l], lScales[l])
    return val

i = 0
j = 2

print kern_ij(S, T, i, j)
print Cov(S,T)[i,j]
                                                             
def getResult( ):
    M = np.zeros((Dim, Dim), dtype='complex')
    for m in range(Dim):
        for n in range(Dim):
            val = 0. + 0j
            for l in range(Dim):
                Ival = TwiceIntegratedSEKernel(S, T, Aeig[m], Beig[n], lScales[l], cScales[l], 0., 0.)
                val += Sens[i,l]*Sens[j,l]*Ival
            M[m, n] = UAinv[m,i]*UB[j,n]*val

    M = np.dot(UA, np.dot(M, UBinv))
    return np.real(M)

def getResult2( ):
    M = np.zeros((Dim, Dim), dtype='complex')
    for l in range(Dim):
        # Potential for vectorisation and ravelling in here
        inner_res = np.zeros((Dim, Dim), dtype='complex')
        for m in range(Dim):
            for n in range(Dim):
                Ival = TwiceIntegratedSEKernel(S, T,
                                               Aeig[m],
                                               Beig[n],
                                               lScales[l],
                                               cScales[l],
                                               0., 0.)
                inner_res[m,n] += UAinv[m, i]*UB[j, n]*Ival
        M += Sens[i, l]*Sens[j, l]*inner_res

    M = np.dot(UA, np.dot(M, UBinv))
    return np.real(M)

from scipy.integrate import dblquad
def getResultSlow( ):
    def integrand(y,x,m,n):
        DA = np.diag(np.exp(Aeig*(S-x)))
        DB = np.diag(np.exp(Beig*(T-y)))

        eA = np.dot(UA, np.dot(DA, UAinv))
        eB = np.dot(UB, np.dot(DB, UBinv))
        
        Kij = np.zeros((Dim, Dim))
        Kij[i,j] = kern_ij(x,y,i,j)
        
        return np.dot(eA, np.dot(Kij, eB))[m,n]

    result = np.zeros((Dim, Dim))
    for m in range(Dim):
        for n in range(Dim):
            result[m,n] = dblquad(integrand,
                                  0., S, lambda x: 0, lambda x: T,
                                  args=(m,n))[0]
    return result

dim = Dim
def getResult3(i, j):
    N1 = 1
    N2 = 1

    Aeig_aug = np.array([Aeig for n in range(N1)]).ravel()
    Beig_aug = np.array([Beig for n in range(N2)]).ravel()

    # Expand the time vectors for ravelling 
    ss_aug = S*np.ones(dim)
    tt_aug = T*np.ones(dim)


    T_, S_ = np.meshgrid(tt_aug, ss_aug)
    eigValB, eigValA = np.meshgrid(Beig_aug, Aeig_aug)

    # Array to store the result, finished output will be cast to real
    res = np.zeros((N1*dim, N2*dim), dtype='complex')

    # These points will return a nan under the NON_ZERO_ flag
    posnegPairs = eigValA.ravel() + eigValB.ravel() == 0

    # Every square submatrices gets pre and post multiplied by
    diagMat1 = np.diag(UAinv[:,i])
    mat1 = np.dot(UA, diagMat1)

    diagMat2 = np.diag(UB[j,:])
    mat2 = np.dot(diagMat2, UBinv)
        
    with np.errstate(divide='ignore', invalid='ignore'):
        for k in range(dim):
            if cScales[k] > 0:
                
                M1 = TwiceIntegratedSEKernel(S_.ravel(), T_.ravel(),
                                             eigValA.ravel(), eigValB.ravel(),
                                             lScales[k], cScales[k],
                                             0., 0., NON_ZERO_=True).reshape(res.shape)

                M1 = np.dot(mat1, np.dot(M1, mat2)) 
                res += Sens[i, k]*Sens[j, k]*M1

    return np.real(res)

def getResult4(ss, tt, i, j):
    N1 = ss.size
    N2 = tt.size

    Aeig_aug = np.array([Aeig for n in range(N1)]).ravel()
    Beig_aug = np.array([Beig for n in range(N2)]).ravel()

    # Expand the time vectors for ravelling
    ss_aug = np.column_stack((ss for d in range(dim) )).ravel()
    tt_aug = np.column_stack((tt for d in range(dim) )).ravel()

    eigValB, eigValA = np.meshgrid(Beig_aug, Aeig_aug)
    T_, S_ = np.meshgrid(tt_aug, ss_aug)

    # Array to store the result, finished output will be real
    res = np.zeros((N1*dim, N2*dim), dtype='complex')

    mat1 = np.dot(UA, np.diag(UAinv[:,i]))
    PreMat = mat1
    for n in range(N1-1):
        PreMat = scipy.linalg.block_diag( PreMat, mat1 )
    
    mat2 = np.dot(np.diag(UB[j,:]), UBinv)
    PostMat = mat2
    for n in range(N2-1):
        PostMat = scipy.linalg.block_diag( PostMat, mat2 )

    # These points will return a nan under the NON_ZERO_ flag
    posnegPairs = eigValA.ravel() + eigValB.ravel() == 0

    if np.sum(posnegPairs) > 0:
       IsPosNeg = True
    else:
        IsPosNeg = False

    
    # The actual result is calculated now
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
                    M[np.isnan(M) ] = 0.
                    M = M*(1-posnegPairs) + M2*posnegPairs
                
                res += Sens[i, k]*Sens[j, k]*np.dot(PreMat, np.dot(M.reshape(res.shape), PostMat))

    return np.real(res)

#res = np.zeros((dim, dim))
#for i in range(dim):
#    for j in range(dim):
#        res += getResult3(i, j)

#print res

ss = np.array([S])
tt = np.array([T])

print getResult3(i, j)
print getResult4(ss, tt, i, j)
print getResult4(ss, tt, j, i)
#print getResult()
#print getResult2()
#print getResultSlow()



"""
def makecov1(S, T,
             Adecomp, Bdecomp,
             lScales, cScales,
             dim, s0, t0):

    Aeig, UA, UAinv = Adecomp
    Beig, UB, UBinv = Bdecomp

    res = np.zeros((dim, dim), dtype='complex')

    for k in range(dim):
        if cScales[k] > 0:
            C = np.zeros((dim, dim), dtype='complex')
            for i in range(dim):
                for j in range(dim):
                    C[i, j] = TwiceIntegratedSEKernel(S, T,
                                                      Aeig[i], Beig[j],
                                                      lScales[k], cScales[k],
                                                      s0, t0)
                    C[i, j] *= UAinv[i, k]*UB[k, j]
            res += C
    return res


def makecov2(S, T,
             Adecomp, Bdecomp,
             lScales, cScales,
             dim, s0, t0):

    Aeig, UA, UAinv = Adecomp
    Beig, UB, UBinv = Bdecomp

    res = np.zeros((dim, dim), dtype='complex')
    
    for k in range(dim):
        if cScales[k] > 0:
            C = np.zeros((dim, dim), dtype='complex')
            for i in range(dim):
                for j in range(dim):
                    C[i, j] = TwiceIntegratedSEKernel(S, T,
                                                      Aeig[i], Beig[j],
                                                      lScales[k], cScales[k],
                                                      s0, t0)
            UAinvk = np.diag(UAinv[:,k])
            UBk = np.diag(UB[k,:])
            C = np.dot(UAinvk, np.dot(C, UBk))

            res += C
    return res

def makecov3(S, T,
             Adecomp, Bdecomp,
             lScales, cScales,
             dim, s0, t0):

    Aeig, UA, UAinv = Adecomp
    Beig, UB, UBinv = Bdecomp

    res = np.zeros((dim, dim), dtype='complex')

    evalB, evalA = np.meshgrid(Beig, Aeig)
    for k in range(dim):
        if cScales[k] > 0:
            # ravel Aeig, Beig
            M = TwiceIntegratedSEKernel(S, T,
                                        evalA.ravel(), evalB.ravel(),
                                        lScales[k], cScales[k],
                                        s0, t0, NON_ZERO_ = True).reshape(dim, dim)
            UAinvk = np.diag(UAinv[:,k])
            UBk = np.diag(UB[k,:])
            res += np.dot(UAinvk, np.dot(M, UBk))

    res = np.dot(UA, np.dot(res, UBinv))
    return res

def makeCov3(ss, tt,
             Adecomp, Bdecomp,
             lScales, cScales,
             dim, s0, t0):

    N1 = ss.size
    N2 = tt.size

    Aeig, UA, UAinv = Adecomp
    Beig, UB, UBinv = Bdecomp

    # Expand the vectors for raveling 
    Aeig_aug = np.array([Aeig for n in range(N1)])
    Beig_aug = np.array([Beig for n in range(N2)])

    ss_aug = np.column_stack((ss for d in range(dim) )).ravel()
    tt_aug = np.column_stack((tt for d in range(dim) )).ravel()
    
    #ss_aug = np.array([ss for d in range(dim)])
    #tt_aug = np.array([tt for d in range(dim)])

    evalB, evalA = np.meshgrid(Beig_aug, Aeig_aug)
    T_, S_  = np.meshgrid(tt_aug, ss_aug)

    res = np.zeros((N1*dim, N2*dim), dtype='complex')

    PreMats = []
    PostMats = []
    for k in range(dim):
        pmat = np.diag(np.array( [UAinv[:,k] for n in range(N1)], dtype='complex').ravel())
        pomat = np.diag(np.array( [UBinv[:,k] for n in range(N2)], dtype='complex').ravel()) 
        PreMats.append( pmat )
        PostMats.append( pomat )

    PreMats2 = []
    PostMats2 = []
    for k in range(dim):
        diagMat1 = np.diag(UAinv[:,k])
        mat1 = np.dot(UA, diagMat1)
        pmat = mat1
        for n in range(N1-1):
            pmat = scipy.linalg.block_diag( pmat, mat1 )

        diagMat2 = np.diag(UB[k,:])
        mat2 = np.dot(diagMat2, UBinv)
        pomat = mat2
        for n in range(N2-1):
            pomat = scipy.linalg.block_diag( pomat, mat2 )

        PreMats2.append(pmat)
        PostMats2.append(pomat)
    
    for k in range(dim):
        if cScales[k] > 0:
            M = TwiceIntegratedSEKernel(S_.ravel(), T_.ravel(),
                                        evalA.ravel(), evalB.ravel(),
                                        lScales[k], cScales[k],
                                        s0, t0, NON_ZERO_ = True).reshape(res.shape)
            res += np.dot(PreMats2[k], np.dot(M, PostMats2[k] ))

    return np.real(res)
                                        
    

lScales = np.array([0.5, 1., 1.3])
cScales = np.array([1., 1., 1.1])

B = A.T
Aeig, UA = np.linalg.eig(A)
UAinv = np.linalg.inv(UA)
Beig, UB = np.linalg.eig(B)
UBinv = np.linalg.inv(UB)

Adecomp = (Aeig, UA, UAinv)
Bdecomp = (Beig, UB, UBinv)


S = 1.
T = 0.9

SS = np.linspace(0., 10., 10000)

ss = np.linspace(0.3, 0.5, 2)
tt = np.linspace(0.1, 1., 2)

#makecov3(ss[0], tt[0], Adecomp, Bdecomp, lScales, cScales, 3, 0., 0.)
#covQuick = makeCov3(ss, ss, Adecomp, Bdecomp, lScales, cScales, 3, 0., 0.)

#for s in SS:
#    makecov1(s, T, Adecomp, Bdecomp, lScales, cScales, 3, 0., 0.)


#for s in SS:
#    makecov3(s, T, Adecomp, Bdecomp, lScales, cScales, 3, 0., 0.)

from core import linODEGP
x0 = np.random.uniform(size=3)
gp = linODEGP(A, x0, 0.)
gp.setKernel(kpar=(cScales, lScales),
             ktype='sq_exp_kern')

#cov = gp.makeCov(ss)

#print cov - covQuick
"""



#makecov1(S, T, Adecomp, Bdecomp, lScales, cScales, 3, 0., 0.)
#makecov2(S, T, Adecomp, Bdecomp, lScales, cScales, 3, 0., 0.)
#makecov3(S, T, Adecomp, Bdecomp, lScales, cScales, 3, 0., 0.)
#print makeCov3(np.array(S), np.array(T), Adecomp, Bdecomp, lScales, cScales, 3, 0., 0.)

#print gp.integratedCovarksqExp(S, T)

#cov1 = gp.makeCov(ss)
#ss = np.asarray(ss[0])
#tt = np.asarray(tt[0])
#print gp.integratedCovarksqExp(ss[0], tt[0])
#print ""

#ss = np.linspace(0., 5., 100)
#cov = gp.makeCov(ss)
#cov = makeCov3(ss, ss, Adecomp, Bdecomp, lScales, cScales, 3, 0., 0.)

#cov2 = gp.makeCov_faster(ss)

#print np.max(cov - cov2)
