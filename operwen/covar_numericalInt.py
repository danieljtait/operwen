import numpy as np
import scipy.linalg
from core import MaketvcTransformationMatrix
from covariance_util import ksqexp

np.random.seed(21)
cScales = np.random.uniform(size=2)
lScales = np.random.uniform(size=2)
SensMat = np.diag(np.ones(2))# np.random.uniform(size=4).reshape(2, 2)


def At(t):
    return np.array([[-0.3, 1.1],
                     [ 0.0, 1.0]])

#    return np.array([[  -1.,  0.],
#                     [0.3*t, -t**2]])

tt = np.linspace(0., 3., 50)

tta = tt[:-1]
ttb = tt[1:]
ttm = 0.5*(tta+ttb)

Att = [At(t) for t in ttm]

def kfunc(s, t):
    return ksqexp(s, t, cScales, lScales, SensMat)

def Jvec_mat(ss, ta, A, tb, kernel, dim):
    res = np.zeros((dim*ss.size, dim*ta.size))

    EA = [ scipy.linalg.expm(a*(t2-t1)) for (a,t2,t1) in zip(A, tb, ta) ]    
    kk = kernel(ss, tb)
    
    for i in range(ss.size):
        res[i*dim:(i+1)*dim, : ] = np.column_stack((
            np.dot( kk[i*dim:(i+1)*dim, j*dim:(j+1)*dim], EA[j]) for j in range(tb.size) ))

    return res


def makeCov(S, T, A, B, kernel, dim):
    sa, sm, sb = S
    ta, tm, tb = T

    K11 = kernel(sa, ta)
    K12 = kernel(sa, tm)
    K13 = kernel(sa, tb)

    K21 = kernel(sm, ta)
    K22 = kernel(sm, tm)
    K23 = kernel(sm, tb)

    K31 = kernel(sb, ta)
    K32 = kernel(sb, tm)
    K33 = kernel(sb, tb)

    N1 = sa.size
    N2 = ta.size

    result = np.zeros((N1*dim, N2*dim))
 
    for i in range(N1):

        eAi1 = scipy.linalg.expm(A[i]*(sb[i]-sa[i]))
        eAi2 = scipy.linalg.expm(A[i]*(sb[i]-sm[i]))

        for j in range(i+1):

            eBj1 = scipy.linalg.expm(B[j]*(tb[j]-ta[j]))
            eBj2 = scipy.linalg.expm(B[j]*(tb[j]-tm[j]))

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

            ival = (sb[i]-sa[i])*(tb[j]-ta[j])*(expr1 + 4*expr2 + expr3)/36.
            
            result[i*dim:(i+1)*dim, j*dim:(j+1)*dim] = ival
            result[j*dim:(j+1)*dim, i*dim:(i+1)*dim] = ival.T

    return result

"""
j = Jvec_mat(tta, tta, Att, ttb, kfunc, 2)

xx = np.random.uniform(size=2*4).reshape((4,2))

Bbt = [a.T for a in Att]

makeCov( (tta, ttm, ttb),
         (tta, ttm, ttb),
         Att, Bbt, kfunc, 2)

def result(tt, tk):
    T = MaketvcTransformationMatrix(tt, tk, At, 2)

    tta = []
    ttb = []
    Att = []
    Btt = []
    for t in tt:
        tau = tk[tk < t][-1]
        tta.append(tau)
        ttb.append(t)
        Att.append(At(0.5*(tau+t)))
        Btt.append(Att[-1].T)
    for i in range(tk.size-1):
        tta.append(tk[i])
        ttb.append(tk[i+1])
        Att.append(At(0.5*(tk[i]+tk[i+1])))
        Btt.append(Att[-1].T)
    tta = np.array(tta)
    ttb = np.array(ttb)
    ttm = 0.5*(tta + ttb)

    print T.shape
    print sum(tk < tt[-1]) - 1
    Cgg = makeCov( (tta, ttm, ttb), (tta, ttm, ttb), Att, Btt, kfunc, 2)
    M = np.column_stack((np.diag(np.ones(tt.size*2)), T))

    print Cgg.shape
    print M.shape

    print np.dot(M, np.dot(Cgg[:34,:34], M.T)) 

tt = np.array([0.2, 0.5, 0.9])
tk = np.linspace(0., 3., 50)

result(tt, tk)

print ""
print "------------"
print ""
from covariance_util import cov

Aeig, UA = np.linalg.eig(At(0.))
UAinv = np.linalg.inv(UA)

Adecomp = (Aeig, UA, UAinv)
Bdecomp = (Aeig, UAinv.T, UA.T)

print cov(tt[1], tt[2], SensMat, Adecomp, Bdecomp, 2, cScales, lScales)
#print "covar check"
#print kfunc(tt[1], tt[2])

from scipy.integrate import dblquad
def result2():
    S = tt[1]
    T = tt[2]

    def integrand(y,x,i,j):
        eA = scipy.linalg.expm(At(0.)*(S-x))
        eB = scipy.linalg.expm(At(0.).T*(T-y))
        k = kfunc(x,y)
        return np.dot(eA, np.dot(k, eB))[i, j]

    Sigma = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            Sigma[i, j] = dblquad(integrand, 0., S, lambda x: 0., lambda x: T, args=(i, j))[0]

    print Sigma

print ""
result2()
"""
