import numpy as np
from core import linODEGP
from core import NumDblIntQuad, MaketvcTransformationMatrix
from scipy.stats import multivariate_normal
import covariance_util as covutil

dim = 5
A = np.random.uniform(size=dim*dim).reshape(dim, dim)
S = np.random.uniform(size=dim*dim).reshape(dim, dim)
x0 = np.random.uniform(size=dim)

lScales = np.random.uniform(size=dim)
cScales = np.random.uniform(size=dim)

def kernel(s, t):
    k = [c*np.exp(-0.5*(s-t)**2/l**2) for (c,l) in zip(cScales, lScales) ]
    return np.dot(S, np.dot(np.diag(k), S.T))

gp = linODEGP(A, x0, 0., True, S=S, withSens=True)
gp.setKernel( (cScales, lScales),
              'sq_exp_kern')

Adecomp = gp.Aeigval, gp.U, gp.Uinv
Bdecomp = gp.Aeigval, gp.Uinv.T, gp.U.T

ss = np.linspace(0.5, 1., 7)
tt = np.linspace(0.5, 1., 7)

C = gp.makeCov_faster2(tt)
m = gp.mean(tt)

#Z = multivariate_normal.rvs(mean=m.ravel(), cov=C)


## 
# Alternative calculuation of the covariance matrix
tt_ = np.linspace(0.5, 1., 25)
tta = tt_[:-1]
ttb = tt_[1:]
Att = [A for k in range(tta.size)]

TM = MaketvcTransformationMatrix(tt, tt_, lambda x: A, dim=dim) 

from covariance_util import ksqexp 

def kfunc(s, t):
    return ksqexp(s, t, lScales, cScales, S)

#C = NumDblIntQuad(tta, Att, ttb, kfunc) 


print TM.shape
#print C.shape
#print Z.reshape(m.shape)
print "==========="


