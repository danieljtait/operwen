import numpy as np
from core import linODEGP
from scipy.stats import multivariate_normal
import covariance_util as covutil

dim = 5
A = np.random.uniform(size=dim*dim).reshape(dim, dim)
S = np.random.uniform(size=dim*dim).reshape(dim, dim)
x0 = np.random.uniform(size=dim)

lScales = np.random.uniform(size=dim)
cScales = np.random.uniform(size=dim)

gp = linODEGP(A, x0, 0., True, S=S, withSens=True)
gp.setKernel( (cScales, lScales),
              'sq_exp_kern')

Adecomp = gp.Aeigval, gp.U, gp.Uinv
Bdecomp = gp.Aeigval, gp.Uinv.T, gp.U.T

ss = np.linspace(0.5, 1., 7)
tt = np.linspace(0.5, 1., 7)

C = gp.makeCov_faster2(tt)
m = gp.mean(tt)

Z = multivariate_normal.rvs(mean=m.ravel(), cov=C)

print Z.reshape(m.shape)
print "==========="


