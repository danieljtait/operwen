import numpy as np
import scipy.linalg
from core import MaketvcTransformationMatrix
from covariance_util import ksqexp
from covar_numericalInt import makeCov, kfunc
from timevarying_covar import time_discretisation_handler, makeCov_sym

evalt = np.array([0.3, 0.5, .9, 1.5, 2.6])
tk = np.linspace(0., 1., 15)

dim = 2
def At(t):
    return np.array([[0.3, 1.],
                     [0.5,-1.]])

T = MaketvcTransformationMatrix(evalt, tk, At, dim)

tta, ttm, ttb, Att = time_discretisation_handler(evalt, tk, At, 2)
Btt = [a.T for a in Att]
CGG = makeCov( (tta, ttm, ttb), (tta, ttm, ttb), Att, Btt, kfunc, dim)
CGG2 = makeCov_sym( (tta, ttm, ttb), Att, kfunc, dim)

M = np.column_stack((np.diag(np.ones(evalt.size*dim)), T))


cov = np.dot(M, np.dot(CGG, M.T))

print(CGG - CGG2)

#print(len(tta))
#print(evalt.size*dim)
#print(CGG.shape)
