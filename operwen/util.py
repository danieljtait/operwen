import numpy as np
from scipy.stats import multivariate_normal


class ProbabilityDistribution:
    def __init__(self):
        pass


class MultivariateNormal:
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        self.L = np.linalg.cholesky(cov)
        self.pars = []
        
    def pdf(self, y):
        return multivariate_normal.pdf(y, self.mean, self.cov)

    def logpdf(self, y):
        return multivariate_normal.logpdf(y, self.mean, self.cov)

    def logpdf_muGrad(self, y):
        eta = y - self.mean
        return np.linalg.solve(self.L.T, np.linalg.solve(self.L, eta))

    def logpdf_covGrad(self, y):
        eta = y - self.mean
        mat = np.outer(eta,eta.T)
        cinv = np.linalg.inv(self.cov)
        expr1 = np.dot(cinv, np.dot(mat, cinv))
        return -0.5*(cinv - expr1)

    def addPar(self, par):
        self.pars.append(par)

    def logpdf_parGrad(self, y, par_val, ind):
        par = self.pars[ind]
        if par.pointsTo == 'mean':
            dLdm = self.logpdf_muGrad(y)
            dmdp = par.grad(par_val)
            return np.dot(dLdm, dmdp.T)
        elif par.pointsTo == 'cov':
            dLdC = self.logpdf_covGrad(y)
            dCdp = par.grad(par_val)
            return [np.dot(dLdC.ravel(), dc.ravel())
                    for dc in dCdp]

class Mvtnorm_parametrisation:
    def __init__(self, f, pointsTo):
        self.parf = f
        self.pointsTo = pointsTo # one of 'mean', 'cov' or 'both' 

# example
def mu_f(ts):
    t = ts[0]
    s = ts[1]
    return np.array([t*s, s**2])

def mu_f_grad(ts):
    return np.array([[ts[1], 0.],
                     [ts[0], 2*ts[1]]])

def cov_f(t):
    u = np.array([1., 0.5])
    u /= np.sqrt(sum(u**2))
    ut = np.array([-u[1], u[0]])
    l = np.array([2*t, 0.3*t])
    U = np.column_stack((u, ut))
    D = np.diag(l)
    return np.dot(U, np.dot(D, U.T))

def cov2_f(t):
    C = cov_f(1.)
    D = np.diag([2*t[0], t[1]])
    return C + D

def cov2_f_grad(t):
    C = cov_f(1.)
    D1 = np.diag([2., 0.])
    D2 = np.diag([0., 1.]) 
    return D1, D2

def f(t):
    return mu_f, cov_f

par = Mvtnorm_parametrisation(mu_f, 'mean')
par.grad = mu_f_grad

par2 = Mvtnorm_parametrisation(cov2_f, 'cov')
par2.grad = cov2_f_grad

mean0 = mu_f([1., .3])
cov0 = cov2_f([1., .3])
np.random.seed(11)
y = np.random.normal(size=2)

MvNorm = MultivariateNormal(mean0, cov0)

MvNorm.addPar(par2)
MvNorm.logpdf_parGrad(y, [1., 0.3], 0)

def result():
    EPS = 1e-6
    t0 = 1.
    s0 = .3

    mean = mu_f([t0, s0])
    cov = cov2_f([t0, s0])
    covp = cov2_f([t0, s0 + EPS])

    MvNorm.mean = mean
    MvNorm.cov = cov
    F = MvNorm.logpdf(y)
    MvNorm.cov = covp
    Fp = MvNorm.logpdf(y)

    return (Fp - F)/EPS

from scipy.linalg import expm
A = np.random.uniform(size=4).reshape(2,2)
x0 = np.random.uniform(size=2)

def result2(t):
    
    def m(A):
        return np.dot(expm(A*t),x0)

    def m_grad_FD(A, i, j):
        eps = 1e-6
        Eij = np.zeros(A.shape)
        Eij[i,j] = eps
        return (m(A + Eij) - m(A))/eps

    def m_grad_CStep_ij(A, i, j):
        h = 1e-8
        iEij = np.zeros(A.shape, dtype='complex')
        iEij[i,j] = 1j*h
        return np.imag(m(A + iEij)/h)

    def m_grad_CStep(A):
        return [m_grad_CStep_ij(A, ind1, ind2) for ind1 in range(2) for ind2 in range(2)]
    
    i = 0
    j = 1
#    print m_grad_FD(A, i, j)
    print m_grad_CStep_ij(A, i, j)
#    print m_grad_CStep(A)[i*2 + j]

def result3(tt):

    def m(A,tt):
        return np.array([np.dot(expm(A*t), x0) for t in tt])

    ####
    # Returns the derivative the mean function of the 
    # Gaussian process evaluated at time t
    # with respect to the ijth element
    # of the matrix A
    def m_grad_CStep_ij(A, tt, i, j):
        h = 1e-8
        iEij = np.zeros(A.shape, dtype='complex')
        iEij[i,j] = 1j*h
        return np.imag(m(A + iEij, tt)/h)        

    def m_grad_CStep(A, tt):
        return [m_grad_CStep_ij(A, tt, ind1, ind2).ravel() for ind1 in range(2) for ind2 in range(2)]

    for vec in m_grad_CStep(A, tt):
        print vec


result2(1.0)
result3(np.array([0.5, 1.0]))


    
