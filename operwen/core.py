import numpy as np
import scipy.linalg
from scipy.special import erf

def status():
    print "Success!"

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

#################################################################
# Computes the integral                                         #
#                                                               #
#    /S                                                         #
#    |  d1(S-s)                                                 #
#    | e       k(s,t)ds,    k(s,t) = c*exp(-0.5*(s-t)**2/l**2 ) #
#   /s_0                                                        #
#                                                               #
#################################################################
def IntegratedSEKernel(t, S, d1, s0, l_scalePar, c_scalePar):
    rootTwol = rootTwo*l_scalePar
    upperTerm = erf( (S + d1*l_scalePar*l_scalePar - t)/rootTwol )
    lowerTerm = erf( (s0 + d1*l_scalePar*l_scalePar - t)/rootTwol )
    const = rootPi*rootTwoRecip*l_scalePar
    expr = np.exp(0.5*d1*(d1*l_scalePar**2 + 2*S - 2*t))
    return c_scalePar*const*expr*(upperTerm - lowerTerm)

################################################################
#                                                              #
# Requires checking [ ]                                        #
#                                                              #
#  - Seems corrent                                             #
#                                                              #
################################################################
def TwiceIntegratedSEKernel(S, T,
                            d1, d2,
                            l_scalePar, c_scalePar,
                            s0, t0):
    # expr1: int_t0^t exp(d2(T-t) * erf( (S+d1*l**2 - t)/rootTwo*l )
    a1 = -(d1+d2)*(S + d1*l_scalePar**2)
    b1 = (d1+d2)*rootTwo*l_scalePar
    lower1 = (S + d1*l_scalePar**2 - t0)*rootTwoRecip/l_scalePar
    upper1 = (S + d1*l_scalePar**2 - T)*rootTwoRecip/l_scalePar
    expr1 = -rootTwo*l_scalePar*linexperfInt(a1, b1, lower1, upper1)
    expr1 *= np.exp(d2*T)
    
    # expr2: int_t0*t exp(d2(T-t) (s0 + d1*l**2 - t)/rootTwo*l )
    a2 = -(d1+d2)*(s0 + d1*l_scalePar**2)
    b2 = (d1+d2)*rootTwo*l_scalePar
    lower2 = (s0 + d1*l_scalePar**2 - t0)*rootTwoRecip/l_scalePar
    upper2 = (s0 + d1*l_scalePar**2 - T)*rootTwoRecip/l_scalePar
    expr2 = -rootTwo*l_scalePar*linexperfInt(a2, b2, lower2, upper2)
    expr2 *= np.exp(d2*T)

    Const1 = rootPi*rootTwoRecip*l_scalePar
    Const2 = np.exp(0.5*d1*d1*l_scalePar*l_scalePar + d1*S)
    
    return c_scalePar*Const1*Const2*(expr1 - expr2)


class SquareExpKernel:
    def __init__(self, cscales, lscales):
        self.cscales = cscales
        self.lscales = lscales
        self.dim = len(cscales)

    def sek_(self,s,t,c,l):
        return c*np.exp(-0.5*(s-t)**2/(l*l))

    def __call__(self, ss, tt):
        if self.dim == 1:
            return self.sek_(s,t,self.cscales[0], self.lscales[0])
        else:
            res = np.zeros((ss.size*self.dim, tt.size*self.dim))
            for i in range(ss.size):
                for j in range(i+1):
                    kval = np.diag([self.sek_(ss[i],tt[j],c,l) for (c,l) in zip(self.cscales,self.lscales)])
                    res[i*self.dim:(i+1)*self.dim, j*self.dim:(j+1)*self.dim] = res[j*self.dim:(j+1)*self.dim,i*self.dim:(i+1)*self.dim] = kval
            return res
                    


################################################################
# Gaussian Process representation of the process defined       #
# by                                                           #
#                                                              #
#                      dX/dt = AX + g(t)                       #
#                                                              #
# where g(t) is a Gaussian Process                             #
################################################################
class linODEGP:
    def __init__(self, A, initValue, initTime, specDecomp=True):
        self.A = A
        self.specDecomp = specDecomp

        # If the matrix A is normal, which for now we require, then
        # perform the spectral decomposition of A and save it
        if self.specDecomp:
            self.Aeigval, self.U = np.linalg.eig(self.A)
            self.Uinv = np.linalg.inv(self.U)

        self.dim = A.shape[0]

        # Initial conditions for the specified ODE model
        self.initTime = initTime
        self.initValue = initValue

    def setKernel(self,kpar,ktype='sq_exp_kern'):
        if(ktype == 'sq_exp_kern'):
            self.ktype = 'sq_exp_kern'
            self.setKernel_sq_exp_kern(kpar)

    def setKernel_sq_exp_kern(self, kpar):
        self.cscales = kpar[0]
        self.lscales = kpar[1]
        
    def integratedCovarksqExp(self, s, t):
        D = self.dim
        # Use numpy's complex data type for safe broadcasting
        # of imaginary numbers
        if np.any(np.iscomplex(self.Aeigval)):
            res = np.zeros((D, D), dtype=complex)
            complexDtype = True
        else:
            res = np.zeros((D, D))
            complexDtype = False

        for k in range(D):
            # Ignore components of state variable with no latent
            # force function
            if self.cscales[k] != 0:
                if complexDtype:
                    C = np.zeros((D, D), dtype=complex)
                else:
                    C = np.zeros((D, D))

                for i in range(D):
                    for j in range(D):
                        C[i, j] = TwiceIntegratedSEKernel(s, t,
                                                          self.Aeigval[i], self.Aeigval[j],
                                                          self.lscales[k], self.cscales[k],
                                                          self.initTime, self.initTime)
                        C[i, j] *= self.Uinv[i, k]*self.Uinv[j, k]
                res += C
        if complexDtype:
            return np.real(np.dot(self.U, np.dot(res, self.U.T)))
        else:
            return np.dot(self.U, np.dot(res, self.U.T))

    def makeCov(self, tt):
        D = self.dim
        if self.ktype == 'sq_exp_kern':
            Cov = np.zeros((tt.size*D, tt.size*D))
            for i in range(tt.size):
                for j in range(i+1):
                    k = self.integratedCovarksqExp(tt[i], tt[j])
                    Cov[i*D:(i+1)*D, j*D:(j+1)*D] = k
                    Cov[j*D:(j+1)*D, i*D:(i+1)*D] = k.T
            return Cov




"""

============= TO DO ===============

 - Sort out handling of covariance functions of the form 

 |s11 s12 | | k11 0   | |s11 s21|
 |s21 s22 | | 0   k22 | |s12 s22|

  =  |s11k11 s12k22 | |s11 s21 |
     |s21k11 s22k22 | |s12 s22 |

  =  |s11*s11*k11 + s12*s12*k22  s12*s21*k11 + 
"""
