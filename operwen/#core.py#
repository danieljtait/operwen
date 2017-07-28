import numpy as np
import scipy.linalg
from scipy.special import erf
import covariance_util

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

##
# Vectorised implementation for the covariance matrix in the 
# linear latent GP ode model with no sensitivity matrix
#
# To Do
#   [ ] place the eig val check on the pre aug eigenvalues, and then ravel the resulting boolean if necessary
# 
def makeCovarMat_sqExpk_specDecomp_noSens_2(ss, tt,
                                            Adecomp, Bdecomp,
                                            lScales, cScales,
                                            dim, s0, t0):
    N1 = ss.size
    N2 = tt.size

    Aeig, UA, UAinv = Adecomp
    Beig, UB, UBinv = Bdecomp

    # Expand the eigenvalues for ravelling
    Aeig_aug = np.array([Aeig for n in range(N1)])
    Beig_aug = np.array([Beig for n in range(N2)])

    # Expand the time vectors for ravelling
    ss_aug = np.column_stack((ss for d in range(dim) )).ravel()
    tt_aug = np.column_stack((tt for d in range(dim) )).ravel()

    eigValB, eigValA = np.meshgrid(Beig_aug, Aeig_aug)
    T_, S_ = np.meshgrid(tt_aug, ss_aug)

    # Array to store the result, finished output will be real
    res = np.zeros((N1*dim, N2*dim), dtype='complex')

    # These points will return a nan under the NON_ZERO_ flag
    posnegPairs = eigValA.ravel() + eigValB.ravel() == 0

    # Each of the N1 x N2 (dxd) square submatrix get multiplied by 
    PreMats = []
    PostMats = []
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

        PreMats.append(pmat)
        PostMats.append(pomat)

    with np.errstate(divide='ignore', invalid='ignore'):
        for k in range(dim):
            # setting cScales[k] = 0 will mask the force for the kth component 
            if cScales[k] > 0:
                M1 = TwiceIntegratedSEKernel(S_.ravel(), T_.ravel(),
                                             eigValA.ravel(), eigValB.ravel(),
                                             lScales[k], cScales[k],
                                             s0, t0, NON_ZERO_ = True)

                M2 = TwiceIntegratedSEKernel(S_.ravel(), T_.ravel(),
                                             eigValA.ravel(), eigValB.ravel(),
                                             lScales[k], cScales[k],
                                             s0, t0, ZERO_ = True)

                # Catch the nans in M1
                M1[np.isnan(M1) ] = 0.

                M = M1*(1-posnegPairs) + M2*posnegPairs
                M = M.reshape(res.shape)

                res += np.dot(PreMats[k], np.dot(M, PostMats[k] ))

    return np.real(res)


##
# Vectorised implementation for the covariance matrix in the 
# linear latent GP ode model with no sensitivity matrix
def makeCovarMat_sqExpk_specDecomp_noSens(ss, tt,
                                          Adecomp, Bdecomp,
                                          lScales, cScales,
                                          dim, s0, t0):
    N1 = ss.size
    N2 = tt.size

    Aeig, UA, UAinv = Adecomp
    Beig, UB, UBinv = Bdecomp

    # Expand the eigenvalues for ravelling
    Aeig_aug = np.array([Aeig for n in range(N1)])
    Beig_aug = np.array([Beig for n in range(N2)])

    # Expand the time vectors for ravelling
    ss_aug = np.column_stack((ss for d in range(dim) )).ravel()
    tt_aug = np.column_stack((tt for d in range(dim) )).ravel()

    eigValB, eigValA = np.meshgrid(Beig_aug, Aeig_aug)
    T_, S_ = np.meshgrid(tt_aug, ss_aug)

    # Array to store the result, finished output will be real
    res = np.zeros((N1*dim, N2*dim), dtype='complex')

    # Each of the N1 x N2 (dxd) square submatrix get multiplied by 
    PreMats = []
    PostMats = []
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

        PreMats.append(pmat)
        PostMats.append(pomat)

    # The actual result is calculated here
    # There is an implicit assumption that
    #
    #         eigval(A)[k] != -eigval(B)[k]
    #
    for k in range(dim):
        # setting cScales[k] = 0 will mask the force for the kth component 
        if cScales[k] > 0:
            M = TwiceIntegratedSEKernel(S_.ravel(), T_.ravel(),
                                        eigValA.ravel(), eigValB.ravel(),
                                        lScales[k], cScales[k],
                                        s0, t0, NON_ZERO_ = True).reshape(res.shape)

            res += np.dot(PreMats[k], np.dot(M, PostMats[k] ))

    return np.real(res)

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
    def __init__(self, A, initValue, initTime, specDecomp=True, S = None, withSens=False):
        self.A = A
        self.specDecomp = specDecomp

        self.withSens = withSens
        if self.withSens:
            self.SensMat = S

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

    def mean(self, t):
        return np.array([np.dot(scipy.linalg.expm(self.A*(t_-self.initTime)), self.initValue) for t_ in t])
        
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

    def makeCov_faster(self, tt):
        if not self.withSens:
            return makeCovarMat_sqExpk_specDecomp_noSens(tt, tt,
                                                         (self.Aeigval, self.U, self.Uinv),
                                                         (self.Aeigval, self.Uinv.T, self.U.T),
                                                         self.lscales, self.cscales,
                                                         self.dim, self.initTime, self.initTime)

    def makeCov_faster2(self, tt):
        if not self.withSens:
            return makeCovarMat_sqExpk_specDecomp_noSens_2(tt, tt,
                                                           (self.Aeigval, self.U, self.Uinv),
                                                           (self.Aeigval, self.Uinv.T, self.U.T),
                                                           self.lscales, self.cscales,
                                                           self.dim, self.initTime, self.initTime)

        else:
            return covariance_util.covarLinODEGP_SLF(tt,
                                                     (self.Aeigval, self.U, self.Uinv),
                                                     self.SensMat,
                                                     self.cscales, self.lscales)


    def fit(self, Y, input_times, noise=None):
        self.input_times = input_times
        self.Ydata = Y

        self.covarMatrix = self.makeCov_faster(input_times)
        if noise != None:
            self.withNoise = True
            self.covarMatrix += np.diag(noise*np.ones(input_times.size*self.dim))

        

    
################################################################
# Gaussian Process representation of the process defined       #
# by                                                           #
#                                                              #
#                      dX/dt = A(t)X + g(t)                    #
#                                                              #
# where g(t) is a Gaussian Process                             #
################################################################

class tvlinODEGP:
    def __init__(self, At, initValue, initTime,
                 AtType='func', At_tknots = None, At_eval=None):
        self.x0 = initValue
        self.t0 = initTime
        self.dim = initValue.size

        self.At = At
        self.tknots = At_tknots

        self.kernelIsSet = False
        self.augmentedVarsetIsMade = False

        # Store a potentially very large covariance matrix
        self.storeAugmentedCovar = True

    def fit(self, Data, InputTimes):
        self.evalt = InputTimes
        self.Data = Data
        self.G_time_setup()

    ##
    # G variables are ordered as
    #  [ G(max(tk < t_i ) ) i in evalt 
    def G_time_setup(self):
        tta = []
        ttb = []
        Att = []
        for t in self.evalt:
            tau = self.tknots[self.tknots < t][-1]
            tta.append(tau)
            ttb.append(t)
            Att.append(self.At(0.5*(tau+t)))
        for i in range(self.tknots.size-1):
            tta.append(self.tknots[i])
            ttb.append(self.tknots[i+1])
            Att.append(self.At(0.5*(self.tknots[i]+self.tknots[i+1])))
        self.tta = np.array(tta)
        self.ttb = np.array(ttb)
        self.Att = np.array(Att)

        self.augmentedVarsetMade = True
        
    def mean(self, t):
        pass 
        # the recursive definition of exponential matrix
        # needs to be sped up

    # Set kernel, if A(t) is rapidly varying then we may carry out
    # numerical differentiation of the kernel for no additional cost
    # else if A(t) is constant over long intervals we either need to
    # subpartition the numerical integration routine or better
    # provide analytically integrable covariance functions 
    def setKernel(self, kernel, kpar, ktype='userSupplied'):
        self.kernel = kernel
        self.kpar = kpar
        self.ktype=ktype

        self.kernelIsSet = True

    def makeCov(self, evalt=None):
        if(not self.kernelIsSet):
            print "Kernel function for the model has not been set."
            return 0
        
        if (evalt == None) :
            self.fit([], evalt)

        # Look for stored copies of the variables
        try:
            CGG = self.CGG
        except:
            CGG = NumDblIntQuad(self.tta, self.Att, self.ttb, self.kernel)
            if self.storeAugmentedCovar:
                self.CGG = CGG
        try:
            Tmat = self.Tmat
        except:
            Tmat = MaketvcTransformationMatrix(self.evalt, self.tknots, self.At, self.dim)
            if self.storeTmat:
                self.Tmat = Tmat

        M = np.column_stack((np.diag(np.ones(self.evalt.size*self.dim)), Tmat))
        return np.dot(M, np.dot(CGG, M.T))



# ============================================================================================ #

def MaketvcTransformationMatrix(tt, tk, At, dim=1):
    NG = sum(tk < tt[-1]) - 1
    Ndata = tt.size
    T = np.zeros((Ndata*dim, NG*dim))

    ts = []
    ss = tk[1:]

    nts = 0

    if dim==1:
        for nt in range(Ndata):

            # This could be improved because tt is sorted and so there is a redundant
            tsNew = [sval for sval in ss[nts:] if sval < tt[nt]]
            ts = np.concatenate((ts, tsNew))

            nts += len(tsNew)
            SS = np.concatenate((ts, [tt[nt]]))

            k = SS.size - 1

            T[nt, :k] = np.ones(k)

            for j in range(k):
                ta = SS[j]
                tb = SS[j+1]
                ea = np.exp(At(0.5*(ta+tb))*(tb-ta))
                T[nt, :j+1] *= ea

    else:
        Id = np.diag(np.ones(dim))
        for nt in range(Ndata):
            tsNew = [sval for sval in ss[nts:] if sval < tt[nt]]
            ts = np.concatenate((ts, tsNew))

            nts += len(tsNew)
            SS = np.concatenate((ts, [tt[nt]]))

            k = SS.size - 1
            if k > 0:
                T[nt*dim:(nt+1)*dim, :k*dim] = np.column_stack((Id for k_ in range(k)))

            for j in range(k):
                ta = SS[j]
                tb = SS[j+1]
                ea = scipy.linalg.expm(At(0.5*(ta+tb))*(tb-ta))
                #####
                # Need to work out the best way of performing
                #
                # eA * [ M1, M2, ... MN ]
                #Mvec = T[nt*dim:(nt+1)*dim, :(j+1)*dim]
                ## seems convoluted but...
                M = T[nt*dim:(nt+1)*dim, :(j+1)*dim]
                Mlist = M.T.reshape(j+1, dim, dim) # Gives the array of [ Mi . T ]
                result = np.column_stack(( np.dot(ea, mat.T) for mat in Mlist ))

                T[nt*dim:(nt+1)*dim, :(j+1)*dim] = result
#                T[nt*dim:(nt+1)*dim, :(j+1)*dim] = [np.dot(ea, M) for M in T[nt*dim:(nt+1)*dim, :(j+1)*dim]]



    return T

################################################################
#                                                              #
# approximates the covariance given by                         #
#                                                              #
#                /   /  As(sb-s)         At(tb-t)              #
# Cov(Gs, Gt ) = |   | e         k(s,t) e         dsdt         #
#                /sa /ta                                       #
#                                                              #
# under the assumption that everywhere tb-ta is small so that  #
# the double integral can be replaced by a quadrature rule     #
#                                                              #
################################################################
def NumDblIntQuad(tta, Att, ttb, kernel):
    ttm = 0.5*(tta + ttb)

    I1 = Idblvec(tta, tta, Att, ttb, kernel)
    I2 = Idblvec(ttm, tta, Att, ttb, kernel)
    I3 = Idblvec(ttb, tta, Att, ttb, kernel)

    ea1 = np.exp(Att*(ttb-tta))
    ea2 = np.exp(Att*0.5*(ttb-tta))
    for i in range(tta.size):
        I1[:,i] *= ea1
        I2[:,i] *= ea2
        I1[i,:] *= (ttb[i]-tta[i])
        I2[i,:] *= (ttb[i]-tta[i])
        I3[i,:] *= (ttb[i]-tta[i])

    return (I1 + 4*I2 + I3)/6.


def Idblvec(ss, tta, As, ttb, kernel):
    w1 = J1vec(ss, tta, As, ttb, kernel)
    w2 = J2vec(ss, 0.5*(tta+ttb), As, ttb, kernel)
    w3 = J3vec(ss, ttb, As, ttb, kernel)

    return (ttb-tta)*(w1 + 4*w2 + w3)/6

##
# Returns the ss.size x xa.size result array where
#
#  res[i,j] = kernel(ss[i], xa[j]) * np.exp(A[j]*(tb[j]-xa[j]))
#  
def J1vec(ss, xa, A, tb, kernel):
    ss_, xa_ = np.meshgrid(ss, xa)
    K = kernel(ss_.ravel(), xa_.ravel()).reshape(ss_.shape).T
    return K*np.exp(A*(tb-xa))

####
# This is still going to be a little slow, 
# assuming the kernel function has been vectorised may help somewhat
# - ultimately these functions should probably be rewritten in C/C++
def J1vec_mat(ss, xa, A, tb, kernel, dim=1):
    res = np.zeros((dim*ss.size, dim*xa.size))
    # make EAs first
    EA = [ scipy.linalg.expm(a*(t2-t1)) for (a,t2,t1) in zip(A,tb, xa) ]

    for i in range(ss.size):
        res[i*dim:(i+1)*dim,:] = np.column_stack((
            np.dot(kernel(ss[i],xa[j]), EA[j]) for j in range(xa.size) ))

    return res
    
def J2vec(ss, xm, A, tb, kernel):
    ss_, xm_ = np.meshgrid(ss, xm)
    K = kernel(ss_.ravel(), xm_.ravel()).reshape(ss_.shape).T
    return K*np.exp(A*(tb-xm))

def J2vec_mat(ss, xm, A, tb, kernel, dim=1):
    res = np.zeros((dim*ss.size, dim*xm.size))
    # make EAs first
    EA = [ scipy.linalg.expm(a*(t2-t1)) for (a,t2,t1) in zip(A,tb, xm) ]

    for i in range(ss.size):
        res[i*dim:(i+1)*dim,:] = np.column_stack((
            np.dot(kernel(ss[i],xm[j]), EA[j]) for j in range(xm.size) ))

    return res

def J3vec(ss, xb, A, tb, kernel):
    ss_, xb_ = np.meshgrid(ss, xb)
    K = kernel(ss_.ravel(), xb_.ravel()).reshape(ss_.shape).T
    return K

def J3vec_mat(ss, xb, A, tb, kernel, dim=1):
    res = np.zeros((dim*ss.size, dim*xm.size))

    for i in range(ss.size):
        res[i*dim:(i+1)*dim,:] = np.column_stack((
            kernel(ss[i],xm[j]) for j in range(xm.size) ))

    return res

def Idblvec_mat(ss, tta, As, ttb, kernel,dim):
    w1 = J1vec(ss, tta, As, ttb, kernel,dim)
    w2 = J2vec(ss, 0.5*(tta+ttb), As, ttb, kernel,dim)
    w3 = J3vec(ss, ttb, As, ttb, kernel,dim)

    return (ttb-tta)*(w1 + 4*w2 + w3)/6

def NumDblIntQuad_mat(tta, Att, ttb, kernel, dim):
    ttm = 0.5*(tta + ttb)

    I1 = Idblvec(tta, tta, Att, ttb, kernel, dim)
    I2 = Idblvec(ttm, tta, Att, ttb, kernel, dim)
    I3 = Idblvec(ttb, tta, Att, ttb, kernel, dim)

    ea1 = []
    ea2 = []
    for (a,t2,t1) in zip(Att, ttb, tta) :
        ea1.append( scipy.linalg.expm(a*(t2-t1)) )
        ea2.append( scipy.linalg.expm(a*0.5*(t2-t1)) )

    # This needs carefully inspected because some of these matrices
    # should be transposed - do later and do properly.
    #for i in range(tta.size):
    #    I1[:,i*dim:(i+1)*dim] = 
    
    """
    ea1 = np.exp(Att*(ttb-tta))
    ea2 = np.exp(Att*0.5*(ttb-tta))
    for i in range(tta.size):
        I1[:,i] *= ea1
        I2[:,i] *= ea2
        I1[i,:] *= (ttb[i]-tta[i])
        I2[i,:] *= (ttb[i]-tta[i])
        I3[i,:] *= (ttb[i]-tta[i])

    return (I1 + 4*I2 + I3)/6.
    """


"""

============= TO DO ===============

 - Sort out handling of covariance functions of the form 

 |s11 s12 | | k11 0   | |s11 s21|
 |s21 s22 | | 0   k22 | |s12 s22|

  =  |s11k11 s12k22 | |s11 s21 |
     |s21k11 s22k22 | |s12 s22 |

  =  |s11*s11*k11 + s12*s12*k22  s12*s21*k11 + 
"""

## Example
"""
def At(t):
    return -t**2

def At2(t):
    return np.array([[-t**2, -t],
                     [ 0., 0.1*t]])

def kernel(s,t):
    k11 = np.exp(-3*(s-t)**2)
    return k11

def kernel2(s,t):
    k11 = np.exp(-3*(s-t)**2)
    k22 = np.exp(-0.5*(s-t)**2)
    return np.array([[k11,0],[0,k22]])

x0 = np.array([1.])
x02 = np.array([1.,0.])
tk = np.linspace(0., 1., 100)

gp = tvlinODEGP(At, x0, 0., At_tknots = tk)
gp2 = tvlinODEGP(At2, x02, 0., At_tknots = tk)

evalt = np.linspace(0.01, 1., 7)

gp.fit([], evalt)
gp.setKernel(kernel, None)

#T = MaketvcTransformationMatrix(gp.evalt, gp.tknots, gp.At, gp.dim)

gp2.fit([], evalt)
gp2.setKernel(kernel2, None)


C1 = NumDblIntQuad(gp.tta, gp.Att, gp.ttb, gp.kernel)


tta = np.linspace(0., 0.8, 4)
ttb = np.linspace(0.2, 1., 4)
ttm = 0.5*(tta + ttb)
Att = At(ttm)

# Handle this slighly differently 
Att2 = [ At2(t) for t in ttm]

##
# J1 should calculate 

ss = np.linspace(0., 1., 3)

print J1vec(ss, tta, Att, ttb, kernel)

res = np.zeros((ss.size, tta.size))
for i in range(ss.size):
    for j in range(tta.size):
        res[i,j] = kernel(ss[i],tta[j])*np.exp(Att[j]*(ttb[j]-tta[j]))

print "====================================="
dim = 2


res2 = np.zeros((ss.size*dim, tta.size*dim))
for i in range(ss.size):
    for j in range(tta.size):
        mat = At2(ttm[j])*(ttb[j]-tta[j])
        val = np.dot(kernel2(ss[i], tta[j]), scipy.linalg.expm(mat))
        res2[i*dim:(i+1)*dim, j*dim:(j+1)*dim] = val

print res2 - J1vec_mat(ss, tta, Att2, ttb, kernel2, dim)
"""
