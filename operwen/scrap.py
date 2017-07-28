


        
"""    
Sens = np.random.uniform(size=9).reshape(3,3)

def cov(s,t):
    k11 = np.exp(-0.5*(s-t)**2/(2**2))
    k22 = np.exp(-0.5*(s-t)**2/(0.3**2))
    k33 = np.exp(-0.5*(s-t)**2/(1**2))
    C = np.diag([k11,k22,k33])
    return np.dot(Sens, np.dot(C,Sens.T))

def Kij(s,t,i,j):
    c = cov(s,t)[i,j]
    res = np.zeros((3,3))
    res[i,j] = c
    return res

B = np.random.uniform(size=9).reshape(3,3)
l, U = np.linalg.eig(B)

i = 0
j = 1

A = np.random.uniform(size=9).reshape(3,3)
eigA, UA = np.linalg.eig(A)
UAinv = np.linalg.inv(UA)
"""


def result(i,j):
    row = np.array([ U[j, ind]*np.exp(l[ind]) for ind in range(3) ])
    res = np.row_stack(( np.exp(eigA[ind])*UAinv[ind, i]*row for ind in range(3) ))
    res *= cov(0.3,0.3)[i,j]
    return res



def result2(S, T, i, j, A, B, dim, lScales, cScales, Sens, s0=0., t0=0.):

    ## Spectral decomposition
    eigA, UA = np.linalg.eig(A)
    UAinv = np.linalg.inv(UA)
    eigB, UB = np.linalg.eig(B)
    UBinv = np.linalg.inv(UB)

    ### Triple nested for loop :s

    # Move the final nested expression out, 
    dblImat = np.zeros((dim, dim), dtype='complex')
    for ind1 in range(dim):
        for ind2 in range(dim):
            for n in range(dim):
                dblImat[ind1, ind2] += Sens[i,n]*Sens[j,n]*TwiceIntegratedSEKernel(S, T, eigA[ind1], eigB[ind2], lScales[n], cScales[n], s0, t0)


    #
    #for ind1 in range(dim):
    #    for ind2 in range(dim):
    #        dblImat[ind1, ind2] *= UAinv[ind1, i]*UB[j, ind2]
            
    for ind in range(dim):
        dblImat[:,ind] *= UB[j, ind]
        dblImat[ind,:] *= UAinv[ind, i]

    return np.real(np.dot(UA, np.dot(dblImat, UBinv)))



def result3(S, T, Adecomp, Bdecomp, lScales, cScales, Sens, dim, s0=0., t0=0.):
    eigA = Adecomp[0]
    UA = Adecomp[1]
    UAinv = Adecomp[2]

    eigB = Bdecomp[0]
    UB = Bdecomp[1]
    UBinv = Bdecomp[2]

    result = np.zeros((dim, dim), dtype='complex')

    for i in range(dim):
        for j in range(i+1):

            dblImat = np.zeros((dim, dim), dtype='complex')
            for ind1 in range(dim):
                for ind2 in range(dim):
                    for n in range(dim):
                        if cScales[n] != 0:
                            dblImat[ind1, ind2] += Sens[i, n]*Sens[j, n]*TwiceIntegratedSEKernel(S, T, eigA[ind1], eigB[ind2], lScales[n], cScales[n], s0, t0)

            for ind in range(dim):
                dblImat[:,ind] *= UB[j, ind]
                dblImat[ind,:] *= UAinv[ind, i]

            result += dblImat

    return np.real(np.dot(UA, np.dot(result, UBinv)))
    



lScales = [2., 0.3, 1.]
cScales = [1., 1., 1.]
S = 0.9
T = 0.9
#print result2(S, T, i, j, A, B, 3, lScales, cScales, Sens)
result2(S, T, j, i, A, B, 3, lScales, cScales, Sens)

def Kij(s,t,i,j):
    # method1
    k = np.array([c*np.exp(-0.5*(s-t)**2/(l**2)) for (c,l) in zip(cScales, lScales)])
    C = np.diag(k)
    Sigma = np.dot(Sens, np.dot(C, Sens.T))

    # Method 2
    val = 0.
    for n in range(3):
        val += Sens[i,n]*Sens[j,n]*k[n]
