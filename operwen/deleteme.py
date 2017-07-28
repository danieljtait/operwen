import numpy as np
import scipy.linalg
from covariance_util import ksqexp

cScales = np.array([0.3, 0.5])
lScales = np.array([0.5, 1.1])

def kernel(s, t):
    k = np.column_stack([c*np.exp(-0.5*(s-t)**2/l**2) for (c,l) in zip(cScales, lScales)])
    print k
        

ss = np.array([0.5, 1.1])
#kernel(ss, ss)

xx = np.array([0.3, 0.6])

M = np.array([[1., 0., 2. ,0. ],
              [0., 0., 0. ,0. ],
              [2., 0., 3. ,0. ],
              [0., 0., 0. ,0. ]])

x = np.array([[ 0, 1, 2],
              [ 3, 4, 5],
              [ 6, 7, 8],
              [ 9,10,11]])

rows = np.array([[0, 0],
                 [3, 3]], dtype=np.intp)

columns = np.array([[0, 2],
                    [0, 2]], dtype=np.intp)

print x[rows, columns]

Mrows = np.array([[0, 0],
                  [2, 2]], dtype=np.intp)
Mcols = np.array([[0, 2],
                  [0, 2]], dtype=np.intp)

print M[Mrows, Mcols]


S = np.random.uniform(size=9).reshape(3, 3)
K = np.array([[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]])

print np.dot(S, np.dot(K, S.T))
print np.array([[S[m,0]*S[n,0] for n in range(3)] for m in range(3)])

print np.outer(S[:,0],S[:,0])
def ksqexp(s, t, cScales, lScales, S):
    dim = cScales.size
    N1 = s.size
    N2 = t.size

    i = 0

    Si = np.outer(S[:,i], S[:,i])
    S_ = np.row_stack(( np.column_stack(( Si for ind2 in range(N2) )) for ind1 in range(N1) ))

    Spre = S.copy()
    Spost = S.T.copy()
    for l in range(N1-1):
        Spre = scipy.linalg.block_diag(Spre, S)
    for l in range(N2-1):
        Spost = scipy.linalg.block_diag(S.T, Spost)

    # ----
    t_, s_ = np.meshgrid(t, s)

    result = np.zeros((N1*dim, N2*dim))
    for i in range(dim):
        rows = np.row_stack(( (nt*dim+i)*np.ones(N2, dtype=np.intp) for nt in range(N1) ))
        cols = np.column_stack(( (nt*dim+i)*np.ones(N1, dtype=np.intp) for nt in range(N2) ))
        
        result[rows, cols] = cScales[i]*np.exp(-0.5*(s_.ravel()-t_.ravel())**2/lScales[i]**2).reshape((N1, N2))
    
    return np.dot(Spre, np.dot(result, Spost))

ss = np.array([0.3, 1.3, 2.3])
tt = np.array([0.5, 1.1, 1.3])
print "================================"
"""
S = np.array([[1., 1.0],
              [0.0,1.0]])
S = np.random.uniform(size=4).reshape(2, 2)
print S

#cScales = np.array([1., 2.])
cScales = np.random.uniform(size=2)
lScales = np.array([0.3,0.5])
lScales = np.random.uniform(size=2)
K = ksqexp(ss, ss, cScales, lScales, S)
print cScales
print lScales
print " "
print K[:2, 4:]

k1 = cScales[0]*np.exp(-0.5*(ss[0] - ss[2])**2/(lScales[0]**2))
k2 = cScales[1]*np.exp(-0.5*(ss[0] - ss[2])**2/(lScales[1]**2))

K1 = np.array([[k1, 0.], [0., 0.]])
K2 = np.array([[0., 0.], [0., k2]])

print np.dot(S, np.dot(K1, S.T)) + np.dot(S, np.dot(K2, S.T))

K2 = ksqexp(ss, ss, cScales, lScales, S)
print K2 
"""
