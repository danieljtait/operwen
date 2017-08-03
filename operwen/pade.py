import numpy as np

A = np.random.uniform(size=4).reshape(2, 2)

def matPower(A, p):
    if p == 0:
        return np.diag(np.ones(A.shape[0]))
    else:
        res = A.copy()
        for i in range(p-1):
            res = np.dot(A, res)
        return res

def p(A, m):
    res = np.zeros(A.shape)

    mfact = np.math.factorial(m)
    twomfact = np.math.factorial(2*m)
    
    for j in range(m+1):
        expr1 = np.math.factorial(2*m - j)*mfact
        expr2 = np.math.factorial(m-j)*twomfact*np.math.factorial(j)
        res += (1.*expr1/expr2)*matPower(A, j)
        
    return res



X1 = p(A, 5)
X2 = p(-A, 5)

print np.dot(np.linalg.inv(X2), X1)

from scipy.linalg import expm

print ""
print expm(A)
