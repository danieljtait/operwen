import numpy as np
from scipy.linalg import expm

tk = np.array([0., 1., 2., 3., 4.])
tt = np.array([0.5, 1.3, 2.6, 3.5])

def func(tt, tk, Atk, dim):

    NG = sum(tk < tt[-1]) - 1

    result = np.zeros((tt.size*dim, NG*dim))
    Id = np.diag(np.ones(dim))

    tauSet = []
    M = 0
    for i in range(tt.size):
        if tt[i] > tk[1]:

            tauSet = np.concatenate(( tauSet, [s for s in tk[M+1:] if s < tt[i] ] ))
            M = len(tauSet)

            result[i*dim:(i+1)*dim, :M*dim] = np.column_stack((Id for nt in range(M) ))

            for k in range(M-1):
                eA = expm( Atk[k+1]*(tauSet[k+1] - tauSet[k]) )
                for j in range(k+1):
                    #eA = expm( Atk[j+1]*(tauSet[j+1] - tauSet[j]) )
                    result[i*dim:(i+1)*dim, j*dim:(j+1)*dim] = np.dot(eA, result[i*dim:(i+1)*dim, j*dim:(j+1)*dim])

            eAi = expm(Atk[M]*(tt[i] - tauSet[-1]))
            for k in range(M):
                result[i*dim:(i+1)*dim, k*dim:(k+1)*dim] = np.dot(eAi, result[i*dim:(i+1)*dim, k*dim:(k+1)*dim])
            
            print "====="
    return result
            

def At(t):
    return np.array([[0., 3.],
                     [1., -t]])

###
# Model is actually solved on tm[i] = 0.5*(tk[i] + tk[i+1])
Atk = [At(t) for t in tk]

T = func(tt, tk, Atk, 2)

class handler:
    def __init__(self):
        pass

    def setup_tvecs(self, tt, tknots):
        self.tti = tt.copy()
        self.ttk = tknots[tknots < self.tti[-1]]

        self.ttk_mid = self.ttk + 0.5*np.diff(tknots[:self.ttk.size+1])
        self.tt_aug = np.linspace(self.ttk[0], self.tti[-1], 51)
        
        self.ttk_full = np.concatenate(( self.ttk_mid, self.tt_aug ))
        
        self.ttk_solve_sort_inds = np.argsort(self.ttk_full)
        self.ttk_inds = [np.where(self.ttk_solve_sort_inds == i)[0][0] for i in range(self.ttk.size) ]

        
    def get_Atk(self, solveObj, Amap):
        sol = solveObj.solve(self.ttk_full[self.ttk_solve_sort_inds])
        return Amap( sol[self.ttk_inds, :], self.ttk_mid )
        
print "-----"
obj = handler()
obj.setup_tvecs(tt, tk)

