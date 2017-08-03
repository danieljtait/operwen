import numpy as np
import clean_src as src
import matplotlib.pyplot as plt

###
# Step 1) We need to define an ODE model, in this case a linear ODE
#
#
def A(x, t):
    return np.array([[ -t, 1.],
                     [  0., -t**2]])

def dXdt(X, t):
    return np.dot(A(X,t),X)

x0 = np.array([1., 0.])
odem = src.ODE(initTime=0., initCond=x0, dXdt=dXdt, dXdt_xJac=A, isLinear=True)

##
# Step 2) Import some data, or simulate some

evalt = np.array([ 0.33, 1.32, 1.42, 2.35])
tknots = np.array([0., 1., 2., 3.])


##
# Step 3) Setup up some model parameters 

lScales = np.array([1., 1.])
cScales = np.array([0., 1.])



##
# Step 4)

gpode = src.tvODE(tknots, odem, 2) # Construction of the ode gp object
gpode.setup_time_vec(evalt)

C = gpode.get_cov(cScales, lScales, evalt)



"""
dim = 2
ttk = gpode.ttk
Att = gpode.get_Atk()

tstar = 1.1

Tcur = gpode.get_transformation_matrix(evalt)

Tnew_row = src.makeTransformation(np.array([tstar]), ttk, Att, dim)
if Tnew_row == []:
    Tnew_row = np.zeros((dim, Tcur.shape[1]))
else:
    Tnew_row = np.column_stack((Tnew_row,
                                np.zeros((dim, Tcur.shape[1] - Tnew_row.shape[1]))
                                ))

Tnew = np.row_stack((Tnew_row, Tcur))

##
n = sum(ttk < tstar)
Cov_new_row = get_cov_row( ttk[n-1], tstar, Att[n-1],
"""
print " "


Cnew = gpode.get_Gcov_row(evalt[0], cScales, lScales, evalt)
#print Cnew[:, 2:8]
#print gpode.CGG[:2,:6]



CGG_cur = gpode.CGG
TMat_cur = gpode.TMat

crow_new = gpode.get_Gcov_row(1.68, cScales, lScales, evalt)
Trow_new = gpode.get_Trow(np.array([1.68]))

Tnew = np.row_stack((Trow_new, TMat_cur))

TMat = np.column_stack(( np.diag(np.ones((evalt.size+1)*2)),
                         Tnew))

CGG_new = np.row_stack(( crow_new[:,2:], CGG_cur ))

dim = 2
cc = np.zeros(crow_new.shape).T
for i in range(crow_new.shape[1]/dim):
    cc[i*dim:(i+1)*dim, :] = crow_new[:,i*dim:(i+1)*dim].T


CGG_new = np.column_stack(( cc, CGG_new))

print "=============="

res = np.dot(CGG_new, TMat.T)

kk = gpode.get_kvec(1.32, cScales, lScales)
print kk[:2,:2]
for i in range(evalt.size):
    print evalt[i]
    print kk[:,(i+1)*dim:(i+2)*dim]
