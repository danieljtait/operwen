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
print np.linalg.eig(C)[0]
