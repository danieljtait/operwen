import numpy as np
import clean_src as src
import matplotlib.pyplot as plt

###
# Step 1) We need to define an ODE model, in this case a linear ODE
#
#
def A(x, t):
    return np.array([[ -t, 0.],
                     [ 1., -t**2]])

def dXdt(X, t):
    return np.dot(A(X,t),X)

x0 = np.array([1., 0.])
odem = src.ODE(initTime=0., initCond=x0, dXdt=dXdt, dXdt_xJac=A, isLinear=True)

##
# Step 2) Import some data, or simulate some
evalt = np.linspace(0.1, 2.6, 5)

##
# Step 3) Setup up some model parameters 

lScales = np.array([1., 1.])
cScales = np.array([0., 1.])

tknots = np.linspace(0., 3., 11.)

##
# Step 4)

gpode = src.tvODE(tknots, odem, 2) # Construction of the ode gp object
gpode.setup_time_vec(evalt)
T = gpode.get_transformation_matrix(evalt)

