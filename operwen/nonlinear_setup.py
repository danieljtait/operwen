
def set_it_up(odeModelObj, evalt, tknots):
    pass

class tvODE:
    def __init__(self, tknots, odeModelObj):
        self.tknots = tknots
        self.odeModelObj
        self.isSetup = False

    ####
    # collects:
    #
    # - the relevant knot points,
    # - the mid point of the knot intervals
    # - augmented vector of sorted times at which the ode will get solved
    # - indices of the knot mid points in the augmented vector
    ###
    def setup_time_vec(self, inputTimes):
        self.ttk, self.ttk_mid, self.ttk_full, self.ttk_inds = setup_time_vecs(inputTimes[0], inputTimes[-1], self.tknots)        

    ###
    # Returns a list of matrices [ A(t) t in self.ttk_mids ] where
    # A is the Jacobian of dXdt in the specified ODE model
    def get_Atk(self):
        if self.odeModelObj.isLinear:
            # Linear model the Jacobian will simply be A(t) and so independent
            # of the X argument
            return [self.odeModelObj.dXdt_Jac(None, t) for t in self.ttk_mid]
        else:
            sol = self.odeModelObj.solve(self.ttk_full)
            sol_tk = sol[self.ttk_inds,:]
            return [self.odeModelObj.dXdt_Jac(x, t) for x,t in zip(sol_tk, self.ttk_mid)]
        
    def fit(Y, inputTimes):
        self.inputTimes
        
        if not self.isSetup:
            # need to set up the points at which the model will be solved
            self.setup_time_vec(inputTimes)

        if not Tismade:
            # makes a transformation matrix
            pass
        
        if not CggisMade:
            # calculate the covariance






        


def setup_time_vecs(tmin, tmax, tknots):
    if tmin < tknots[0] or tmax > tknots[-1]:
        print "input point outside the range of knot points"
        return None,_

    else:

        ttk = tknots[tknots < tmax]
        ttk_mid = ttk + 0.5*np.diff(tknots[:ttk.size + 1])
        
        ttk_full = np.concatenate(( ttk_mid, [] )) # Consider augmenting the size of tvec

        ttk_solve_sort_inds = np.argsort(ttk_full)
        ttk_inds = [np.where(ttk_solve_sort_inds == i)[0][0] for i in range(ttk.size) ]

        return ttk, ttk_mid, ttk_full[ttk_solve_sort_inds], ttk_inds
    

    
