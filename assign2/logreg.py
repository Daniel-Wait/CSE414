class LR: #softmax classifier
    """
    LGISTIC REGRESSION CLASS AND FUNCTIONS
    """
    def fiterate(self, data, classes, k):
        """
        w_k+1 = w_k - inv(hessian(w_k))*grad_l(w_k)
        until abs((w_k+1 - w_k)/w_k) < 1e-5
        """
        self.lam = 2
        
        self.y = classes
        self.k = k
        
        self.N = data.shape[1]
        ones = np.ones((1, self.N))
        self.X = np.vstack((ones, data))
        self.dim = (self.X).shape[0]
        
        self.w_k = np.zeros(shape = (self.dim, k))
        w_kn = np.ones(shape = (self.dim, k))
        check = np.zeros(shape = (self.dim,))
        check = 1
        while check >= 1e-5:
            print("ITERATION")
            for j in range(0, self.k):
                hess = self.hessian(self.w_k[:,j])
                inv_hess = np.linalg.inv(hess)
                l = self.grad_l(j)
                ihl = np.reshape(inv_hess.dot(l), newshape = (3,))
                check = abs(np.sum(ihl))
                print("Inv_HESS")
                print(inv_hess)
                print("GRAD L")
                print(l)
                print("HESS.dot(GRAD L)")
                print(ihl)
                print("")
                w_kn[:,j] = self.w_k[:,j] - ihl
                self.w_k[:,j] = w_kn[:,j]
                

    def sig(self, a):
        return (1/(1+math.exp(-a)))            
        
        
    def hessian(self, w):
        """
        H(w) = sum_N{ sig(w.T*x_n)(1-sig(w.T*x_n))(x_n*x_n.T) + (1/lambda)*I }
        """
        h = np.zeros(shape = (self.dim, self.dim))
        
        for n in range(0, self.N):
            a = np.dot(w, self.X[:,n])
            b = self.sig(a)*(1-self.sig(a))
            c = np.outer(self.X[:,n],self.X[:,n])
            h = h+b*c
        h += np.eye(self.dim, self.dim)/self.lam
        return h
        
        
    def grad_l(self,j): #j = class number for which the grad l vector is calculated
        """
        grad_l(w_j) = sum_N{ t_nj[ 1 - exp(w_j.T*x_n)/sum_k{ exp(w_i.T*x_n) } ] }
        """
        
        subset = self.X[:, self.y  == j]
        a = np.zeros(shape = (self.k,))
        #a = (self.w_k.T).dot(subset)
        #exp_a = np.exp(a)
        coeff = np.zeros(shape = (subset.shape[1], ))
        gl = np.zeros(shape = (self.dim,))
        
        for n in range(0, subset.shape[1]):
            for i in np.array(range(0, self.k)):
                a[i] = np.dot(self.w_k[:, i], subset[:,n])
                
            exp_a = np.exp(a)
            coeff[n] = 1-(exp_a[j]/np.sum(exp_a))
            v = coeff[n]*subset[:,n]
            gl = gl+v
            
        return gl[:,np.newaxis]

    
    def predict(self, data):
        """
        a_j = (wj.T)x
        P(Cn|x, wi) = exp(a_n(x))/sum_j_for_k(exp(a_j(x)))
        class = argmax(P(Cn|x_i))
        y_out.shape = (N,)
        
        get: a_j, exp(a_j)_, P(Cn|x_i), class, y_out 
        """
        N = data.shape[1]
        ones = np.ones((1, self.N))
        X = np.vstack((ones, data))
        
        
        #init var.s
        a = np.zeros(shape = (self.k,))
        exp_a = np.zeros(shape = (self.k,))
        pcx = np.empty(shape = (self.k,)) #posterior vector, P(C|x), 0<=j<k
        y = np.empty(shape = (self.N,))
        
        for i in np.array(range(0, self.N)):
            for j in np.array(range(0, self.k)):
                a[j] = np.dot(self.w_k[:, j], X[:,i])
            
            #get exp(a_j(x)) for all classes
            exp_a = np.exp(a)
            #vector of posterior prob.s for each class
            pcx = exp_a/np.sum(exp_a)
            #class to which i-th entry belongs = index of max posterior
            y[i] = (np.where(pcx == pcx.max()))[0][0]

            #print(pcx, np.sum(pcx))
        
        return y