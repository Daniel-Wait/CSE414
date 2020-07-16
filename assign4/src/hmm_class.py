'''
Module implementing Hidden Markov model parameter estimation.

To avoid repeated warnings of the form "Warning: divide by zero encountered in log", 
it is recommended that you use the command "np.seterr(divide="ignore")" before 
invoking methods in this module.  This warning arises from the code using the 
fact that python sets log 0 to "-inf", to keep the code simple.

Initial version created on Mar 28, 2012

@author: kroon, herbst
'''

from warnings import warn
import numpy as np
from scipy.stats import multivariate_normal as mvn_pdf
from gaussian import Gaussian
np.seterr(divide="ignore")

class HMM(object):
    '''
    Class for representing and using hidden Markov models.
    Currently, this class only supports left-to-right topologies and Gaussian
    emission densities.

    The HMM is defined for n_states emitting states (i.e. states with 
    observational pdf's attached), and an initial and final non-emitting state (with no 
    pdf's attached). The emitting states always use indices 0 to (n_states-1) in the code.
    Indices -1 and n_states are used for the non-emitting states (-1 for the initial and
    n_state for the terminal non-emitting state). Note that the number of emitting states
    may change due to unused states being removed from the model during model inference.

    To use this class, first initialize the class, then either use load() to initialize the
    transition table and emission densities, or fit() to initialize these by fitting to
    provided data.  Once the model has been fitted, one can use viterbi() for inferring
    hidden state sequences, forward() to compute the likelihood of signals, score() to
    calculate likelihoods for observation-state pairs, and sample()
    to generate samples from the model.
        
    Attributes:
    -----------
    data : (d,n_obs) ndarray 
        An array of the trainining data, consisting of several different
        sequences.  Thus: Each observation has d features, and there are a total of n_obs
        observation.   An alternative view of this data is in the attribute signals.

    diagcov: boolean
        Indicates whether the Gaussians emission densities returned by training
        should have diagonal covariance matrices or not.
        diagcov = True, estimates diagonal covariance matrix
        diagcov = False, estimates full covariance matrix

    dists: (n_states,) list
        A list of Gaussian objects defining the emitting pdf's, one object for each 
        emitting state.

    maxiters: int
        Maximum number of iterations used in Viterbi re-estimation.
        A warning is issued if 'maxiters' is exceeded. 

    rtol: float
        Error tolerance for Viterbi re-estimation.
        Threshold of estimated relative error in log-likelihood (LL).

    signals : ((d, n_obs_i),) list
        List of the different observation sequences used to train the HMM. 
        'd' is the dimension of each observation.
        'n_obs_i' is the number of observations in the i-th sequence.
        An alternative view of thise data is in the attribute data.
            
    trans : (n_states+1,n_states+1) ndarray
        The left-to-right transition probability table.  The rightmost column contains probability
        of transitioning to final state, and the last row the initial state's
        transition probabilities.   Note that all the rows need to add to 1. 
    
    Methods:
    --------
    fit():
        Fit an HMM model to provided data using Viterbi re-estimation (i.e. the EM algorithm).

    forward():
        Calculate the log-likelihood of the provided observation.

    load():
        Initialize an HMM model with a provided transition matrix and emission densities
    
    sample():
        Generate samples from the HMM
    
    viterbi():
        Calculate the optimal state sequence for the given observation 
        sequence and given HMM model.
    
    Example (execute the class to run the example as a doctest)
    -----------------------------------------------------------
    >>> import numpy as np
    >>> from gaussian import Gaussian
    >>> signal1 = np.array([[ 1. ,  1.1,  0.9, 1.0, 0.0,  0.2,  0.1,  0.3,  3.4,  3.6,  3.5]])
    >>> signal2 = np.array([[0.8, 1.2, 0.4, 0.2, 0.15, 2.8, 3.6]])
    >>> data = np.hstack([signal1, signal2])
    >>> lengths = [11, 7]
    >>> hmm = HMM()
    >>> hmm.fit(data,lengths, 3)
    >>> trans, dists = hmm.trans, hmm.dists
    >>> means = [d.get_mean() for d in dists]
    >>> covs = [d.get_cov() for d in dists]
    >>> covs = np.array(covs).flatten()
    >>> means = np.array(means).flatten()
    >>> print(trans)
    [[ 0.66666667  0.33333333  0.          0.        ]
     [ 0.          0.71428571  0.28571429  0.        ]
     [ 0.          0.          0.6         0.4       ]
     [ 1.          0.          0.          0.        ]]
    >>> print(covs)
    [ 0.02        0.01702381  0.112     ]
    >>> print(means)
    [ 1.          0.19285714  3.38      ]
    >>> signal = np.array([[ 0.9515792,   0.9832767,   1.04633007,  1.01464327,  0.98207072,  1.01116689, 0.31622856,  0.20819263,  3.57707616]])
    >>> vals, ll = hmm.viterbi(signal)
    >>> print(vals)  
    [0 0 0 0 0 0 1 1 2]
    >>> print(ll)
    2.23881485357
    >>> hmm.load(trans, dists)
    >>> vals, ll = hmm.viterbi(signal)
    >>> print(vals)
    [0 0 0 0 0 0 1 1 2]
    >>> print(ll)
    2.23881485357
    >>> print(hmm.score(signal, vals))
    2.23881485357
    >>> print(hmm.forward(signal))
    2.23882615241
    >>> signal = np.array([[ 0.9515792,   0.832767,   3.57707616]])
    >>> vals, ll = hmm. (signal)
    >>> print(vals)
    [0 1 2]
    >>> print(ll)
    -13.1960946635
    >>> samples, states = hmm.sample()
    '''

    def __init__(self, diagcov=True, maxiters=20, rtol=1e-4): 
        '''
        Create an instance of the HMM class, with n_states hidden emitting states.
        
        Parameters
        ----------
        diagcov: boolean
            Indicates whether the Gaussians emission densities returned by training
            should have diagonal covariance matrices or not.
            diagcov = True, estimates diagonal covariance matrix
            diagcov = False, estimates full covariance matrix

        maxiters: int
            Maximum number of iterations used in Viterbi re-estimation
            Default: maxiters=20

        rtol: float
            Error tolerance for Viterbi re-estimation
            Default: rtol = 1e-4
        '''
        
        self.diagcov = diagcov
        self.maxiters = maxiters
        self.rtol = rtol
        
    def fit(self, data, lengths, n_states):
        '''
        Fit a left-to-right HMM model to the training data provided in `data`.
        The training data consists of l different observaion sequences, 
        each sequence of length n_obs_i specified in `lengths`. 
        The fitting uses Viterbi re-estimation (an EM algorithm).

        Parameters
        ----------
        data : (d,n_obs) ndarray 
            An array of the training data, consisting of several different
            sequences. 
            Note: Each observation has d features, and there are a total of n_obs
            observation. 

        lengths: (l,) int ndarray 
            Specifies the length of each separate observation sequence in `data`
            There are l difference training sequences.

        n_states : int
            The number of hidden emitting states to use initially. 
        '''
        
        # Split the data into separate signals and pass to class
        self.data = data
        newstarts = np.cumsum(lengths)[:-1]
        self.signals = np.hsplit(data, newstarts) 
        self.trans = HMM._ltrtrans(n_states)
        self.trans, self.dists, newLL, iters = self._em(self.trans, self._ltrinit())

    def load(self, trans, dists):
        '''
        Initialize an HMM model using the provided data.

        Parameters
        ----------
        dists: (n_states,) list
            A list of Gaussian objects defining the emitting pdf's, one object for each 
            emitting state.

        trans : (n_states+1,n_states+1) ndarray
            The left-to-right transition probability table.  The rightmost column contains probability
            of transitioning to final state, and the last row the initial state's
            transition probabilities.   Note that all the rows need to add to 1. 
    
        '''

        self.trans, self.dists = trans, dists

    def _n_states(self):
        '''
        Get the number of emitting states used by the model.

        Return
        ------
        n_states : int
        The number of hidden emitting states to use initially. 
        '''

        return self.trans.shape[0]-1

    def _n_obs(self):
        '''
        Get the total number of observations in all signals in the data associated with the model.

        Return
        ------
        n_obs: int 
            The total number of observations in all the sequences combined.
        '''

        return self.data.shape[1]

    @staticmethod
    def _ltrtrans(n_states):
        '''
        Intialize the transition matrix (self.trans) with n_states emitting states (and an initial and 
        final non-emitting state) enforcing a left-to-right topology.  This means 
        broadly: no transitions from higher-numbered to lower-numbered states are 
        permitted, while all other transitions are permitted. 
        All legal transitions from a given state should be equally likely.

        The following exceptions apply:
        -The initial state may not transition to the final state
        -The final state may not transition (all transition probabilities from 
         this state should be 0)
    
        Parameter
        ---------
        n_states : int
            Number of emitting states for the transition matrix

        Return
        ------
        trans : (n_states+1,n_states+1) ndarray
            The left-to-right transition probability table initialized as described below.
        '''

        trans = np.zeros((n_states + 1, n_states + 1))
        trans[-1, :-1] = 1. / n_states
        for row in range(n_states):
            prob = 1./(n_states + 1 - row)
            for col in range(row, n_states+1):
                trans[row, col] = prob
        return trans

    def _ltrinit(self):
        '''
        Initial allocation of the observations to states in a left-to-right manner.
        It uses the observation data that is already available to the class.
    
        Note: Each signal consists of a number of observations. Each observation is 
        allocated to one of the n_states emitting states in a left-to-right manner
        by splitting the observations of each signal into approximately equally-sized 
        chunks of increasing state number, with the number of chunks determined by the 
        number of emitting states.
        If 'n' is the number of observations in signal, the allocation for signal is specified by:
        np.floor(np.linspace(0, n_states, n, endpoint=False))
    
        Returns
        ------
        states : (n_obs, n_states) ndarray
            Initial allocation of signal time-steps to states as a one-hot encoding.  Thus
            'states[:,j]' specifies the allocation of all the observations to state j.
        '''

        states = np.zeros((self._n_obs(), self._n_states()))
        i = 0
        for s in self.signals:
            vals = np.floor(np.linspace(0, self._n_states(), num=s.shape[1], endpoint=False))
            for v in vals:
                states[i][int(v)] = 1
                i += 1

        return np.array(states,dtype = bool)

    def viterbi(self, signal):
        '''
        See documentation for _viterbi()
        '''
        return HMM._viterbi(signal, self.trans, self.dists)

    @staticmethod
    def _viterbi(signal, trans, dists):
        '''
        Apply the Viterbi algorithm to the observations provided in 'signal'.
        Note: `signal` is a SINGLE observation sequence.
    
        Returns the maximum likelihood hidden state sequence as well as the
        log-likelihood of that sequence.

        Note that this function may behave strangely if the provided sequence
        is impossible under the model - e.g. if the transition model requires
        more observations than provided in the signal.
    
        Parameters
        ----------
        signal : (d,n) ndarray
            Signal for which the optimal state sequence is to be calculated.
            d is the dimension of each observation (number of features)
            n is the number of observations 
        
        trans : (n_states+1,n_states+1) ndarray
            The transition probability table.  The rightmost column contains probability
            of transitioning to final state, and the last row the initial state's
            transition probabilities.   Note that all the rows need to add to 1. 
    
        dists: (n_states,) list
            A list of Gaussian objects defining the emitting pdf's, one object for each 
            emitting  state.

        Return
        ------
        seq : (n,) ndarray
            The optimal state sequence for the signal (excluding non-emitting states)

        ll : float
            The log-likelihood associated with the sequence 
        '''
        # In this function, you may want to take log 0 and obtain -inf.
        # To avoid warnings about this, you can use np.seterr.
        N = trans.shape[0]-1
        T = signal.shape[1]
        viterbi = np.zeros(shape = (N,T), dtype = np.float64)
        backpointer = np.zeros(shape = (N,T), dtype = np.int8)
            
        for j in range(0, N): #initialization step
            viterbi[j,0] = np.log(trans[-1,j]) + dists[j].loglik(signal[:,0])
            backpointer[j,0] = -1
            
        for t in range(1,T): #recursion step
            for j in range(0, N):
                tt = np.log(trans[0:-1,j])
                vv = viterbi[:,t-1]
                prod_lik = tt+vv
                backpointer[j,t] = np.argmax(prod_lik)    
                maximum = prod_lik.max()
                viterbi[j,t] = dists[j].loglik(signal[:,t]) + maximum
        
        bestpathpointer = np.argmax(viterbi[:,T-1])
        bestpathprob = viterbi[bestpathpointer,T-1] + np.log(trans[bestpathpointer, N])
        
        bestpath = np.zeros(shape = (T,), dtype = np.int8)
        bestpath[T-1] = bestpathpointer
        
        for t in range(T-2,-1,-1):
            bestpath[t] = backpointer[bestpath[t+1], t+1]

        return bestpath, bestpathprob


    def score(self, signal, seq):
        '''
        See documentation for _score()
        '''
        return HMM._score(signal, seq, self.trans, self.dists)

    @staticmethod
    def _score(signal, seq, trans, dists):
        '''
        Calculate the likelihood of an observation sequence and hidden state correspondence.
        Note: signal is a SINGLE observation sequence, and seq is the corresponding series of
        emitting states being scored.
    
        Returns the log-likelihood of the observation-states correspondence.

        Parameters
        ----------
        signal : (d,n) ndarray
            Signal for which the optimal state sequence is to be calculated.
            d is the dimension of each observation (number of features)
            n is the number of observations 
        
        seq : (n,) ndarray
            The state sequence provided for the signal (excluding non-emitting states)

        trans : (n_states+1,n_states+1) ndarray
            The transition probability table.  The rightmost column contains probability
            of transitioning to final state, and the last row the initial state's
            transition probabilities.   Note that all the rows need to add to 1. 
    
        dists: (n_states,) list
            A list of Gaussian objects defining the emitting pdf's, one object for each 
            emitting  state.

        Return
        ------
        
        
        ll : float
            The log-likelihood associated with the observation and state sequence under the model.
        
        
        
                    if(a_i_j > 0) or (x_giv_s > 0):
                print("cnt = ", cnt)
                print("state  = ", cur)
                print("state_next  = ", new)
                print("a_i_j = ", trans[cur,new])
                print("ln(a_i_j) = ", a_i_j)
                print("signal = ", signal[:,cnt])
                print("X|S = ",x_giv_s)
                print("")  
        
        seq = [0 ... n]
        
           0   1   2   3  
        0              0
        1              0
        2             .4 
        3  1   0   0   0
        
        '''
        #init
        loglike = np.log(trans[-1,seq[0]])
        cnt = 0
        x_giv_s = 0.0
        a_i_j = 0.0
        n = len(seq)
        num_states = np.unique(seq).size
        
        while cnt < n-1:
            cur = seq[cnt]
            new = seq[cnt+1]
            x_giv_s = dists[cur].loglik(signal[:,cnt])
            a_i_j = np.log( trans[cur,new] )   
            loglike += (x_giv_s + a_i_j)
            cnt += 1
            
        x_giv_s = dists[seq[n-1]].loglik(signal[:,n-1])
        a_i_j = np.log( trans[seq[n-1], num_states] )
        loglike += (x_giv_s + a_i_j)
        
        return loglike
            
    def forward(self, signal):
        '''
        See documentation for _forward()
        '''
        return HMM._forward(signal, self.trans, self.dists)

    @staticmethod
    def _forward(signal, trans, dists):
        '''
        Apply the forward algorithm to the observations provided in 'signal' to
        calculate its likelihood.
        Note: `signal` is a SINGLE observation sequence.
    
        Returns the log-likelihood of the observation.

        Parameters
        ----------
        signal : (d,n) ndarray
            Signal for which the optimal state sequence is to be calculated.
            d is the dimension of each observation (number of features)
            n is the number of observations 
        
        trans : (n_states+1,n_states+1) ndarray
            The transition probability table.  The rightmost column contains probability
            of transitioning to final state, and the last row the initial state's
            transition probabilities.   Note that all the rows need to add to 1. 
    
        dists: (n_states,) list
            A list of Gaussian objects defining the emitting pdf's, one object for each 
            emitting  state.

        Return
        ------
        ll : float
            The log-likelihood associated with the observation under the model.
            
            
            
        np.logaddexp(x1,x2) = log( exp(x1) + exp(x2) )
        '''
        # TODO: Implement this function
        N = trans.shape[0]-1
        T = signal.shape[1]
        forward = np.zeros(shape = (N,T), dtype = np.float64) #initialize forward likelihoods (alpha array)
        
        #initialization step: ln{alpha_0(j)} = ln{a_-1(j)} + ln{ p(x_0|s_0 = j)}
        for j in range(0, N):
            forward[j,0] = np.log(trans[-1,j]) + dists[j].loglik(signal[:,0]) 
        
        #recursion step: ln{alpha_t(j)} = ln{ sum_i=0toN-1{ a_ij * alpha_t-1(i)  } } + ln{ p(x_t|s_t = j)}
        for t in range(1,T):
            for j in range(0, N):
                forward[j, t] = np.log(trans[0, j]) + forward[0, t-1] #ln{ a_0j * alpha_t-1(0) }               
                for i in range(1, N):
                    templl = np.log(trans[i, j]) + forward[i, t-1] #ln{ a_ij * alpha_t-1(i)  }    
                    forward[j, t] = np.logaddexp(forward[j, t], templl) 
                forward[j,t] += dists[j].loglik(signal[:,t]) #ln{ p(x_t|s_t = j)}
                    
        ll = np.log(trans[0,N]) + forward[0, T-1] 
        for j in range(1, N): 
            temp = np.log(trans[j,N]) + forward[j, T-1] 
            ll = np.logaddexp(ll, temp)
            
        return ll
     

    def _calcstates(self, trans, dists):
        '''
        Calculate state sequences on the 'signals' maximizing the likelihood for 
        the given HMM parameters.
        
        Calculate the state sequences for each of the given 'signals', maximizing the 
        likelihood of the given parameters of a HMM model. This allocates each of the
        observations, in all the sequences, to one of the states. 
    
        Use the state allocation to calculate an updated transition matrix.   
    
        IMPORTANT: As part of this updated transition matrix calculation, emitting states which 
        are not used in the new state allocation are removed. 
    
        In what follows, n_states is the number of emitting states described in trans, 
        while n_states' is the new number of emitting states.
        
        Note: signals consists of ALL the training sequences and is available
        through the class.
        
        Parameters
        ----------        
        trans : (n_states+1,n_states+1) ndarray
            The transition probability table.  The rightmost column contains probability
            of transitioning to final state, and the last row the initial state's
            transition probabilities.   Note that all the rows need to add to 1. 
    
        dists: (n_states,) list
            A list of Gaussian objects defining the emitting pdf's, one object for each 
            emitting  state.
    
        Return
        ------    
        states : bool (n_obs,n_states') ndarray
            The updated state allocations of each observation in all signals
        trans : (n_states'+ 1,n_states'+1) ndarray
            Updated transition matrix 
        ll : float
            Log-likelihood of all the data
        '''
        # The core of this function involves applying the _viterbi function to each signal stored in the model.
        # Remember to remove emitting states not used in the new state allocation.       
        
        N = trans.shape[0]-1
        return_states = np.empty((1,N), dtype = bool)
        return_trans = np.zeros(shape = (N+1,N+1), dtype = np.float64)
        return_ll = 0.0
        

        for sig in self.signals:
        # BEGIN loop through individual signals
            path,ll = self._viterbi(sig, trans, dists)
            return_ll += ll
            n_obs = path.shape[0]
            # initialize states table for individual signal
            states = np.zeros(shape = (n_obs, N), dtype=bool)
            for n in range(0, n_obs):
                states[n, path[n]] = True

            # tally transitions between states with state tables 
            return_trans[-1, path[0]] += 1
            for i in range(0,N):
                for n in range(0, n_obs-1): 
                    if(states[n,i] == True):
                        for j in range(i,N): #use 'i' for computational efficiency given forward model
                            if(states[n+1,j] == True):
                                return_trans[i,j] += 1   
            return_trans[path[n_obs-1], N] += 1
            
            # add states for individual signal to full states table (i.e. for all signals)
            return_states = np.append(return_states,states, axis = 0)
        # END loop through individual signals
        
        # normalize the tallied trasitions to get estimated transition probabilities
        indices = []
        a = np.sum(return_trans, axis=1)[:,np.newaxis]
        for i in range(0,N):
            if(a[i] == 0):
                indices.append(i) 
        return_trans = np.delete(return_trans, indices, 1)
        return_trans = np.delete(return_trans, indices, 0)
        return_trans /= np.sum(return_trans, axis=1)[:,np.newaxis]
        
        
        # remove dummy obs used for init
        return_states = return_states[1:]
        # remove unused states
        indices = []
        for i in range(0,N):
            if(np.sum(return_states[:,i]) == 0):
                indices.append(i)
        return_states = np.delete(return_states, indices, 1)
            
        return return_states, return_trans, return_ll
        
    def _updatecovs(self, states):
        '''
        Update estimates of the means and covariance matrices for each HMM state
    
        Estimate the covariance matrices for each of the n_states emitting HMM states for 
        the given allocation of the observations in self.data to states. 
        If self.diagcov is true, diagonal covariance matrices are returned.

        Parameters
        ----------
        states : bool (n_obs,n_states) ndarray
            Current state allocations for self.data in model
        
        Return
        ------
        covs: (n_states, d, d) ndarray
            The updated covariance matrices for each state

        means: (n_states, d) ndarray
            The updated means for each state
        '''

        # In this method, if a class has no observations, assign it a mean of zero
        # In this method, estimate a full covariance matrix and discard the non-diagonal elements
        # if a diagonal covariance matrix is required.
        # In this method, if a zero covariance matrix is obtained, assign an identity covariance matrix
        N = states.shape[1]
        d = self.data.shape[0]
        means = np.zeros(shape = (N,d), dtype = np.float64)
        covs = np.zeros(shape = (N, d, d))
        
        for n in range(0,N): #iterate through states
            if(np.sum(states[:,n]) != 0):
                # extract data assoc. w/ each state
                x = self.data[:, states[:,n] == True]
                
                # get means of data in each state
                means[n,:] = np.mean(x,axis=1)
                '''
                EDIT
                # get covariances of data in each state
                seq_obs = x.shape[1]
                working = np.zeros((seq_obs, d, d))
                for i in range(0,seq_obs):
                    a = (x[:,i] - means[n,:])
                    working[i] = np.outer(a,a)
                covs[n,:,:] = np.sum(working, axis = 0)/seq_obs
                '''
                covs[n,:,:] = np.cov(x)
               
                # diagonalize cov. matrix, if desired
                if(self.diagcov == True):
                    a = covs[n]
                    diag = np.einsum('ii->i', a)
                    save = diag.copy()
                    a[...] = 0
                    diag[...] = save
                    covs[n] = a
                    
                # if cov. matrix all zero, return I_(dxd)
                is_all_zero = not np.any(covs[n])
                if(is_all_zero):
                    covs[n] = np.eye(d,d)
        
        return covs, means
        
               
    def _em(self, trans, states):
        '''
        Perform parameter estimation for a hidden Markov model (HMM).
    
        Perform parameter estimation for an HMM using multi-dimensional Gaussian 
        states.  The training observation sequences, signals,  are available 
        to the class, and states designates the initial allocation of emitting states to the
        signal time steps.   The HMM parameters are estimated using Viterbi 
        re-estimation. 
        
        Note: It is possible that some states are never allocated any 
        observations.  Those states are then removed from the states table, effectively redusing
        the number of emitting states. In what follows, n_states is the original 
        number of emitting states, while n_states' is the final number of 
        emitting states, after those states to which no observations were assigned,
        have been removed.
    
        Parameters
        ----------
        trans : (n_states+1,n_states+1) ndarray
            The left-to-right transition probability table.  The rightmost column contains probability
            of transitioning to final state, and the last row the initial state's
            transition probabilities.   Note that all the rows need to add to 1. 
        
        states : (n_obs, n_states) ndarray
            Initial allocation of signal time-steps to states as a one-hot encoding.  Thus
            'states[:,j]' specifies the allocation of all the observations to state j.
        
        Return
        ------
        trans : (n_states'+1,n_states'+1) ndarray
            Updated transition probability table

        dists : (n_states',) list
            Gaussian object of each component.

        newLL : float
            Log-likelihood of parameters at convergence.

        iters: int
            The number of iterations needed for convergence
        '''

        covs, means = self._updatecovs(states) # Initialize the covariances and means using the initial state allocation                
        dists = [Gaussian(mean=means[i], cov=covs[i]) for i in range(len(covs))]
        oldstates, trans, oldLL = self._calcstates(trans, dists)
        converged = False
        iters = 0
        while not converged and iters <  self.maxiters:
            covs, means = self._updatecovs(oldstates)
            dists = [Gaussian(mean=means[i], cov=covs[i]) for i in range(len(covs))]
            newstates, trans, newLL = self._calcstates(trans, dists)
            if abs(newLL - oldLL) / abs(oldLL) < self.rtol:
                converged = True
            oldstates, oldLL = newstates, newLL
            iters += 1
        if iters >= self.maxiters:
            warn("Maximum number of iterations reached - HMM parameters may not have converged")
        return trans, dists, newLL, iters
        
        
        
        
        
    def sample(self):
        '''
        Draw samples from the HMM using the present model parameters. The sequence
        terminates when the final non-emitting state is entered. For the
        left-to-right topology used, this should happen after a finite number of 
        samples is generated, modeling a finite observation sequence. 
        
        Returns
        -------
        samples: (n,) ndarray
            The samples generated by the model
        states: (n,) ndarray
            The state allocation of each sample. Only the emitting states are 
            recorded. The states are numbered from 0 to n_states-1.

        Sample usage
        ------------
        Example below commented out, since results are random and thus not suitable for doctesting.
        However, the example is based on the model fit in the doctests for the class.
        #>>> samples, states = hmm.samples()
        #>>> print(samples)
        #[ 0.9515792   0.9832767   1.04633007  1.01464327  0.98207072  1.01116689
        #  0.31622856  0.20819263  3.57707616]           
        #>>> print(states)   #These will differ for each call
        #[1 1 1 1 1 1 2 2 3]
        '''
        
        #######################################################################
        import scipy.interpolate as interpolate
        def draw_discrete_sample(discr_prob):
            '''
            Draw a single discrete sample from a probability distribution.
            
            Parameters
            ----------
            discr_prob: (n,) ndarray
                The probability distribution.
                Note: sum(discr_prob) = 1
                
            Returns
            -------
            sample: int
                The discrete sample.
                Note: sample takes on the values in the set {0,1,n-1}, where
                n is the the number of discrete probabilities.
            '''

            if not np.sum(discr_prob) == 1:
                raise ValueError('The sum of the discrete probabilities should add to 1')
            x = np.cumsum(discr_prob)
            x = np.hstack((0.,x))
            y = np.array(range(len(x)))
            fn = interpolate.interp1d(x,y)           
            r = np.random.rand(1)
            return np.array(np.floor(fn(r)),dtype=int)[0]
        #######################################################################
        
         # Using the function defined above, draw samples from the HMM 
        
        #use trans matrix to iteratively get states; 
        #use state distr params to get a sample; 
        #add sample and state number to lists
        states = []
        samples = []
        n = 0
        states.append(-1)
        
        while states[n] != self._n_states():
            trans_probs = self.trans[states[n]]
            next_state = draw_discrete_sample( trans_probs )
            states.append(next_state)
            
            if (next_state != self._n_states()):
                next_sample = self.dists[next_state].sample()
                samples.append(next_sample)
            n += 1
           
        return np.array(samples), np.array(states[1:n])
    
if __name__ == "__main__":
    import doctest
    doctest.testmod() 
