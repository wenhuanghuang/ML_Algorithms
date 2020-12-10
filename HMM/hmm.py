from __future__ import print_function
import json
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_{t-1} = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: A dictionary mapping each observation symbol to its index 
        - state_dict: A dictionary mapping each state to its index
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array where alpha[i, t-1] = P(Z_t = s_i, X_{1:t}=x_{1:t})
                 (note that this is alpha[i, t-1] instead of alpha[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        alpha = np.zeros([S, L])
        ######################################################
        # compute and return the forward messages alpha
        ######################################################
        alpha[:,0] = self.B[:, O[0]] * self.pi
        for t in range(1, L):
            alpha[:, t] = np.dot(self.A.T, alpha[:, t-1]) * self.B[:, O[t]]
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array where beta[i, t-1] = P(X_{t+1:T}=x_{t+1:T} | Z_t = s_i)
                    (note that this is beta[i, t-1] instead of beta[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        beta = np.zeros([S, L])
        #######################################################
        # compute and return the backward messages beta
        #######################################################
        beta[:, -1] = 1
        for t in range(L - 2, -1, -1):
            beta[:, t] = np.sum((beta[:, t+1] * self.B[:, O[t+1]]).T * self.A, axis=1)
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(X_{1:T}=x_{1:T})
        """
        
        #####################################################
        # compute and return prob = P(X_{1:T}=x_{1:T})
        #   using the forward/backward messages
        #####################################################
        alpha = self.forward(Osequence)
        prob = sum(alpha[:, -1])
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - gamma: (num_state*L) A numpy array where gamma[i, t-1] = P(Z_t = s_i | X_{1:T}=x_{1:T})
		           (note that this is gamma[i, t-1] instead of gamma[i, t])
        """
        ######################################################################
        # compute and return gamma using the forward/backward messages
        ######################################################################
        L = len(Osequence)
        S = len(self.pi)
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        gamma = np.zeros([S,L])
        for t in range(L):
            gamma[:, t] = alpha[:, t] * beta[:, t]
        gamma /= self.sequence_prob(Osequence)
        return gamma 
    
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array where prob[i, j, t-1] = 
                    P(Z_t = s_i, Z_{t+1} = s_j | X_{1:T}=x_{1:T})
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        
        #####################################################################
        # compute and return prob using the forward/backward messages
        #####################################################################
        O = self.find_item(Osequence)
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        for t in range(L - 1):
            prob[:, :, t] = np.dot(alpha[:, t].reshape(S,-1), beta[:, t+1].reshape(1, -1)) * self.A * self.B[:, O[t+1]]
        prob /= np.sum(np.sum(prob, axis=1),axis=0)
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden states (return actual states instead of their indices;
                    you might find the given function self.find_key useful)
        """
        path = []
        ################################################################################
        # implement the Viterbi algorithm and return the most likely state path
        ################################################################################
        S = len(self.state_dict)
        L = len(Osequence)
        delta = np.zeros((S, L))
        ranker = np.zeros((S, L)).astype(int)
                  
        delta[:, 0] = self.pi * self.B[:, self.obs_dict[Osequence[0]]]
        for t in range(1, L):
            for s in range(S):
                probs = delta[:, t - 1] * self.A[:, s] * self.B[s, self.obs_dict[Osequence[t]]]
                ranker[s, t] = np.argmax(probs)
                delta[s, t] = np.max(probs)

        z = np.argmax(delta[:, L-1])
        path.append(self.find_key(self.state_dict, z))
        for T in range(L - 1, 0, -1):
            z = ranker[z, T]
            path.append(self.find_key(self.state_dict, z))
        path.reverse()
        return path

    def find_key(self, obs_dict, idx):
        for item in obs_dict:
            if obs_dict[item] == idx:
                return item

    def find_item(self, Osequence):
        O = []
        for item in Osequence:
            O.append(self.obs_dict[item])
        return O
