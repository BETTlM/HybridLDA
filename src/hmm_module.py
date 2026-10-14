"""
Hidden Markov Model Analysis Module

HMMs and Bayesian Networks:
A Hidden Markov Model (HMM) is a dynamic Bayesian Network (a directed graphical model)
used to represent probability distributions over sequences of observations. It assumes a
Markov process with unobserved (hidden) states. The hidden states form a Markov chain,
and each state generates an observation based on an emission probability.

Sequential/Temporal Uncertainty:
HMMs model sequential and temporal uncertainty by maintaining a belief state (distribution
over hidden states) that is updated as new observations arrive. In topic modeling, we can
model the evolution of dominant topics over time as an HMM, where hidden states are the
underlying dominant topics (which evolve with a transition probability), and the observations
are the empirical word distributions or document topic distributions in each time period.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class HMM:
    def __init__(self, n_states: int, n_obs: int):
        self.n_states = n_states
        self.n_obs = n_obs
        # Initialize uniformly, then add random noise and normalize
        self.pi = np.random.dirichlet(np.ones(n_states))
        self.A = np.random.dirichlet(np.ones(n_states), size=n_states)
        self.B = np.random.dirichlet(np.ones(n_obs), size=n_states)

    def forward(self, obs_seq: List[int]) -> Tuple[float, np.ndarray]:
        """The Forward algorithm (exact inference for P(observations)) with scaling to prevent underflow."""
        T = len(obs_seq)
        alpha = np.zeros((T, self.n_states))
        c = np.zeros(T)
        
        # Initialization
        alpha[0, :] = self.pi * self.B[:, obs_seq[0]]
        c[0] = np.sum(alpha[0, :])
        if c[0] > 0:
            alpha[0, :] = alpha[0, :] / c[0]
        
        # Recursion
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1, :] * self.A[:, j]) * self.B[j, obs_seq[t]]
            c[t] = np.sum(alpha[t, :])
            if c[t] > 0:
                alpha[t, :] = alpha[t, :] / c[t]
                
        # Termination: sum of log scale factors
        log_prob_obs = np.sum(np.log(c + 1e-12))
        return log_prob_obs, alpha

    def viterbi(self, obs_seq: List[int]) -> List[int]:
        """The Viterbi algorithm (most likely state sequence) using log probabilities."""
        T = len(obs_seq)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        log_pi = np.log(self.pi + 1e-12)
        log_A = np.log(self.A + 1e-12)
        log_B = np.log(self.B + 1e-12)
        
        # Initialization
        delta[0, :] = log_pi + log_B[:, obs_seq[0]]
        
        # Recursion
        for t in range(1, T):
            for j in range(self.n_states):
                probs = delta[t-1, :] + log_A[:, j]
                psi[t, j] = int(np.argmax(probs))
                delta[t, j] = np.max(probs) + log_B[j, obs_seq[t]]
                
        # Path backtracking
        path = np.zeros(T, dtype=int)
        path[T-1] = int(np.argmax(delta[T-1, :]))
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
            
        return path.tolist()

    def baum_welch(self, obs_seq: List[int], n_iters: int = 10):
        """The Forward-Backward (Baum-Welch) algorithm for parameter learning."""
        T = len(obs_seq)
        for it in range(n_iters):
            # Forward pass
            alpha = np.zeros((T, self.n_states))
            c = np.zeros(T)
            alpha[0, :] = self.pi * self.B[:, obs_seq[0]]
            c[0] = np.sum(alpha[0, :]) + 1e-12
            alpha[0, :] /= c[0]
            
            for t in range(1, T):
                alpha[t, :] = (alpha[t-1, :] @ self.A) * self.B[:, obs_seq[t]]
                c[t] = np.sum(alpha[t, :]) + 1e-12
                alpha[t, :] /= c[t]
                
            # Backward pass
            beta = np.zeros((T, self.n_states))
            beta[T-1, :] = 1.0
            for t in range(T-2, -1, -1):
                beta[t, :] = (self.A @ (self.B[:, obs_seq[t+1]] * beta[t+1, :])) / c[t+1]
                
            # Compute xi and gamma
            gamma = np.zeros((T, self.n_states))
            xi = np.zeros((T-1, self.n_states, self.n_states))
            
            for t in range(T-1):
                denom = np.sum(alpha[t, :] * beta[t, :]) + 1e-12
                for i in range(self.n_states):
                    gamma[t, i] = (alpha[t, i] * beta[t, i]) / denom
                    for j in range(self.n_states):
                        xi[t, i, j] = (alpha[t, i] * self.A[i, j] * self.B[j, obs_seq[t+1]] * beta[t+1, j]) / c[t+1] / denom

            gamma[T-1, :] = alpha[T-1, :] * beta[T-1, :] / (np.sum(alpha[T-1, :] * beta[T-1, :]) + 1e-12)
            
            # Update parameters
            self.pi = gamma[0, :] / np.sum(gamma[0, :])
            
            for i in range(self.n_states):
                denom = np.sum(gamma[:-1, i]) + 1e-12
                self.A[i, :] = np.sum(xi[:, i, :], axis=0) / denom
                
                denom_B = np.sum(gamma[:, i]) + 1e-12
                for k in range(self.n_obs):
                    mask = (np.array(obs_seq) == k)
                    self.B[i, k] = np.sum(gamma[mask, i]) / denom_B

def run_hmm_analysis(years: pd.Series, theta: np.ndarray, topic_words: List[List[str]], n_states: int) -> Dict[str, Any]:
    """Runs the HMM on temporal data, using actual document theta mapped by year."""
    logger.info("Starting HMM Analysis...")
    
    # Process actual data
    df = pd.DataFrame({"year": years, "topic": np.argmax(theta, axis=1)})
    df = df.dropna().astype({"year": int, "topic": int})
    
    if len(df) < 5:
        return {"error": "Not enough temporal data"}
        
    # Group by year and find the dominant topic of that year (mode)
    yearly_dominant = df.groupby("year")["topic"].agg(lambda x: x.mode()[0] if not x.mode().empty else 0).sort_index()
    obs_seq = yearly_dominant.values.tolist()
    
    if len(obs_seq) < 3:
        return {"error": "Sequence too short"}
        
    n_obs = theta.shape[1]
    hmm = HMM(n_states=n_states, n_obs=n_obs)
    
    # Train parameters
    hmm.baum_welch(obs_seq, n_iters=10)
    
    # Inference
    log_prob, alpha = hmm.forward(obs_seq)
    viterbi_path = hmm.viterbi(obs_seq)
    
    return {
        'log_forward_prob': log_prob,
        'viterbi_path': viterbi_path,
        'transition_matrix': hmm.A.tolist(),
        'emission_probs': hmm.B.tolist(),
        'years_modeled': yearly_dominant.index.tolist(),
        'observed_sequence': obs_seq
    }
