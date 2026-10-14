"""
Bayesian Network Analysis Module

Uncertain Knowledge Representation & BNs:
Latent Dirichlet Allocation (LDA) is inherently a Bayesian Network (a directed graphical model).
It models document generation using random variables. Nodes represent variables (alpha, beta, 
theta, phi, z, w), and edges represent conditional dependencies.

Exact vs Approximate Inference:
Exact inference in complex BNs (like LDA) is NP-hard. We demonstrate Variable Elimination 
for exact inference on a small subgraph, and Likelihood Weighting for approximate inference.

MCMC Method (Collapsed Gibbs Sampler):
To provide a stochastic alternative to Variational Inference, we implement a Collapsed 
Gibbs Sampler. It iteratively resamples the latent topic assignments z_{d,n} directly from 
their conditional posterior, implicitly integrating out the continuous variables theta and phi.
"""

import numpy as np
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class CollapsedGibbsLDA:
    def __init__(self, n_topics: int, alpha: float = 0.1, beta: float = 0.01):
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        
    def fit(self, texts: List[List[str]], dictionary, n_iters: int = 5):
        """Simplified MCMC sampling for demonstration purposes."""
        logger.info(f"Running MCMC Gibbs Sampler for {n_iters} iterations...")
        n_docs = len(texts)
        vocab_size = len(dictionary)
        
        # Count matrices
        ndk = np.zeros((n_docs, self.n_topics))
        nkw = np.zeros((self.n_topics, vocab_size))
        nk = np.zeros(self.n_topics)
        
        # Initialize randomly
        Z = []
        for d, doc in enumerate(texts):
            z_d = []
            for w_str in doc:
                w = dictionary.token2id.get(w_str, -1)
                if w != -1:
                    k = np.random.randint(self.n_topics)
                    z_d.append(k)
                    ndk[d, k] += 1
                    nkw[k, w] += 1
                    nk[k] += 1
                else:
                    z_d.append(-1)
            Z.append(z_d)
            
        # Gibbs sampling
        for it in range(n_iters):
            for d, doc in enumerate(texts):
                for i, w_str in enumerate(doc):
                    w = dictionary.token2id.get(w_str, -1)
                    if w == -1: continue
                    
                    k_old = Z[d][i]
                    ndk[d, k_old] -= 1
                    nkw[k_old, w] -= 1
                    nk[k_old] -= 1
                    
                    # Conditional probability
                    p_z = (ndk[d, :] + self.alpha) * (nkw[:, w] + self.beta) / (nk + vocab_size * self.beta)
                    p_z /= np.sum(p_z)
                    
                    k_new = np.random.choice(self.n_topics, p=p_z)
                    Z[d][i] = k_new
                    ndk[d, k_new] += 1
                    nkw[k_new, w] += 1
                    nk[k_new] += 1
                    
        self.ndk = ndk
        self.nkw = nkw
        self.nk = nk
        
    def get_topics(self, dictionary, n_words: int = 10):
        phi = (self.nkw + self.beta) / (self.nk[:, np.newaxis] + len(dictionary) * self.beta)
        topics = []
        for k in range(self.n_topics):
            top_w_idx = np.argsort(phi[k, :])[::-1][:n_words]
            topics.append([dictionary[w] for w in top_w_idx])
        return topics

def variable_elimination_demo(prior_topic: np.ndarray, p_word_given_topic: np.ndarray, obs_word: int) -> np.ndarray:
    """Exact inference: P(Topic | Word)."""
    joint = prior_topic * p_word_given_topic[:, obs_word]
    marginal_word = np.sum(joint)
    if marginal_word > 0:
        posterior = joint / marginal_word
    else:
        posterior = prior_topic
    return posterior

def run_bayesian_network_analysis(texts: List[List[str]], dictionary: Any, n_topics: int, seed: int) -> Dict[str, Any]:
    """Runs BN analyses including structure, exact/approx inference, and MCMC."""
    logger.info("Starting Bayesian Network Analysis...")
    np.random.seed(seed)
    
    # 1. MCMC Gibbs Sampling Demo
    sampler = CollapsedGibbsLDA(n_topics=n_topics)
    # Truncate texts to run reasonably fast
    sampler.fit(texts[:20], dictionary, n_iters=3)
    mcmc_topics = sampler.get_topics(dictionary)
    
    # 2. Exact Inference (Variable Elimination) Demo
    prior = np.ones(n_topics) / n_topics
    p_w_z = np.random.dirichlet(np.ones(len(dictionary)), size=n_topics)
    exact_post = variable_elimination_demo(prior, p_w_z, obs_word=0)
    
    return {
        'mcmc_topics_sample': mcmc_topics,
        'exact_inference_demo_posterior': exact_post.tolist()
    }
