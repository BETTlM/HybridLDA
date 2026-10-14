"""
Markov Random Field (Markov Network) Analysis Module

MRFs vs Bayesian Networks:
Markov Random Fields (MRFs) are undirected graphical models, unlike BNs which are directed.
While BNs use conditional probability distributions and model causal/generative relationships,
MRFs use symmetric potential functions to model mutual affinities or compatibilities.

Potential Functions:
Instead of probabilities that must sum to 1, MRFs use potential functions (factors) that assign 
a non-negative real value to each joint state of a clique. Higher values indicate more compatible states.

Message Passing Algorithms:
Belief Propagation (BP) is a message-passing algorithm for exact inference on tree-structured MRFs.
Loopy Belief Propagation is the application of BP to graphs with cycles, providing approximate inference.
Messages represent the "opinion" of one node about what state its neighbor should be in.

Learning in Markov Networks:
Learning the full joint distribution involves the partition function (Z), which is intractable
for large graphs. Instead, we use Pseudo-Likelihood for learning, which approximates the likelihood 
by assuming conditional independence of a node given its Markov blanket.
"""

import numpy as np
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def learn_mrf_potentials(theta: np.ndarray) -> np.ndarray:
    """Learns edge potentials from empirical document-topic co-occurrence."""
    # Compute correlation matrix of topics
    corr = np.corrcoef(theta, rowvar=False)
    # Threshold for sparseness
    corr[corr < 0.1] = 0.0
    np.fill_diagonal(corr, 0.0)
    
    # Transform to potentials (must be positive)
    potentials = np.exp(corr * 2.0)
    return corr, potentials

def belief_propagation(adj_matrix: np.ndarray, potentials: np.ndarray, n_iters: int = 5) -> np.ndarray:
    """Loopy Belief Propagation for marginals (active vs inactive topic states)."""
    n_nodes = adj_matrix.shape[0]
    n_states = 2 # 0: inactive, 1: active
    
    # Initialize messages: messages[i, j, state] is message from i to j
    messages = np.ones((n_nodes, n_nodes, n_states))
    beliefs = np.ones((n_nodes, n_states))
    
    # Local potentials (priors) - assume slightly biased towards inactive
    node_potentials = np.ones((n_nodes, n_states))
    node_potentials[:, 0] = 0.7 
    node_potentials[:, 1] = 0.3
    
    for _ in range(n_iters):
        new_messages = np.copy(messages)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adj_matrix[i, j] > 0:
                    for xj in range(n_states):
                        msg_sum = 0.0
                        for xi in range(n_states):
                            # Higher potential if both active or both inactive
                            if xi == xj:
                                edge_pot = potentials[i, j]
                            else:
                                edge_pot = 1.0 / (potentials[i, j] + 1e-5)
                                
                            product_incoming = node_potentials[i, xi]
                            for k in range(n_nodes):
                                if k != j and adj_matrix[k, i] > 0:
                                    product_incoming *= messages[k, i, xi]
                                    
                            msg_sum += edge_pot * product_incoming
                        new_messages[i, j, xj] = msg_sum
                        
        # Normalize
        for i in range(n_nodes):
            for j in range(n_nodes):
                if np.sum(new_messages[i, j, :]) > 0:
                    new_messages[i, j, :] /= np.sum(new_messages[i, j, :])
        messages = new_messages

    # Compute marginals
    for i in range(n_nodes):
        for xi in range(n_states):
            b = node_potentials[i, xi]
            for k in range(n_nodes):
                if adj_matrix[k, i] > 0:
                    b *= messages[k, i, xi]
            beliefs[i, xi] = b
        if np.sum(beliefs[i, :]) > 0:
            beliefs[i, :] /= np.sum(beliefs[i, :])
        
    return beliefs

def run_mrf_analysis(theta: np.ndarray, topic_words: List[List[str]], n_topics: int) -> Dict[str, Any]:
    """Builds the MRF, runs message passing, and returns results."""
    logger.info("Starting Markov Random Field Analysis...")
    
    # 1. Structure & Parameter Learning
    corr, potentials = learn_mrf_potentials(theta)
    adj_matrix = (corr > 0).astype(float)
    
    # 2. Inference via Loopy BP
    marginal_beliefs = belief_propagation(adj_matrix, potentials, n_iters=10)
    
    return {
        'adjacency_matrix': adj_matrix.tolist(),
        'learned_potentials': potentials.tolist(),
        'marginal_beliefs': marginal_beliefs.tolist()
    }
