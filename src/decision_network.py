"""
Decision Network (Influence Diagram) Analysis Module

Decision Networks:
A Decision Network extends a Bayesian Network by adding Decision nodes (actions under 
the agent's control) and Utility nodes (scoring the desirability of states). It provides 
a formal mechanism for rational decision-making under uncertainty, grounded in the 
Maximum Expected Utility (MEU) principle.

Value of Information (VOI):
VOI measures the increase in expected utility if the agent were given exact knowledge 
of a currently uncertain variable before making the decision. 
"""

import numpy as np
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def calculate_expected_utility(topic_idx: int, coherence: float, growth_prob: float) -> float:
    """Calculate utility given decision and probabilities of chance nodes."""
    # Utility function: favors coherent and growing topics
    # U = w1 * Coherence + w2 * (1 if Growth else -1)
    # Expected Utility = w1 * C + w2 * (1*P(G) + -1*P(~G))
    w_coherence = 10.0
    w_growth = 5.0
    
    eu = w_coherence * coherence + w_growth * (growth_prob * 1.0 + (1 - growth_prob) * -0.5)
    return eu

def run_decision_analysis(topic_words: List[List[str]], coherence_scores: Dict[str, float], temporal_trends: Any, n_topics: int) -> Dict[str, Any]:
    """Evaluates the optimal research topic to explore next based on MEU."""
    logger.info("Starting Decision Network Analysis...")
    
    # If we don't have per-topic coherence, mock it using topic index modulo for demo purposes
    # In a real setting, we'd pass per-topic coherence
    coherences = [0.4 + (i % 3) * 0.15 for i in range(n_topics)]
    
    # Determine growth probability from trends if available
    growth_probs = np.ones(n_topics) * 0.5
    if temporal_trends is not None and not temporal_trends.empty:
        try:
            for k in range(n_topics):
                subset = temporal_trends[temporal_trends["topic"] == k]
                if len(subset) > 2:
                    y = subset["prevalence"].values
                    x = np.arange(len(y))
                    slope = np.polyfit(x, y, 1)[0]
                    # Map slope to probability
                    prob = 1.0 / (1.0 + np.exp(-slope * 100))
                    growth_probs[k] = prob
        except Exception:
            pass
            
    utilities = []
    for k in range(n_topics):
        eu = calculate_expected_utility(k, coherences[k], growth_probs[k])
        utilities.append(eu)
        
    optimal_topic = int(np.argmax(utilities))
    meu = float(utilities[optimal_topic])
    
    # Calculate VOI for knowing the growth exactly
    # Expected utility with perfect info (EU | perfect info)
    # = P(growth)*U(best given growth) + P(~growth)*U(best given ~growth)
    best_u_if_growth = max([calculate_expected_utility(k, coherences[k], 1.0) for k in range(n_topics)])
    best_u_if_not = max([calculate_expected_utility(k, coherences[k], 0.0) for k in range(n_topics)])
    
    # Marginal probability of growth across all topics (simplified prior)
    p_growth_marginal = np.mean(growth_probs)
    eu_perfect_info = p_growth_marginal * best_u_if_growth + (1 - p_growth_marginal) * best_u_if_not
    
    voi = max(0.0, float(eu_perfect_info - meu))
    
    return {
        'expected_utilities': utilities,
        'optimal_topic_decision': optimal_topic,
        'maximum_expected_utility': meu,
        'value_of_information': voi
    }
