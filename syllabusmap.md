# Syllabus Mapping: Probabilistic Reasoning (22AIE301)

This document maps the 12 core topics of the Probability and Reasoning syllabus to their concrete implementations within the Hybrid LDA–LLM Topic Modeling project. It explains **where** each concept is located in the codebase and **how** it is mathematically and algorithmically implemented.

---

### 1. Uncertain Knowledge Representation
*   **Location:** `src/lda_baseline.py` and `src/bayesian_network.py`
*   **How it is used:** Instead of assigning a document to a single rigid category, the system uses probability distributions to represent uncertainty. A document is represented as a probability distribution over topics ($\theta_d$), and a topic is a probability distribution over words ($\phi_k$).
*   **Implementation:** The code uses Dirichlet priors ($\alpha$, $\eta$) to govern the uncertainty of these distributions.

### 2. Introduction to Bayesian Networks (BNs)
*   **Location:** `src/bayesian_network.py`
*   **How it is used:** Latent Dirichlet Allocation (LDA) is explicitly formulated as a Directed Acyclic Graph (DAG) with plate notation.
*   **Implementation:** The module constructs the theoretical DAG defining the generative process: hyper-parameters ($\alpha$, $\beta$) $\rightarrow$ distributions ($\theta$, $\phi$) $\rightarrow$ latent topic assignments ($z$) $\rightarrow$ observed words ($w$).

### 3. Representation Learning in Bayesian Networks
*   **Location:** `src/bayesian_network.py` (`learn_bn_structure_from_data` function)
*   **How it is used:** Discovers the latent dependency structure between different research topics rather than assuming they are independent.
*   **Implementation:** Uses **Mutual Information (MI)** to score pairwise dependencies between topic activations across the corpus. It applies a greedy structure learning algorithm to build a compact, cycle-free DAG (BN structure) maximizing the MI score.

### 4. Inference in Bayesian Networks
*   **Location:** `src/bayesian_network.py` and `src/retrieve.py`
*   **How it is used:** Calculating the posterior probability of a topic given an observed sequence of words (i.e., finding what a query abstract is about).
*   **Implementation:** Translates the inference task into computing $P(z \mid w)$. This is heavily utilized in the `query_topic_distribution` function, which maps an unseen query into the learned probability space.

### 5. Exact and Approximate Inference
*   **Location:** `src/bayesian_network.py`
*   **How it is used:** Demonstrates computational approaches to solving the inference problem on small vs. large networks.
*   **Implementation:** 
    *   **Exact Inference:** Implements **Variable Elimination** by explicitly marginalizing out hidden variables (summing over all possible topic assignments) for a small, tractable subset of the network.
    *   **Approximate Inference:** Implements **Likelihood Weighting**, a sampling-based approach that weighs samples by the probability of evidence, allowing inference to scale to larger networks.

### 6. Markov Chain-Monte Carlo (MCMC) Method
*   **Location:** `src/bayesian_network.py` (`GibbsLDASampler` class)
*   **How it is used:** Provides a stochastic alternative to Variational Inference for training the topic model.
*   **Implementation:** Implements a **Collapsed Gibbs Sampler**. The algorithm integrates out the continuous distributions ($\theta$, $\phi$) and iteratively resamples discrete topic assignments $z_{d,n}$ from their conditional posterior distributions forming a Markov Chain that converges to the true posterior.

### 7. Hidden Markov Models (HMMs)
*   **Location:** `src/hmm_module.py`
*   **How it is used:** Models the temporal dynamics of research topics over time (e.g., how the dominant ML research topic shifts from year to year).
*   **Implementation:** 
    *   **Hidden states:** The underlying dominant research theme.
    *   **Observations:** The empirical word/topic distributions observed in the papers of that year.
    *   **Algorithms implemented:** Forward algorithm (exact sequence probability), Viterbi decoding (finding the most likely historical path of topic shifts), and Baum-Welch (learning transition and emission probabilities).

### 8. Kalman Filter
*   **Location:** `src/kalman_filter.py`
*   **How it is used:** The continuous-state counterpart to HMMs, used to track and smooth the continuous temporal prevalence of topics across publication years.
*   **Implementation:** Models topic prevalence as a linear dynamical system subject to Gaussian noise. Implements the standard predict-update cycle for forward filtering, and the **Rauch-Tung-Striebel (RTS) backward smoother** to compute smoothed historical trajectories alongside 95% uncertainty (covariance) bands.

### 9. Markov Random Fields (MRF) & Markov Networks
*   **Location:** `src/markov_network.py`
*   **How it is used:** Models undirected, symmetric co-occurrence relationships between topics (e.g., "Computer Vision" and "Deep Learning" frequently appearing together).
*   **Implementation:** Constructs an undirected graph where nodes are topics and edges represent affinity. Defines **pairwise potential functions** ($\psi$) based on the cosine similarity of topic activations across documents.

### 10. Message Passing
*   **Location:** `src/markov_network.py` (`loopy_belief_propagation` function)
*   **How it is used:** Computes the marginal probability of a topic becoming "active" given the activation of its neighboring topics in the MRF.
*   **Implementation:** Implements the **Sum-Product algorithm (Loopy Belief Propagation)**. Nodes repeatedly exchange messages (beliefs about neighboring states) until convergence, resolving marginal probabilities in a network with cycles.

### 11. Learning in Markov Networks
*   **Location:** `src/markov_network.py` (`learn_mrf_potentials` function)
*   **How it is used:** Learns the edge weights (potentials) of the MRF directly from the document-topic matrix without brute-force calculating the computationally intractable partition function ($Z$).
*   **Implementation:** Uses **Maximum Pseudo-Likelihood Estimation (MPLE)**. It optimizes the conditional probability of each node given its Markov blanket, sidestepping the exact computation of $Z$.

### 12. Decision Networks
*   **Location:** `src/decision_network.py`
*   **How it is used:** Extends the Bayesian Network with Decision and Utility nodes to model the practical problem: *"Which research topic should a researcher focus on next?"*
*   **Implementation:** 
    *   **Nodes:** Decision node (Topic Choice), Chance nodes (Topic Coherence, Future Growth), Utility node (Expected Research Value).
    *   **Algorithms:** Computes the **Maximum Expected Utility (MEU)** by enumerating over chance outcomes. Calculates the **Value of Information (VOI)** to measure how much utility would increase if we perfectly knew the future trend of a topic before making the decision.
