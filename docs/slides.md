---
marp: true
theme: default
paginate: true
---

# Hybrid LDA-BERT Topic Modeling
## A Comprehensive Implementation of Probabilistic Reasoning

**Course:** 22AIE301 (Probability and Reasoning)

---

## Introduction

- **Goal:** Develop a robust topic modeling system merging neural embeddings (BERT) with Probabilistic Graphical Models (PGMs).
- **Motivation:** Demonstrate the practical application of the 12 core syllabus topics in a unified natural language processing pipeline.
- **Approach:** Use LDA for probabilistic topic discovery and augment it with HMMs, MRFs, Kalman Filters, and Decision Networks.

---

## Syllabus Coverage Map

| Topic | Implementation in Project |
|---|---|
| 1. Uncertain Knowledge Rep. | Latent topics and document-topic distributions |
| 2. Intro to Bayesian Networks | LDA formulated as a directed graphical model |
| 3. Representation Learning in BNs | Inferring latent structures from text data |
| 4. Inference in BNs | Estimating posteriors over topics |
| 5. Exact & Approximate Inference | Variable Elimination vs. Likelihood Weighting |
| 6. Markov Chain-Monte Carlo | Collapsed Gibbs Sampling for LDA inference |

---

## Syllabus Coverage Map (cont.)

| Topic | Implementation in Project |
|---|---|
| 7. Hidden Markov Models | Tracking temporal topic shifts over discrete time |
| 8. Kalman Filter | Smoothing continuous topic prevalence trajectories |
| 9. Markov Random Fields | Undirected models for topic co-occurrence |
| 10. Message Passing | Belief Propagation for marginals in MRFs |
| 11. Learning in Markov Networks | Parameter estimation for topic cliques |
| 12. Decision Networks | Research planning via Expected Utility & VOI |

---

## Bayesian Network Formulation of LDA

- LDA is a classic **Bayesian Network** (directed acyclic graph).
- **Generative Process:**
  - Draw topic distribution $\theta_d \sim \text{Dirichlet}(\alpha)$
  - Draw word distribution $\phi_k \sim \text{Dirichlet}(\beta)$
  - For each word $w_{d,n}$:
    - Draw topic $z_{d,n} \sim \text{Multinomial}(\theta_d)$
    - Draw word $w_{d,n} \sim \text{Multinomial}(\phi_{z_{d,n}})$

---

## Exact vs Approximate Inference

- **Exact Inference:** Variable Elimination is computationally intractable for LDA due to the massive state space of latent variables.
- **Approximate Inference:** We employ sampling methods (Likelihood Weighting) and variational techniques to approximate the intractable posterior distribution $P(Z, \Theta, \Phi | W, \alpha, \beta)$.

---

## Markov Chain-Monte Carlo (MCMC)

- **Gibbs Sampling:** A specific MCMC algorithm used to infer the hidden topic assignments.
- Iteratively samples the topic for each word conditional on the current topic assignments of all other words.
- Converges to the true posterior distribution over time, allowing for effective **Uncertain Knowledge Representation**.

---

## Hidden Markov Models (HMM)

- **Purpose:** Model the dynamic evolution of a document's dominant topic over sequential segments (e.g., temporal bins).
- **Components:**
  - Hidden States: Dominant topics
  - Observations: Text features/words
- **Algorithms:** Forward-Backward (smoothing), Viterbi (decoding), Baum-Welch (learning).

---

## Kalman Filter for Topic Smoothing

- **Purpose:** Smooth the noisy trajectories of topic prevalence over time.
- Treats topic prevalence as a continuous latent state variable.
- Uses linear transition models with Gaussian noise to estimate the true underlying trend of a topic's popularity.

---

## MRF & Message Passing

- **Markov Random Fields (MRFs):** Undirected graphical models capturing the co-occurrence probabilities between different topics.
- **Message Passing:** We implement Belief Propagation (Sum-Product algorithm) to efficiently compute marginal distributions over topic nodes in the MRF.
- **Learning:** Estimating the clique potentials from document-topic empirical counts.

---

## Decision Networks for Research Planning

- Extends BNs by adding **Decision Nodes** (Action: which topic to research) and **Utility Nodes** (Payoff: impact factor, relevance).
- **Expected Utility:** Computes the best research direction based on probabilistic topic trends.
- **Value of Information (VOI):** Determines the benefit of acquiring more data before making a research decision.

---

## Methodology Pipeline

1. **Preprocessing:** Tokenization, stop-word removal.
2. **Embeddings:** SBERT for dense representation.
3. **Topic Modeling:** LDA via Gibbs Sampling (BN, MCMC).
4. **Temporal Dynamics:** HMM & Kalman Filter.
5. **Topic Relationships:** MRFs & Belief Propagation.
6. **Decision Making:** Decision Networks.

---

## Results: Topic Coherence

- Hybrid LDA-BERT achieves higher NPMI coherence scores than baseline LDA.
- BERT embeddings effectively guide the Gibbs sampler in the BN, mitigating the sparsity problem.

---

## Results: Dynamic & Relational Modeling

- **HMMs** successfully segment document streams into coherent thematic phases.
- **Kalman Filtering** reveals clear, denoised trends in topic popularity over time.
- **MRFs** accurately identify synergistic topic clusters (e.g., "Machine Learning" strongly links to "Data Science").

---

## Conclusion

- A complete end-to-end NLP system built entirely on the foundations of Probabilistic Reasoning.
- Successfully integrates all 12 core syllabus topics, from basic uncertain knowledge representation to complex decision-theoretic planning.
- Demonstrates that classic PGMs (BNs, MRFs, HMMs) remain powerful when augmented with modern representation learning (BERT).
