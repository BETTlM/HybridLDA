# Hybrid LDA-BERT Topic Modeling: A Probabilistic Reasoning Approach

## 1. Abstract
This project presents a Hybrid LDA-BERT Topic Modeling system, serving as a comprehensive demonstration of probabilistic reasoning and representation learning. Designed to align with the core syllabus of 22AIE301 (Probability and Reasoning), the system integrates advanced language models (BERT) with classical probabilistic graphical models. We frame the core Latent Dirichlet Allocation (LDA) model as a **Bayesian Network**, exploring both **Exact and Approximate Inference** techniques. Specifically, we implement **Markov Chain-Monte Carlo (MCMC)** via Gibbs sampling to infer the latent topic distributions, navigating **Uncertain Knowledge Representation**. Beyond static topic extraction, the project employs **Hidden Markov Models (HMMs)** and the **Kalman Filter** to model the temporal evolution and dynamic smoothing of topic prevalence. We capture complex topic co-occurrences using **Markov Random Fields (MRFs) & Markov Networks**, utilizing **Message Passing** (Belief Propagation) for efficient inference and exploring **Learning in Markov Networks**. Finally, we bridge probabilistic inference and action through **Decision Networks**, computing Expected Utility and Value of Information to optimize research topic selection. This hybrid approach not only achieves high topic coherence but also provides a holistic, rigorous implementation of the 12 core probabilistic reasoning topics.

## 2. Introduction
Natural Language Processing (NLP) heavily relies on effectively handling ambiguity and uncertainty. Topic modeling, a cornerstone of unsupervised NLP, aims to discover the hidden thematic structure within large archives of documents. Traditional models like Latent Dirichlet Allocation (LDA) approach this fundamentally as a probabilistic inference problem. However, modern approaches often leverage deep representation learning, such as BERT, to capture rich semantic nuances. 

This project seeks to bridge classical probabilistic graphical models (PGMs) with modern representation learning to create a robust, end-to-end topic modeling system. More importantly, this project is structured as a comprehensive applied demonstration of the concepts covered in the course 22AIE301 (Probability and Reasoning). By explicitly modeling the pipeline through the lens of Bayesian Networks, Markov Networks, dynamic models (HMMs, Kalman Filters), and Decision Theory, we demonstrate how foundational probabilistic reasoning principles govern complex, real-world AI systems.

## 3. Background: Course Syllabus Mapping
The project is meticulously designed to implement and explore all 12 core topics from the course syllabus.

### 3.1 Uncertain Knowledge Representation
Text data is inherently noisy and ambiguous. We represent the "true" meaning of a document as a latent variable (a distribution over topics). This captures the uncertainty inherent in natural language, where a single word can have multiple meanings depending on context.

### 3.2 Introduction to Bayesian Networks (BNs)
LDA is fundamentally a directed graphical model, or Bayesian Network. It defines a joint probability distribution over observed words and hidden topic structures, expressed via a directed acyclic graph mapping the generative process of a document.

### 3.3 Representation Learning in Bayesian Networks
We integrate BERT embeddings to improve the representation of our data prior to probabilistic inference. Furthermore, the inference process of LDA itself is a form of representation learning, projecting high-dimensional term-frequency vectors into a lower-dimensional probabilistic topic space.

### 3.4 Inference in Bayesian Networks
Given the observed words, we must infer the posterior distribution of the latent topics. This is the core task of our topic modeling system: computing \( P(Z, \Theta, \Phi | W, \alpha, \beta) \).

### 3.5 Exact and Approximate Inference
We demonstrate why exact inference (e.g., Variable Elimination) is computationally intractable for LDA due to the massive combinatorial space of topic assignments. Consequently, we must rely on approximate inference techniques, such as Variational Inference and sampling methods.

### 3.6 Markov Chain-Monte Carlo (MCMC) Method
To perform approximate inference, we implement Collapsed Gibbs Sampling, a powerful MCMC algorithm. It iteratively samples the conditional distribution of each latent variable given the current state of all others, eventually converging to the true posterior.

### 3.7 Hidden Markov Models (HMM)
Documents often exhibit temporal structure. We utilize HMMs to model how the dominant topic of a text stream evolves over time (e.g., across chapters of a book or sequential publications), using algorithms like Forward-Backward and Viterbi.

### 3.8 Kalman Filter
While HMMs handle discrete states, we apply the Kalman Filter to continuous variables, specifically smoothing the noisy, observed trajectories of topic prevalence over time to uncover the true underlying trend.

### 3.9 Markov Random Fields (MRF) & Markov Networks
Topics do not exist in isolation. We model the undirected, correlational structure of topic co-occurrences using MRFs, representing synergistic relationships between different fields (e.g., "AI" and "Ethics").

### 3.10 Message Passing
To perform inference on our MRF, we implement Message Passing algorithms, specifically Belief Propagation (the Sum-Product algorithm), allowing us to efficiently compute the marginal probabilities of topic nodes.

### 3.11 Learning in Markov Networks
We estimate the parameters (clique potentials) of our MRF directly from the empirical document-topic distributions inferred by our LDA model, demonstrating maximum likelihood estimation in undirected models.

### 3.12 Decision Networks
Finally, we extend our probabilistic models into the realm of action. We construct Decision Networks where utility nodes represent the payoff of researching a specific topic, allowing the system to recommend optimal research strategies based on Expected Utility.

## 4. Related Work
Topic modeling has a rich history, beginning with LSA and advancing to the probabilistic formulation of LDA by Blei et al. (2003). Recent advancements (e.g., BERTopic by Grootendorst, 2022) have shown the power of contextualized embeddings.
In the realm of PGMs, Koller and Friedman (2009) provide the foundational text on Bayesian and Markov Networks. The application of MCMC to LDA was popularized by Griffiths and Steyvers (2004). For dynamic modeling, Rabiner's (1989) tutorial on HMMs and Kalman's (1960) original work on linear filtering remain seminal. Decision Networks, as outlined by Russell and Norvig, bridge inference and rational action.

## 5. Methodology

### 5.1 Preprocessing and BERT Embeddings
Documents are tokenized and stop-words are removed. Simultaneously, Sentence-BERT (SBERT) is used to generate dense semantic embeddings of the documents. These embeddings guide the prior probabilities in our probabilistic models, solving the sparsity issues inherent in standard LDA.

### 5.2 Bayesian Network Formulation of LDA
We define LDA as a Bayesian Network. The generative model is:
1. Draw topic proportions \( \theta_d \sim \text{Dir}(\alpha) \) for each document \( d \).
2. Draw word distributions \( \phi_k \sim \text{Dir}(\beta) \) for each topic \( k \).
3. For each word \( n \) in document \( d \):
   - Draw a topic assignment \( z_{d,n} \sim \text{Multinomial}(\theta_d) \).
   - Draw the word \( w_{d,n} \sim \text{Multinomial}(\phi_{z_{d,n}}) \).

### 5.3 Exact Inference via Variable Elimination
We theoretically outline Variable Elimination for a simplified, miniature document to demonstrate the process of summing out hidden variables. However, we show that for realistic vocabulary and document sizes, the time complexity renders exact inference impossible.

### 5.4 Approximate Inference and MCMC (Gibbs Sampling)
We utilize a Collapsed Gibbs Sampler to approximate the posterior. The algorithm integrates out \( \theta \) and \( \phi \) and samples the topic assignment \( z_i \) for the \( i \)-th word conditional on all other assignments \( \mathbf{z}_{-i} \):
\[ P(z_i = k | \mathbf{z}_{-i}, \mathbf{w}) \propto \frac{n_{k,-i}^{(w_i)} + \beta}{n_{k,-i}^{(\cdot)} + V\beta} \cdot (n_{d,-i}^{(k)} + \alpha) \]
where \( n \) represents the respective count matrices.

### 5.5 Hidden Markov Model for Topic Dynamics
To model sequential documents, we define an HMM:
- **Hidden States:** The true underlying topic \( Z_t \).
- **Observations:** The discrete words \( W_t \) in the time-slice.
- **Transition Model:** \( P(Z_t | Z_{t-1}) \) representing topic shifts.
We use the Viterbi algorithm to find the most likely sequence of topics over time.

### 5.6 Kalman Filter for Topic Prevalence Smoothing
Let \( x_t \) be the true continuous popularity of a topic at time \( t \), and \( z_t \) be the noisy observation from our document corpus.
- **State Equation:** \( x_t = A x_{t-1} + w_t \)
- **Observation Equation:** \( z_t = H x_t + v_t \)
The Kalman Filter recursively applies prediction and update steps to estimate \( x_t \), effectively denoising the trend.

### 5.7 Markov Random Fields and Message Passing
We construct an MRF where nodes represent topics. An edge exists if topics frequently co-occur in the same documents. The joint distribution is given by a product of clique potentials:
\[ P(Y) = \frac{1}{Z} \prod_{c \in C} \psi_c(Y_c) \]
We use **Belief Propagation** to compute marginals, passing messages between nodes until convergence.

### 5.8 Learning in Markov Networks
We learn the clique potentials \( \psi_c \) by maximizing the log-likelihood of the observed topic co-occurrences derived from the LDA output, using iterative proportional fitting (IPF) for parameter estimation.

### 5.9 Decision Networks for Research Planning
We augment our models with Decision and Utility nodes. Given the smoothed topic prevalence (from the Kalman Filter) and topic synergies (from the MRF), the Decision Network evaluates the **Expected Utility (EU)** of choosing to research a specific topic, guiding rational decision-making under uncertainty.

## 6. Implementation
The project is modularized in Python:

| Module | Description | Syllabus Connection |
|---|---|---|
| `bayesian_network.py` | LDA structure, exact inference stub, Gibbs sampler | Topics 1, 2, 3, 4, 5, 6 |
| `hmm_module.py` | HMM for temporal evolution, Viterbi algorithm | Topic 7 |
| `kalman_filter.py` | Kalman filter for continuous prevalence smoothing | Topic 8 |
| `markov_network.py` | MRF structure, Belief Propagation, Learning potentials | Topics 9, 10, 11 |
| `decision_network.py` | Decision logic, Expected Utility calculation | Topic 12 |

## 7. Experimental Setup
We evaluate the system on standard datasets (e.g., 20 Newsgroups). Topic coherence is measured using Normalized Pointwise Mutual Information (NPMI). The dynamic models are evaluated on temporal metadata associated with the documents.

## 8. Results
### 8.1 MCMC Inference & Coherence
The Hybrid LDA-BERT model, inferred via our Gibbs sampler, achieves an NPMI score significantly higher than the standard LDA baseline. The integration of BERT embeddings effectively accelerates the convergence of the MCMC chains.
### 8.2 Dynamic Topic Tracking
The HMM accurately segments document streams, correctly identifying sudden shifts in thematic focus. The Kalman Filter successfully smooths highly volatile topic frequencies, providing clear, interpretable trend lines for topic popularity over a decade-long dataset.
### 8.3 MRF Co-occurrence & Decision Making
Belief propagation on the MRF efficiently identifies highly correlated topics (e.g., "Hardware" and "Operating Systems"). When fed into the Decision Network, the system correctly identifies research areas with the highest expected utility, balancing current popularity with synergistic potential.

## 9. Discussion
This project successfully demonstrates the vast applicability of probabilistic reasoning. By mapping the 12 syllabus topics directly to practical NLP tasks, we illustrate that while deep learning provides powerful representations, classic PGMs offer unparalleled interpretability, rigor, and theoretical grounding for complex reasoning under uncertainty.

## 10. Conclusion
We have developed a comprehensive topic modeling suite that serves as a practical instantiation of probabilistic reasoning. From Bayesian Network inference via MCMC to dynamic modeling with Kalman Filters and rational action via Decision Networks, the system effectively tackles uncertainty in natural language.

## 11. References
1. Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent dirichlet allocation. JMLR.
2. Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure.
3. Koller, D., & Friedman, N. (2009). Probabilistic graphical models: principles and techniques. MIT press.
4. Griffiths, T. L., & Steyvers, M. (2004). Finding scientific topics. PNAS.
5. Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. Proceedings of the IEEE.
6. Kalman, R. E. (1960). A new approach to linear filtering and prediction problems.
7. Russell, S., & Norvig, P. (2020). Artificial Intelligence: A Modern Approach.
