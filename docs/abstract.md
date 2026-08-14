# Original project abstract (22AIE301)

Source: Probability Reasoning Abstract (uploaded).

## Title

Probabilistic Topic Modeling for Research Trend Analysis: A Hybrid LDA–LLM Approach

## Abstract

Understanding how research interest evolves within a scientific domain is critical for researchers, funding bodies, and institutions seeking to identify emerging directions and avoid redundant work. Traditional literature review is manual, time-consuming, and does not scale to the thousands of papers published annually. This project proposes a hybrid probabilistic topic modeling framework that combines Latent Dirichlet Allocation (LDA), a generative probabilistic graphical model, with modern LLM-based semantic embeddings to analyze and visualize thematic trends within a research corpus.

LDA treats each document as a probabilistic mixture of latent topics, and each topic as a probability distribution over words, offering an interpretable generative structure grounded in Bayesian inference — directly reflecting core course concepts of latent variable models and probabilistic inference. However, LDA's reliance on word co-occurrence often produces topics that are statistically valid but semantically incoherent. To address this, the project augments LDA with dense embeddings from a pretrained LLM (e.g., Sentence-BERT or a similar encoder), which capture contextual meaning beyond raw word frequency. These embeddings will be used to (i) improve topic coherence through embedding-guided clustering, and (ii) enable semantic similarity search across papers, so a query paper can be probabilistically mapped to its most relevant existing topics.

The pipeline will be applied to a corpus of research papers on a domain of interest. The system will output: (a) a set of interpretable latent topics with associated probability distributions over terms, (b) a temporal trend graph showing how topic prevalence shifts across publication years, and (c) a document-to-topic probability assignment enabling users to explore corpus structure interactively.

## Objectives

Expected outcome: a working prototype demonstrating that combining classical probabilistic generative models with modern LLM embeddings yields more interpretable and coherent topic discovery than either approach alone.

Pipeline:

1. Implement a baseline LDA model and evaluate topic coherence using standard metrics (e.g., C_v).
2. Integrate LLM embeddings to refine and re-rank topics, comparing coherence and interpretability against the baseline.
3. Build a temporal analysis module to visualize how identified topics trend over time.
4. Deploy an interactive dashboard where users can query a topic or paper and receive probabilistically ranked related work.

## Professor feedback addressed

Augmenting LDA with semantic embeddings should significantly improve the coherence of the identified topics in large scientific corpora. **Calculating a Coherence Score to quantitatively compare hybrid results against a baseline LDA model.**
