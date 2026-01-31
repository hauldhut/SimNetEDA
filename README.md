# SimNetEDA

Identifying disease-associated enhancers is critical for understanding the regulatory mechanisms underlying complex diseases, yet remains challenging due to the sparse and heterogeneous nature of available data. In this study, we propose SimNetRLEDA, a similarity network–based representation learning framework for predicting disease–enhancer associations. SimNetRLEDA integrates enhancer and disease similarity networks with graph attention–based encoders to learn informative representations while avoiding information leakage from known associations. The learned embeddings are subsequently used by an XGBoost classifier to infer potential disease–enhancer links. Extensive experiments demonstrate that SimNetRLEDA achieves robust and superior performance across multiple evaluation metrics, outperforming alternative network encoders and classifiers. Ablation studies confirm the effectiveness of each model component, including multi-head attention, dropout regularization, similarity-driven embeddings, and gradient-boosted classification. Furthermore, biological validation shows that a subset of top-ranked predicted enhancers is supported by genome-wide association studies, while additional predictions without direct GWAS support are enriched in disease-relevant pathways, highlighting their potential functional relevance. Together, these results indicate that SimNetRLEDA provides an effective and biologically meaningful approach for prioritizing disease-associated enhancers and offers a valuable tool for regulatory genomics and disease mechanism studies.

![SimNetEDA](https://github.com/hauldhut/SimNetEDA/blob/main/Figure1_SimNetEDA_v4.png)

## Repo structure
- **Data**: Contains all data 
- **Code**: Contains all source code to reproduce all the results
- **Results**: To store simulation results

## How to run
- Download the repo
- Follow instructions (README.md) in the folder **Code** to run
