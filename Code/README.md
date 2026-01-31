## Environment Setup
- **For Python-based Code**
  - *conda env create -f SimNetRLMDA.yml*: To run SimNetRLMDA
  - *conda env create -f MAMFGAT.yml*: To run MAMFGAT
  - *conda env create -f MHXGMDA.yml*: To run MHXGMDA
- **For R-based Code**: To run MHMDA, and visualize disease-miRNA associations
  - *Install R packages: RandomWalkRestartMH, igraph, foreach, doParallel, ROCR, ggplot2, Metrics, hash*

## Simulations
- **Generate Embeddings**
  - *generate_embeddings_for_DiNet.py*: Generate embeddings for diseases from disease similarity.
  - *generate_embeddings_for_miRNet.py*: Generate embeddings for miRNAs from the miRNA similarity network.
 
- **Evaluate**:
  - *evaluate.py*: For various combinations of disease and miRNA similarity networks, embedding sizes, and epochs

- **Predict**:
  - *predict.py*: For prediction of novel disease-miRNA associations

## Summary
  - *summarize.py*: To summarize and create heatmaps for various combinations of disease and miRNA similarity networks, embedding sizes, and epochs
  - *create_Figures.py*: To create Figure 2 and 3
  - *visualize_Disease-miRNA_by_SharedGenes_final.R*: To visualize disease-miRNA associations via shared genes

## Comparison
  - *MHMDA_KFold_Balanced_Final.R*: To compare with MHMDA (https://github.com/hauldhut/MHMDA)
  - *MAMFGAT.zip*: All data and code used to compare with MAMFGAT
  - *MHXGMDA.zip*: All data and code used to compare with MHXGMDA
  


