## Environment Setup
- **For Python-based Code**
  - *conda env create -f SimNetEDA.yml*: To run SimNetEDA

- **For R-based Code**: To run RWDisEnh+
  - *Install R packages: RandomWalkRestartMH, igraph, ROCR, Metrics, hash*

## Simulations
- **Generate Embeddings**
  - *generate_embeddings_for_DiNet.py*: Generate embeddings for diseases from disease similarity.
  - *generate_embeddings_for_EnhNet.py*: Generate embeddings for enhancers from the enhancer networks.
 
- **Evaluate**:
  - *evaluate.py*: For various combinations of disease and enhancer embeddings, embedding sizes, and epochs

- **Predict**:
  - *predict.py*: For prediction of novel disease-enhancer associations

## Summary
  - *summarize.py*: To summarize and create heatmaps for various combinations of disease and enhancer embeddings, embedding sizes, and epochs

## Comparison
  - *RWDisEnhPlus_KFold_Final_Balanced.R*: To compare with RWDisEnhPlus  (https://github.com/hauldhut/RWDisEnhPlus)
  


