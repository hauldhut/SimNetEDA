import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import os
import csv
from sklearn.decomposition import PCA


# Set model name
mn = "DNABERT-2"
pooling = "mean"#mean/max
if mn=="DNABERT-2":#original_dim = 768
    model_name = "zhihan1996/DNABERT-2-117M" #dim = 768  
elif mn=="DNABERT-S":#original_dim = 1024
    model_name = "zhihan1996/DNABERT-S" #dim = 1024
elif mn=="DNA_bert_6":#orignal_dim = 768
    model_name = "zhihan1996/DNA_bert_6" #need token | DNABERT (v1) pretrained using 6-mer tokenization  

print(f"{mn}-{pooling}: {model_name}")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
model.eval()  # Set to evaluation mode
model.to('cpu')  # Ensure CPU-only mode for macOS

# Read Enh sequences
import pandas as pd
from Bio import SeqIO

# Input FASTA file
fasta_file = "../Data/enh2disease-1.0.2.txt.bed.fasta"

# Read FASTA and collect records
records = []
for record in SeqIO.parse(fasta_file, "fasta"):
    records.append({
        "SeqID": record.id,
        "Sequence": str(record.seq)
    })

# Create DataFrame
dfSeq = pd.DataFrame(records)

# Show basic info
print(dfSeq.head())
print(dfSeq.shape)

seqs = dfSeq['Sequence'].tolist()

# Generate embeddings
def get_seq_embedding(seq):
    inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True, max_length=512).to('cpu')
    with torch.no_grad():  # Disable gradients for inference
        hidden_states = model(**inputs)[0]  # [1, sequence_length, 768]
    
    # Mean pooling
    embedding_mean = torch.mean(hidden_states[0], dim=0).detach().numpy()
    
    # Max pooling
    embedding_max = torch.max(hidden_states[0], dim=0)[0].detach().numpy()
    
    return embedding_mean, embedding_max

# Generate embeddings for all sequences
embeddings_mean = []
embeddings_max = []
for seq in seqs:
    mean_emb, max_emb = get_seq_embedding(seq)
    embeddings_mean.append(mean_emb)
    embeddings_max.append(max_emb)

# Convert to NumPy arrays
embeddings_mean = np.array(embeddings_mean)
embeddings_max = np.array(embeddings_max)

# Use mean embeddings for similarity (or switch to embeddings_max as needed)
if pooling == "mean":
    embeddings = embeddings_mean
else:
    embeddings = embeddings_max

original_dim = embeddings.shape[1]


print(f"Original embedding dimension: {original_dim}")
# PCA target dimensions
pca_dims = [128, 256, 512]

for target_dim in pca_dims:
    if target_dim >= original_dim:
        print(f"Skip PCA-{target_dim}: target_dim >= original_dim")
        continue

    print(f"Running PCA to reduce to {target_dim} dimensions...")

    pca = PCA(n_components=target_dim, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)

    print(f"Explained variance ratio (sum) for PCA-{target_dim}: "
          f"{np.sum(pca.explained_variance_ratio_):.4f}")

    out_file = f"../Data/EnhEmbS_initVec_{mn}-{pooling}_PCA{target_dim}.tsv"
    print(f"Saving PCA-{target_dim} embeddings to: {out_file}")

    with open(out_file, "w", newline='') as f:
        writer = csv.writer(f, delimiter="\t")
        for i, emb in enumerate(embeddings_pca):
            row = [dfSeq.iloc[i]["SeqID"]] + [f"{x:.6f}" for x in emb]
            writer.writerow(row)
