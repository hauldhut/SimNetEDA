import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

emb_met = "gcn" #gat|gcn
cls_method = "XGB" #MLP|XGB|RF
print(f"cls_method: {cls_method}")

# Define filtering criteria
Dis_nets = ["DODisSimNet"]

Enh_nets = ["EnhNetG_EnhEmbS_initVec_DNABERT-2-mean", #Only for gat
            "EnhNetG"
            ]
Enh_nets = ["EnhNetG"]


disease_emb_methods = [emb_met] 
emb_methods = [emb_met]

embedding_sizes = [128, 256, 512]
epochs = [100, 200, 400]



# input_file = f"../Results/Perf_sumstats_{Dis_nets[0]}_{Enh_nets[0]}_{cls_method}.csv"
input_file = f"../Results/Perf_sumstats_{Dis_nets[0]}_{Enh_nets[0]}_{disease_emb_methods[0]}_{cls_method}_MatchedEE.csv"
print(f"input_file: {input_file}")

# Generate Enh_emb_methods
Enh_emb_methods = []
for Enh_net in Enh_nets:
    for emb_method in emb_methods:
        for emb_size in embedding_sizes:
            for epoch in epochs:
                Enh_emb_methods.append(f"{Enh_net}_{emb_method}_d_{emb_size}_e_{epoch}")

# Metrics to analyze
metrics = ["auroc", "auprc", "f1", "accuracy"]
dict_metrics = dict()
dict_metrics = {"auroc":"AUROC",
                    "auprc":"AUPRC",
                    "f1":"F1",
                    "accuracy":"Accuracy"
}

# Read the merged CSV file

if not os.path.exists(input_file):
    print(f"Error: File {input_file} not found")
    exit()

df = pd.read_csv(input_file)

# Filter rows based on disease_emb_file and Enh_emb_file
def filter_rows(row):
    disease_match = any(Dis_net in row["disease_emb_file"] for Dis_net in Dis_nets) and \
                 any(method in row["disease_emb_file"] for method in disease_emb_methods) and \
                 any(str(size) in row["disease_emb_file"] for size in embedding_sizes) and \
                 any(str(epoch) in row["disease_emb_file"] for epoch in epochs)
    Enh_match = any(method in row["Enh_emb_file"] for method in Enh_emb_methods)
    return disease_match and Enh_match

filtered_df = df[df.apply(filter_rows, axis=1)].copy()

if filtered_df.empty:
    print("No data matches the filtering criteria")
    exit()

# Extract relevant components from disease_emb_file
def extract_disease_components(disease_emb_file):
    Dis_net = next((d for d in Dis_nets if d in disease_emb_file), None)
    method = next((m for m in disease_emb_methods if m in disease_emb_file), None)
    size = next((str(s) for s in embedding_sizes if str(s) in disease_emb_file), None)
    epoch = next((str(e) for e in epochs if str(e) in disease_emb_file), None)
    return Dis_net, method, size, epoch

# Extract relevant components from Enh_emb_file
def extract_Enh_components(Enh_emb_file):
    Enh_net = next((net for net in Enh_nets if net in Enh_emb_file), None)
    emb_method = next((m for m in emb_methods if m in Enh_emb_file), None)
    return Enh_net, emb_method

filtered_df[["Dis_net", "dis_emb_met", "embedding_size", "epochs"]] = filtered_df["disease_emb_file"].apply(
    lambda x: pd.Series(extract_disease_components(x))
)
filtered_df[["Enh_net", "Enh_emb_method"]] = filtered_df["Enh_emb_file"].apply(
    lambda x: pd.Series(extract_Enh_components(x))
)

# Compute average values for CSV output
grouped = filtered_df.groupby(["Dis_net", "dis_emb_met", "Enh_net", "Enh_emb_method"]).agg({
    "auroc_mean": "mean",
    "auroc_std": "mean",
    "auprc_mean": "mean",
    "auprc_std": "mean",
    "f1_mean": "mean",
    "f1_std": "mean",
    "accuracy_mean": "mean",
    "accuracy_std": "mean"
}).reset_index()

# Save to CSV
output_csv = os.path.join("performance_heatmaps_size_epochs_filter", f"{Dis_nets[0]}_{Enh_net}_{disease_emb_methods[0]}_{cls_method}_summary_avg_MatchedEE.csv")
grouped.to_csv(output_csv, index=False, float_format="%.3f")
print(f"Average metrics saved to {output_csv}")

# Set up plotting style
# sns.set(style="whitegrid")

# Set up plotting style with global font size
plt.rcParams.update({'font.size': 14})  # Set global font size to 12

# Create a directory for saving plots
output_dir = "performance_heatmaps_size_epochs_filter"
os.makedirs(output_dir, exist_ok=True)

# Plot heatmaps for each Dis_net and disease method
for Dis_net in Dis_nets:
    for dis_emb_met in disease_emb_methods:
        subset_df = filtered_df[
            (filtered_df["Dis_net"] == Dis_net) & 
            (filtered_df["dis_emb_met"] == dis_emb_met)
        ]
        if subset_df.empty:
            print(f"No data for {Dis_net} with {dis_emb_met}")
            continue

        for metric in metrics:
            # Create pivot table for heatmap
            pivot = subset_df.pivot_table(
                values=f"{metric}_mean",
                index="embedding_size",
                columns="epochs",
                aggfunc="mean"
            )
            
            # Ensure all embedding sizes and epochs are present
            pivot = pivot.reindex(
                index=[str(size) for size in sorted(embedding_sizes, reverse=True)],  # 512 to 128
                columns=[str(epoch) for epoch in sorted(epochs)],
                fill_value=np.nan
            )
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                pivot, 
                annot=False,  # Disable default annotation
                cmap="viridis", 
                vmin=0.5, 
                vmax=1.0, 
                cbar_kws={"label": f"{metric.upper()} Mean"},
                annot_kws={"size": 14}  # Larger font size for annotations
            )
            
            # Customize annotations with only annot_text and larger font
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    mean_val = pivot.iloc[i, j]
                    if pd.notna(mean_val):
                        std_val = subset_df[
                            (subset_df["embedding_size"] == pivot.index[i]) & 
                            (subset_df["epochs"] == pivot.columns[j])
                        ][f"{metric}_std"].mean()
                        annot_text = f"{mean_val:.3f}\n(\u00B1{std_val:.3f})"
                        plt.text(j + 0.5, i + 0.5, annot_text, 
                                ha="center", va="center", color="black", fontsize=14)

            plt.title(f"{dict_metrics[metric]}")
            plt.xlabel("Epochs")
            plt.ylabel("Embedding Size")
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, f"{Dis_net}_{Enh_net}_{dis_emb_met}_{cls_method}_{metric}_heatmap_MatchedEE.png"), dpi=600, bbox_inches="tight")
            plt.close()

print(f"Heatmaps saved in {output_dir} directory")