import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve, precision_recall_curve
from xgboost import XGBClassifier
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
import os
import sys
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

cls_method = "XGB"

# Custom class to redirect print output to both console and file
class Tee:
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout
    
    def write(self, message):
        self.file.write(message)
        self.stdout.write(message)
    
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    
    def close(self):
        self.file.close()

# Step 1: Load embeddings and disease-Enh pairs, filter invalid pairs
def load_data(disease_embeddings_file, Enh_embeddings_file, disease_Enh_file):
    print("Loading embeddings...")
    disease_embeddings_df = pd.read_csv(disease_embeddings_file)
    Enh_embeddings_df = pd.read_csv(Enh_embeddings_file)
    
    valid_diseases = set(disease_embeddings_df[disease_embeddings_df['type'] == 'disease']['node_id'])
    valid_Enhs = set(Enh_embeddings_df[Enh_embeddings_df['type'] == 'Enh']['node_id'])
    
    disease_emb = disease_embeddings_df[disease_embeddings_df['type'] == 'disease'].set_index('node_id')
    Enh_emb = Enh_embeddings_df[Enh_embeddings_df['type'] == 'Enh'].set_index('node_id')
    
    disease_emb_cols = [col for col in disease_embeddings_df.columns if col.startswith('dim_')]
    embedding_size_disease = len(disease_emb_cols)
    Enh_emb_cols = [col for col in Enh_embeddings_df.columns if col.startswith('dim_')]
    embedding_size_Enh = len(Enh_emb_cols)
    
    disease_emb = disease_emb[disease_emb_cols].to_numpy()
    Enh_emb = Enh_emb[Enh_emb_cols].to_numpy()
    
    diseases = list(valid_diseases)
    Enhs = list(valid_Enhs)
    
    print(f"Number of diseases with embeddings: {len(diseases)}")
    print(f"Number of Enhs with embeddings: {len(Enhs)}")
    
    print("Loading positive disease-Enh pairs...")
    disease_Enh_df = pd.read_csv(disease_Enh_file)
    
    positive_pairs = []
    total_pairs = len(disease_Enh_df)
    for _, row in disease_Enh_df.iterrows():
        disease, Enh = row['disease'], row['Enh']
        if disease in valid_diseases and Enh in valid_Enhs:
            positive_pairs.append((disease, Enh))
    
    positive_pairs = set(positive_pairs)
    skipped_pairs = total_pairs - len(positive_pairs)
    print(f"Number of positive pairs loaded: {len(positive_pairs)}")
    print(f"Number of pairs skipped (missing embeddings): {skipped_pairs}")
    
    if not positive_pairs:
        raise ValueError("No valid positive pairs found after filtering. Check embeddings_file and disease_Enh_file.")
    
    return diseases, Enhs, disease_emb, Enh_emb, positive_pairs, embedding_size_disease, embedding_size_Enh

# Step 2: Generate feature vectors and labels with balanced negative sampling
def generate_features_labels(
    diseases,
    Enhs,
    disease_emb,
    Enh_emb,
    positive_pairs,
    embedding_size_disease,
    embedding_size_Enh,
    selNeg_file=None
):
    print("Generating feature vectors and labels with balanced negative sampling...")

    disease_idx = {disease: i for i, disease in enumerate(diseases)}
    Enh_idx = {Enh: i for i, Enh in enumerate(Enhs)}

    positive_pairs_list = list(positive_pairs)
    num_positive = len(positive_pairs_list)
    print(f"Number of positive pairs: {num_positive}")

    # ---------- NEGATIVE PAIRS ----------
    if selNeg_file is not None and os.path.exists(selNeg_file):
        print(f"Loading selected negative pairs from: {selNeg_file}")
        neg_df = pd.read_csv(selNeg_file, sep="\t", header=None, names=["disease", "Enh"])
        selected_negative_pairs = list(zip(neg_df["disease"], neg_df["Enh"]))
    else:
        print("Sampling negative pairs...")
        all_pairs = itertools.product(diseases, Enhs)
        negative_pairs = [p for p in all_pairs if p not in positive_pairs]

        selected_negative_pairs = random.sample(negative_pairs, num_positive)

        if selNeg_file is not None:
            print(f"Saving selected negative pairs to: {selNeg_file}")
            pd.DataFrame(selected_negative_pairs).to_csv(
                selNeg_file, sep="\t", index=False, header=False
            )

    print(f"Number of negative pairs used: {len(selected_negative_pairs)}")

    # ---------- BUILD FEATURES ----------
    features, labels, pairs = [], [], []

    selected_pairs = positive_pairs_list + selected_negative_pairs
    random.shuffle(selected_pairs)

    for disease, Enh in tqdm(selected_pairs, desc="Generating features for pairs"):
        pairs.append((disease, Enh))
        label = 1 if (disease, Enh) in positive_pairs else 0
        labels.append(label)

        disease_vec = disease_emb[disease_idx[disease]]
        Enh_vec = Enh_emb[Enh_idx[Enh]]
        features.append(np.concatenate([disease_vec, Enh_vec]))

    return np.array(features), np.array(labels), pairs


# Step 3: Train and evaluate XGBoost with 10-fold cross-validation
kfold = 5
def evaluate_model(features, labels, base_name, embedding_size, epochs):
    print("Training and evaluating XGBoost model...")
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)
    
    auroc_scores = []
    auprc_scores = []
    f1_scores = []
    accuracy_scores = []
    
    roc_data = []
    pr_data = []
    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)
    
    tprs = []
    precisions = []

    # === Create figures ===
    # Set up plotting style with global font size
    plt.rcParams.update({'font.size': 14})  # Set global font size to 12

    fig_roc, ax_roc = plt.subplots(figsize=(6, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(6, 6))
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels), 1):
        print(f"Processing fold {fold}/{kfold}...")
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        xgb.fit(X_train, y_train)
        y_pred_proba = xgb.predict_proba(X_test)[:, 1]
        
        auroc = roc_auc_score(y_test, y_pred_proba)
        auprc = average_precision_score(y_test, y_pred_proba)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        
        auroc_scores.append(auroc)
        auprc_scores.append(auprc)
        f1_scores.append(f1)
        accuracy_scores.append(accuracy)
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
        roc_data.append(pd.DataFrame({'fold': fold, 'fpr': fpr, 'tpr': tpr}))
        
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        precision_interp = np.interp(mean_recall, recall[::-1], precision[::-1])
        precisions.append(precision_interp)
        pr_data.append(pd.DataFrame({'fold': fold, 'recall': recall, 'precision': precision}))
        
        # === ROC curve ===
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        ax_roc.plot(
            fpr, tpr,
            lw=2,
            label=f"Fold {fold} (AUROC={auroc:.3f})"
        )

        # === PR curve ===
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        ax_pr.plot(
            recall, precision,
            lw=2,
            label=f"Fold {fold} (AUPRC={auprc:.3f})"
        )

        print(f"Fold {fold} - AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
    
    auroc_mean, auroc_std = np.mean(auroc_scores), np.std(auroc_scores)
    auprc_mean, auprc_std = np.mean(auprc_scores), np.std(auprc_scores)
    f1_mean, f1_std = np.mean(f1_scores), np.std(f1_scores)
    accuracy_mean, accuracy_std = np.mean(accuracy_scores), np.std(accuracy_scores)
    
    print("\nFinal Results:")
    print(f"AUROC: {auroc_mean:.4f} ± {auroc_std:.4f}")
    print(f"AUPRC: {auprc_mean:.4f} ± {auprc_std:.4f}")
    print(f"F1-score: {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"Accuracy: {accuracy_mean:.4f} ± {accuracy_std:.4f}")
    
    # === Final ROC plot ===
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right", fontsize=12)
    # ax_roc.set_title(f"Embedding Size: {embedding_size}, Epochs: {epochs}")
    ax_roc.set_title(f"AUROC: {auroc_mean:.3f} ± {auroc_std:.3f}")

    roc_path = f"../Results/Detail/{base_name}_ROC.png"
    fig_roc.savefig(roc_path, dpi=600, bbox_inches="tight")
    plt.close(fig_roc)

    # === Final PR plot ===
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="lower left", fontsize=12)
    # ax_pr.set_title(f"Embedding Size: {embedding_size}, Epochs: {epochs}")
    ax_pr.set_title(f"AUPRC: {auprc_mean:.3f} ± {auprc_std:.3f}")

    pr_path = f"../Results/Detail/{base_name}_PR.png"
    fig_pr.savefig(pr_path, dpi=600, bbox_inches="tight")
    plt.close(fig_pr)

    print(f"ROC curves saved to {roc_path}")
    print(f"PR curves saved to {pr_path}")
    
    ####################
    
    return auroc_mean, auroc_std, auprc_mean, auprc_std, f1_mean, f1_std, accuracy_mean, accuracy_std


# Main function
def main():
    disease_emb_files = []

    emb_method = "gat" #gat|gcn
    print(f"emb_method: {emb_method}")

    DiSimNet = "DODisSimNet"
    
    # Enh_net = "EnhNetG_EnhEmbS_initVec_DNABERT-2-mean" #ONLY for gat
    Enh_net = "EnhNetG"

    print(f"Enh_net: {Enh_net}")

    results = []
    embedding_size_list = [128, 256, 512]
    epochs_list = [100, 200, 400]
    for epochs in epochs_list:    
        for embedding_size in embedding_size_list:
            print(f"embedding_size: {embedding_size}, epochs: {epochs}")
            disease_emb_file = f"{DiSimNet}_{emb_method}_d_{embedding_size}_e_{epochs}"

            Enh_emb_file = f"{Enh_net}_{emb_method}_d_{embedding_size}_e_{epochs}"
            
            # #For initVec
            # Enh_emb_file = f"{Enh_net}{embedding_size}_{emb_method}_d_{embedding_size}_e_{epochs}"

            disease_embeddings_file = f"../Results/Embeddings/{disease_emb_file}.csv"
            Enh_embeddings_file = f"../Results/Embeddings/{Enh_emb_file}.csv"

            if DiSimNet == f"{DiSimNet}":
                disease_Enh_file = os.path.expanduser(f"../Data/EDRelation.csv")
            
            base_name_disease = os.path.splitext(os.path.basename(disease_embeddings_file))[0]
            base_name_Enh = os.path.splitext(os.path.basename(Enh_embeddings_file))[0]
            
            base_name = base_name_disease + "_" + base_name_Enh + f"_Balanced_{cls_method}"

            print(f"\nProcessing pair:")
            print(f"disease_embeddings_file: {disease_embeddings_file}")
            print(f"Enh_embeddings_file: {Enh_embeddings_file}")
            print(f"disease-Enh file: {disease_Enh_file}")
            
            output_file = f'../Results/Detail/{base_name}_output_MatchedEE.txt'
            selNeg_file = f'../Results/Detail/{base_name}_selNeg_MatchedEE.txt'
            tee = Tee(output_file)
            sys.stdout = tee
            
            try:
                diseases, Enhs, disease_emb, Enh_emb, positive_pairs, disease_emb_size, Enh_emb_size = load_data(
                    disease_embeddings_file, Enh_embeddings_file, disease_Enh_file
                )
                
                features, labels, pairs = generate_features_labels(
                    diseases, Enhs, disease_emb, Enh_emb, positive_pairs, disease_emb_size, Enh_emb_size, selNeg_file
                )
                
                # from sklearn.manifold import TSNE
                # import seaborn as sns
                # tsne = TSNE(n_components=2, random_state=42)
                # embeddings_2d = tsne.fit_transform(features)
                # sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=labels)
                # plt.savefig(f'{base_name}_tsne.png')
                # plt.close()
                
                auroc_mean, auroc_std, auprc_mean, auprc_std, f1_mean, f1_std, accuracy_mean, accuracy_std = evaluate_model(features, labels, base_name, embedding_size, epochs)
                
                # Collect results
                results.append({
                    'disease_emb_file': disease_emb_file,
                    'Enh_emb_file': Enh_emb_file,
                    'auroc_mean': auroc_mean,
                    'auroc_std': auroc_std,
                    'auprc_mean': auprc_mean,
                    'auprc_std': auprc_std,
                    'f1_mean': f1_mean,
                    'f1_std': f1_std,
                    'accuracy_mean': accuracy_mean,
                    'accuracy_std': accuracy_std
                })
            
            except Exception as e:
                print(f"Error processing: {str(e)}")
            
            finally:
                sys.stdout = tee.stdout
                tee.close()
            
    # Save summary results to a CSV file
    # emb_str = "_".join(emb_methods)
    summary_df = pd.DataFrame(results)
    summary_file = f"../Results/Perf_sumstats_{DiSimNet}_{Enh_net}_{emb_method}_{cls_method}_MatchedEE.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary results saved to {summary_file}")

# Execute
if __name__ == "__main__":
    main()
