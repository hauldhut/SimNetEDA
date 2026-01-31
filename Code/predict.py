import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from tqdm import tqdm
import itertools
import os
import sys
import random
import heapq

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

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

    diseases = sorted(valid_diseases)
    Enhs = sorted(valid_Enhs)

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

# def logit(p, eps=1e-8):
#     p = np.clip(p, eps, 1 - eps)
#     return np.log(p / (1 - p))

# Step 3: Train XGB model and predict novel associations (memory safe)


# Step 3: Train XGB model and predict novel associations (memory safe)
def train_and_predict(diseases, Enhs, disease_emb, Enh_emb, positive_pairs,
                      embedding_size_disease, embedding_size_Enh,
                      disease_embeddings_file, Enh_embeddings_file,
                      base_name, topK=100, batch_size=500_000):

    print("Training XGB model on all labeled data...")

    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist"   # faster & deterministic on CPU
    )

    selNeg_file = f'../Results/Detail/{base_name}_selNeg_MatchedEE.txt'

    # Generate training data
    features, labels, pairs = generate_features_labels(
        diseases, Enhs, disease_emb, Enh_emb,
        positive_pairs, embedding_size_disease,
        embedding_size_Enh, selNeg_file
    )

    # Train model
    xgb.fit(features, labels)

    # Evaluate model on training data
    y_pred_proba = xgb.predict_proba(features)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auroc = roc_auc_score(labels, y_pred_proba)
    auprc = average_precision_score(labels, y_pred_proba)
    f1 = f1_score(labels, y_pred)
    accuracy = accuracy_score(labels, y_pred)

    print("\nTraining Data Performance:")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    metrics_df = pd.DataFrame([{
        'disease_emb_file': disease_embeddings_file,
        'Enh_emb_file': Enh_embeddings_file,
        'auroc_mean': auroc,
        'auroc_std': 0.0,
        'auprc_mean': auprc,
        'auprc_std': 0.0,
        'f1_mean': f1,
        'f1_std': 0.0,
        'accuracy_mean': accuracy,
        'accuracy_std': 0.0
    }])

    metrics_csv = f'../Prediction/{base_name}_top_{topK}_model_metrics.csv'
    metrics_df.to_csv(metrics_csv, index=False, float_format="%.4f")
    print(f"Saved model metrics to {metrics_csv}")

    # --- Predict for ALL novel pairs in batches ---
    print("Generating predictions for novel disease-Enh associations (batched)...")

    disease_idx = {disease: i for i, disease in enumerate(diseases)}
    Enh_idx = {Enh: i for i, Enh in enumerate(Enhs)}

    all_pairs_iter = itertools.product(diseases, Enhs)
    novel_pairs_iter = (pair for pair in all_pairs_iter if pair not in positive_pairs)

    top_k_heap = []
    batch_pairs = []
    batch_features = []

    total_novel = len(diseases) * len(Enhs) - len(positive_pairs)

    for pair in tqdm(novel_pairs_iter, total=total_novel, desc="Processing batches"):
        disease, Enh = pair
        disease_vec = disease_emb[disease_idx[disease]]
        Enh_vec = Enh_emb[Enh_idx[Enh]]
        batch_features.append(np.concatenate([disease_vec, Enh_vec]))
        batch_pairs.append(pair)

        if len(batch_features) >= batch_size:
            probs = xgb.predict_proba(np.array(batch_features))[:, 1]

            for p, prob in zip(batch_pairs, probs):
                if len(top_k_heap) < topK:
                    heapq.heappush(top_k_heap, (prob, p))
                else:
                    heapq.heappushpop(top_k_heap, (prob, p))

            batch_pairs.clear()
            batch_features.clear()

    # Last batch
    if batch_features:
        probs = xgb.predict_proba(np.array(batch_features))[:, 1]
        for p, prob in zip(batch_pairs, probs):
            if len(top_k_heap) < topK:
                heapq.heappush(top_k_heap, (prob, p))
            else:
                heapq.heappushpop(top_k_heap, (prob, p))

    # Sort results
    top_k_heap.sort(reverse=True, key=lambda x: x[0])
    top_k_probs, top_k_pairs = zip(*top_k_heap)

    predictions_df = pd.DataFrame({
        'disease': [pair[0] for pair in top_k_pairs],
        'Enh': [pair[1] for pair in top_k_pairs],
        'predicted_probability': top_k_probs
    })

    predictions_csv = f'../Prediction/{base_name}_top_{topK}_predictions.csv'
    predictions_df.to_csv(predictions_csv, index=False, float_format="%.4f")
    print(f"Saved top {topK} predictions to {predictions_csv}")

    return auroc, 0.0, auprc, 0.0, f1, 0.0, accuracy, 0.0


# Main function
def main():
    cls_method = "XGB"

    Enh_net = "EnhNetG_EnhEmbS_initVec_DNABERT-2-meam"
    # Enh_net = "EnhNetG"
    emb_method = "gat"
    emb_size = 128
    epoch = 100

    topK = 1153472 #max = 2152*536 = 1153472 - 550*2(Number of Pos and Neg pairs) = 1152372
    print(f"topK: {topK}")
    
    disease_embeddings_file = f"../Results/Embeddings/DODisSimNet_{emb_method}_d_{emb_size}_e_{epoch}.csv"
    Enh_emb_file = f"{Enh_net}_{emb_method}_d_{emb_size}_e_{epoch}"
    Enh_embeddings_file = f"../Results/Embeddings/{Enh_emb_file}.csv"
    
    print(f"disease_embeddings_file: {disease_embeddings_file}")
    print(f"Enh_embeddings_file: {Enh_embeddings_file}")

    disease_Enh_file = os.path.expanduser("../Data/EDRelation.csv")

    base_name_disease = os.path.splitext(os.path.basename(disease_embeddings_file))[0]
    base_name_Enh = os.path.splitext(os.path.basename(Enh_embeddings_file))[0]
    base_name = base_name_disease + "_" + base_name_Enh + f"_Balanced_{cls_method}"

    output_file = f'../Prediction/{base_name}_top_{topK}_output.txt'
    tee = Tee(output_file)
    sys.stdout = tee

    try:
        diseases, Enhs, disease_emb, Enh_emb, positive_pairs, disease_emb_size, Enh_emb_size = load_data(
            disease_embeddings_file, Enh_embeddings_file, disease_Enh_file
        )

        train_and_predict(
            diseases, Enhs, disease_emb, Enh_emb, positive_pairs, disease_emb_size, Enh_emb_size,
            disease_embeddings_file, Enh_embeddings_file, base_name, topK, batch_size=500_000
        )

    except Exception as e:
        print(f"Error processing: {str(e)}")

    finally:
        sys.stdout = tee.stdout
        tee.close()

if __name__ == "__main__":
    main()
