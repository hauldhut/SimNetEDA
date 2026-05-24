import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import networkx as nx
import os

# =====================================================
# Config
# =====================================================
emb_method = "gat"
torch.manual_seed(42)
np.random.seed(42)

# =====================================================
# Load initial enhancer feature vectors
# =====================================================
def load_enh_init_vectors(init_vec_file):
    """
    TSV format:
    col0 = enhancer ID
    col1..n = initial feature vector
    """
    df = pd.read_csv(init_vec_file, sep="\t", header=None)

    enh_ids = df.iloc[:, 0].astype(str).tolist()
    feats = df.iloc[:, 1:].astype(float).values

    assert not np.isnan(feats).any(), "NaN detected in enhancer init vectors"

    # L2 normalization (critical for stability)
    feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)

    print(f"Loaded enhancer features (normalized): {feats.shape}")
    return dict(zip(enh_ids, feats))


# =====================================================
# Load enhancer similarity network
# =====================================================
def load_Enh_network(Enh_sim_file):
    print(f"Loading Enh similarity network: {Enh_sim_file}")

    df = pd.read_csv(
        Enh_sim_file,
        sep="\t",
        header=None,
        names=["u", "sim", "v"]
    )

    G = nx.Graph()

    removed = 0
    for _, r in df.iterrows():
        w = float(r["sim"])
        if not np.isfinite(w) or w <= 0:
            removed += 1
            continue
        G.add_edge(str(r["u"]), str(r["v"]), weight=w)

    for n in G.nodes():
        G.nodes[n]["type"] = "Enh"

    if removed > 0:
        print(f"Removed {removed} invalid edges")

    return G, list(G.nodes())


# =====================================================
# Convert to PyG Data (USING INIT VECTORS)
# =====================================================
def graph_to_pyg_data(G, enh_feat_dict):
    print(f"Graph: {len(G.nodes())} nodes, {len(G.edges())} edges")

    node_to_idx = {n: i for i, n in enumerate(G.nodes())}

    edge_index, edge_weight = [], []
    for u, v, d in G.edges(data=True):
        w = float(d["weight"])
        edge_index += [[node_to_idx[u], node_to_idx[v]],
                       [node_to_idx[v], node_to_idx[u]]]
        edge_weight += [w, w]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    edge_weight /= (edge_weight.max() + 1e-8)

    feat_dim = next(iter(enh_feat_dict.values())).shape[0]
    x = torch.zeros((len(G.nodes()), feat_dim), dtype=torch.float)

    for n, idx in node_to_idx.items():
        if n not in enh_feat_dict:
            raise ValueError(f"Missing init vector for enhancer {n}")
        x[idx] = torch.from_numpy(enh_feat_dict[n])

    assert torch.isfinite(x).all(), "NaN in node features"

    data = Data(x=x, edge_index=edge_index)
    data.edge_weight = edge_weight

    print(f"Node feature matrix: {x.shape}")
    return data, node_to_idx


# =====================================================
# Weighted GAT layer
# =====================================================
class WeightedGATConv(GATConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, add_self_loops=False, **kwargs)

    def forward(self, x, edge_index, edge_weight):
        return super().forward(
            x, edge_index,
            edge_attr=edge_weight.view(-1, 1)
        )


# =====================================================
# GAT model
# =====================================================
class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=4, dropout=0.5):
        super().__init__()
        self.g1 = WeightedGATConv(
            in_dim, hidden_dim,
            heads=heads, concat=True,
            dropout=dropout, edge_dim=1
        )
        self.g2 = WeightedGATConv(
            hidden_dim * heads, out_dim,
            heads=1, concat=False,
            dropout=dropout, edge_dim=1
        )
        self.dropout = dropout

    def forward(self, d):
        x = F.elu(self.g1(d.x, d.edge_index, d.edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.g2(x, d.edge_index, d.edge_weight)


# =====================================================
# Train GAT (unsupervised)
# =====================================================
def train_gat(data, emb_dim, epochs, lr=0.001):
    model = GAT(data.x.size(1), 64, emb_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    pos_edges = data.edge_index
    model.train()

    for ep in tqdm(range(epochs), desc="Training GAT"):
        opt.zero_grad()
        z = model(data)

        # Positive edges
        pos_sim = F.cosine_similarity(
            z[pos_edges[0]], z[pos_edges[1]]
        )

        # Negative sampling
        neg_u = torch.randint(0, data.num_nodes, (pos_edges.size(1),))
        neg_v = torch.randint(0, data.num_nodes, (pos_edges.size(1),))
        neg_sim = F.cosine_similarity(z[neg_u], z[neg_v])

        loss = (
            -torch.log(torch.sigmoid(pos_sim) + 1e-8).mean()
            -torch.log(1 - torch.sigmoid(neg_sim) + 1e-8).mean()
        )

        if torch.isnan(loss):
            raise RuntimeError("NaN loss detected")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (ep + 1) % 20 == 0:
            print(f"Epoch {ep+1}/{epochs} | Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        return model(data).cpu().numpy()


# =====================================================
# Save embeddings
# =====================================================
def save_embeddings(emb, nodes, node_to_idx, G, out_file):
    rows = []
    for n in nodes:
        rows.append(
            [n, G.nodes[n]["type"]] +
            emb[node_to_idx[n]].tolist()
        )

    dim = emb.shape[1]
    cols = ["node_id", "type"] + [f"dim_{i+1}" for i in range(dim)]
    pd.DataFrame(rows, columns=cols).to_csv(out_file, index=False)
    print(f"Saved embeddings → {out_file}")


# =====================================================
# Main
# =====================================================
def main(emb_dim, epochs, net_file, init_vec_file):
    enh_feat_dict = load_enh_init_vectors(init_vec_file)
    G, nodes = load_Enh_network(net_file)

    missing = set(nodes) - set(enh_feat_dict.keys())
    if missing:
        raise ValueError(f"Missing init vectors for {len(missing)} enhancers")

    data, node_to_idx = graph_to_pyg_data(G, enh_feat_dict)
    emb = train_gat(data, emb_dim, epochs)

    base = os.path.splitext(os.path.basename(net_file))[0]
    base_initVec = os.path.splitext(os.path.basename(init_vec_file))[0]
    out = f"../Results/Embeddings/{base}_{base_initVec}_{emb_method}_d_{emb_dim}_e_{epochs}.csv"
    save_embeddings(emb, nodes, node_to_idx, G, out)


# =====================================================
# Run
# =====================================================
if __name__ == "__main__":
    Enh_sim_file = "../Data/EnhNetG.txt"

    print(f"Enh_sim_file: {Enh_sim_file}")

    embedding_size_list = [128, 256, 512]
    # embedding_size_list = [512]
    epochs_list = [100, 200, 400]
    # epochs_list = [100]
    for epochs in epochs_list:    
        for embedding_size in embedding_size_list:
            init_vec_file = f"../Data/EnhEmbS_initVec_DNABERT-2-max_PCA{embedding_size}.tsv"
            print(f"init_vec_file: {init_vec_file}")
            print(f"\nembedding_size={embedding_size}, epochs={epochs}")
            main(embedding_size, epochs, Enh_sim_file, init_vec_file)
