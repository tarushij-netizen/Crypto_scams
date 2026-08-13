"""
gnn_model.py
============
Graph Neural Network for detecting scam wallets from transaction graphs.

Architecture: GraphSAGE (inductive, works on unseen wallets at inference time)
  - 2 GraphSAGE layers with ReLU + Dropout
  - Binary output: scam (1) vs legit (0)

Pipeline:
  1. Load wallet_features.csv + scam_transactions.csv
  2. Build a NetworkX graph (nodes = wallets, edges = transactions)
  3. Convert to PyTorch Geometric Data object
  4. Train GraphSAGE on labeled nodes (known scam addresses)
  5. Predict scam probability for ALL wallets in the graph
  6. Save predictions → gnn_predictions.csv

Install:
    pip install torch torch_geometric networkx scikit-learn

For CPU-only PyTorch (lighter, no CUDA):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install torch_geometric
"""

import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ── 1. Build transaction graph ────────────────────────────────────────────────

def build_graph(tx_df: pd.DataFrame, features_df: pd.DataFrame):
    """
    Build a directed graph from transaction data.

    Nodes  = unique wallet addresses
    Edges  = transactions (from_address → to_address)
    Weight = ETH value transferred

    Returns:
        G          : NetworkX DiGraph
        node_index : dict mapping address → integer node id
    """
    print("[GNN] Building transaction graph...")

    G = nx.DiGraph()

    # Add all wallets as nodes
    all_addresses = set(tx_df["from_address"]) | set(tx_df["to_address"])
    G.add_nodes_from(all_addresses)

    # Add edges (one edge per unique from→to pair, aggregated value)
    edge_data = (
        tx_df.groupby(["from_address", "to_address"])
        .agg(total_eth=("value_eth", "sum"), tx_count=("hash", "count"))
        .reset_index()
    )

    for _, row in edge_data.iterrows():
        G.add_edge(
            row["from_address"],
            row["to_address"],
            weight=float(row["total_eth"]),
            tx_count=int(row["tx_count"])
        )

    # Attach node features
    feat_dict = features_df.set_index("address").to_dict("index")
    for node in G.nodes():
        if node in feat_dict:
            G.nodes[node].update(feat_dict[node])
        else:
            # Unknown wallet — zero features
            G.nodes[node].update({
                "total_tx": 0, "total_sent_eth": 0, "total_received_eth": 0,
                "avg_sent_value": 0, "unique_counterparties": 0,
                "error_rate": 0, "is_known_scam": 0
            })

    node_index = {addr: i for i, addr in enumerate(G.nodes())}

    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")
    print(f"  Known scam nodes: {sum(1 for n in G.nodes if G.nodes[n].get('is_known_scam', 0) == 1)}")

    return G, node_index


# ── 2. Convert to PyTorch Geometric format ────────────────────────────────────

FEATURE_COLS = [
    "total_tx", "total_sent_eth", "total_received_eth",
    "avg_sent_value", "unique_counterparties", "error_rate"
]

def graph_to_pyg(G: nx.DiGraph, node_index: dict) -> Data:
    """
    Convert NetworkX graph → PyTorch Geometric Data object.

    Returns:
        data.x       : Node feature matrix  [num_nodes, num_features]
        data.edge_index : Edge index         [2, num_edges]
        data.y       : Node labels          [num_nodes]  (1=scam, 0=legit)
        data.train_mask, data.test_mask : Boolean masks
    """
    num_nodes = len(node_index)

    # Build feature matrix
    feature_matrix = np.zeros((num_nodes, len(FEATURE_COLS)))
    labels         = np.zeros(num_nodes, dtype=int)

    for addr, idx in node_index.items():
        node_data = G.nodes[addr]
        for j, col in enumerate(FEATURE_COLS):
            feature_matrix[idx, j] = node_data.get(col, 0)
        labels[idx] = int(node_data.get("is_known_scam", 0))

    # Normalise features
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)

    # Build edge index
    src, dst = [], []
    for u, v in G.edges():
        src.append(node_index[u])
        dst.append(node_index[v])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x          = torch.tensor(feature_matrix, dtype=torch.float)
    y          = torch.tensor(labels,         dtype=torch.long)

    # Train/test masks — only labeled (known scam) nodes get a mask
    labeled_mask = y == 1
    labeled_idx  = labeled_mask.nonzero(as_tuple=True)[0]

    # 70/30 split on labeled nodes; unlabeled nodes are in neither mask
    n_labeled   = labeled_idx.size(0)
    n_train     = max(1, int(0.7 * n_labeled))
    perm        = torch.randperm(n_labeled)
    train_idx   = labeled_idx[perm[:n_train]]
    test_idx    = labeled_idx[perm[n_train:]]

    train_mask  = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask   = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx]   = True

    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, test_mask=test_mask)

    print(f"\n[GNN] PyG Data object:")
    print(f"  Feature shape : {data.x.shape}")
    print(f"  Labels        : {y.sum().item()} scam / {(y==0).sum().item()} legit")
    print(f"  Train nodes   : {train_mask.sum().item()}")
    print(f"  Test nodes    : {test_mask.sum().item()}")

    return data, scaler


# ── 3. GraphSAGE Model ────────────────────────────────────────────────────────

class GraphSAGE(torch.nn.Module):
    """
    Two-layer GraphSAGE for node-level binary classification.

    GraphSAGE aggregates features from a node's local neighbourhood
    (rather than using the full graph Laplacian like vanilla GCN),
    making it inductive — it can score wallets not seen during training.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv1   = SAGEConv(in_channels, hidden_channels)
        self.conv2   = SAGEConv(hidden_channels, hidden_channels // 2)
        self.linear  = torch.nn.Linear(hidden_channels // 2, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Classification head
        x = self.linear(x)
        return x


# ── 4. Training loop ──────────────────────────────────────────────────────────

def train_gnn(data: Data, epochs: int = 200, lr: float = 0.005) -> tuple:
    """
    Train GraphSAGE on the transaction graph.

    Returns:
        model       : trained GraphSAGE
        train_losses: list of loss per epoch
    """
    model     = GraphSAGE(in_channels=data.x.size(1))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # Class weights to handle imbalance (scam nodes are rare)
    n_legit   = (data.y == 0).sum().float()
    n_scam    = (data.y == 1).sum().float()
    weight    = torch.tensor([1.0, (n_legit / n_scam).item()])
    criterion = torch.nn.CrossEntropyLoss(weight=weight)

    print(f"\n[GNN] Training GraphSAGE for {epochs} epochs...")
    train_losses = []

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if epoch % 50 == 0:
            print(f"  Epoch {epoch:>3} | Loss: {loss.item():.4f}")

    print("  Training complete.")
    return model, train_losses


# ── 5. Evaluation ─────────────────────────────────────────────────────────────

def evaluate_gnn(model: GraphSAGE, data: Data) -> dict:
    """Evaluate on test nodes and return metrics + per-node probabilities."""
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs  = F.softmax(logits, dim=1)[:, 1].numpy()
        preds  = logits.argmax(dim=1).numpy()

    if data.test_mask.sum() > 0:
        y_test    = data.y[data.test_mask].numpy()
        y_pred    = preds[data.test_mask.numpy()]
        y_proba   = probs[data.test_mask.numpy()]

        print("\n[GNN] Test set evaluation:")
        # print(classification_report(y_test, y_pred, target_names=["Legit", "Scam"], zero_division=0))
        # NEW
        print(classification_report(y_test, y_pred, target_names=["Legit", "Scam"], 
                                    labels=[0, 1], zero_division=0))
        try:
            auc = roc_auc_score(y_test, y_proba)
            print(f"  ROC-AUC: {auc:.4f}")
        except Exception:
            auc = None
    else:
        print("  No test nodes available for evaluation.")
        auc = None

    return {"probs": probs, "preds": preds, "auc": auc}


# ── 6. Save predictions ───────────────────────────────────────────────────────

def save_predictions(
    node_index:  dict,
    eval_results: dict,
    output_file: str = "gnn_predictions.csv"
) -> pd.DataFrame:
    """Save per-wallet GNN scam probability to CSV."""
    idx_to_addr = {v: k for k, v in node_index.items()}
    rows = [
        {
            "address":      idx_to_addr[i],
            "gnn_scam_prob": eval_results["probs"][i],
            "gnn_pred_label": int(eval_results["preds"][i])
        }
        for i in range(len(eval_results["probs"]))
    ]
    df = pd.DataFrame(rows).sort_values("gnn_scam_prob", ascending=False)
    df.to_csv(output_file, index=False)
    print(f"\n[GNN] Predictions saved → {output_file}")
    print(df.head(10).to_string(index=False))
    return df


# ── 7. Visualise training loss ────────────────────────────────────────────────

def plot_training_loss(train_losses: list):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, color="steelblue", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("GraphSAGE Training Loss")
    plt.grid(True, linewidth=0.5)
    plt.tight_layout()
    plt.savefig("gnn_training_loss.png", dpi=150)
    plt.show()
    print("Plot saved → gnn_training_loss.png")


# ── 8. Visualise subgraph around top predicted scam wallets ──────────────────

def visualise_scam_subgraph(G: nx.DiGraph, predictions_df: pd.DataFrame, top_n: int = 30):
    """
    Plot the transaction subgraph for the top-N highest-risk wallets.
    Red = predicted scam, grey = connected legit wallets.
    """
    top_scam = set(predictions_df.head(top_n)["address"])

    # Get 1-hop neighbours
    neighbours = set()
    for addr in top_scam:
        if addr in G:
            neighbours.update(G.predecessors(addr))
            neighbours.update(G.successors(addr))

    subgraph_nodes = top_scam | neighbours
    SG = G.subgraph(subgraph_nodes)

    colors = ["#E24B4A" if n in top_scam else "#B4B2A9" for n in SG.nodes()]

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(SG, seed=42, k=0.5)
    nx.draw_networkx(
        SG, pos,
        node_color=colors,
        node_size=80,
        with_labels=False,
        arrows=True,
        arrowsize=8,
        edge_color="#D3D1C7",
        width=0.5,
        alpha=0.9
    )

    from matplotlib.patches import Patch
    legend = [
        Patch(color="#E24B4A", label=f"Predicted scam (top {top_n})"),
        Patch(color="#B4B2A9", label="Connected wallets"),
    ]
    plt.legend(handles=legend, loc="upper left")
    plt.title(f"Transaction Subgraph — Top {top_n} Predicted Scam Wallets")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("gnn_scam_subgraph.png", dpi=150)
    plt.show()
    print("Plot saved → gnn_scam_subgraph.png")


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_gnn_pipeline(
    tx_csv:       str = "scam_transactions.csv",
    features_csv: str = "wallet_features.csv",
    epochs:       int = 200
) -> pd.DataFrame:
    """
    End-to-end GNN pipeline:
      load CSVs → build graph → train GraphSAGE → predict → save

    Returns DataFrame of per-wallet scam predictions.
    """
    tx_df       = pd.read_csv(tx_csv)
    features_df = pd.read_csv(features_csv)

    from onchain_fetcher import extract_onchain_features
    if features_df.empty:
        features_df = extract_onchain_features(tx_df)

    G, node_index         = build_graph(tx_df, features_df)
    data, scaler          = graph_to_pyg(G, node_index)
    model, train_losses   = train_gnn(data, epochs=epochs)
    eval_results          = evaluate_gnn(model, data)
    predictions_df        = save_predictions(node_index, eval_results)

    plot_training_loss(train_losses)
    visualise_scam_subgraph(G, predictions_df)

    return predictions_df


if __name__ == "__main__":
    predictions = run_gnn_pipeline()
    print(f"\nDone. Top 5 highest-risk wallets:")
    print(predictions.head())
