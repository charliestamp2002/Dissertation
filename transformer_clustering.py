import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from run_segmentation import resample_coords

class TrajectoryTransformerEncoder(nn.Module):
    def __init__(self, input_dim=2, embed_dim=64, nhead=4, num_layers=2, latent_dim=32, seq_len=50):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(embed_dim, latent_dim)

    def forward(self, x):
        x = self.input_proj(x)  # [B, seq_len, embed_dim]
        x = self.transformer_encoder(x)  # [B, seq_len, embed_dim]
        x = x.transpose(1, 2)  # [B, embed_dim, seq_len]
        x = self.pool(x).squeeze(-1)  # [B, embed_dim]
        z = self.projection(x)  # [B, latent_dim]
        return z


def run_transformer_clustering(final_runs_df, num_points=50, k_clusters=70):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare run tensors
    run_ids = final_runs_df["run_id"].unique()
    run_tensors, run_id_lookup = [], []

    for run_id in run_ids:
        run_df = final_runs_df[final_runs_df["run_id"] == run_id]
        coords = run_df[["x_mirror_c", "y_mirror_c"]].values
        if coords.shape[0] < 2:
            continue
        resampled = resample_coords(coords, num_points)
        run_tensors.append(resampled)
        run_id_lookup.append(run_id)

    run_tensor = np.stack(run_tensors, axis=0)  # [N, seq_len, 2]
    input_tensor = torch.tensor(run_tensor, dtype=torch.float32)

    # Setup DataLoader
    batch_size = 64
    dataset = TensorDataset(input_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model setup
    model = TrajectoryTransformerEncoder(seq_len=num_points).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Pre-train using autoencoder-style training (optional)
    class TransformerAutoencoder(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = nn.Sequential(
                nn.Linear(encoder.projection.out_features, 64),
                nn.ReLU(),
                nn.Linear(64, num_points * 2),
                nn.Unflatten(1, (num_points, 2))
            )

        def forward(self, x):
            z = self.encoder(x)
            recon = self.decoder(z)
            return recon, z

    ae_model = TransformerAutoencoder(model).to(device)

    # Train loop
    epochs = 50
    for epoch in range(epochs):
        ae_model.train()
        total_loss = 0
        for (batch,) in loader:
            batch = batch.to(device)
            recon, _ = ae_model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

    # Inference to get embeddings
    ae_model.eval()
    with torch.no_grad():
        all_z = model(input_tensor.to(device)).cpu().numpy()

    # KMeans clustering
    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    cluster_ids = kmeans.fit_predict(all_z)

    # Package cluster assignments
    transformer_assignments = []
    for run_id, cluster_id in zip(run_id_lookup, cluster_ids):
        run_df = final_runs_df[final_runs_df["run_id"] == run_id]
        if run_df.empty:
            continue

        assignment = {
            "run_id": run_id,
            "transformer_cluster": cluster_id,
            "playerId": run_df["playerId"].iloc[0],
            "player_name": run_df["player_name"].iloc[0],
            "position": run_df["position"].iloc[0],
            "team_role": run_df["team_role"].iloc[0],
            "match_id": run_df["match_id"].iloc[0],
        }
        transformer_assignments.append(assignment)

    assignments_df = pd.DataFrame(transformer_assignments)
    return assignments_df, model
