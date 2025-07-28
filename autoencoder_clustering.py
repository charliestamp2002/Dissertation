import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from run_segmentation import resample_coords



class TrajectoryAutoencoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, latent_dim=16, seq_len=50):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len * input_dim),
            nn.Unflatten(1, (seq_len, input_dim))
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

def run_autoencoder_clustering(final_runs_df, num_points=25, k_clusters=70, 
                                batch_size=64, epochs=100, lr=1e-3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_ids = final_runs_df["run_id"].unique()
    run_tensors = []
    run_id_lookup = []

    for run_id in run_ids:
        run_df = final_runs_df[final_runs_df["run_id"] == run_id]
        resampled = resample_coords(run_df[["x_mirror_c", "y_mirror_c"]].values, num_points)
        run_tensors.append(resampled)
        run_id_lookup.append(run_id)

    run_tensor = np.stack(run_tensors, axis=0)  # [N, num_points, 2]

    # Prepare dataset
    traj_tensor = torch.tensor(run_tensor, dtype=torch.float32)
    dataset = TensorDataset(traj_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define model and optimizer
    model = TrajectoryAutoencoder(seq_len=num_points).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Train
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for (batch,) in loader:
            batch = batch.to(device)
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

    # Extract embeddings
    model.eval()
    with torch.no_grad():
        all_z = model.encoder(traj_tensor.to(device)).cpu().numpy()

    # Cluster
    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    cluster_ids = kmeans.fit_predict(all_z)

    # Assign cluster metadata
    autoencoder_assignments = []
    for run_id, cluster_id in zip(run_id_lookup, cluster_ids):
        run_df = final_runs_df[final_runs_df["run_id"] == run_id]
        if run_df.empty:
            continue

        assignment = {
            "run_id": run_id,
            "ae_cluster": cluster_id,
            "playerId": run_df["playerId"].iloc[0],
            "player_name": run_df["player_name"].iloc[0],
            "position": run_df["position"].iloc[0],
            "team_role": run_df["team_role"].iloc[0],
            "match_id": run_df["match_id"].iloc[0],
        }
        autoencoder_assignments.append(assignment)

    autoencoder_assignments_df = pd.DataFrame(autoencoder_assignments)
    return autoencoder_assignments_df, model


