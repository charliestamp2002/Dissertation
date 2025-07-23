import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, silhouette_score, davies_bouldin_score
from bezier_clustering import evaluate_bezier_curve

def compute_reconstruction_loss(model, data_tensor):
    """Computes MSE reconstruction loss for autoencoder."""
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        recon, _ = model(data_tensor.to(device))
    mse = mean_squared_error(
        data_tensor.cpu().numpy().reshape(len(data_tensor), -1),
        recon.cpu().numpy().reshape(len(recon), -1)
    )
    return mse

def evaluate_clustering_metrics(features, labels, name=""):
    """Computes silhouette score and Davies-Bouldin Index for given features and cluster labels."""
    sil = silhouette_score(features, labels)
    dbi = davies_bouldin_score(features, labels)
    print(f"\n{name} Clustering Evaluation:")
    print(f"  Silhouette Score        : {sil:.4f}")
    print(f"  Davies-Bouldin Index    : {dbi:.4f}")
    return sil, dbi

def evaluate_bezier_l1_distance(assignments_df, control_points_dict):
    """Computes average L1 distance between each resampled run and its cluster's Bézier control points."""
    total_l1 = 0
    count = 0
    for _, row in assignments_df.iterrows():
        traj = row["resampled_traj"]  # shape: (50, 2)
        cluster_id = row["assigned_cluster"]
        control_pts = control_points_dict[cluster_id]
        if control_pts is None:
            continue
        curve = evaluate_bezier_curve(control_pts, num_points=25)
        diff = np.abs(traj - curve)
        total_l1 += np.sum(diff)
        count += traj.shape[0]
    mean_l1 = total_l1 / count if count > 0 else np.nan
    print(f" Bézier Mean L1 Distance to Cluster Center: {mean_l1:.4f}")
    return mean_l1

def run_all_metrics(
    autoencoder_model,
    ae_tensor,
    ae_embeddings,
    ae_assignments_df,
    bezier_assignments_df,
    transformer_embeddings,
    transformer_assignments_df,
    bezier_control_points=None
):
    # === Autoencoder ===
    print("\n Autoencoder Evaluation:")
    recon_loss = compute_reconstruction_loss(autoencoder_model, ae_tensor)
    print(f"  Reconstruction Loss (MSE): {recon_loss:.4f}")
    evaluate_clustering_metrics(ae_embeddings, ae_assignments_df["ae_cluster"], name="Autoencoder")

    # === Bézier ===
    print("\n Bézier Clustering Evaluation:")
    run_trajs = np.stack(bezier_assignments_df["resampled_traj"].to_numpy())  # shape (N, 25, 2)
    run_trajs_flat = run_trajs.reshape(len(run_trajs), -1)
    labels = bezier_assignments_df["assigned_cluster"]
    evaluate_clustering_metrics(run_trajs_flat, labels, name="Bézier")

    if bezier_control_points is not None:
        control_dict = {i: cp for i, cp in enumerate(bezier_control_points)}
        evaluate_bezier_l1_distance(bezier_assignments_df, control_dict)

    # === Transformer ===
    print("\n Transformer Clustering Evaluation:")
    evaluate_clustering_metrics(transformer_embeddings, transformer_assignments_df["transformer_cluster"], name="Transformer")

        # Return metrics dictionary
    return {
        "ae_recon_loss": recon_loss,
        "ae_silhouette": silhouette_score(ae_embeddings, ae_assignments_df["ae_cluster"]),
        "ae_dbi": davies_bouldin_score(ae_embeddings, ae_assignments_df["ae_cluster"]),
        "bezier_silhouette": silhouette_score(run_trajs_flat, labels),
        "bezier_dbi": davies_bouldin_score(run_trajs_flat, labels),
        "bezier_mean_l1": evaluate_bezier_l1_distance(bezier_assignments_df, control_dict) if bezier_control_points is not None else np.nan,
        "transformer_silhouette": silhouette_score(transformer_embeddings, transformer_assignments_df["transformer_cluster"]),
        "transformer_dbi": davies_bouldin_score(transformer_embeddings, transformer_assignments_df["transformer_cluster"])
    }

# === Cluster Size Experiment ===




    