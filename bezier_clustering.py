import numpy as np
import random
from scipy.special import comb
from scipy.interpolate import interp1d
import pandas as pd
from run_segmentation import resample_coords

# Equation (5): Bézier basis function
def bernstein_poly(p, P, t):
    return comb(P - 1, p) * (t**p) * ((1 - t)**(P - 1 - p))

# Equation (7): Design matrix for Bézier fitting
def bezier_design_matrix(num_points, num_control_points):
    t_vals = np.linspace(0, 1, num_points)
    X = np.stack([bernstein_poly(p, num_control_points, t_vals) for p in range(num_control_points)], axis=1)
    return X  # shape: [num_points, num_control_points]

# Equation (6): Fit Bézier curve via least squares
def fit_bezier_curve(coords, num_control_points):
    N = coords.shape[0]
    X = bezier_design_matrix(N, num_control_points)  # shape [N, P]
    theta_x, _, _, _ = np.linalg.lstsq(X, coords[:, 0], rcond=None)
    theta_y, _, _, _ = np.linalg.lstsq(X, coords[:, 1], rcond=None)
    control_points = np.stack([theta_x, theta_y], axis=1)  # shape: [P, 2]
    return control_points

# Equation (4): Evaluate Bézier curve at t using control points θ
def evaluate_bezier_curve(control_points, num_points=25):
    P = control_points.shape[0]
    X = bezier_design_matrix(num_points, P)
    return X @ control_points

def compute_l1_distance(traj, bezier_curve):
    return np.mean(np.abs(traj - bezier_curve))

def bezier_kmeans_clustering(final_runs_df, adjusted_runs_list, k_clusters=70, num_control_points=4, num_points=25, max_iterations=10, tolerance=1e-3):
    random.seed(42)
    initial_centroids = random.sample(adjusted_runs_list, k_clusters)
    cluster_control_points = []

    for run in initial_centroids:
        coords = run[["x_mirror_c", "y_mirror_c"]].values
        control_pts = fit_bezier_curve(coords, num_control_points)
        cluster_control_points.append(control_pts)

    cluster_control_points = np.array(cluster_control_points)
    previous_objective = float('inf')

    for it in range(max_iterations):
        assignments = []
        objective_distances = []

        for run_id, run_df in final_runs_df.groupby("run_id"):
            coords = run_df[["x_mirror_c", "y_mirror_c"]].values
            resampled_coords = resample_coords(coords, num_points=num_points)
            if resampled_coords is None:
                continue

            min_dist = float("inf")
            assigned_cluster = -1

            for cluster_idx, control_pts in enumerate(cluster_control_points):
                bezier_curve = evaluate_bezier_curve(control_pts, num_points=num_points)
                dist = compute_l1_distance(resampled_coords, bezier_curve)
                if dist < min_dist:
                    min_dist = dist
                    assigned_cluster = cluster_idx

            assignments.append({
                "run_id": run_id,
                "assigned_cluster": assigned_cluster,
                "min_distance": min_dist,
                "playerId": run_df["playerId"].iloc[0],
                "player_name": run_df["player_name"].iloc[0],
                "position": run_df["position"].iloc[0],
                "team_role": run_df["team_role"].iloc[0],
                "match_id": run_df["match_id"].iloc[0],
                "resampled_traj": resampled_coords
            })

            objective_distances.append(min_dist)

        assignments_df = pd.DataFrame(assignments)
        objective = np.mean(objective_distances)
        print(f"Iteration {it}: Mean objective = {objective:.4f}")

        improvement = previous_objective - objective
        if improvement < tolerance:
            print(f"Converged (Δ={improvement:.6f}) at iteration {it}")
            break
        previous_objective = objective

        new_cluster_control_points = []

        for cluster_idx in range(k_clusters):
            assigned_indices = [i for i, a in enumerate(assignments) if a["assigned_cluster"] == cluster_idx]
            if not assigned_indices:
                new_cluster_control_points.append(cluster_control_points[cluster_idx])
                continue

            cluster_coords = []
            for idx in assigned_indices:
                run_id = assignments_df.loc[idx, "run_id"]
                run_df = final_runs_df[final_runs_df["run_id"] == run_id]
                coords = run_df[["x_mirror_c", "y_mirror_c"]].values
                resampled = resample_coords(coords, num_points=num_points)
                if resampled is not None:
                    cluster_coords.append(resampled)

            cluster_coords = np.stack(cluster_coords, axis=0)
            mean_coords = np.mean(cluster_coords, axis=0)
            new_control_pts = fit_bezier_curve(mean_coords, num_control_points)
            new_cluster_control_points.append(new_control_pts)

        cluster_control_points = np.array(new_cluster_control_points)

    return cluster_control_points, pd.DataFrame(assignments)

