import json, gzip, pandas as pd, pathlib
import matplotlib.pyplot as plt
from scipy.special import comb
import numpy as np
import hdbscan
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from matplotlib import cm

# Path to the tracking-compressed folder
TRACKING_DIR = pathlib.Path(__file__).parent / "tracking-compressed"

# Choose one file to start with (first in folder)
json_gz_path = sorted(TRACKING_DIR.glob("tracking_*.json.gz"))[0]

# Read the compressed JSON lines
with gzip.open(json_gz_path, "rt") as f:
    records = [json.loads(line) for line in f]

# Expand into frame and player tables
frames_flat = []
players_nested = []

for rec in records:
    rec_frame = {k: rec[k] for k in
                 ("period", "frameIdx", "gameClock", "live", "lastTouch")}
    bx, by, bz = rec["ball"]["xyz"]
    rec_frame.update({"ball_x": bx, "ball_y": by, "ball_z": bz,
                      "ball_speed": rec["ball"]["speed"]})
    frames_flat.append(rec_frame)

    for side in ("homePlayers", "awayPlayers"):
        for p in rec[side]:
            px, py, pz = p["xyz"]
            players_nested.append({
                "period": rec["period"],
                "frameIdx": rec["frameIdx"],
                "side": "home" if side == "homePlayers" else "away",
                "playerId": p["playerId"],
                "number": p["number"],
                "x": px, "y": py, "z": pz,
                "speed": p["speed"]
            })

# Create tidy dataframes
frames_df = pd.DataFrame(frames_flat)
players_df = pd.DataFrame(players_nested)

print("Frames loaded:", len(frames_df))
print("Player entries:", len(players_df))
#print(players_df.head(100))

# Step 1: Mirror Direction so all attacks go left-to-right. 

MIRROR = True  # Set to True if your team attacks right-to-left in this half

if MIRROR:
    players_df["y"] = -players_df["y"]

# Translate by Team Centroid (Translate co-ordinates relative to team centroid at run start)

# Drop keeper (typically number == 1)
outfield_df = players_df[players_df["number"] != 1]

# Compute centroid per team per frame
centroids = (
    outfield_df
    .groupby(["period", "frameIdx", "side"])
    .agg(cx=("x", "mean"), cy=("y", "mean"))
    .reset_index()
)

# Merge centroid info back into player table
players_df = players_df.merge(centroids, on=["period", "frameIdx", "side"])

# Translate positions relative to team centroid
players_df["x_c"] = players_df["x"] - players_df["cx"] # x_c = centroid relative positions (x-axis)
players_df["y_c"] = players_df["y"] - players_df["cy"] # y_c = centroid relative positions (y-axis)

# Sanity check: after coordinate adjustment, team centroid should be near (0, 0)
centroid_check = (
    players_df[players_df["number"] != 1]  # exclude keeper
    .groupby(["period", "frameIdx", "side"])
    .agg(mean_x_c=("x_c", "mean"), mean_y_c=("y_c", "mean"))
    .reset_index()
)

# Compute deviations from origin
centroid_check["distance_from_origin"] = np.sqrt(centroid_check["mean_x_c"]**2 + centroid_check["mean_y_c"]**2)

# Report max deviation and sample frames with large errors
max_dev = centroid_check["distance_from_origin"].max()
mean_dev = centroid_check["distance_from_origin"].mean()

print(f"✅ Team centroid check complete:")
print(f"- Max deviation from (0,0): {max_dev:.5f} m")
print(f"- Mean deviation: {mean_dev:.5f} m")

# Optionally show top 5 largest deviations if any are significant
if max_dev > 0.01:
    print("\n⚠️ Frames with largest deviation from (0,0):")
    print(centroid_check.sort_values("distance_from_origin", ascending=False).head(5))

# Histogram of distances from origin
plt.figure(figsize=(8, 4))
plt.hist(centroid_check["distance_from_origin"], bins=50, color="steelblue", edgecolor="black")
plt.title("Histogram: Distance of Team Centroid from (0, 0)")
plt.xlabel("Distance from Origin (m)")
plt.ylabel("Number of Frames")
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# Scatter plot of (mean_x_c, mean_y_c)
plt.figure(figsize=(6, 6))
plt.scatter(centroid_check["mean_x_c"], centroid_check["mean_y_c"], alpha=0.5, s=10)
plt.title("Scatter: Mean Team Centroid Coordinates After Adjustment")
plt.xlabel("mean_x_c")
plt.ylabel("mean_y_c")
plt.axhline(0, color="black", linewidth=1, linestyle="--")
plt.axvline(0, color="black", linewidth=1, linestyle="--")
plt.grid(True, linestyle='--', alpha=0.3)
plt.axis("equal")
plt.tight_layout()
plt.show()

# MACROSCOPIC PLOTS

# import matplotlib.pyplot as plt
# import seaborn as sns

# plt.figure(figsize=(10, 6))
# sns.histplot(players_df["speed"], bins=100, kde=True, color="steelblue")

# plt.title("Distribution of Player Speeds (Entire Match)", fontsize=14)
# plt.xlabel("Speed (m/s)")
# plt.ylabel("Frame Count")
# plt.grid(True, linestyle='--', alpha=0.4)
# plt.tight_layout()
# plt.show()

# # Average speed across all players per frame
# avg_speed_per_frame = (
#     players_df
#     .groupby("frameIdx")["speed"]
#     .mean()
#     .reset_index()
# )

# plt.figure(figsize=(12, 5))
# plt.plot(avg_speed_per_frame["frameIdx"], avg_speed_per_frame["speed"], alpha=0.8)

# plt.title("Average Player Speed Over Time", fontsize=14)
# plt.xlabel("Frame Index")
# plt.ylabel("Mean Speed (m/s)")
# plt.grid(True, linestyle='--', alpha=0.3)
# plt.tight_layout()
# plt.show()

# Segmenting each player's trajectory into distinct "runs"
MIN_FRAMES = 20  # minimum duration of a run (0.8s at 25Hz)
SPEED_THRESHOLD = 2.0

# Sort for consecutive frame detection
players_df = players_df.sort_values(["playerId", "period", "frameIdx"]).reset_index(drop=True)

# Identify fast frames
players_df["is_fast"] = players_df["speed"] > SPEED_THRESHOLD

# Mark breaks in continuity (i.e. when a new run starts)
players_df["new_run"] = (
    (players_df["is_fast"] != players_df["is_fast"].shift(1)) |
    (players_df["frameIdx"].diff() != 1) |
    (players_df["playerId"] != players_df["playerId"].shift(1))
).astype(int)

# Assign run IDs
players_df["run_id"] = players_df["new_run"].cumsum()

# Keep only fast sequences
fast_frames = players_df[players_df["is_fast"]].copy()

# Count how many frames per run
run_lengths = fast_frames.groupby("run_id").size()
valid_run_ids = run_lengths[run_lengths >= MIN_FRAMES].index

# Filter to valid runs only
runs_df = fast_frames[fast_frames["run_id"].isin(valid_run_ids)].copy()

# # Pick N sample run_ids to visualize
# sample_run_ids = runs_df["run_id"].drop_duplicates().sample(n=5, random_state=42)

# plt.figure(figsize=(10, 8))

# for run_id in sample_run_ids:
#     run = runs_df[runs_df["run_id"] == run_id]
#     plt.plot(run["x_c"], run["y_c"], marker="o", label=f"Run {run_id}")

# plt.title("Sample Off-Ball Runs (Centroid-Relative)", fontsize=14)
# plt.xlabel("x_c (m)")
# plt.ylabel("y_c (m)")
# plt.grid(True, linestyle="--", alpha=0.3)
# plt.legend()
# plt.gca().set_aspect("equal")  # keep proportions square
# plt.tight_layout()
# plt.show()


def bernstein_basis(p, P, t):
    return comb(P, p) * (1 - t)**(P - p) * t**p

def fit_bezier_curve(x, y, degree=3, n_samples=20):
    # Normalize time over [0, 1]
    t = np.linspace(0, 1, n_samples)
    
    # Build design matrix B: shape (n_samples, degree+1)
    B = np.stack([bernstein_basis(p, degree, t) for p in range(degree + 1)], axis=1)

    # Fit separate least squares models for x(t) and y(t)
    theta_x = np.linalg.lstsq(B, x, rcond=None)[0]
    theta_y = np.linalg.lstsq(B, y, rcond=None)[0]
    
    return np.stack([theta_x, theta_y], axis=1)  # shape (4, 2)

bezier_runs = []

# Number of points to interpolate each run to
n_samples = 20

for run_id, run_data in runs_df.groupby("run_id"):
    # Resample the run to fixed length
    coords = run_data[["x_c", "y_c"]].to_numpy()
    if len(coords) < n_samples:
        continue  # Skip short runs just in case

    idxs = np.linspace(0, len(coords) - 1, n_samples).astype(int)
    coords_resampled = coords[idxs]
    x, y = coords_resampled[:, 0], coords_resampled[:, 1]

    # Fit Bézier curve
    control_pts = fit_bezier_curve(x, y)
    
    bezier_runs.append({
        "run_id": run_id,
        "playerId": run_data["playerId"].iloc[0],
        "control_points": control_pts
    })

# print(bezier_runs)


def plot_run_with_bezier(x, y, control_points, run_id=None, cluster_id=None):
    t_vals = np.linspace(0, 1, 100)
    bernstein = np.stack([bernstein_basis(p, 3, t_vals) for p in range(4)], axis=1)
    bezier_curve = bernstein @ control_points  # (100, 2)

    plt.plot(x, y, 'o-', label="Original run", alpha=0.6)
    plt.plot(bezier_curve[:, 0], bezier_curve[:, 1], '-', label="Fitted Bézier", linewidth=2)
    plt.plot(control_points[:, 0], control_points[:, 1], 'x--', label="Control points")

    if run_id is not None:
        title = f"Run {run_id}"
        if cluster_id is not None:
            title += f" | Cluster {cluster_id}"
        plt.title(title)
    plt.xlabel("x_c (m)")
    plt.ylabel("y_c (m)")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()

# # Example: Plot 3 random runs
# for run in np.random.choice(bezier_runs, size=3, replace=False):
#     run_id = run["run_id"]
#     run_data = runs_df[runs_df["run_id"] == run_id]
#     coords = run_data[["x_c", "y_c"]].to_numpy()
#     idxs = np.linspace(0, len(coords) - 1, 20).astype(int)
#     coords_resampled = coords[idxs]
#     x, y = coords_resampled[:, 0], coords_resampled[:, 1]
    
#     plot_run_with_bezier(x, y, run["control_points"], run_id=run_id)


control_matrix = np.array([run["control_points"].flatten() for run in bezier_runs])
control_df = pd.DataFrame(control_matrix, columns=[f"cp{i}_{axis}" for i in range(4) for axis in ("x", "y")])
control_df["run_id"] = [run["run_id"] for run in bezier_runs]
control_df["playerId"] = [run["playerId"] for run in bezier_runs]

# CLUSTERING: 

#OPTION A: K means (simple but assumes spherical)

# Choose clustering method
CLUSTER_METHOD = "gmm"  # Options: "kmeans", "gmm", "hdbscan"

if CLUSTER_METHOD == "kmeans":
    k = 70
    model = KMeans(n_clusters=k, random_state=42)
    control_df["cluster"] = model.fit_predict(control_matrix)

elif CLUSTER_METHOD == "gmm":
    k = 15
    model = GaussianMixture(n_components=k, random_state=42)
    control_df["cluster"] = model.fit_predict(control_matrix)

elif CLUSTER_METHOD == "hdbscan":
    model = hdbscan.HDBSCAN(min_cluster_size=20)
    control_df["cluster"] = model.fit_predict(control_matrix)

else:
    raise ValueError("Unknown clustering method. Choose from: kmeans, gmm, hdbscan")

# Number of examples per cluster to plot
n_examples = 3

# Unique clusters (excluding noise, e.g. -1 from HDBSCAN)
unique_clusters = sorted(c for c in control_df["cluster"].unique() if c != -1)

for cluster_id in unique_clusters:
    runs_in_cluster = control_df[control_df["cluster"] == cluster_id]
    sample_runs = runs_in_cluster.sample(n=min(n_examples, len(runs_in_cluster)), random_state=42)

    print(f"\n📊 Cluster {cluster_id} — {len(runs_in_cluster)} total runs")

    for _, row in sample_runs.iterrows():
        run_id = row["run_id"]
        run_data = runs_df[runs_df["run_id"] == run_id]

        coords = run_data[["x_c", "y_c"]].to_numpy()
        idxs = np.linspace(0, len(coords) - 1, 20).astype(int)
        coords_resampled = coords[idxs]
        x, y = coords_resampled[:, 0], coords_resampled[:, 1]

        bezier_index = [r["run_id"] for r in bezier_runs].index(run_id)
        control_pts = bezier_runs[bezier_index]["control_points"]

        # PLOTTING EACH BEZIER RUN (NOT NEEDED FOR NOW)
        # plt.figure(figsize=(6, 5))
        # plot_run_with_bezier(x, y, control_pts, run_id=run_id, cluster_id=cluster_id)
        # plt.tight_layout()
        # plt.show()


def plot_pitch(ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(-52.5, 52.5)
    ax.set_ylim(-34, 34)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.set_title("Runs on Full Pitch")
    ax.grid(True, linestyle='--', alpha=0.2)

    # Optional: add halfway line
    ax.axvline(0, color='black', linewidth=1, alpha=0.3)
    return ax

def plot_cluster_on_pitch(cluster_id, control_df, bezier_runs, n_max=1000, color=None):
    runs_in_cluster = control_df[control_df["cluster"] == cluster_id].head(n_max)

    fig, ax = plt.subplots(figsize=(10, 7))
    plot_pitch(ax)

    for _, row in runs_in_cluster.iterrows():
        run_id = row["run_id"]
        bezier = bezier_runs[[r["run_id"] for r in bezier_runs].index(run_id)]
        control_pts = bezier["control_points"]

        t_vals = np.linspace(0, 1, 100)
        B = np.stack([bernstein_basis(p, 3, t_vals) for p in range(4)], axis=1)
        curve = B @ control_pts

        ax.plot(curve[:, 0], curve[:, 1], alpha=0.4, color=color, linewidth=2)

    ax.set_title(f"Cluster {cluster_id} — {len(runs_in_cluster)} runs")
    plt.tight_layout()
    plt.show()


# Define colors for clusters (e.g. tab10)
# Exclude -1 (HDBSCAN noise) from cluster count if present
valid_clusters = sorted(c for c in control_df["cluster"].unique() if c != -1)
n_clusters = len(valid_clusters)

# Get colormap that can handle large k
colors = cm.get_cmap("tab20", n_clusters)  # or "tab20b", "Set3", etc.
color_map = {cid: colors(i % n_clusters) for i, cid in enumerate(valid_clusters)}
#for cid in sorted(c for c in control_df["cluster"].unique() if c != -1):
top_k = 20
cluster_sizes = control_df[control_df["cluster"] != -1]["cluster"].value_counts().head(top_k).index.tolist()
selected_clusters = cluster_sizes

for cid in selected_clusters: 
    plot_cluster_on_pitch(cid, control_df, bezier_runs, n_max=100, color=color_map.get(cid, "black"))

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

import numpy as np
import pandas as pd

# Your control_matrix of shape (n_runs, 8)
X = control_matrix

results = []

k_values = range(5, 80, 5)  # Tune this as needed

for k in k_values:
    # ----- KMeans -----
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)

    sil_k = silhouette_score(X, kmeans_labels)
    ch_k = calinski_harabasz_score(X, kmeans_labels)
    db_k = davies_bouldin_score(X, kmeans_labels)

    results.append({
        "method": "kmeans",
        "k": k,
        "silhouette": sil_k,
        "calinski_harabasz": ch_k,
        "davies_bouldin": db_k,
        "bic": np.nan,
        "aic": np.nan
    })

    # ----- GMM -----
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm_labels = gmm.fit_predict(X)

    sil_g = silhouette_score(X, gmm_labels)
    ch_g = calinski_harabasz_score(X, gmm_labels)
    db_g = davies_bouldin_score(X, gmm_labels)
    bic = gmm.bic(X)
    aic = gmm.aic(X)

    results.append({
        "method": "gmm",
        "k": k,
        "silhouette": sil_g,
        "calinski_harabasz": ch_g,
        "davies_bouldin": db_g,
        "bic": bic,
        "aic": aic
    })

# Results as a dataframe
results_df = pd.DataFrame(results)

for method in ["kmeans", "gmm"]:
    subset = results_df[results_df["method"] == method]
    plt.plot(subset["k"], subset["silhouette"], label=method)

plt.xlabel("Number of Clusters / Components")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs k")
plt.legend()
plt.grid(True)
plt.show()