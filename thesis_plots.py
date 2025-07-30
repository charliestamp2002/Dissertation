# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the CSV
# df = pd.read_csv("results/clustering_metrics_by_kclusters_2407.csv")

# new_row = pd.DataFrame([{
#     "ae_recon_loss": 0.0659,
#     "ae_silhouette": 0.1719,
#     "ae_dbi": 1.3419,
#     "bezier_silhouette": 0.1810,
#     "bezier_dbi": 1.2655,
#     "bezier_mean_l1": 5.1762,
#     "transformer_silhouette": 0.2948,
#     "transformer_dbi": 0.9525,
#     "k_clusters": 70
# }])

# df = pd.concat([df, new_row], ignore_index=True)
# df = df.sort_values("k_clusters").reset_index(drop=True)

# # Set plotting style
# plt.style.use("seaborn-v0_8-whitegrid")
# plt.rcParams.update({
#     "font.size": 14,
#     "axes.labelsize": 16,
#     "axes.titlesize": 18,
#     "xtick.labelsize": 12,
#     "ytick.labelsize": 12,
#     "legend.fontsize": 12,
#     "figure.figsize": (8, 6),
#     "lines.linewidth": 2,
#     "axes.spines.top": False,
#     "axes.spines.right": False,
# })

# # Define k values (clusters)
# k = df["k_clusters"]

# # Plot 1: Silhouette Scores vs Clusters
# plt.figure()
# plt.plot(k, df["ae_silhouette"], marker="o", label="Autoencoder")
# plt.plot(k, df["bezier_silhouette"], marker="s", label="Bezier")
# plt.plot(k, df["transformer_silhouette"], marker="^", label="Transformer")
# plt.xlabel("Number of Clusters (k)")
# plt.ylabel("Silhouette Score")
# plt.title("Silhouette Score vs Number of Clusters")
# plt.legend()
# plt.tight_layout()
# plt.savefig("silhouette_vs_clusters.png", dpi=300)
# plt.show()

# # Plot 2: Davies-Bouldin Index (DBI) vs Clusters
# plt.figure()
# plt.plot(k, df["ae_dbi"], marker="o", label="Autoencoder")
# plt.plot(k, df["bezier_dbi"], marker="s", label="Bezier")
# plt.plot(k, df["transformer_dbi"], marker="^", label="Transformer")
# plt.xlabel("Number of Clusters (k)")
# plt.ylabel("Davies-Bouldin Index")
# plt.title("DBI vs Number of Clusters")
# plt.legend()
# plt.tight_layout()
# plt.savefig("dbi_vs_clusters.png", dpi=300)
# plt.show()

# # Plot 3: Autoencoder Reconstruction Loss vs Clusters
# plt.figure()
# plt.plot(k, df["ae_recon_loss"], marker="o", color="tab:blue")
# plt.xlabel("Number of Clusters (k)")
# plt.ylabel("Reconstruction Loss")
# plt.title("Autoencoder Reconstruction Loss vs Number of Clusters")
# plt.tight_layout()
# plt.savefig("recon_loss_vs_clusters.png", dpi=300)
# plt.show()

# # Plot 4: Bezier Mean L1 vs Clusters
# plt.figure()
# plt.plot(k, df["bezier_mean_l1"], marker="s", color="tab:green")
# plt.xlabel("Number of Clusters (k)")
# plt.ylabel("Bezier Mean L1 Distance")
# plt.title("Bezier Mean L1 Distance vs Number of Clusters")
# plt.tight_layout()
# plt.savefig("bezier_l1_vs_clusters.png", dpi=300)
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from pathlib import Path

# # Configuration for publication-quality plots
# plt.rcParams.update({
#     'font.family': 'serif',
#     'font.serif': ['Times New Roman', 'DejaVu Serif'],
#     'font.size': 11,
#     'axes.labelsize': 12,
#     'axes.titlesize': 14,
#     'xtick.labelsize': 10,
#     'ytick.labelsize': 10,
#     'legend.fontsize': 10,
#     'figure.figsize': (7, 5),
#     'lines.linewidth': 1.5,
#     'lines.markersize': 6,
#     'axes.linewidth': 0.8,
#     'axes.spines.top': False,
#     'axes.spines.right': False,
#     'axes.grid': True,
#     'grid.alpha': 0.3,
#     'grid.linewidth': 0.5,
#     'savefig.dpi': 300,
#     'savefig.bbox': 'tight',
#     'savefig.pad_inches': 0.1,
# })

# def load_and_prepare_data():
#     """Load clustering metrics data and add additional data point."""
#     # Load the CSV data
#     df = pd.read_csv("results/clustering_metrics_by_kclusters_2407.csv")
    
#     # Add the additional data point
#     new_row = pd.DataFrame([{
#         "ae_recon_loss": 0.0659,
#         "ae_silhouette": 0.1719,
#         "ae_dbi": 1.3419,
#         "bezier_silhouette": 0.1810,
#         "bezier_dbi": 1.2655,
#         "bezier_mean_l1": 5.1762,
#         "transformer_silhouette": 0.2948,
#         "transformer_dbi": 0.9525,
#         "k_clusters": 70
#     }])
    
#     # Combine and sort data
#     df = pd.concat([df, new_row], ignore_index=True)
#     df = df.sort_values("k_clusters").reset_index(drop=True)
    
#     return df

# def create_comparison_plots(df, output_dir="figures"):
#     """Create professional comparison plots for clustering metrics."""
    
#     # Create output directory
#     Path(output_dir).mkdir(exist_ok=True)
    
#     k = df["k_clusters"]
    
#     # Define consistent colors and markers for each method
#     colors = {'Autoencoder': '#1f77b4', 'Bézier': '#ff7f0e', 'Transformer': '#2ca02c'}
#     markers = {'Autoencoder': 'o', 'Bézier': 's', 'Transformer': '^'}
    
#     # Plot 1: Silhouette Scores Comparison
#     fig, ax = plt.subplots(figsize=(7, 5))
    
#     ax.plot(k, df["ae_silhouette"], 
#            color=colors['Autoencoder'], marker=markers['Autoencoder'], 
#            label='Autoencoder', markerfacecolor='white', markeredgewidth=1.2)
#     ax.plot(k, df["bezier_silhouette"], 
#            color=colors['Bézier'], marker=markers['Bézier'], 
#            label='Bézier', markerfacecolor='white', markeredgewidth=1.2)
#     ax.plot(k, df["transformer_silhouette"], 
#            color=colors['Transformer'], marker=markers['Transformer'], 
#            label='Transformer', markerfacecolor='white', markeredgewidth=1.2)
    
#     ax.set_xlabel('Number of Clusters ($k$)')
#     ax.set_ylabel('Silhouette Score')
#     ax.set_title('Clustering Performance: Silhouette Score Analysis')
#     ax.legend(frameon=True, fancybox=True, shadow=True)
#     ax.set_xlim(k.min() - 2, k.max() + 2)
    
#     # Add minor ticks
#     ax.minorticks_on()
    
#     plt.savefig(f"{output_dir}/silhouette_comparison.png")
#     plt.savefig(f"{output_dir}/silhouette_comparison.pdf")  # Vector format for thesis
#     plt.show()
    
#     # Plot 2: Davies-Bouldin Index Comparison
#     fig, ax = plt.subplots(figsize=(7, 5))
    
#     ax.plot(k, df["ae_dbi"], 
#            color=colors['Autoencoder'], marker=markers['Autoencoder'], 
#            label='Autoencoder', markerfacecolor='white', markeredgewidth=1.2)
#     ax.plot(k, df["bezier_dbi"], 
#            color=colors['Bézier'], marker=markers['Bézier'], 
#            label='Bézier', markerfacecolor='white', markeredgewidth=1.2)
#     ax.plot(k, df["transformer_dbi"], 
#            color=colors['Transformer'], marker=markers['Transformer'], 
#            label='Transformer', markerfacecolor='white', markeredgewidth=1.2)
    
#     ax.set_xlabel('Number of Clusters ($k$)')
#     ax.set_ylabel('Davies-Bouldin Index')
#     ax.set_title('Clustering Performance: Davies-Bouldin Index Analysis')
#     ax.legend(frameon=True, fancybox=True, shadow=True)
#     ax.set_xlim(k.min() - 2, k.max() + 2)
    
#     # Add minor ticks
#     ax.minorticks_on()
    
#     plt.savefig(f"{output_dir}/dbi_comparison.png")
#     plt.savefig(f"{output_dir}/dbi_comparison.pdf")
#     plt.show()

# def create_individual_metric_plots(df, output_dir="figures"):
#     """Create individual plots for method-specific metrics."""
    
#     k = df["k_clusters"]
    
#     # Plot 3: Autoencoder Reconstruction Loss
#     fig, ax = plt.subplots(figsize=(7, 5))
    
#     ax.plot(k, df["ae_recon_loss"], 
#            color='#1f77b4', marker='o', 
#            markerfacecolor='white', markeredgewidth=1.2)
    
#     ax.set_xlabel('Number of Clusters ($k$)')
#     ax.set_ylabel('Reconstruction Loss')
#     ax.set_title('Autoencoder Performance: Reconstruction Loss vs. Cluster Count')
#     ax.set_xlim(k.min() - 2, k.max() + 2)
#     ax.minorticks_on()
    
#     # Add trend annotation if useful
#     if len(k) > 3:
#         # Add a subtle trend line
#         z = np.polyfit(k, df["ae_recon_loss"], 1)
#         p = np.poly1d(z)
#         ax.plot(k, p(k), "--", alpha=0.5, color='red', linewidth=1, 
#                label=f'Trend (slope: {z[0]:.4f})')
#         ax.legend()
    
#     plt.savefig(f"{output_dir}/autoencoder_reconstruction_loss.png")
#     plt.savefig(f"{output_dir}/autoencoder_reconstruction_loss.pdf")
#     plt.show()
    
#     # Plot 4: Bézier Mean L1 Distance
#     fig, ax = plt.subplots(figsize=(7, 5))
    
#     ax.plot(k, df["bezier_mean_l1"], 
#            color='#ff7f0e', marker='s', 
#            markerfacecolor='white', markeredgewidth=1.2)
    
#     ax.set_xlabel('Number of Clusters ($k$)')
#     ax.set_ylabel('Mean L1 Distance')
#     ax.set_title('Bézier Method Performance: L1 Distance vs. Cluster Count')
#     ax.set_xlim(k.min() - 2, k.max() + 2)
#     ax.minorticks_on()
    
#     # Add trend annotation if useful
#     if len(k) > 3:
#         z = np.polyfit(k, df["bezier_mean_l1"], 1)
#         p = np.poly1d(z)
#         ax.plot(k, p(k), "--", alpha=0.5, color='red', linewidth=1, 
#                label=f'Trend (slope: {z[0]:.4f})')
#         ax.legend()
    
#     plt.savefig(f"{output_dir}/bezier_l1_distance.png")
#     plt.savefig(f"{output_dir}/bezier_l1_distance.pdf")
#     plt.show()

# def create_summary_statistics_table(df):
#     """Generate summary statistics for the thesis."""
    
#     methods = ['ae', 'bezier', 'transformer']
#     metrics = {
#         'ae': ['silhouette', 'dbi', 'recon_loss'],
#         'bezier': ['silhouette', 'dbi', 'mean_l1'],
#         'transformer': ['silhouette', 'dbi']
#     }
    
#     print("\n" + "="*60)
#     print("CLUSTERING PERFORMANCE SUMMARY STATISTICS")
#     print("="*60)
    
#     for method in methods:
#         method_name = method.replace('ae', 'Autoencoder').replace('bezier', 'Bézier').replace('transformer', 'Transformer')
#         print(f"\n{method_name} Method:")
#         print("-" * 30)
        
#         for metric in metrics[method]:
#             col_name = f"{method}_{metric}"
#             if col_name in df.columns:
#                 values = df[col_name].dropna()
#                 print(f"{metric.replace('_', ' ').title():15}: "
#                       f"Mean={values.mean():.4f}, "
#                       f"Std={values.std():.4f}, "
#                       f"Min={values.min():.4f}, "
#                       f"Max={values.max():.4f}")

# def main():
#     """Main execution function."""
    
#     print("Loading and preparing clustering metrics data...")
#     df = load_and_prepare_data()
    
#     print(f"Data loaded successfully. Shape: {df.shape}")
#     print(f"Cluster range: {df['k_clusters'].min()} to {df['k_clusters'].max()}")
    
#     print("\nCreating comparison plots...")
#     create_comparison_plots(df)
    
#     print("Creating individual metric plots...")
#     create_individual_metric_plots(df)
    
#     print("Generating summary statistics...")
#     create_summary_statistics_table(df)
    
#     print(f"\nAll plots saved in 'figures/' directory")
#     print("Both PNG (for viewing) and PDF (for thesis) formats generated")

# if __name__ == "__main__":
#     main()


