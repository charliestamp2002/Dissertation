import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def extract_tactical_features_for_refinement(zones_df):
    """
    Extract tactical features as numerical array for clustering refinement
    """
    tactical_features = []
    
    for _, row in zones_df.iterrows():
        features = [
            # Tactical phase features (boolean -> int)
            int(row.get('build_up_phase', False)),
            int(row.get('attacking_third_entry', False)),
            int(row.get('counter_attack', False)),
            int(row.get('set_piece_situation', False)),
            int(row.get('pressing_phase', False)),
            
            # Ball progression features
            int(row.get('creates_passing_lane', False)),
            int(row.get('stretches_defense', False)),
            int(row.get('creates_space', False)),
            int(row.get('supports_ball_carrier', False)),
            
            # Existing tactical features
            int(row.get('tactical_overlap', False)),
            int(row.get('tactical_underlap', False)),
            int(row.get('tactical_diagonal', False)),
            
            # Event context features
            int(row.get('under_pressure', False)),
            row.get('concurrent_passes', 0),
            row.get('pressure_events', 0),
            int(row.get('runner_received_pass', False)),
            
            # Phase and positioning
            1 if row.get('phase_of_play') == 'attack' else 0,
            int(row.get('in_possession', False)),
            int(row.get('run_forward', False))
        ]
        
        tactical_features.append(features)
    
    return np.array(tactical_features)

def calculate_tactical_coherence(cluster_runs_tactical_features):
    """
    Calculate how tactically coherent a cluster is
    Lower variance = higher coherence
    """
    if len(cluster_runs_tactical_features) < 2:
        return 0.0
    
    # Calculate mean variance across all tactical features
    feature_variances = np.var(cluster_runs_tactical_features, axis=0)
    mean_variance = np.mean(feature_variances)
    
    # Convert to coherence score (higher is better)
    coherence = 1.0 / (1.0 + mean_variance)
    return coherence

def tactical_refinement(base_cluster_assignments, zones_df, min_cluster_size=8, max_subclusters=4):
    """
    Refine trajectory-based clusters using tactical context
    
    Args:
        base_cluster_assignments (pd.DataFrame): DataFrame with 'run_id' and base cluster column
        zones_df (pd.DataFrame): DataFrame with tactical features
        min_cluster_size (int): Minimum size to refine a cluster
        max_subclusters (int): Maximum number of sub-clusters to create
    
    Returns:
        pd.DataFrame: Refined cluster assignments
    """
    
    # Determine the base cluster column name
    cluster_col = None
    for col in base_cluster_assignments.columns:
        if 'cluster' in col.lower() and col != 'run_id':
            cluster_col = col
            break
    
    if cluster_col is None:
        raise ValueError("No cluster column found in base_cluster_assignments")
    
    print(f"Refining clusters using tactical context...")
    print(f"Base clustering has {base_cluster_assignments[cluster_col].nunique()} clusters")
    
    # Merge with tactical features
    merged_df = base_cluster_assignments.merge(zones_df, on='run_id', how='inner')
    
    # Extract tactical features
    tactical_features_full = extract_tactical_features_for_refinement(zones_df)
    
    # Create mapping from run_id to tactical features
    run_id_to_features = dict(zip(zones_df['run_id'], tactical_features_full))
    
    refined_assignments = []
    refinement_stats = []
    
    scaler = StandardScaler()
    
    for base_cluster_id in sorted(merged_df[cluster_col].unique()):
        cluster_runs = merged_df[merged_df[cluster_col] == base_cluster_id]
        
        print(f"Processing base cluster {base_cluster_id}: {len(cluster_runs)} runs")
        
        if len(cluster_runs) < min_cluster_size:
            # Keep small clusters as single units
            for _, run in cluster_runs.iterrows():
                refined_assignments.append({
                    'run_id': run['run_id'],
                    'base_cluster': base_cluster_id,
                    'refined_cluster': f"{base_cluster_id}_0",
                    'refinement_applied': False
                })
            
            refinement_stats.append({
                'base_cluster': base_cluster_id,
                'n_runs': len(cluster_runs),
                'n_subclusters': 1,
                'refinement_applied': False,
                'reason': 'too_small'
            })
            continue
        
        # Extract tactical features for this cluster
        cluster_tactical_features = np.array([
            run_id_to_features[run_id] for run_id in cluster_runs['run_id']
        ])
        
        # Calculate original tactical coherence
        original_coherence = calculate_tactical_coherence(cluster_tactical_features)
        
        # Determine optimal number of sub-clusters
        n_subclusters = min(max_subclusters, max(2, len(cluster_runs) // 4))
        
        # Scale features for clustering
        scaled_features = scaler.fit_transform(cluster_tactical_features)
        
        # Perform sub-clustering
        sub_kmeans = KMeans(n_clusters=n_subclusters, random_state=42, n_init=10)
        sub_cluster_labels = sub_kmeans.fit_predict(scaled_features)
        
        # Calculate silhouette score for sub-clustering
        if n_subclusters > 1 and len(np.unique(sub_cluster_labels)) > 1:
            silhouette = silhouette_score(scaled_features, sub_cluster_labels)
        else:
            silhouette = -1
        
        # Evaluate if refinement improves tactical coherence
        refined_coherences = []
        for sub_cluster_id in range(n_subclusters):
            sub_cluster_mask = sub_cluster_labels == sub_cluster_id
            if np.sum(sub_cluster_mask) > 1:
                sub_features = cluster_tactical_features[sub_cluster_mask]
                coherence = calculate_tactical_coherence(sub_features)
                refined_coherences.append(coherence)
        
        avg_refined_coherence = np.mean(refined_coherences) if refined_coherences else 0
        
        # Apply refinement if it improves coherence and silhouette is reasonable
        apply_refinement = (avg_refined_coherence > original_coherence * 1.1 and 
                          silhouette > 0.2)
        
        if apply_refinement:
            # Create refined cluster assignments
            for i, (_, run) in enumerate(cluster_runs.iterrows()):
                sub_cluster_id = sub_cluster_labels[i]
                refined_assignments.append({
                    'run_id': run['run_id'],
                    'base_cluster': base_cluster_id,
                    'refined_cluster': f"{base_cluster_id}_{sub_cluster_id}",
                    'refinement_applied': True
                })
            
            refinement_stats.append({
                'base_cluster': base_cluster_id,
                'n_runs': len(cluster_runs),
                'n_subclusters': n_subclusters,
                'refinement_applied': True,
                'original_coherence': original_coherence,
                'refined_coherence': avg_refined_coherence,
                'silhouette_score': silhouette,
                'reason': 'improved_coherence'
            })
            
            print(f"  ✓ Refined into {n_subclusters} sub-clusters (coherence: {original_coherence:.3f} → {avg_refined_coherence:.3f})")
        
        else:
            # Keep as single cluster
            for _, run in cluster_runs.iterrows():
                refined_assignments.append({
                    'run_id': run['run_id'],
                    'base_cluster': base_cluster_id,
                    'refined_cluster': f"{base_cluster_id}_0",
                    'refinement_applied': False
                })
            
            refinement_stats.append({
                'base_cluster': base_cluster_id,
                'n_runs': len(cluster_runs),
                'n_subclusters': 1,
                'refinement_applied': False,
                'original_coherence': original_coherence,
                'refined_coherence': avg_refined_coherence,
                'silhouette_score': silhouette,
                'reason': 'no_improvement'
            })
            
            print(f"  ✗ No refinement (coherence: {original_coherence:.3f}, silhouette: {silhouette:.3f})")
    
    refined_df = pd.DataFrame(refined_assignments)
    stats_df = pd.DataFrame(refinement_stats)
    
    # Print summary
    n_refined_clusters = len([s for s in refinement_stats if s['refinement_applied']])
    total_refined_subclusters = refined_df['refined_cluster'].nunique()
    
    print(f"\nRefinement Summary:")
    print(f"  Original clusters: {base_cluster_assignments[cluster_col].nunique()}")
    print(f"  Refined clusters: {total_refined_subclusters}")
    print(f"  Clusters that were refined: {n_refined_clusters}")
    print(f"  Total runs processed: {len(refined_df)}")
    
    return refined_df, stats_df

def analyze_refined_clusters(refined_assignments, zones_df):
    """
    Analyze the tactical characteristics of refined clusters
    """
    
    merged_df = refined_assignments.merge(zones_df, on='run_id', how='inner')
    
    cluster_analysis = []
    
    for refined_cluster in sorted(merged_df['refined_cluster'].unique()):
        cluster_runs = merged_df[merged_df['refined_cluster'] == refined_cluster]
        
        if len(cluster_runs) == 0:
            continue
        
        # Calculate tactical feature percentages
        analysis = {
            'refined_cluster': refined_cluster,
            'base_cluster': refined_cluster.split('_')[0],
            'sub_cluster': refined_cluster.split('_')[1],
            'n_runs': len(cluster_runs),
            
            # Tactical phase characteristics
            'build_up_phase_pct': cluster_runs['build_up_phase'].mean() * 100,
            'counter_attack_pct': cluster_runs['counter_attack'].mean() * 100,
            'attacking_third_entry_pct': cluster_runs['attacking_third_entry'].mean() * 100,
            'set_piece_situation_pct': cluster_runs['set_piece_situation'].mean() * 100,
            'pressing_phase_pct': cluster_runs['pressing_phase'].mean() * 100,
            
            # Ball progression characteristics
            'creates_passing_lane_pct': cluster_runs['creates_passing_lane'].mean() * 100,
            'stretches_defense_pct': cluster_runs['stretches_defense'].mean() * 100,
            'creates_space_pct': cluster_runs['creates_space'].mean() * 100,
            'supports_ball_carrier_pct': cluster_runs['supports_ball_carrier'].mean() * 100,
            
            # Tactical run types
            'tactical_overlap_pct': cluster_runs['tactical_overlap'].mean() * 100,
            'tactical_underlap_pct': cluster_runs['tactical_underlap'].mean() * 100,
            'tactical_diagonal_pct': cluster_runs['tactical_diagonal'].mean() * 100,
            
            # Event context
            'under_pressure_pct': cluster_runs['under_pressure'].mean() * 100,
            'avg_concurrent_passes': cluster_runs['concurrent_passes'].mean(),
            'runner_received_pass_pct': cluster_runs['runner_received_pass'].mean() * 100,
            
            # Physical and positional
            'avg_run_length': cluster_runs['run_length_m'].mean(),
            'avg_speed': cluster_runs['mean_speed'].mean(),
            'forward_runs_pct': cluster_runs['run_forward'].mean() * 100,
            'attack_phase_pct': (cluster_runs['phase_of_play'] == 'attack').mean() * 100,
            
            # Most common characteristics
            'most_common_position': cluster_runs['position'].mode().iloc[0] if not cluster_runs['position'].mode().empty else 'Unknown',
            'most_common_trigger': cluster_runs['triggered_by_event'].mode().iloc[0] if not cluster_runs['triggered_by_event'].mode().empty else 'unknown',
            'most_common_outcome': cluster_runs['run_outcome'].mode().iloc[0] if not cluster_runs['run_outcome'].mode().empty else 'unknown'
        }
        
        cluster_analysis.append(analysis)
    
    return pd.DataFrame(cluster_analysis)

def create_tactical_cluster_labels(cluster_analysis_df):
    """
    Create human-readable tactical labels for clusters based on their characteristics
    """
    
    cluster_labels = {}
    
    for _, cluster in cluster_analysis_df.iterrows():
        cluster_id = cluster['refined_cluster']
        
        # Initialize label components
        labels = []
        
        # Phase-based labels
        if cluster['counter_attack_pct'] > 50:
            labels.append("Counter-Attack")
        elif cluster['build_up_phase_pct'] > 50:
            labels.append("Build-Up")
        elif cluster['set_piece_situation_pct'] > 30:
            labels.append("Set-Piece")
        elif cluster['pressing_phase_pct'] > 50:
            labels.append("Pressing")
        
        # Tactical type labels
        if cluster['tactical_overlap_pct'] > 60:
            labels.append("Overlap")
        elif cluster['tactical_underlap_pct'] > 60:
            labels.append("Underlap")
        elif cluster['tactical_diagonal_pct'] > 60:
            labels.append("Diagonal")
        
        # Ball progression labels
        if cluster['stretches_defense_pct'] > 60:
            labels.append("Defense-Stretching")
        elif cluster['creates_passing_lane_pct'] > 60:
            labels.append("Lane-Creating")
        elif cluster['creates_space_pct'] > 60:
            labels.append("Space-Creating")
        
        # Outcome-based labels
        if cluster['runner_received_pass_pct'] > 70:
            labels.append("Receiving")
        
        # Positional labels
        if cluster['attacking_third_entry_pct'] > 70:
            labels.append("Penetrating")
        
        # Create final label
        if labels:
            final_label = " + ".join(labels[:2])  # Take top 2 characteristics
        else:
            # Fallback based on position and phase
            position = cluster['most_common_position']
            phase = "Attack" if cluster['attack_phase_pct'] > 50 else "Defend"
            final_label = f"{position} {phase} Run"
        
        cluster_labels[cluster_id] = final_label
    
    return cluster_labels