import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

def compare_algorithms():
    """
    Compare results from all three algorithms by reading their CSV files
    and generating comparison visualizations.
    """
    # Check if results directory exists, if not create it
    results_dir = 'comparison_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Load result data from each algorithm's experiment
    try:
        euclidean_results = pd.read_csv('data/euclidean_results.csv')
        kmeans_results = pd.read_csv('data/kmeans_results.csv')
        ga_results = pd.read_csv('data/ga_results.csv')
        
        print(f"Loaded results data:")
        print(f"  Euclidean: {len(euclidean_results)} iterations")
        print(f"  K-Means: {len(kmeans_results)} iterations")
        print(f"  Genetic Algorithm: {len(ga_results)} iterations")
    except FileNotFoundError as e:
        print(f"Error: One or more result files not found. {e}")
        print("Make sure to run each algorithm first to generate the result files.")
        return
    
    # Define colors for algorithms for consistent visualization
    colors = {
        'Euclidean': '#4285F4',  # Google Blue
        'KMeans': '#EA4335',     # Google Red
        'GA': '#34A853'          # Google Green
    }
    
    # ------ Metrics Comparison Plots ------
    
    # 1. Validity Comparison (Bar Chart)
    plt.figure(figsize=(10, 6))
    algorithms = ['Euclidean', 'KMeans', 'GA']
    validities = [
        euclidean_results['validity_percentage'].mean(),
        kmeans_results['validity_percentage'].mean(),
        ga_results['validity_percentage'].mean()
    ]
    std_validities = [
        euclidean_results['validity_percentage'].std(),
        kmeans_results['validity_percentage'].std(),
        ga_results['validity_percentage'].std()
    ]
    
    plt.bar(algorithms, validities, yerr=std_validities, color=list(colors.values()), capsize=10)
    plt.title('Validity Score Comparison', fontsize=14)
    plt.ylabel('Validity Percentage (%)', fontsize=12)
    plt.ylim(top=min(max(validities) * 1.1, 100))  # Set y-limit with some padding
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add actual values on top of bars
    for i, v in enumerate(validities):
        plt.text(i, v + 1, f"{v:.2f}%", ha='center', fontsize=10)
    
    plt.savefig(f"{results_dir}/validity_comparison.png", dpi=300)
    plt.close()
    
    # 2. Similarity Score Comparison (Bar Chart)
    plt.figure(figsize=(10, 6))
    similarities = [
        euclidean_results['avg_similarity_score'].mean(),
        kmeans_results['avg_similarity_score'].mean(),
        ga_results['avg_similarity_score'].mean()
    ]
    std_similarities = [
        euclidean_results['avg_similarity_score'].std(),
        kmeans_results['avg_similarity_score'].std(),
        ga_results['avg_similarity_score'].std()
    ]
    
    plt.bar(algorithms, similarities, yerr=std_similarities, color=list(colors.values()), capsize=10)
    plt.title('Average Similarity Score Comparison', fontsize=14)
    plt.ylabel('Average Similarity Score (%)', fontsize=12)
    plt.ylim(top=min(max(similarities) * 1.1, 100))  # Set y-limit with some padding
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add actual values on top of bars
    for i, v in enumerate(similarities):
        plt.text(i, v + 1, f"{v:.2f}%", ha='center', fontsize=10)
    
    plt.savefig(f"{results_dir}/similarity_comparison.png", dpi=300)
    plt.close()
    
    # 3. Execution Time Comparison (Bar Chart)
    plt.figure(figsize=(10, 6))
    times = [
        euclidean_results['execution_time'].mean(),
        kmeans_results['execution_time'].mean(),
        ga_results['execution_time'].mean()
    ]
    std_times = [
        euclidean_results['execution_time'].std(),
        kmeans_results['execution_time'].std(),
        ga_results['execution_time'].std()
    ]
    
    plt.bar(algorithms, times, yerr=std_times, color=list(colors.values()), capsize=10)
    plt.title('Execution Time Comparison', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add actual values on top of bars
    for i, v in enumerate(times):
        plt.text(i, v + (max(times) * 0.03), f"{v:.2f}s", ha='center', fontsize=10)
    
    plt.savefig(f"{results_dir}/execution_time_comparison.png", dpi=300)
    plt.close()
    
    # 4. Match Count Comparison (Bar Chart)
    plt.figure(figsize=(10, 6))
    matches = [
        euclidean_results['total_matches'].mean(),
        kmeans_results['total_matches'].mean(),
        ga_results['total_matches'].mean()
    ]
    std_matches = [
        euclidean_results['total_matches'].std(),
        kmeans_results['total_matches'].std(),
        ga_results['total_matches'].std()
    ]
    
    plt.bar(algorithms, matches, yerr=std_matches, color=list(colors.values()), capsize=10)
    plt.title('Total Matches Comparison', fontsize=14)
    plt.ylabel('Number of Matches', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add actual values on top of bars
    for i, v in enumerate(matches):
        plt.text(i, v + (max(matches) * 0.03), f"{v:.0f}", ha='center', fontsize=10)
    
    plt.savefig(f"{results_dir}/matches_comparison.png", dpi=300)
    plt.close()
    
    # 5. Combined Metrics (Multi-bar chart)
    plt.figure(figsize=(12, 8))
    x = np.arange(len(algorithms))
    width = 0.25
    
    # Normalize metrics to same scale for comparison (out of 100)
    # Convert all to percentages
    validity_norm = [
        euclidean_results['validity_percentage'].mean(),
        kmeans_results['validity_percentage'].mean(),
        ga_results['validity_percentage'].mean()
    ]
    
    similarity_norm = [
        euclidean_results['avg_similarity_score'].mean(),
        kmeans_results['avg_similarity_score'].mean(),
        ga_results['avg_similarity_score'].mean()
    ]
    
    # Scale execution time to 0-100 range (lower is better)
    max_time = max(times)
    speed_norm = [(1 - (t / max_time)) * 100 for t in times]  # Invert so higher is better
    
    plt.bar(x - width, validity_norm, width, label='Validity %', color='skyblue')
    plt.bar(x, similarity_norm, width, label='Similarity %', color='lightgreen')
    plt.bar(x + width, speed_norm, width, label='Speed %', color='salmon')
    
    plt.ylabel('Score (%)', fontsize=12)
    plt.title('Algorithm Performance Comparison', fontsize=14)
    plt.xticks(x, algorithms)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/combined_metrics.png", dpi=300)
    plt.close()
    
    # 6. Radar Chart for Algorithm Comparison (shows all metrics in one view)
    plt.figure(figsize=(10, 10))
    
    # Function to create radar chart
    def radar_chart(labels, data, title):
        # Number of variables
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        
        # Close the polygon by repeating the first point
        angles += angles[:1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        for i, algorithm in enumerate(['Euclidean', 'KMeans', 'GA']):
            values = data[i]
            values += values[:1]  # Close the polygon
            
            ax.plot(angles, values, 'o-', linewidth=2, label=algorithm, color=colors[algorithm])
            ax.fill(angles, values, alpha=0.1, color=colors[algorithm])
        
        # Fix axis to go clockwise and start from top
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Set labels and grid
        plt.xticks(angles[:-1], labels)
        ax.set_rlabel_position(0)
        plt.yticks([25, 50, 75, 100], ["25", "50", "75", "100"], color="grey", size=8)
        plt.ylim(0, 100)
        
        # Add title and legend
        plt.title(title, size=16, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        return fig
    
    # Prepare data for radar chart
    radar_labels = ['Validity', 'Similarity', 'Speed', 'Match Rate', 'Valid Match %']
    
    # Valid match percentage (number of valid matches / total matches)
    valid_match_pct = [
        (euclidean_results['valid_matches'].mean() / euclidean_results['total_matches'].mean()) * 100,
        (kmeans_results['valid_matches'].mean() / kmeans_results['total_matches'].mean()) * 100,
        (ga_results['valid_matches'].mean() / ga_results['total_matches'].mean()) * 100
    ]
    
    # Match rate (as percentage of theoretical maximum - 1000 mentees if all matched)
    match_rate = [
        (euclidean_results['total_matches'].mean() / 1000) * 100,
        (kmeans_results['total_matches'].mean() / 1000) * 100,
        (ga_results['total_matches'].mean() / 1000) * 100
    ]
    
    radar_data = [
        [validity_norm[0], similarity_norm[0], speed_norm[0], match_rate[0], valid_match_pct[0]],
        [validity_norm[1], similarity_norm[1], speed_norm[1], match_rate[1], valid_match_pct[1]],
        [validity_norm[2], similarity_norm[2], speed_norm[2], match_rate[2], valid_match_pct[2]]
    ]
    
    radar_fig = radar_chart(radar_labels, radar_data, 'Algorithm Performance Comparison')
    radar_fig.savefig(f"{results_dir}/radar_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(radar_fig)
    
    # 7. Box Plots for Statistical Distribution Comparison
    metrics = ['validity_percentage', 'avg_similarity_score', 'execution_time']
    metric_titles = ['Validity (%)', 'Similarity Score (%)', 'Execution Time (s)']
    
    for i, metric in enumerate(metrics):
        plt.figure(figsize=(10, 6))
        
        # Gather data for box plot
        data = [
            euclidean_results[metric],
            kmeans_results[metric],
            ga_results[metric]
        ]
        
        # Create box plot
        boxplot = plt.boxplot(data, labels=algorithms, patch_artist=True)
        
        # Fill boxes with colors
        for patch, color in zip(boxplot['boxes'], list(colors.values())):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.title(f'{metric_titles[i]} Distribution Across Algorithms', fontsize=14)
        plt.ylabel(metric_titles[i], fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(f"{results_dir}/boxplot_{metric.replace('_', '')}.png", dpi=300)
        plt.close()
    
    # 8. Statistical Analysis - T-tests between algorithms
    print("\nStatistical Analysis (t-tests):")
    for metric in metrics:
        print(f"\nMetric: {metric}")
        # Euclidean vs KMeans
        t_stat, p_value = stats.ttest_ind(euclidean_results[metric], kmeans_results[metric])
        sig = "Significant" if p_value < 0.05 else "Not significant"
        print(f"  Euclidean vs KMeans: p={p_value:.4f} ({sig})")
        
        # Euclidean vs GA
        t_stat, p_value = stats.ttest_ind(euclidean_results[metric], ga_results[metric])
        sig = "Significant" if p_value < 0.05 else "Not significant"
        print(f"  Euclidean vs GA: p={p_value:.4f} ({sig})")
        
        # KMeans vs GA
        t_stat, p_value = stats.ttest_ind(kmeans_results[metric], ga_results[metric])
        sig = "Significant" if p_value < 0.05 else "Not significant"
        print(f"  KMeans vs GA: p={p_value:.4f} ({sig})")
    
    # Print summary statistics to console
    print("\nSummary Statistics:")
    
    print("\nValidity Percentage:")
    for algo, results in zip(['Euclidean', 'KMeans', 'GA'], [euclidean_results, kmeans_results, ga_results]):
        print(f"{algo}: {results['validity_percentage'].mean():.2f}% ± {results['validity_percentage'].std():.2f}%")
    
    print("\nAverage Similarity Score:")
    for algo, results in zip(['Euclidean', 'KMeans', 'GA'], [euclidean_results, kmeans_results, ga_results]):
        print(f"{algo}: {results['avg_similarity_score'].mean():.2f}% ± {results['avg_similarity_score'].std():.2f}%")
    
    print("\nExecution Time:")
    for algo, results in zip(['Euclidean', 'KMeans', 'GA'], [euclidean_results, kmeans_results, ga_results]):
        print(f"{algo}: {results['execution_time'].mean():.4f}s ± {results['execution_time'].std():.4f}s")
    
    print("\nTotal Matches:")
    for algo, results in zip(['Euclidean', 'KMeans', 'GA'], [euclidean_results, kmeans_results, ga_results]):
        print(f"{algo}: {results['total_matches'].mean():.2f} ± {results['total_matches'].std():.2f}")
    
    print("\nValid Matches:")
    for algo, results in zip(['Euclidean', 'KMeans', 'GA'], [euclidean_results, kmeans_results, ga_results]):
        print(f"{algo}: {results['valid_matches'].mean():.2f} ± {results['valid_matches'].std():.2f}")
    
    # Create overall algorithm ranking based on combined metrics
    print("\nAlgorithm Ranking:")
    # Create a weighted score (adjust weights based on importance)
    scores = {}
    
    # Calculate normalized scores (0-1 range)
    for algo, results, time_value in zip(
        ['Euclidean', 'KMeans', 'GA'], 
        [euclidean_results, kmeans_results, ga_results],
        times
    ):
        # Higher is better for validity and similarity
        validity_score = results['validity_percentage'].mean() / 100
        similarity_score = results['avg_similarity_score'].mean() / 100
        
        # Lower is better for time, so invert the score
        time_score = 1 - (time_value / max(times))
        
        # Calculate weighted total (adjust weights as needed)
        total_score = (0.4 * validity_score) + (0.3 * similarity_score) + (0.3 * time_score)
        scores[algo] = total_score
    
    # Rank algorithms by score
    ranked_algos = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (algo, score) in enumerate(ranked_algos, 1):
        print(f"{rank}. {algo}: {score:.4f}")
    
    print(f"\nComparison charts saved to '{results_dir}/' directory.")

if __name__ == "__main__":
    compare_algorithms()