import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

def compare_metrics_with_line_graphs():
    """
    Compare results from all three algorithms using line graphs with improved visibility
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
    
    # Define colors and markers for algorithms
    colors = {
        'Euclidean': '#4285F4',  # Google Blue
        'KMeans': '#EA4335',     # Google Red
        'GA': '#34A853'          # Google Green
    }
    
    markers = {
        'Euclidean': 'o',
        'KMeans': 's',
        'GA': '^'
    }
    
    # Create a better visualization approach using histograms and separate graphs
    metrics = ['validity_percentage', 'avg_similarity_score', 'execution_time']
    metric_titles = ['Validity (%)', 'Similarity Score (%)', 'Execution Time (s)']
    
    # 1. Simple bar chart comparison of average values
    plt.figure(figsize=(14, 6))
    algorithms = ['Euclidean', 'KMeans', 'GA']
    
    # List to store the average values for each metric and algorithm
    all_means = []
    all_stds = []
    
    # Calculate means for all metrics
    for metric in metrics:
        means = [
            euclidean_results[metric].mean(),
            kmeans_results[metric].mean(),
            ga_results[metric].mean() if len(ga_results) > 0 else np.nan
        ]
        stds = [
            euclidean_results[metric].std(),
            kmeans_results[metric].std(),
            ga_results[metric].std() if len(ga_results) > 0 else np.nan
        ]
        all_means.append(means)
        all_stds.append(stds)
    
    # Create bar positions
    bar_width = 0.25
    r1 = np.arange(len(algorithms))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    # Create bars
    plt.bar(r1, all_means[0], width=bar_width, edgecolor='grey', label=metric_titles[0], color='lightblue', yerr=all_stds[0], capsize=5)
    plt.bar(r2, all_means[1], width=bar_width, edgecolor='grey', label=metric_titles[1], color='lightgreen', yerr=all_stds[1], capsize=5)
    plt.bar(r3, all_means[2], width=bar_width, edgecolor='grey', label=metric_titles[2], color='salmon', yerr=all_stds[2], capsize=5)
    
    # Add labels
    plt.xlabel('Algorithm', fontweight='bold', fontsize=12)
    plt.ylabel('Value', fontweight='bold', fontsize=12)
    plt.xticks([r + bar_width for r in range(len(algorithms))], algorithms)
    plt.title('Average Performance Metrics by Algorithm', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/average_metrics_comparison.png", dpi=300)
    plt.close()
    
    # 2. Algorithm distribution comparison - Create separate plots for each metric
    for i, metric in enumerate(metrics):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        fig.suptitle(f'Distribution of {metric_titles[i]} Across Algorithms', fontsize=16)
        
        # Set common y-axis label
        fig.text(0.04, 0.5, 'Frequency', va='center', rotation='vertical', fontsize=12)
        
        for j, (algo, results) in enumerate(zip(algorithms, [euclidean_results, kmeans_results, ga_results])):
            if len(results) > 0:  # Only plot if there's data
                # Create histogram with KDE
                try:
                    values = results[metric].values
                    axes[j].hist(values, bins=30, alpha=0.7, color=colors[algo], density=True)
                    
                    # Add a vertical line for the mean
                    mean_val = np.mean(values)
                    axes[j].axvline(x=mean_val, color='black', linestyle='--', linewidth=1)
                    axes[j].text(mean_val, 0, f'Mean: {mean_val:.2f}', rotation=90, verticalalignment='bottom')
                    
                    # Add a title showing the number of data points
                    axes[j].set_title(f'{algo} (n={len(results)})')
                    axes[j].set_xlabel(metric_titles[i])
                    
                    # If very few data points, add scatter points below histogram
                    if len(results) < 10:
                        y_pos = -0.1  # Position below x-axis
                        for val in values:
                            axes[j].scatter(val, y_pos, color='red', s=50, zorder=10)
                            axes[j].text(val, y_pos*2, f'{val:.2f}', ha='center', fontsize=9)
                except Exception as e:
                    print(f"Error plotting {algo} for {metric}: {e}")
            else:
                axes[j].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axes[j].transAxes)
                axes[j].set_title(algo)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Adjust for the suptitle
        plt.savefig(f"{results_dir}/distribution_{metric}.png", dpi=300)
        plt.close()
    
    # 3. Create scatter plots comparing two metrics at a time
    metric_pairs = [
        ('validity_percentage', 'avg_similarity_score', 'Validity vs Similarity'),
        ('validity_percentage', 'execution_time', 'Validity vs Execution Time'),
        ('avg_similarity_score', 'execution_time', 'Similarity vs Execution Time')
    ]
    
    for x_metric, y_metric, title in metric_pairs:
        plt.figure(figsize=(10, 8))
        
        for algo, results, color, marker in zip(algorithms, [euclidean_results, kmeans_results, ga_results], 
                                              list(colors.values()), list(markers.values())):
            if len(results) > 0:
                x = results[x_metric].values
                y = results[y_metric].values
                
                # Plot scatter with large points and low alpha for better visibility
                plt.scatter(x, y, color=color, marker=marker, s=60, alpha=0.4, label=algo)
                
                # Add a larger point for the mean
                mean_x = np.mean(x)
                mean_y = np.mean(y)
                plt.scatter(mean_x, mean_y, color=color, marker='X', s=200, edgecolor='black', linewidth=1.5, label=f"{algo} Mean")
        
        plt.title(title, fontsize=14)
        plt.xlabel(x_metric.replace('_', ' ').title(), fontsize=12)
        plt.ylabel(y_metric.replace('_', ' ').title(), fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{results_dir}/scatter_{x_metric}_vs_{y_metric}.png", dpi=300)
        plt.close()
    
    # 4. Create a radar chart for overall comparison
    # Only if all algorithms have at least one data point
    if len(euclidean_results) > 0 and len(kmeans_results) > 0 and len(ga_results) > 0:
        # Create radar chart data
        categories = ['Validity', 'Similarity', 'Speed']
        
        # Get the means
        e_means = [
            euclidean_results['validity_percentage'].mean(),
            euclidean_results['avg_similarity_score'].mean(),
            100 * (1 - euclidean_results['execution_time'].mean() / max(
                euclidean_results['execution_time'].mean(),
                kmeans_results['execution_time'].mean(),
                ga_results['execution_time'].mean()
            ))
        ]
        
        k_means = [
            kmeans_results['validity_percentage'].mean(),
            kmeans_results['avg_similarity_score'].mean(),
            100 * (1 - kmeans_results['execution_time'].mean() / max(
                euclidean_results['execution_time'].mean(),
                kmeans_results['execution_time'].mean(),
                ga_results['execution_time'].mean()
            ))
        ]
        
        g_means = [
            ga_results['validity_percentage'].mean(),
            ga_results['avg_similarity_score'].mean(),
            100 * (1 - ga_results['execution_time'].mean() / max(
                euclidean_results['execution_time'].mean(),
                kmeans_results['execution_time'].mean(),
                ga_results['execution_time'].mean()
            ))
        ]
        
        # Set up radar chart
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Add values to the angles
        e_means += e_means[:1]
        k_means += k_means[:1]
        g_means += g_means[:1]
        
        # Plot radar chart
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        ax.plot(angles, e_means, 'o-', linewidth=2, color=colors['Euclidean'], label='Euclidean')
        ax.fill(angles, e_means, color=colors['Euclidean'], alpha=0.1)
        
        ax.plot(angles, k_means, 'o-', linewidth=2, color=colors['KMeans'], label='KMeans')
        ax.fill(angles, k_means, color=colors['KMeans'], alpha=0.1)
        
        ax.plot(angles, g_means, 'o-', linewidth=2, color=colors['GA'], label='GA')
        ax.fill(angles, g_means, color=colors['GA'], alpha=0.1)
        
        # Set category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # Add value labels at each point
        for i, angle in enumerate(angles[:-1]):
            ax.text(angle, e_means[i] + 5, f"{e_means[i]:.1f}", color=colors['Euclidean'], ha='center')
            ax.text(angle, k_means[i] + 5, f"{k_means[i]:.1f}", color=colors['KMeans'], ha='center')
            ax.text(angle, g_means[i] + 5, f"{g_means[i]:.1f}", color=colors['GA'], ha='center')
        
        ax.set_ylim(0, 100)
        plt.title('Algorithm Performance Comparison', size=15)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(f"{results_dir}/radar_chart.png", dpi=300)
        plt.close()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    
    print("\nValidity Percentage:")
    for algo, results in zip(algorithms, [euclidean_results, kmeans_results, ga_results]):
        if len(results) > 0:
            print(f"{algo}: {results['validity_percentage'].mean():.2f}% ± {results['validity_percentage'].std():.2f}%")
        else:
            print(f"{algo}: No data")
    
    print("\nAverage Similarity Score:")
    for algo, results in zip(algorithms, [euclidean_results, kmeans_results, ga_results]):
        if len(results) > 0:
            print(f"{algo}: {results['avg_similarity_score'].mean():.2f}% ± {results['avg_similarity_score'].std():.2f}%")
        else:
            print(f"{algo}: No data")
    
    print("\nExecution Time:")
    for algo, results in zip(algorithms, [euclidean_results, kmeans_results, ga_results]):
        if len(results) > 0:
            print(f"{algo}: {results['execution_time'].mean():.4f}s ± {results['execution_time'].std():.4f}s")
        else:
            print(f"{algo}: No data")
    
    print(f"\nComparison graphs saved to '{results_dir}/' directory.")

if __name__ == "__main__":
    compare_metrics_with_line_graphs()