import os
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import importlib
import shutil
from multiprocessing import Pool, cpu_count

# Import your algorithms - ensure these are accessible
from euclideanalgorithm import run_experiment as euclidean_run
from kclusteringalgorithm import run_experiment as kmeans_run
from genetic_algorithm import run_experiment as ga_run

class TestingFramework:
    def __init__(self, base_output_dir="results"):
        """
        Initialize the comprehensive testing framework
        """
        self.base_output_dir = base_output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config = None
        
        # Create the main output directory
        os.makedirs(base_output_dir, exist_ok=True)
        
        # Subdirectories for organized results
        self.dirs = {
            'constraints': os.path.join(base_output_dir, "constraint_analysis"),
            'scalability': os.path.join(base_output_dir, "scalability_analysis"),
            'visualizations': os.path.join(base_output_dir, "visualizations"),
            'raw_data': os.path.join(base_output_dir, "raw_data"),
        }
        
        # Create subdirectories
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        
        # Define algorithm configurations
        self.algorithms = {
            'Euclidean': {
                'run_func': euclidean_run,
                'color': '#4285F4',  # Google Blue
                'marker': 'o',
                'markersize': 10,    # Larger marker
                'linestyle': '-',    # Solid line
                'linewidth': 3,      # Thicker line
                'zorder': 10         # Draw on top of other lines
            },
            'KMeans': {
                'run_func': kmeans_run,
                'color': '#EA4335',  # Google Red
                'marker': 's',
                'markersize': 8,
                'linestyle': '--',   # Dashed line
                'linewidth': 2,
                'zorder': 5
            },
            'GA': {
                'run_func': ga_run,
                'color': '#34A853',  # Google Green
                'marker': '^',
                'markersize': 8,
                'linestyle': ':',    # Dotted line
                'linewidth': 2,
                'zorder': 1,
                'default_params': {
                    'generations': 5,      #adjust paremters for GA runs in the test_framework
                    'pop_size': 100,
                    'crossover_rate': 0.9,
                    'mutation_rate': 0.01,
                    'tournament_size': 5
                }
            }
        }
        
        # Define default configurations
        self.default_config = {
            # Basic run configuration
            'mentors_file': 'mentors_dataset.csv',
            'mentees_file': 'mentees_dataset.csv',
            'constraint_iterations': 5,  # Number of iterations for each constraint configuration
            'scalability_iterations': 5,  # Number of iterations for each dataset size
            
            # Dataset sizes for scalability testing
            'dataset_sizes': [
                {'mentors': 500, 'mentees': 1000, 'label': 'Small (500/1000)'},
                {'mentors': 1000, 'mentees': 2000, 'label': 'Medium (1000/2000)'},
                {'mentors': 2000, 'mentees': 4000, 'label': 'Large (2000/4000)'},
                {'mentors': 4000, 'mentees': 4000, 'label': 'Max Equal (4000/4000)'}
            ],
            
            # Constraint configurations to test
            'constraints': [
                {
                    'label': 'Basic (2 constraints)',
                    'constraints': {
                        'max_mentees_per_mentor': 2,
                        'min_similarities': 2
                    }
                },
                {
                    'label': 'Industry Match (3 constraints)',
                    'constraints': {
                        'max_mentees_per_mentor': 2,
                        'min_similarities': 2,
                        'industry_match': True
                    }
                },
                {
                    'label': 'Experience & Education (5 constraints)',
                    'constraints': {
                        'max_mentees_per_mentor': 2,
                        'min_similarities': 2,
                        'industry_match': True,
                        'min_experience_gap': 2,
                        'education_level': True
                    }
                },
                {
                    'label': 'Communication & Availability (7 constraints)',
                    'constraints': {
                        'max_mentees_per_mentor': 2,
                        'min_similarities': 2,
                        'industry_match': True,
                        'min_experience_gap': 2,
                        'education_level': True,
                        'communication_match': True,
                        'availability_match': True
                    }
                },
                {
                    'label': 'Language & Mentoring (10 constraints)',
                    'constraints': {
                        'max_mentees_per_mentor': 2,
                        'min_similarities': 2,
                        'industry_match': True,
                        'min_experience_gap': 2,
                        'education_level': True,
                        'communication_match': True,
                        'availability_match': True,
                        'language_match': True,
                        'secondary_language_match': True,
                        'mentoring_preferences_match': True
                    }
                },
                {
                    'label': 'Expertise & Position (15 constraints)',
                    'constraints': {
                        'max_mentees_per_mentor': 2,
                        'min_similarities': 2,
                        'industry_match': True,
                        'min_experience_gap': 2,
                        'education_level': True,
                        'communication_match': True,
                        'availability_match': True,
                        'language_match': True,
                        'secondary_language_match': True,
                        'mentoring_preferences_match': True,
                        'expertise_overlap': True,
                        'position_progression': True,
                        'min_position_level_gap': 1,
                        'complementary_roles': True,
                        'required_mentor_expertise': 'Machine Learning'  # Example specific expertise
                    }
                },
                {
                    'label': 'Full Constraints (20 constraints)',
                    'constraints': {
                        'max_mentees_per_mentor': 2,
                        'min_similarities': 2,
                        'industry_match': True,
                        'min_experience_gap': 2,
                        'education_level': True,
                        'communication_match': True,
                        'availability_match': True,
                        'language_match': True,
                        'secondary_language_match': True,
                        'mentoring_preferences_match': True,
                        'expertise_overlap': True,
                        'position_progression': True,
                        'min_position_level_gap': 1,
                        'complementary_roles': True,
                        'required_mentor_expertise': 'Machine Learning',
                        'gender_match': True,
                        'location_match': True,
                        'workstyle_match': True,
                        'min_industry_years': 5,
                        'mentoring_style_match': True,
                        'max_distance': 5,
                        'communication_frequency': 'Weekly'
                    }
                }
            ]
        }
        
    def load_config(self, config_file=None):
        """
        Load configuration from file or use default
        """
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = json.load(f)
            print(f"Configuration loaded from {config_file}")
        else:
            self.config = self.default_config
            print("Using default configuration")
            
        # Save the configuration for reference
        with open(os.path.join(self.base_output_dir, f"config_{self.timestamp}.json"), 'w') as f:
            json.dump(self.config, f, indent=4)
            
    
    def run_constraint_analysis(self):
        """
        Run all algorithms with varying constraints
        """
        print("\n" + "="*70)
        print("RUNNING CONSTRAINT ANALYSIS")
        print("="*70)
        
        # Create output directory for constraint results
        output_dir = os.path.join(self.dirs['constraints'], f"constraints_{self.timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        all_results = {}
        
        # For each algorithm
        for algo_name, algo_config in self.algorithms.items():
            all_results[algo_name] = []
            
            # For each constraint configuration
            for constraint_config in self.config['constraints']:
                constraint_label = constraint_config['label']
                constraints = constraint_config['constraints']
                
                print(f"\nRunning {algo_name} with {constraint_label}...")
                
                # Parameters specific to GA
                kwargs = {}
                if algo_name == 'GA':
                    kwargs = algo_config['default_params']
                
                try:
                    # Start timing
                    start_time = time.time()
                    
                    # Run the algorithm WITH CONSTRAINTS PROPERLY PASSED
                    results = algo_config['run_func'](
                        self.config['mentors_file'],
                        self.config['mentees_file'],
                        num_randomizations=self.config['constraint_iterations'],
                        constraints=constraints,  # THIS LINE IS CRITICAL
                        **kwargs
                    )
                    
                    # Save detailed results
                    results_df = results['all_results']
                    results_df.to_csv(
                        os.path.join(output_dir, f"{algo_name.lower()}_{constraint_label.replace(' ', '_').lower()}.csv"), 
                        index=False
                    )
                    
                    # Store summary results
                    constraint_result = {
                        'algorithm': algo_name,
                        'constraint_label': constraint_label,
                        'num_constraints': len(constraints),
                        'constraints': json.dumps(constraints),
                        'validity_percentage': results['avg_validity'],
                        'avg_similarity_score': results['avg_similarity'],
                        'execution_time': results['avg_execution_time'],
                        'total_matches': results['total_matches'] / self.config['constraint_iterations'],
                        'valid_matches': results['total_valid_matches'] / self.config['constraint_iterations']
                    }
                    
                    all_results[algo_name].append(constraint_result)
                    
                    duration = time.time() - start_time
                    print(f"  ✓ Completed in {duration:.2f}s, Validity: {results['avg_validity']:.2f}%")
                    
                except Exception as e:
                    print(f"  ✗ Error: {str(e)}")
        
        # Combine all results and save
        all_algorithms_results = []
        for algo_name, results in all_results.items():
            all_algorithms_results.extend(results)
            
        combined_df = pd.DataFrame(all_algorithms_results)
        combined_df.to_csv(os.path.join(output_dir, "all_constraints_results.csv"), index=False)
        
        # Create constraint impact visualizations
        self._create_constraint_visualizations(combined_df, output_dir)
        
        return combined_df
    
    def _create_constraint_visualizations(self, combined_results, output_dir):
        """
        Create visualizations showing impact of constraints
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            sns.set_style("whitegrid")
            
            # Group data by algorithm and constraint
            algorithms = combined_results['algorithm'].unique()
            
            # 1. Plot validity vs number of constraints
            plt.figure(figsize=(12, 6))
            
            for algo in algorithms:
                algo_data = combined_results[combined_results['algorithm'] == algo]
                plt.plot(
                    algo_data['num_constraints'], 
                    algo_data['validity_percentage'],
                    label=algo,
                    color=self.algorithms[algo]['color'],
                    marker=self.algorithms[algo]['marker'], 
                    markersize=self.algorithms[algo].get('markersize', 8),
                    linestyle=self.algorithms[algo].get('linestyle', '-'),
                    linewidth=self.algorithms[algo].get('linewidth', 2),
                    zorder=self.algorithms[algo].get('zorder', 1)
                )
                            
                # Add data labels
                for i, row in algo_data.iterrows():
                    plt.annotate(
                        f"{row['avg_similarity_score']:.1f}%", 
                        (row['num_constraints'], row['avg_similarity_score']),
                        textcoords="offset points", 
                        xytext=(0, 10), 
                        ha='center'
                    )
            
            plt.title('Impact of Constraints on Similarity Score', fontsize=14)
            plt.xlabel('Number of Constraints', fontsize=12)
            plt.ylabel('Average Similarity Score (%)', fontsize=12)
            # Update x-axis labels in the visualization code
            plt.xticks(combined_results['num_constraints'].unique(), 
                    ['Basic (2)', 'Industry (3)', 'Exp+Edu (5)', 'Comm+Avail (7)', 
                        'Lang+Mentor (10)', 'Exp+Pos (15)', 'Full (20)'],
                    rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "constraints_similarity_impact.png"), dpi=300)
            plt.savefig(os.path.join(self.dirs['visualizations'], "constraints_similarity_impact.png"), dpi=300)
            plt.close()
            
            # 3. Plot execution time vs number of constraints
            plt.figure(figsize=(12, 6))
            
            for algo in algorithms:
                algo_data = combined_results[combined_results['algorithm'] == algo]
                plt.plot(
                    algo_data['num_constraints'], 
                    algo_data['execution_time'], 
                    'o-', 
                    label=algo,
                    color=self.algorithms[algo]['color'],
                    marker=self.algorithms[algo]['marker'],
                    linewidth=2
                )
                
                # Add data labels
                for i, row in algo_data.iterrows():
                    plt.annotate(
                        f"{row['execution_time']:.2f}s", 
                        (row['num_constraints'], row['execution_time']),
                        textcoords="offset points", 
                        xytext=(0, 10), 
                        ha='center'
                    )
            
            plt.title('Impact of Constraints on Execution Time', fontsize=14)
            plt.xlabel('Number of Constraints', fontsize=12)
            plt.ylabel('Average Execution Time (seconds)', fontsize=12)
            plt.xticks(combined_results['num_constraints'].unique())
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "constraints_execution_time.png"), dpi=300)
            plt.savefig(os.path.join(self.dirs['visualizations'], "constraints_execution_time.png"), dpi=300)
            plt.close()
            
            # 4. Plot match coverage vs number of constraints
            plt.figure(figsize=(12, 6))
            
            for algo in algorithms:
                algo_data = combined_results[combined_results['algorithm'] == algo]
                match_rates = (algo_data['total_matches'] / self.config['mentees_file']) * 100 
                plt.plot(
                    algo_data['num_constraints'], 
                    match_rates, 
                    'o-', 
                    label=algo,
                    color=self.algorithms[algo]['color'],
                    marker=self.algorithms[algo]['marker'],
                    linewidth=2
                )
                
                # Add data labels
                for i, (x, y) in enumerate(zip(algo_data['num_constraints'], match_rates)):
                    plt.annotate(
                        f"{y:.1f}%", 
                        (x, y),
                        textcoords="offset points", 
                        xytext=(0, 10), 
                        ha='center'
                    )
            
            plt.title('Impact of Constraints on Match Coverage', fontsize=14)
            plt.xlabel('Number of Constraints', fontsize=12)
            plt.ylabel('Percentage of Mentees Matched (%)', fontsize=12)
            plt.xticks(combined_results['num_constraints'].unique())
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "constraints_match_coverage.png"), dpi=300)
            plt.savefig(os.path.join(self.dirs['visualizations'], "constraints_match_coverage.png"), dpi=300)
            plt.close()
            
            # 5. Create radar charts for each constraint level
            from matplotlib.cm import get_cmap
            
            constraint_levels = sorted(combined_results['num_constraints'].unique())
            
            for num_constraints in constraint_levels:
                plt.figure(figsize=(10, 8))
                ax = plt.subplot(111, polar=True)
                
                # Filter data for this constraint level
                level_data = combined_results[combined_results['num_constraints'] == num_constraints]
                
                # Define categories
                categories = ['Validity', 'Similarity', 'Speed', 'Coverage']
                N = len(categories)
                
                # Compute the angle for each category
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1]  # Close the polygon
                
                # Add the first category again at the end to close the circle
                ax.set_theta_offset(np.pi / 2)
                ax.set_theta_direction(-1)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories)
                
                # Draw the polygons for each algorithm
                for i, algo in enumerate(algorithms):
                    algo_row = level_data[level_data['algorithm'] == algo]
                    if len(algo_row) == 0:
                        continue
                        
                    # Get values for each metric
                    validity = algo_row['validity_percentage'].values[0]
                    similarity = algo_row['avg_similarity_score'].values[0]
                    
                    # For speed, lower is better, so invert and normalize
                    max_time = level_data['execution_time'].max()
                    speed = (1 - algo_row['execution_time'].values[0] / max_time) * 100
                    
                    # Calculate coverage as percentage of total possible matches
                    coverage = (algo_row['total_matches'].values[0] / 1000) * 100  # Assuming 1000 mentees
                    
                    values = [validity, similarity, speed, coverage]
                    values += values[:1]  # Close the polygon
                    
                    ax.plot(angles, values, linewidth=2, linestyle='solid', label=algo, 
                           color=self.algorithms[algo]['color'])
                    ax.fill(angles, values, alpha=0.1, color=self.algorithms[algo]['color'])
                
                # Add legend and title
                plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                plt.title(f'Algorithm Performance with {num_constraints} Constraints', size=15)
                
                # Save the plot
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"radar_constraints_{num_constraints}.png"), dpi=300)
                plt.savefig(os.path.join(self.dirs['visualizations'], f"radar_constraints_{num_constraints}.png"), dpi=300)
                plt.close()
                
            print("✓ Constraint analysis visualizations created")
            
        except Exception as e:
            print(f"✗ Error creating constraint visualizations: {str(e)}")
    
    def run_scalability_analysis(self):
        """
        Run all algorithms with varying dataset sizes
        """
        print("\n" + "="*70)
        print("RUNNING SCALABILITY ANALYSIS")
        print("="*70)
        
        # Create output directory for scalability results
        output_dir = os.path.join(self.dirs['scalability'], f"scalability_{self.timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get constraints from the last (most comprehensive) configuration
        full_constraints = self.config['constraints'][-1]['constraints']
        print(f"Using full constraints ({len(full_constraints)} constraints) for scalability testing")
        
        all_results = {}
        
        # For each algorithm
        for algo_name, algo_config in self.algorithms.items():
            all_results[algo_name] = []
            
            # For each dataset size
            for size_config in self.config['dataset_sizes']:
                mentor_size = size_config['mentors']
                mentee_size = size_config['mentees']
                size_label = size_config['label']
                
                print(f"\nRunning {algo_name} with dataset size {size_label}...")
                
                # Parameters specific to GA
                kwargs = {}
                if algo_name == 'GA':
                    kwargs = algo_config['default_params']
                
                try:
                    # Start timing
                    start_time = time.time()
                    
                    # Run the algorithm WITH CONSTRAINTS
                    results = algo_config['run_func'](
                        self.config['mentors_file'],
                        self.config['mentees_file'],
                        num_randomizations=self.config['scalability_iterations'],
                        mentor_sample_size=mentor_size,
                        mentee_sample_size=mentee_size,
                        constraints=full_constraints,  # THIS IS THE IMPORTANT LINE - PASSING CONSTRAINTS
                        **kwargs
                    )
                    
                    # Save detailed results
                    results_df = results['all_results']
                    results_df.to_csv(
                        os.path.join(output_dir, f"{algo_name.lower()}_{mentor_size}m_{mentee_size}m.csv"), 
                        index=False
                    )
                    
                    # Store summary results
                    size_result = {
                        'algorithm': algo_name,
                        'mentor_size': mentor_size,
                        'mentee_size': mentee_size,
                        'size_label': size_label,
                        'validity_percentage': results['avg_validity'],
                        'avg_similarity_score': results['avg_similarity'],
                        'execution_time': results['avg_execution_time'],
                        'total_matches': results['total_matches'] / self.config['scalability_iterations'],
                        'valid_matches': results['total_valid_matches'] / self.config['scalability_iterations'],
                        'total_dataset_size': mentor_size + mentee_size
                    }
                    
                    all_results[algo_name].append(size_result)
                    
                    duration = time.time() - start_time
                    print(f"  ✓ Completed in {duration:.2f}s, Validity: {results['avg_validity']:.2f}%")
                    
                except Exception as e:
                    print(f"  ✗ Error: {str(e)}")
        
        # Combine all results and save
        all_algorithms_results = []
        for algo_name, results in all_results.items():
            all_algorithms_results.extend(results)
            
        combined_df = pd.DataFrame(all_algorithms_results)
        combined_df.to_csv(os.path.join(output_dir, "all_scalability_results.csv"), index=False)
        
        # Create scalability visualizations
        self._create_scalability_visualizations(combined_df, output_dir)
        
        return combined_df
    
    def _create_scalability_visualizations(self, combined_results, output_dir):
        """
        Create visualizations showing impact of dataset size
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            sns.set_style("whitegrid")
            
            # Group data by algorithm
            algorithms = combined_results['algorithm'].unique()
            
            # 1. Plot execution time vs dataset size
            plt.figure(figsize=(12, 6))
            
            for algo in algorithms:
                algo_data = combined_results[combined_results['algorithm'] == algo].sort_values('total_dataset_size')
                plt.plot(
                    algo_data['size_label'],
                    algo_data['execution_time'],
                    label=algo,
                    color=self.algorithms[algo]['color'],
                    marker=self.algorithms[algo]['marker'],
                    markersize=self.algorithms[algo].get('markersize', 8),
                    linestyle=self.algorithms[algo].get('linestyle', '-'),
                    linewidth=self.algorithms[algo].get('linewidth', 2),
                    zorder=self.algorithms[algo].get('zorder', 1)
                )
                
                # Add data labels with different vertical positions
                label_offsets = {'Euclidean': 12, 'KMeans': 8, 'GA': 4}
                
                # Add data labels
                for i, row in algo_data.iterrows():
                    plt.annotate(
                        f"{row['execution_time']:.2f}s", 
                        (row['size_label'], row['execution_time']),
                        textcoords="offset points", 
                        xytext=(0, label_offsets.get(algo, 10)), 
                        ha='center',
                        fontweight='bold' if algo == 'Euclidean' else 'normal'
                    )
            
            plt.title('Execution Time vs Dataset Size', fontsize=14)
            plt.xlabel('Dataset Size', fontsize=12)
            plt.ylabel('Average Execution Time (seconds)', fontsize=12)
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "scalability_execution_time.png"), dpi=300)
            plt.savefig(os.path.join(self.dirs['visualizations'], "scalability_execution_time.png"), dpi=300)
            plt.close()
            
            # 2. Log-log plot for time complexity analysis
            plt.figure(figsize=(12, 6))
            
            for algo in algorithms:
                algo_data = combined_results[combined_results['algorithm'] == algo].sort_values('total_dataset_size')
                plt.loglog(
                    algo_data['total_dataset_size'], 
                    algo_data['execution_time'], 
                    'o-', 
                    label=algo,
                    color=self.algorithms[algo]['color'],
                    marker=self.algorithms[algo]['marker'],
                    linewidth=2
                )
                
                # Add data labels
                for i, row in algo_data.iterrows():
                    plt.annotate(
                        f"{row['mentor_size']}m/{row['mentee_size']}m", 
                        (row['total_dataset_size'], row['execution_time']),
                        textcoords="offset points", 
                        xytext=(5, 5), 
                        ha='left',
                        fontsize=8
                    )
            
            # Add reference lines for common complexity classes
            x_range = np.logspace(np.log10(combined_results['total_dataset_size'].min()), 
                                np.log10(combined_results['total_dataset_size'].max()), 100)
            
            # Scale reference lines to be visible on the same plot
            ref_line_scale = combined_results['execution_time'].min() / 10
            
            plt.loglog(x_range, ref_line_scale * np.power(x_range, 1), 'k--', alpha=0.3, label='O(n)')
            plt.loglog(x_range, ref_line_scale * np.power(x_range, 2) / 100, 'k-.', alpha=0.3, label='O(n²)')
            
            plt.title('Time Complexity Analysis (log-log scale)', fontsize=14)
            plt.xlabel('Total Dataset Size (mentors + mentees)', fontsize=12)
            plt.ylabel('Execution Time (seconds)', fontsize=12)
            plt.grid(True, which='both', linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "scalability_complexity.png"), dpi=300)
            plt.savefig(os.path.join(self.dirs['visualizations'], "scalability_complexity.png"), dpi=300)
            plt.close()
            
            # 3. Plot validity vs dataset size
            plt.figure(figsize=(12, 6))
            
            for algo in algorithms:
                algo_data = combined_results[combined_results['algorithm'] == algo].sort_values('total_dataset_size')
                plt.plot(
                    algo_data['size_label'], 
                    algo_data['validity_percentage'], 
                    'o-', 
                    label=algo,
                    color=self.algorithms[algo]['color'],
                    marker=self.algorithms[algo]['marker'],
                    linewidth=2
                )
                
                # Add data labels
                for i, row in algo_data.iterrows():
                    plt.annotate(
                        f"{row['validity_percentage']:.1f}%", 
                        (row['size_label'], row['validity_percentage']),
                        textcoords="offset points", 
                        xytext=(0, 10), 
                        ha='center'
                    )
            
            # Add reference line at 80% validity
            plt.axhline(y=80, color='gray', linestyle='--', alpha=0.7)
            plt.text(0, 81, 'Target Threshold (80%)', fontsize=10)
            
            plt.title('Match Validity vs Dataset Size', fontsize=14)
            plt.xlabel('Dataset Size', fontsize=12)
            plt.ylabel('Validity Percentage (%)', fontsize=12)
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "scalability_validity.png"), dpi=300)
            plt.savefig(os.path.join(self.dirs['visualizations'], "scalability_validity.png"), dpi=300)
            plt.close()
            
            # 4. Create a quality vs speed tradeoff plot
            plt.figure(figsize=(12, 8))
            
            for algo in algorithms:
                algo_data = combined_results[combined_results['algorithm'] == algo].sort_values('total_dataset_size')
                
                plt.scatter(
                    algo_data['execution_time'],
                    algo_data['validity_percentage'],
                    s=100,
                    color=self.algorithms[algo]['color'],
                    marker=self.algorithms[algo]['marker'],
                    label=algo
                )
                
                # Add lines connecting points
                plt.plot(
                    algo_data['execution_time'],
                    algo_data['validity_percentage'],
                    '--',
                    color=self.algorithms[algo]['color'],
                    alpha=0.7
                )
                
                # Add data labels
                for i, row in algo_data.iterrows():
                    plt.annotate(
                        f"{row['size_label']}",
                        (row['execution_time'], row['validity_percentage']),
                        textcoords="offset points", 
                        xytext=(5, 5), 
                        ha='left',
                        fontsize=8
                    )
            
            plt.title('Quality vs Speed Tradeoff Across Dataset Sizes', fontsize=14)
            plt.xlabel('Execution Time (seconds)', fontsize=12)
            plt.ylabel('Validity Percentage (%)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "scalability_tradeoff.png"), dpi=300)
            plt.savefig(os.path.join(self.dirs['visualizations'], "scalability_tradeoff.png"), dpi=300)
            plt.close()
            
            print("✓ Scalability analysis visualizations created")
            
        except Exception as e:
            print(f"✗ Error creating scalability visualizations: {str(e)}")
    
    def run_all_experiments(self):
        """
        Run all experiments in sequence
        """
        print("\n" + "="*70)
        print("STARTING ALGORITHM EVALUATION")
        print("Timestamp:", self.timestamp)
        print("Results directory:", self.base_output_dir)
        print("="*70)
        
        # Run constraint analysis
        constraint_results = self.run_constraint_analysis()
        
        # Run scalability analysis
        scalability_results = self.run_scalability_analysis()
        
        print("\n" + "="*70)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
        print("Results saved to:", self.base_output_dir)
        print("="*70)
        
        # Create summary report
        self._create_summary_report(constraint_results, scalability_results)
        
        return {
            'constraint_results': constraint_results,
            'scalability_results': scalability_results
        }
    
    def _create_summary_report(self, constraint_results, scalability_results):
        """
        Create a summary report of all experiments
        """
        try:
            report_file = os.path.join(self.base_output_dir, f"summary_report_{self.timestamp}.txt")
            
            with open(report_file, 'w') as f:
                f.write("="*80 + "\n")
                f.write("MENTOR-MENTEE MATCHING ALGORITHM EVALUATION SUMMARY\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Configuration Timestamp: {self.timestamp}\n\n")
                
                # Constraint analysis summary
                f.write("-"*80 + "\n")
                f.write("CONSTRAINT ANALYSIS SUMMARY\n")
                f.write("-"*80 + "\n\n")
                
                if isinstance(constraint_results, pd.DataFrame):
                    # Group by algorithm and constraint level
                    grouped = constraint_results.groupby(['algorithm', 'num_constraints'])
                    
                    for (algo, num_constraints), group in grouped:
                        row = group.iloc[0]
                        f.write(f"{algo} with {num_constraints} constraints ({row['constraint_label']}):\n")
                        f.write(f"  Validity: {row['validity_percentage']:.2f}%\n")
                        f.write(f"  Similarity: {row['avg_similarity_score']:.2f}%\n")
                        f.write(f"  Execution Time: {row['execution_time']:.4f}s\n")
                        f.write(f"  Matches: {row['total_matches']:.0f} (Valid: {row['valid_matches']:.0f})\n\n")
                
                # Scalability analysis summary
                f.write("-"*80 + "\n")
                f.write("SCALABILITY ANALYSIS SUMMARY\n")
                f.write("-"*80 + "\n\n")
                
                if isinstance(scalability_results, pd.DataFrame):
                    # Group by algorithm and dataset size
                    grouped = scalability_results.groupby(['algorithm', 'size_label'])
                    
                    for (algo, size), group in grouped:
                        row = group.iloc[0]
                        f.write(f"{algo} with {size} dataset ({row['mentor_size']} mentors, {row['mentee_size']} mentees):\n")
                        f.write(f"  Validity: {row['validity_percentage']:.2f}%\n")
                        f.write(f"  Similarity: {row['avg_similarity_score']:.2f}%\n")
                        f.write(f"  Execution Time: {row['execution_time']:.4f}s\n")
                        f.write(f"  Matches: {row['total_matches']:.0f} (Valid: {row['valid_matches']:.0f})\n\n")
                
                # Overall findings
                f.write("-"*80 + "\n")
                f.write("OVERALL FINDINGS\n")
                f.write("-"*80 + "\n\n")
                
                f.write("1. Best algorithm for quality (highest validity): ")
                # Logic to determine best algorithm goes here
                f.write("\n\n")
                
                f.write("2. Best algorithm for speed (lowest execution time): ")
                # Logic to determine fastest algorithm goes here
                f.write("\n\n")
                
                f.write("3. Best algorithm for scalability: ")
                # Logic to determine most scalable algorithm goes here
                f.write("\n\n")
                
                f.write("4. Best algorithm for constraint handling: ")
                # Logic to determine best algorithm for constraints goes here
                f.write("\n\n")
                
                f.write("-"*80 + "\n")
                f.write("VISUALIZATION FILES\n")
                f.write("-"*80 + "\n\n")
                
                # List all generated visualization files
                viz_files = os.listdir(self.dirs['visualizations'])
                for file in sorted(viz_files):
                    f.write(f"- {file}\n")
                
            print(f"✓ Summary report created: {report_file}")
            
        except Exception as e:
            print(f"✗ Error creating summary report: {str(e)}")

def main():
    """
    Main function to run the comprehensive evaluation
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run comprehensive evaluation of mentor-mentee matching algorithms')
    parser.add_argument('--config', type=str, help='Path to configuration file (JSON)')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--constraints-only', action='store_true', help='Run only the constraint analysis')
    parser.add_argument('--scalability-only', action='store_true', help='Run only the scalability analysis')
    parser.add_argument('--quick-test', action='store_true', help='Run a minimal test to check for errors') 
    parser.add_argument('--visualize-only', action='store_true', help='Only generate visualizations from existing results')

    args = parser.parse_args()
    
    # Initialize the framework
    framework = TestingFramework(args.output_dir)
    
    # Load configuration
    framework.load_config(args.config)

    if args.quick_test:
        # Override configuration with minimal values
        framework.config['constraint_iterations'] = 1
        framework.config['scalability_iterations'] = 1
        # Limit dataset sizes
        framework.config['dataset_sizes'] = [framework.config['dataset_sizes'][0]]  # Just the smallest size
        
    # Run experiments based on command line flags
    if args.constraints_only:
        framework.run_constraint_analysis()
    elif args.scalability_only:
        framework.run_scalability_analysis()
    else:
        # Run all experiments
        framework.run_all_experiments()

if __name__ == "__main__":
    main()