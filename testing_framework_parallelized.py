import os
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import importlib
from multiprocessing import Pool, cpu_count, Manager
import traceback
import sys

sys.path.append(os.getcwd())

# Import your algorithms - ensure these are accessible
from euclideanalgorithm import preprocess_data as euclidean_preprocess, match_mentees_to_mentors as euclidean_match, evaluate_matching as euclidean_evaluate
from kclusteringalgorithm import preprocess_data as kmeans_preprocess, modified_kmeans_matching as kmeans_match, evaluate_matching as kmeans_evaluate
from genetic_algorithm import preprocess_data as ga_preprocess, genetic_algorithm as ga_match, evaluate_matching as ga_evaluate
from DefferedAcceptance_algorithm import preprocess_data as DAcceptance_preprocess, deferred_acceptance_matching as DAcceptance_match, evaluate_matching as DAcceptance_evaluate

def run_single_iteration(args):
    """
    Run a single iteration of the matching algorithm.
    This function will be called by each worker process.
    
    Args:
        args: Tuple containing (
            algo_type: Type of algorithm ('Euclidean', 'KMeans', 'GA')
            iteration: Iteration number for random seed
            mentors_processed: Preprocessed mentors dataframe
            mentees_processed: Preprocessed mentees dataframe
            exact_match_mentors: For exact match calculation (GA only)
            exact_match_mentees: For exact match calculation (GA only)
            features: List of features (GA only)
            similarity_matrix: Precomputed similarity matrix (GA only)
            constraints: Dictionary of constraints to apply
            kwargs: Additional algorithm-specific parameters
        )
        
    Returns:
        Dictionary containing evaluation results for this iteration
    """
    try:
        # Unpack arguments
        (algo_type, iteration, mentors_processed, mentees_processed, 
         exact_match_mentors, exact_match_mentees, features, similarity_matrix,
         constraints, kwargs) = args
        
        # Set random seed based on iteration
        np.random.seed(iteration)
        
        # Shuffle data for this iteration
        shuffled_mentors = mentors_processed.sample(frac=1, random_state=iteration).reset_index(drop=True)
        shuffled_mentees = mentees_processed.sample(frac=1, random_state=iteration+1000).reset_index(drop=True)
        
        # Measure execution time
        start_time = time.time()
        
        # Run the appropriate algorithm
        if algo_type == 'Euclidean':
            matches_df = euclidean_match(
                shuffled_mentors, 
                shuffled_mentees,
                max_mentees_per_mentor=constraints.get('max_mentees_per_mentor', 2),
                constraints=constraints
            )
            results = euclidean_evaluate(matches_df)
            
        elif algo_type == 'KMeans':
            matches_df = kmeans_match(
                shuffled_mentors, 
                shuffled_mentees,
                max_mentees_per_mentor=constraints.get('max_mentees_per_mentor', 2),
                constraints=constraints
            )
            results = kmeans_evaluate(matches_df)
            
        elif algo_type == 'GA':
            # GA requires additional parameters from kwargs
            matches_df = ga_match(
                shuffled_mentors, shuffled_mentees, 
                exact_match_mentors, exact_match_mentees, features,
                similarity_matrix, constraints,
                **kwargs
            )
            results = ga_evaluate(matches_df)

        elif algo_type == 'DAcceptance':
            matches_df = DAcceptance_match(
                shuffled_mentors, 
                shuffled_mentees,
                max_mentees_per_mentor=constraints.get('max_mentees_per_mentor', 2),
                constraints=constraints
            )
            results = DAcceptance_evaluate(matches_df)
        
        
        # Calculate execution time
        execution_time = time.time() - start_time
        results['execution_time'] = execution_time
        results['iteration'] = iteration
        
        return results
        
    except Exception as e:
        # Return error information
        return {
            'error': True,
            'iteration': args[1] if len(args) > 1 else 'unknown',
            'message': str(e),
            'traceback': traceback.format_exc()
        }

# Add this function to optimize batch processing
def run_batched_iterations(algo_type, mentors_processed, mentees_processed, 
                         exact_match_mentors, exact_match_mentees, features, 
                         similarity_matrix, constraints, iterations, batch_size=5, **kwargs):
    """
    Run multiple iterations in a single process to reduce overhead.
    """
    results = []
    
    for i in range(iterations):
        # Set random seed based on iteration
        np.random.seed(i)
        
        # Shuffle data for this iteration - avoid full copy when possible
        shuffled_mentors = mentors_processed.sample(frac=1, random_state=i).reset_index(drop=True)
        shuffled_mentees = mentees_processed.sample(frac=1, random_state=i+1000).reset_index(drop=True)
        
        # Measure execution time
        start_time = time.time()
        
        try:
            # Run the appropriate algorithm with optimized settings
            if algo_type == 'GA':
                # For GA, use fewer generations and smaller population for faster runs
                ga_kwargs = kwargs.copy()
                if 'generations' in ga_kwargs and ga_kwargs['generations'] > 10:
                    # Reduce generations for large iterations
                    if iterations > 100:
                        ga_kwargs['generations'] = max(2, ga_kwargs['generations'] // 10)
                
                matches_df = ga_match(
                    shuffled_mentors, shuffled_mentees, 
                    exact_match_mentors, exact_match_mentees, features,
                    similarity_matrix, constraints,
                    **ga_kwargs
                )
                results_dict = ga_evaluate(matches_df)
            
            elif algo_type == 'DAcceptance':
                matches_df = DAcceptance_match(
                    shuffled_mentors, 
                    shuffled_mentees,
                    max_mentees_per_mentor=constraints.get('max_mentees_per_mentor', 2),
                    constraints=constraints
                )
                results_dict = DAcceptance_evaluate(matches_df)
            
            # Execution time and metadata
            execution_time = time.time() - start_time
            results_dict['execution_time'] = execution_time
            results_dict['iteration'] = i
            results.append(results_dict)
            
        except Exception as e:
            print(f"Error in iteration {i}: {str(e)}")
            results.append({
                'error': True,
                'iteration': i,
                'message': str(e)
            })
    
    return results
  
def worker_batch(args):
    """
    Worker function to process a batch of iterations in a single process.
    Takes a tuple of arguments to work with multiprocessing.
    
    Args:
        args: Tuple containing (
            batch_iters: Number of iterations to perform
            algo_type: Algorithm type ('GA', 'DAcceptance', etc.)
            mentors_processed: Preprocessed mentors dataframe
            mentees_processed: Preprocessed mentees dataframe
            exact_match_mentors: For exact match calculation
            exact_match_mentees: For exact match calculation
            features: List of features
            similarity_matrix: Precomputed similarity matrix
            constraints: Dictionary of constraints to apply
            kwargs: Additional algorithm-specific parameters
        )
        
    Returns:
        List of results from each iteration
    """
    batch_iters, algo_type, mentors_processed, mentees_processed, \
    exact_match_mentors, exact_match_mentees, features, \
    similarity_matrix, constraints, kwargs = args
    
    return run_batched_iterations(
        algo_type, mentors_processed, mentees_processed,
        exact_match_mentors, exact_match_mentees, features,
        similarity_matrix, constraints, batch_iters, **kwargs
    )

def run_parallel_algorithm(algo_type, mentors_file, mentees_file, iterations=1000, 
                         mentor_sample_size=500, mentee_sample_size=1000, 
                         constraints=None, num_processes=None, **kwargs):
    """
    Run algorithm in parallel across multiple processes.
    
    Args:
        algo_type: Type of algorithm ('Euclidean', 'KMeans', 'GA', 'DAcceptance')
        mentors_file: Path to mentors dataset CSV
        mentees_file: Path to mentees dataset CSV
        iterations: Number of randomized iterations to run
        mentor_sample_size: Number of mentors to sample
        mentee_sample_size: Number of mentees to sample
        constraints: Dictionary of constraints to apply
        num_processes: Number of processes to use (default: cpu_count)
        **kwargs: Additional algorithm-specific parameters
        
    Returns:
        Dictionary containing aggregated results
    """
    # Default constraints if none provided
    if constraints is None:
        constraints = {
            'max_mentees_per_mentor': 2,
            'min_similarities': 2
        }
    
    # Set number of processes
    if num_processes is None:
        num_processes = cpu_count()
    
    print(f"\nRunning {algo_type} algorithm with {iterations} iterations using {num_processes} processes...")
    print(f"Applied constraints: {', '.join(constraints.keys())}")
    
    # Start overall timing
    overall_start_time = time.time()
    
    # Preprocess data based on algorithm type
    if algo_type == 'Euclidean':
        mentors_processed, mentees_processed = euclidean_preprocess(
            mentors_file, mentees_file, 
            mentor_sample_size=mentor_sample_size, 
            mentee_sample_size=mentee_sample_size
        )
        # These are not used by Euclidean but needed for consistent arguments
        exact_match_mentors = None
        exact_match_mentees = None
        features = None 
        similarity_matrix = None
        
    elif algo_type == 'KMeans':
        mentors_processed, mentees_processed = kmeans_preprocess(
            mentors_file, mentees_file, 
            mentor_sample_size=mentor_sample_size, 
            mentee_sample_size=mentee_sample_size
        )
        # These are not used by KMeans but needed for consistent arguments
        exact_match_mentors = None
        exact_match_mentees = None
        features = None
        similarity_matrix = None
    
    elif algo_type == 'GA':
        # GA requires additional preprocessing
        mentors_processed, mentees_processed, exact_match_mentors, exact_match_mentees, features = ga_preprocess(
            mentors_file, mentees_file,
            mentor_sample_size=mentor_sample_size,
            mentee_sample_size=mentee_sample_size
        )
        
        # Calculate similarity matrix
        from sklearn.metrics.pairwise import euclidean_distances
        features_list = [col for col in mentors_processed.columns if col != 'User_ID']
        mentor_array = mentors_processed[features_list].values
        mentee_array = mentees_processed[features_list].values
        distances = euclidean_distances(mentee_array, mentor_array)
        max_distance = np.max(distances)
        similarity_matrix = 1 - (distances / max_distance)

    elif algo_type == 'DAcceptance':
        # DAcceptance has simpler preprocessing than GA
        mentors_processed, mentees_processed = DAcceptance_preprocess(
            mentors_file, mentees_file,
            mentor_sample_size=mentor_sample_size,
            mentee_sample_size=mentee_sample_size
        )
        
        # For consistency with other algorithms, set these to None
        exact_match_mentors = None
        exact_match_mentees = None
        features = [col for col in mentors_processed.columns if col != 'User_ID']  # Extract features from processed data
        similarity_matrix = None
    
    # Different processing for GA and DAcceptance vs other algorithms
    if algo_type in ['GA', 'DAcceptance']:
        # Adjust parameters for GA based on dataset size
        if algo_type == 'GA' and 'generations' in kwargs:
            if iterations > 100 or mentor_sample_size > 1000:
                # Reduce generations for large tests to improve performance
                kwargs['generations'] = min(kwargs['generations'], 2)
                kwargs['pop_size'] = min(kwargs.get('pop_size', 100), 50)
                print(f"Adjusted GA parameters for large test: {kwargs}")
    
        # Distribute iterations across processes
        iters_per_process = iterations // num_processes
        remainder = iterations % num_processes
        
        iter_batches = []
        for i in range(num_processes):
            if i < remainder:
                process_iters = iters_per_process + 1
            else:
                process_iters = iters_per_process
                
            if process_iters > 0:
                iter_batches.append(process_iters)
        
        # Prepare arguments for worker_batch
        worker_args = [
            (batch_iters, algo_type, mentors_processed, mentees_processed,
             exact_match_mentors, exact_match_mentees, features,
             similarity_matrix, constraints, kwargs)
            for batch_iters in iter_batches
        ]
        
        # Use multiprocessing to run batches in parallel
        with Pool(processes=num_processes) as pool:
            batch_results = list(pool.map(worker_batch, worker_args))
            
        # Flatten results from all batches
        all_results = [item for sublist in batch_results for item in sublist]
    else:
        # Original code for other algorithms
        iter_args = [
            (algo_type, i, mentors_processed, mentees_processed, 
            exact_match_mentors, exact_match_mentees, features, similarity_matrix,
            constraints, kwargs) 
            for i in range(iterations)
        ]
        
        # Use multiprocessing to run iterations in parallel
        with Pool(processes=num_processes) as pool:
            # Run the iterations and collect results
            all_results = list(pool.imap_unordered(run_single_iteration, iter_args))
    
    # Check for errors
    errors = [r for r in all_results if 'error' in r and r['error']]
    if errors:
        print(f"⚠️ {len(errors)} iterations failed. First error: {errors[0]['message']}")
    
    # Filter out errors and convert to DataFrame
    valid_results = [r for r in all_results if 'error' not in r or not r['error']]
    results_df = pd.DataFrame(valid_results)
    
    # Calculate averages across all iterations
    avg_validity = results_df['validity_percentage'].mean()
    avg_similarity = results_df['avg_similarity_score'].mean()
    avg_execution_time = results_df['execution_time'].mean()
    
    # Calculate total valid matches and total matches
    total_valid_matches = results_df['valid_matches'].sum()
    total_matches = results_df['total_matches'].sum()
    
    # Overall execution time
    overall_execution_time = time.time() - overall_start_time
    
    print(f"  ✓ Completed in {overall_execution_time:.2f}s")
    print(f"  ✓ Average execution time per iteration: {avg_execution_time:.4f}s")
    print(f"  ✓ Validity: {avg_validity:.2f}%, Similarity: {avg_similarity:.2f}%")
    
    return {
        'avg_validity': avg_validity,
        'avg_similarity': avg_similarity,
        'avg_execution_time': avg_execution_time,
        'total_valid_matches': total_valid_matches,
        'total_matches': total_matches,
        'all_results': results_df,
        'overall_execution_time': overall_execution_time
    }

class ParallelTestingFramework:
    def __init__(self, base_output_dir="results", num_processes=None):
        """
        Initialize the parallel testing framework
        """
        self.base_output_dir = base_output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config = None
        self.num_processes = num_processes if num_processes else cpu_count()
        
        print(f"Initializing Parallel Testing Framework with {self.num_processes} processes")
        
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
                'color': '#4285F4',  # Google Blue
                'marker': 'o',
                'markersize': 10,    # Larger marker
                'linestyle': '-',    # Solid line
                'linewidth': 3,      # Thicker line
                'zorder': 10         # Draw on top of other lines
            },
            'KMeans': {
                'color': '#EA4335',  # Google Red
                'marker': 's',
                'markersize': 8,
                'linestyle': '--',   # Dashed line
                'linewidth': 2,
                'zorder': 5
            },
            'GA': {
                'color': '#34A853',  # Google Green
                'marker': '^',
                'markersize': 8,
                'linestyle': ':',    # Dotted line
                'linewidth': 2,
                'zorder': 1,
                'default_params': {
                    'generations': 2,      
                    'pop_size': 100,
                    'crossover_rate': 0.9,
                    'mutation_rate': 0.01,
                    'tournament_size': 5
                }
            },
            'DAcceptance': {
                'color': '#FBBC04',  # Google Yellow
                'marker': 'd',
                'markersize': 8,
                'linestyle': '-.',
                'linewidth': 2,
                'zorder': 3
            }
        }
        
        # Define default configurations
        self.default_config = {
            # Basic run configuration
            'mentors_file': 'mentors_dataset.csv',
            'mentees_file': 'mentees_dataset.csv',
            'constraint_iterations': 1,  # Number of iterations for each constraint configuration
            'scalability_iterations': 1,  # Number of iterations for each dataset size
            
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
        Run all algorithms with varying constraints using parallel processing
        """
        print("\n" + "="*70)
        print("RUNNING CONSTRAINT ANALYSIS WITH PARALLEL PROCESSING")
        print("="*70)
        
        # Create output directory for constraint results
        output_dir = os.path.join(self.dirs['constraints'], f"constraints_{self.timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        all_results = []
        
        # For each algorithm
        for algo_name in self.algorithms.keys():
            # For each constraint configuration
            for constraint_config in self.config['constraints']:
                constraint_label = constraint_config['label']
                constraints = constraint_config['constraints']
                
                print(f"\nRunning {algo_name} with {constraint_label}...")
                
                # Parameters specific to GA
                kwargs = {}
                if algo_name == 'GA':
                    kwargs = self.algorithms['GA']['default_params']
                
                try:
                    # Run the algorithm in parallel
                    results = run_parallel_algorithm(
                        algo_name,
                        self.config['mentors_file'],
                        self.config['mentees_file'],
                        iterations=self.config['constraint_iterations'],
                        constraints=constraints,
                        num_processes=self.num_processes,
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
                        'valid_matches': results['total_valid_matches'] / self.config['constraint_iterations'],
                        'overall_execution_time': results['overall_execution_time']
                    }
                    
                    all_results.append(constraint_result)
                    
                except Exception as e:
                    print(f"  ✗ Error: {str(e)}")
                    traceback.print_exc()
        
        # Combine all results and save
        combined_df = pd.DataFrame(all_results)
        combined_df.to_csv(os.path.join(output_dir, "all_constraints_results.csv"), index=False)
        
        return combined_df
        
    def _create_constraint_visualizations(self, combined_results, output_dir):
      """
      Create visualizations showing impact of constraints with improved visibility
      """
      try:
          import matplotlib.pyplot as plt
          import numpy as np
  
         
          
          # Update algorithm styles to be more professional
          for algo in self.algorithms:
              self.algorithms[algo].update({
                  'marker': None,  # Remove markers
                  'linestyle': {
                      'Euclidean': ':',
                      'KMeans': '--', 
                      'GA': '-.', 
                      'DAcceptance': '-'
                  }[algo]
              })
          
          # Group data by algorithm and constraint
          algorithms = combined_results['algorithm'].unique()
          
          # 1. Plot validity vs number of constraints - with improved line visibility
          plt.figure(figsize=(12, 6))
          
          # Adjust z-order to ensure all lines are visible (reverse algorithm order)
          alg_z_orders = {algo: (10 - i) for i, algo in enumerate(algorithms)}
          
          for algo in algorithms:
              algo_data = combined_results[combined_results['algorithm'] == algo]
              
              # Apply consistent line widths and transparency
              plt.plot(
                  algo_data['num_constraints'], 
                  algo_data['validity_percentage'],
                  label=algo,
                  color=self.algorithms[algo]['color'],
                  marker=self.algorithms[algo]['marker'], 
                  markersize=10,  # Consistent larger marker size
                  linestyle=self.algorithms[algo].get('linestyle', '-'),
                  linewidth=2,    # Consistent line width
                  alpha=0.85,     # Add slight transparency
                  zorder=alg_z_orders[algo]  # Adjust z-order to prevent overlapping
              )
              
              # Add data labels with small offset based on algorithm
              y_offset = 7 + algorithms.tolist().index(algo) * 5  # Stagger labels vertically
              
              for i, row in algo_data.iterrows():
                  plt.annotate(
                      f"{row['validity_percentage']:.1f}%", 
                      (row['num_constraints'], row['validity_percentage']),
                      textcoords="offset points", 
                      xytext=(0, y_offset), 
                      ha='center',
                      fontweight='normal'
                  )
          
          plt.title('Impact of Constraints on Validity Percentage', fontsize=14)
          plt.xlabel('Number of Constraints', fontsize=12)
          plt.ylabel('Validity Percentage (%)', fontsize=12)
          
          # Get unique constraint levels sorted
          constraint_levels = sorted(combined_results['num_constraints'].unique())

          # Create labels from the constraint label column
          constraint_df = combined_results.drop_duplicates(['num_constraints', 'constraint_label'])
          constraint_df = constraint_df.sort_values('num_constraints')
          constraint_labels = [f"{row['constraint_label']}" for _, row in constraint_df.iterrows()]

          # Use the dynamic labels
          plt.xticks(constraint_levels, constraint_labels, rotation=45)
          plt.legend(loc='lower left')
          plt.tight_layout()
          plt.savefig(os.path.join(output_dir, "constraints_validity_impact.png"), dpi=300)
          plt.savefig(os.path.join(self.dirs['visualizations'], "constraints_validity_impact.png"), dpi=300)
          plt.close()
          
          # 2. Plot similarity score vs number of constraints
          plt.figure(figsize=(12, 6))
          
          for algo in algorithms:
              algo_data = combined_results[combined_results['algorithm'] == algo]
              
              # Apply consistent line widths and transparency
              plt.plot(
                  algo_data['num_constraints'], 
                  algo_data['avg_similarity_score'],
                  label=algo,
                  color=self.algorithms[algo]['color'],
                  marker=self.algorithms[algo]['marker'], 
                  markersize=10,  # Consistent larger marker size
                  linestyle=self.algorithms[algo].get('linestyle', '-'),
                  linewidth=2,    # Consistent line width
                  alpha=0.85,     # Add slight transparency
                  zorder=alg_z_orders[algo]  # Adjust z-order to prevent overlapping
              )
              
              # Add data labels with small offset based on algorithm
              y_offset = 7 + algorithms.tolist().index(algo) * 5  # Stagger labels vertically
              
              for i, row in algo_data.iterrows():
                  plt.annotate(
                      f"{row['avg_similarity_score']:.1f}%", 
                      (row['num_constraints'], row['avg_similarity_score']),
                      textcoords="offset points", 
                      xytext=(0, y_offset), 
                      ha='center',
                      fontweight='normal'
                  )
          
          plt.title('Impact of Constraints on Similarity Score', fontsize=14)
          plt.xlabel('Number of Constraints', fontsize=12)
          plt.ylabel('Average Similarity Score (%)', fontsize=12)
          
          # Get unique constraint levels sorted
          constraint_levels = sorted(combined_results['num_constraints'].unique())

          # Create labels from the constraint label column
          constraint_df = combined_results.drop_duplicates(['num_constraints', 'constraint_label'])
          constraint_df = constraint_df.sort_values('num_constraints')
          constraint_labels = [f"{row['constraint_label']}" for _, row in constraint_df.iterrows()]

          # Use the dynamic labels
          plt.xticks(constraint_levels, constraint_labels, rotation=45)
          plt.legend(loc='lower left')
          plt.tight_layout()
          plt.savefig(os.path.join(output_dir, "constraints_similarity_impact.png"), dpi=300)
          plt.savefig(os.path.join(self.dirs['visualizations'], "constraints_similarity_impact.png"), dpi=300)
          plt.close()
          
          # 3. Plot execution time vs number of constraints
          plt.figure(figsize=(12, 6))
          
          for algo in algorithms:
              algo_data = combined_results[combined_results['algorithm'] == algo]
              
              # Apply consistent styling with improved visibility
              plt.plot(
                  algo_data['num_constraints'], 
                  algo_data['execution_time'],
                  label=algo,
                  color=self.algorithms[algo]['color'],
                  marker=self.algorithms[algo]['marker'], 
                  markersize=10,
                  linestyle=self.algorithms[algo].get('linestyle', '-'),
                  linewidth=2,
                  alpha=0.85,
                  zorder=alg_z_orders[algo]
              )
              
              # Add data labels with staggered vertical position
              y_offset = 7 + algorithms.tolist().index(algo) * 5
              
              for i, row in algo_data.iterrows():
                  plt.annotate(
                      f"{row['execution_time']:.2f}s", 
                      (row['num_constraints'], row['execution_time']),
                      textcoords="offset points", 
                      xytext=(0, y_offset), 
                      ha='center',
                      fontweight='normal'
                  )
          
          plt.title('Impact of Constraints on Execution Time', fontsize=14)
          
          # Get unique constraint levels sorted
          constraint_levels = sorted(combined_results['num_constraints'].unique())

          # Create labels from the constraint label column
          constraint_df = combined_results.drop_duplicates(['num_constraints', 'constraint_label'])
          constraint_df = constraint_df.sort_values('num_constraints')
          constraint_labels = [f"{row['constraint_label']}" for _, row in constraint_df.iterrows()]

          # Use the dynamic labels
          plt.xticks(constraint_levels, constraint_labels, rotation=45)
          plt.legend()
          plt.tight_layout()
          plt.savefig(os.path.join(output_dir, "constraints_execution_time.png"), dpi=300)
          plt.savefig(os.path.join(self.dirs['visualizations'], "constraints_execution_time.png"), dpi=300)
          plt.close()
          
          # 4. Plot match coverage vs number of constraints
          plt.figure(figsize=(12, 6))
          
          for algo in algorithms:
              algo_data = combined_results[combined_results['algorithm'] == algo]
              # Use total dataset size from config as denominator
              total_mentees = 1000  # Default from config
              match_rates = (algo_data['total_matches'] / total_mentees) * 100 
              
              plt.plot(
                  algo_data['num_constraints'], 
                  match_rates, 
                  label=algo,
                  color=self.algorithms[algo]['color'],
                  marker=self.algorithms[algo]['marker'],
                  markersize=10,
                  linestyle=self.algorithms[algo].get('linestyle', '-'),
                  linewidth=2,
                  alpha=0.85,
                  zorder=alg_z_orders[algo]
              )
              
              # Add data labels with staggered vertical position
              y_offset = 7 + algorithms.tolist().index(algo) * 5
              
              for i, (x, y) in enumerate(zip(algo_data['num_constraints'], match_rates)):
                  plt.annotate(
                      f"{y:.1f}%", 
                      (x, y),
                      textcoords="offset points", 
                      xytext=(0, y_offset), 
                      ha='center',
                      fontweight='normal'
                  )
          
          plt.title('Impact of Constraints on Match Coverage', fontsize=14)
          plt.xlabel('Number of Constraints', fontsize=12)
          plt.ylabel('Percentage of Mentees Matched (%)', fontsize=12)
          
          # Get unique constraint levels sorted
          constraint_levels = sorted(combined_results['num_constraints'].unique())

          # Create labels from the constraint label column
          constraint_df = combined_results.drop_duplicates(['num_constraints', 'constraint_label'])
          constraint_df = constraint_df.sort_values('num_constraints')
          constraint_labels = [f"{row['constraint_label']}" for _, row in constraint_df.iterrows()]

          # Use the dynamic labels
          plt.xticks(constraint_levels, constraint_labels, rotation=45)
          plt.legend()
          plt.tight_layout()
          plt.savefig(os.path.join(output_dir, "constraints_match_coverage.png"), dpi=300)
          plt.savefig(os.path.join(self.dirs['visualizations'], "constraints_match_coverage.png"), dpi=300)
          plt.close()
          
          # 5.
          # 5. Quality vs Speed Tradeoff Plot
          plt.figure(figsize=(12, 8))

          # Color palette
          color_palette = {
              'Euclidean': '#1E90FF',  # Dodger Blue
              'KMeans': '#FF6347',     # Tomato Red
              'GA': '#2E8B57',         # Sea Green
              'DAcceptance': '#FFD700'    # Gold
          }

          plt.figure(figsize=(12, 8))
          plt.grid(True, linestyle='--', alpha=0.5)

          for algo in algorithms:
              algo_data = combined_results[combined_results['algorithm'] == algo]
              
              # Plot a line connecting the points
              plt.plot(
                  algo_data['execution_time'], 
                  algo_data['validity_percentage'], 
                  color=color_palette[algo],
                  linestyle='-',
                  linewidth=2,
                  alpha=0.7,
                  label=algo
              )

              # Annotate each point with number of constraints
              for i, row in algo_data.iterrows():
                  plt.annotate(
                      f"{int(row['num_constraints'])} constraints", 
                      (row['execution_time'], row['validity_percentage']),
                      xytext=(5, 5),
                      textcoords='offset points',
                      fontsize=8,
                      color=color_palette[algo],
                      alpha=0.7
                  )

          plt.title('Algorithm Performance: Execution Time vs Validity', fontsize=14)
          plt.xlabel('Execution Time (seconds)', fontsize=12)
          plt.ylabel('Validity Percentage (%)', fontsize=12)
          plt.legend()
          plt.tight_layout()

          plt.savefig(os.path.join(output_dir, "constraints_quality_speed_tradeoff.png"), dpi=300)
          plt.savefig(os.path.join(self.dirs['visualizations'], "constraints_quality_speed_tradeoff.png"), dpi=300)
          plt.close()
          
          print("✓ Constraint analysis visualizations created")
      
      except Exception as e:
          print(f"✗ Error creating constraint visualizations: {str(e)}")
          traceback.print_exc()
        
    def run_scalability_analysis(self):
        """
        Run all algorithms with varying dataset sizes using parallel processing
        """
        print("\n" + "="*70)
        print("RUNNING SCALABILITY ANALYSIS WITH PARALLEL PROCESSING")
        print("="*70)
        
        # Create output directory for scalability results
        output_dir = os.path.join(self.dirs['scalability'], f"scalability_{self.timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get constraints from the first (most basic) configuration
        basic_constraints = self.config['constraints'][0]['constraints']
        print(f"Using basic constraints ({len(basic_constraints)} constraints) for scalability testing")
        
        all_results = []
        
        # For each algorithm
        for algo_name in self.algorithms.keys():
            # For each dataset size
            for size_config in self.config['dataset_sizes']:
                mentor_size = size_config['mentors']
                mentee_size = size_config['mentees']
                size_label = size_config['label']
                
                print(f"\nRunning {algo_name} with dataset size {size_label}...")
                
                # Parameters specific to GA
                kwargs = {}
                if algo_name == 'GA':
                    kwargs = self.algorithms['GA']['default_params']
                
                try:
                    # Run the algorithm in parallel
                    results = run_parallel_algorithm(
                        algo_name,
                        self.config['mentors_file'],
                        self.config['mentees_file'],
                        iterations=self.config['scalability_iterations'],
                        mentor_sample_size=mentor_size,
                        mentee_sample_size=mentee_size,
                        constraints=basic_constraints,
                        num_processes=self.num_processes,
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
                        'total_dataset_size': mentor_size + mentee_size,
                        'overall_execution_time': results['overall_execution_time']
                    }
                    
                    all_results.append(size_result)
                    
                except Exception as e:
                    print(f"  ✗ Error: {str(e)}")
                    traceback.print_exc()
        
        # Combine all results and save
        combined_df = pd.DataFrame(all_results)
        combined_df.to_csv(os.path.join(output_dir, "all_scalability_results.csv"), index=False)
        
        return combined_df
    
    def _create_scalability_visualizations(self, combined_results, output_dir):
        """
        Create visualizations showing impact of dataset size with improved visibility
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
           
            # Group data by algorithm
            algorithms = combined_results['algorithm'].unique()
            
            # Adjust z-order to ensure all lines are visible (reverse algorithm order)
            alg_z_orders = {algo: (10 - i) for i, algo in enumerate(algorithms)}
            
            # 1. Plot execution time vs dataset size - with improved visibility
            plt.figure(figsize=(12, 6))
            
            for algo in algorithms:
                algo_data = combined_results[combined_results['algorithm'] == algo].sort_values('total_dataset_size')
                plt.plot(
                    algo_data['size_label'],
                    algo_data['execution_time'],
                    label=algo,
                    color=self.algorithms[algo]['color'],
                    marker=self.algorithms[algo]['marker'],
                    markersize=10,  # Consistent larger marker size
                    linestyle=self.algorithms[algo].get('linestyle', '-'),
                    linewidth=2,    # Consistent line width (make blue line thinner)
                    alpha=0.85,     # Add slight transparency
                    zorder=alg_z_orders[algo]  # Adjust z-order to prevent overlapping
                )
                
                # Add data labels with different vertical positions - larger offset between algorithms
                y_offset = 7 + algorithms.tolist().index(algo) * 7
                
                for i, row in algo_data.iterrows():
                    plt.annotate(
                        f"{row['execution_time']:.2f}s", 
                        (row['size_label'], row['execution_time']),
                        textcoords="offset points", 
                        xytext=(0, y_offset), 
                        ha='center',
                        fontweight='normal'
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
            
            # 2. Log-log plot for time complexity analysis - with improved visibility
            plt.figure(figsize=(12, 6))
            
            for algo in algorithms:
                algo_data = combined_results[combined_results['algorithm'] == algo].sort_values('total_dataset_size')
                plt.loglog(
                    algo_data['total_dataset_size'], 
                    algo_data['execution_time'], 
                    label=algo,
                    color=self.algorithms[algo]['color'],
                    marker=self.algorithms[algo]['marker'],
                    markersize=10,
                    linewidth=2,
                    alpha=0.85,
                    zorder=alg_z_orders[algo]
                )
                
                # Add data labels with slight offset based on algorithm
                x_offset = 0.02 + algorithms.tolist().index(algo) * 0.01
                y_offset = 0.05 + algorithms.tolist().index(algo) * 0.05
                
                for i, row in algo_data.iterrows():
                    plt.annotate(
                        f"{row['mentor_size']}m/{row['mentee_size']}m", 
                        (row['total_dataset_size'] * (1 + x_offset), row['execution_time'] * (1 + y_offset)),
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8)
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
            
            # 3. Plot validity vs dataset size - with improved visibility
            plt.figure(figsize=(12, 6))
            
            for algo in algorithms:
                algo_data = combined_results[combined_results['algorithm'] == algo].sort_values('total_dataset_size')
                plt.plot(
                    algo_data['size_label'], 
                    algo_data['validity_percentage'], 
                    label=algo,
                    color=self.algorithms[algo]['color'],
                    marker=self.algorithms[algo]['marker'],
                    markersize=10,
                    linestyle=self.algorithms[algo].get('linestyle', '-'),
                    linewidth=2,
                    alpha=0.85,
                    zorder=alg_z_orders[algo]
                )
                
                # Add data labels with staggered vertical position
                y_offset = 7 + algorithms.tolist().index(algo) * 7
                
                for i, row in algo_data.iterrows():
                    plt.annotate(
                        f"{row['validity_percentage']:.1f}%", 
                        (row['size_label'], row['validity_percentage']),
                        textcoords="offset points", 
                        xytext=(0, y_offset), 
                        ha='center',
                        fontweight='normal'
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
            
            # 4. Create a quality vs speed tradeoff scatter plot - with improved visibility
            plt.figure(figsize=(12, 8))
            
            # Define marker sizes based on total matches
            max_total_matches = combined_results['total_matches'].max()
            
            for algo in algorithms:
                algo_data = combined_results[combined_results['algorithm'] == algo].sort_values('total_dataset_size')
                
                # Calculate marker sizes based on total matches
                marker_sizes = (algo_data['total_matches'] / max_total_matches) * 300 + 100
                
                plt.scatter(
                    algo_data['execution_time'],
                    algo_data['validity_percentage'],
                    s=marker_sizes,
                    color=self.algorithms[algo]['color'],
                    marker=self.algorithms[algo]['marker'],
                    label=algo,
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=1,
                    zorder=5
                )
                
                # Add lines connecting points with arrow to show size progression
                for i in range(len(algo_data) - 1):
                    plt.annotate('', 
                        xy=(algo_data['execution_time'].iloc[i+1], algo_data['validity_percentage'].iloc[i+1]),
                        xytext=(algo_data['execution_time'].iloc[i], algo_data['validity_percentage'].iloc[i]),
                        arrowprops=dict(arrowstyle='->', color=self.algorithms[algo]['color'], 
                                        alpha=0.7, linewidth=1.5),
                        zorder=1
                    )
                
                # Add data labels showing dataset size
                for i, row in algo_data.iterrows():
                    plt.annotate(
                        f"{row['size_label']}",
                        (row['execution_time'], row['validity_percentage']),
                        textcoords="offset points", 
                        xytext=(5, 5), 
                        ha='left',
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                    )
            
            plt.title('Quality vs Speed Tradeoff Across Dataset Sizes', fontsize=14)
            plt.xlabel('Execution Time (seconds)', fontsize=12)
            plt.ylabel('Validity Percentage (%)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add a legend for marker size
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                    markersize=8, label='Smaller match count'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                    markersize=16, label='Larger match count')
            ]
            
            # Combine algorithm legend with marker size legend
            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend(handles=handles + legend_elements, loc='best')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "scalability_tradeoff.png"), dpi=300)
            plt.savefig(os.path.join(self.dirs['visualizations'], "scalability_tradeoff.png"), dpi=300)
            plt.close()
            
            print("✓ Scalability analysis visualizations created")
            
        except Exception as e:
            print(f"✗ Error creating scalability visualizations: {str(e)}")
            traceback.print_exc()
        
    def run_all_experiments(self):
        """
        Run all experiments in sequence with parallel processing
        """
        print("\n" + "="*70)
        print(f"STARTING ALGORITHM EVALUATION WITH {self.num_processes} PARALLEL PROCESSES")
        print("Timestamp:", self.timestamp)
        print("Results directory:", self.base_output_dir)
        print("="*70)
        
        # Run constraint analysis
        constraint_results = self.run_constraint_analysis()
        
        # Run scalability analysis
        scalability_results = self.run_scalability_analysis()
        
        # Create constraint visualizations
        constraint_output_dir = os.path.join(self.dirs['constraints'], f"constraints_{self.timestamp}")
        self._create_constraint_visualizations(constraint_results, constraint_output_dir)
        
        # Create scalability visualizations
        scalability_output_dir = os.path.join(self.dirs['scalability'], f"scalability_{self.timestamp}")
        self._create_scalability_visualizations(scalability_results, scalability_output_dir)
        
        print("\n" + "="*70)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
        print("Results saved to:", self.base_output_dir)
        print("="*70)
        
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
    Main function to run the parallelized evaluation
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run parallelized evaluation of mentor-mentee matching algorithms')
    parser.add_argument('--config', type=str, help='Path to configuration file (JSON)')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--constraints-only', action='store_true', help='Run only the constraint analysis')
    parser.add_argument('--scalability-only', action='store_true', help='Run only the scalability analysis')
    parser.add_argument('--quick-test', action='store_true', help='Run a minimal test to check for errors')
    parser.add_argument('--processes', type=int, default=None, help='Number of processes to use (default: CPU count)')
    
    args = parser.parse_args()
    
    # Initialize the framework
    framework = ParallelTestingFramework(args.output_dir, args.processes)
    
    # Load configuration
    framework.load_config(args.config)

    if args.quick_test:
        # Override configuration with minimal values
        framework.config['constraint_iterations'] = 10
        framework.config['scalability_iterations'] = 10
        # Limit dataset sizes and constraints
        framework.config['dataset_sizes'] = [framework.config['dataset_sizes'][0]]  # Just the smallest size
        framework.config['constraints'] = [framework.config['constraints'][0]]  # Just the basic constraints
        
    # Run experiments based on command line flags
    if args.constraints_only:
        constraint_results = framework.run_constraint_analysis()
        # Create constraint visualizations
        constraint_output_dir = os.path.join(framework.dirs['constraints'], f"constraints_{framework.timestamp}")
        framework._create_constraint_visualizations(constraint_results, constraint_output_dir)

    elif args.scalability_only:
        scalability_results = framework.run_scalability_analysis()
        # Create scalability visualizations
        scalability_output_dir = os.path.join(framework.dirs['scalability'], f"scalability_{framework.timestamp}")
        framework._create_scalability_visualizations(scalability_results, scalability_output_dir)
    else:
        # Run all experiments
        framework.run_all_experiments()

if __name__ == "__main__":
    main()