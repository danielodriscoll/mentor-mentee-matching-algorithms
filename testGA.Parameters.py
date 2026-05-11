import pandas as pd
import numpy as np
import time
import itertools
from datetime import datetime

from genetic_algorithm import (
    preprocess_data,
    calculate_similarity_matrix,
    genetic_algorithm,
    evaluate_matching,
    run_experiment
)

# Import your GA functions here, or copy them into this file
# from genetic_algorithm import preprocess_data, calculate_similarity_matrix, genetic_algorithm, evaluate_matching

def test_parameters(mentors_file, mentees_file, test_generations=10):
    """
    Test different combinations of GA parameters to find optimal values.
    Uses shorter runs (fewer generations) for efficiency.
    
    Returns a DataFrame with results for each parameter combination.
    """
    # Define parameter ranges to test
    param_grid = {
        'crossover_rate': [0.7, 0.8, 0.9],
        'mutation_rate': [0.01, 0.05, 0.1],
        'tournament_size': [2, 3, 4, 5],
        'population_size': [50, 100, 150]
    }
    
    # Load and preprocess data (only need to do this once)
    print("Preprocessing data...")
    mentors_processed, mentees_processed, exact_match_mentors, exact_match_mentees, features = preprocess_data(
        mentors_file, mentees_file
    )
    
    # Calculate similarity matrix (only need to do this once)
    print("Calculating similarity matrix...")
    similarity_matrix = calculate_similarity_matrix(mentors_processed, mentees_processed)
    
    # Define constraints (same as your baseline)
    constraints = {
        'max_mentees_per_mentor': 2,
        'min_similarities': 2
    }
    
    # Create all combinations of parameters to test
    all_params = []
    for cr in param_grid['crossover_rate']:
        for mr in param_grid['mutation_rate']:
            for ts in param_grid['tournament_size']:
                for ps in param_grid['population_size']:
                    all_params.append({
                        'crossover_rate': cr,
                        'mutation_rate': mr,
                        'tournament_size': ts,
                        'population_size': ps
                    })
    
    print(f"Testing {len(all_params)} parameter combinations...")
    
    # Store results
    results = []
    
    # Test each parameter combination
    for i, params in enumerate(all_params):
        print(f"\nTesting combination {i+1}/{len(all_params)}: {params}")
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        
        # Time this run
        start_time = time.time()
        
        # Run genetic algorithm with these parameters
        matches_df = genetic_algorithm(
            mentors_processed, mentees_processed, 
            exact_match_mentors, exact_match_mentees, features,
            similarity_matrix, constraints, 
            pop_size=params['population_size'], 
            generations=test_generations,
            crossover_rate=params['crossover_rate'], 
            mutation_rate=params['mutation_rate'], 
            tournament_size=params['tournament_size']
        )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Evaluate the results
        eval_results = evaluate_matching(matches_df)
        
        # Store all results
        results.append({
            'crossover_rate': params['crossover_rate'],
            'mutation_rate': params['mutation_rate'],
            'tournament_size': params['tournament_size'],
            'population_size': params['population_size'],
            'validity_percentage': eval_results['validity_percentage'],
            'avg_similarity_score': eval_results['avg_similarity_score'],
            'valid_matches': eval_results['valid_matches'],
            'total_matches': eval_results['total_matches'],
            'execution_time': execution_time
        })
        
        print(f"Validity: {eval_results['validity_percentage']:.2f}%, " 
              f"Similarity: {eval_results['avg_similarity_score']:.2f}%, "
              f"Time: {execution_time:.1f}s")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/ga_parameter_tuning_{timestamp}.csv"
    results_df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")
    
    return results_df

def select_best_parameters(results_df, weight_validity=0.5, weight_similarity=0.4, weight_time=0.1):
    """
    Select the best parameter combination based on weighted criteria.
    
    Args:
        results_df: DataFrame with parameter testing results
        weight_validity: Weight for validity percentage (0-1)
        weight_similarity: Weight for similarity score (0-1)
        weight_time: Weight for execution time (0-1), lower times are better
    
    Returns:
        Dictionary with the best parameter values
    """
    # Normalize each metric to 0-1 scale
    results_df['norm_validity'] = results_df['validity_percentage'] / 100  # Already 0-100
    results_df['norm_similarity'] = results_df['avg_similarity_score'] / 100  # Already 0-100
    
    # For execution time, lower is better, so invert the normalization
    max_time = results_df['execution_time'].max()
    min_time = results_df['execution_time'].min()
    time_range = max_time - min_time
    
    if time_range > 0:
        results_df['norm_time'] = 1 - ((results_df['execution_time'] - min_time) / time_range)
    else:
        # If all times are the same
        results_df['norm_time'] = 1
    
    # Calculate weighted score
    results_df['score'] = (
        (weight_validity * results_df['norm_validity']) + 
        (weight_similarity * results_df['norm_similarity']) + 
        (weight_time * results_df['norm_time'])
    )
    
    # Find the best parameter combination
    best_idx = results_df['score'].idxmax()
    best_params = results_df.loc[best_idx]
    
    print("\n" + "="*50)
    print("BEST PARAMETER COMBINATION:")
    print("="*50)
    print(f"Crossover Rate: {best_params['crossover_rate']}")
    print(f"Mutation Rate: {best_params['mutation_rate']}")
    print(f"Tournament Size: {best_params['tournament_size']}")
    print(f"Population Size: {best_params['population_size']}")
    print(f"Validity: {best_params['validity_percentage']:.2f}%")
    print(f"Similarity: {best_params['avg_similarity_score']:.2f}%")
    print(f"Execution Time: {best_params['execution_time']:.1f}s")
    print(f"Overall Score: {best_params['score']:.4f}")
    
    return {
        'crossover_rate': best_params['crossover_rate'],
        'mutation_rate': best_params['mutation_rate'],
        'tournament_size': best_params['tournament_size'],
        'population_size': best_params['population_size']
    }

def simplified_parameter_test(mentors_file, mentees_file):
    """
    A simplified approach testing fewer combinations to save time
    """
    # Test a subset of parameter combinations
    print("Running simplified parameter test with fewer combinations...")
    
    # Load and preprocess data 
    mentors_processed, mentees_processed, exact_match_mentors, exact_match_mentees, features = preprocess_data(
        mentors_file, mentees_file
    )
    
    # Calculate similarity matrix
    similarity_matrix = calculate_similarity_matrix(mentors_processed, mentees_processed)
    
    # Define constraints
    constraints = {
        'max_mentees_per_mentor': 2,
        'min_similarities': 2
    }
    
    # Define parameter combinations to test
    param_combinations = [
        {'name': 'Baseline', 'cr': 0.8, 'mr': 0.05, 'ts': 3, 'ps': 100},
        {'name': 'High Exploration', 'cr': 0.7, 'mr': 0.1, 'ts': 2, 'ps': 100},
        {'name': 'High Exploitation', 'cr': 0.9, 'mr': 0.01, 'ts': 5, 'ps': 100},
        {'name': 'Balanced Alternative', 'cr': 0.8, 'mr': 0.03, 'ts': 4, 'ps': 150},
    ]
    
    results = []
    
    for params in param_combinations:
        print(f"\nTesting: {params['name']}")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Time this run
        start_time = time.time()
        
        # Run genetic algorithm with these parameters (10 generations)
        matches_df = genetic_algorithm(
            mentors_processed, mentees_processed, 
            exact_match_mentors, exact_match_mentees, features,
            similarity_matrix, constraints, 
            pop_size=params['ps'], 
            generations=10,  # Short run for quick testing
            crossover_rate=params['cr'], 
            mutation_rate=params['mr'], 
            tournament_size=params['ts']
        )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Evaluate the results
        eval_results = evaluate_matching(matches_df)
        
        # Add to results
        params.update({
            'validity': eval_results['validity_percentage'],
            'similarity': eval_results['avg_similarity_score'],
            'time': execution_time
        })
        
        results.append(params)
        
        print(f"Validity: {eval_results['validity_percentage']:.2f}%, "
              f"Similarity: {eval_results['avg_similarity_score']:.2f}%, "
              f"Time: {execution_time:.1f}s")
    
    # Sort by validity first, then similarity
    sorted_results = sorted(results, key=lambda x: (-x['validity'], -x['similarity']))
    
    print("\n" + "="*50)
    print("PARAMETER TEST RESULTS (SORTED BY PERFORMANCE):")
    print("="*50)
    
    for i, params in enumerate(sorted_results):
        print(f"{i+1}. {params['name']}")
        print(f"   Crossover: {params['cr']}, Mutation: {params['mr']}, Tournament: {params['ts']}, Population: {params['ps']}")
        print(f"   Validity: {params['validity']:.2f}%, Similarity: {params['similarity']:.2f}%, Time: {params['time']:.1f}s")
    
    print("\nRecommended parameters:")
    print(f"Crossover Rate: {sorted_results[0]['cr']}")
    print(f"Mutation Rate: {sorted_results[0]['mr']}")
    print(f"Tournament Size: {sorted_results[0]['ts']}")
    print(f"Population Size: {sorted_results[0]['ps']}")
    
    return sorted_results[0]

# Example usage
if __name__ == "__main__":
    try:
        print("Starting GA parameter tuning...")
        
        # Choose between comprehensive or simplified testing
        comprehensive_testing = False
        
        if comprehensive_testing:
            # Full parameter grid testing (will take longer)
            results_df = test_parameters('mentors_dataset.csv', 'mentees_dataset.csv', test_generations=10)
            best_params = select_best_parameters(results_df)
        else:
            # Simplified testing with fewer combinations
            best_params = simplified_parameter_test('mentors_dataset.csv', 'mentees_dataset.csv')
        
        print("\nParameter tuning complete!")
        
    except Exception as e:
        print(f"Error in execution: {str(e)}")
        raise