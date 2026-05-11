import os
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import traceback
import pickle

# Import your algorithms
from euclideanalgorithm import run_experiment as euclidean_run
from kclusteringalgorithm import run_experiment as kmeans_run
from genetic_algorithm import run_experiment as ga_run

def parse_constraints(constraint_str):
    """Parse constraint string into a dictionary"""
    if not constraint_str:
        return {
            'max_mentees_per_mentor': 2,
            'min_similarities': 2
        }
    
    try:
        # Try to parse as JSON
        return json.loads(constraint_str)
    except:
        # Parse as key=value pairs
        constraints = {'max_mentees_per_mentor': 2, 'min_similarities': 2}
        pairs = constraint_str.split(',')
        for pair in pairs:
            if '=' in pair:
                key, value = pair.split('=', 1)
                # Convert value to appropriate type
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '', 1).isdigit():
                    value = float(value)
                constraints[key.strip()] = value
        return constraints

def main():
    """Run a single algorithm with specified parameters"""
    parser = argparse.ArgumentParser(description='Run a single algorithm iteration with specific parameters')
    
    # Algorithm selection
    parser.add_argument('--algorithm', type=str, required=True, choices=['Euclidean', 'KMeans', 'GA'], 
                        help='Algorithm to run')
    
    # Input files
    parser.add_argument('--mentors-file', type=str, default='mentors_dataset.csv', 
                        help='Path to mentors dataset CSV')
    parser.add_argument('--mentees-file', type=str, default='mentees_dataset.csv', 
                        help='Path to mentees dataset CSV')
    
    # Test parameters
    parser.add_argument('--iterations', type=int, default=100, 
                        help='Number of randomized iterations to run')
    parser.add_argument('--start-iteration', type=int, default=0,
                        help='Starting iteration number (for distributed runs)')
    parser.add_argument('--constraints', type=str, default='', 
                        help='Constraints to apply (comma-separated key=value pairs or JSON string)')
    
    # Dataset size parameters
    parser.add_argument('--mentor-size', type=int, default=500, 
                        help='Number of mentors to sample')
    parser.add_argument('--mentee-size', type=int, default=1000, 
                        help='Number of mentees to sample')
    
    # GA-specific parameters
    parser.add_argument('--generations', type=int, default=50, 
                        help='Number of generations (GA only)')
    parser.add_argument('--pop-size', type=int, default=100, 
                        help='Population size (GA only)')
    parser.add_argument('--crossover-rate', type=float, default=0.9, 
                        help='Crossover rate (GA only)')
    parser.add_argument('--mutation-rate', type=float, default=0.01, 
                        help='Mutation rate (GA only)')
    parser.add_argument('--tournament-size', type=int, default=5, 
                        help='Tournament size (GA only)')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='distributed_results', 
                        help='Output directory')
    parser.add_argument('--output-prefix', type=str, default='', 
                        help='Prefix for output files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse constraints
    constraints = parse_constraints(args.constraints)
    
    # Set output prefix
    if not args.output_prefix:
        num_constraints = len(constraints)
        if 'dataset_size' in args.output_prefix:
            args.output_prefix = f"{args.algorithm.lower()}_{args.mentor_size}m_{args.mentee_size}m"
        else:
            args.output_prefix = f"{args.algorithm.lower()}_constraints_{num_constraints}"
    
    # Set output file
    output_file = os.path.join(args.output_dir, f"{args.output_prefix}_{args.start_iteration}_to_{args.start_iteration + args.iterations - 1}.pkl")
    
    print(f"Running {args.algorithm} algorithm with iterations {args.start_iteration} to {args.start_iteration + args.iterations - 1}")
    print(f"Mentor size: {args.mentor_size}, Mentee size: {args.mentee_size}")
    print(f"Applied constraints: {', '.join(constraints.keys())}")
    
    # Select algorithm
    if args.algorithm == 'Euclidean':
        run_func = euclidean_run
        kwargs = {}
    elif args.algorithm == 'KMeans':
        run_func = kmeans_run
        kwargs = {}
    elif args.algorithm == 'GA':
        run_func = ga_run
        kwargs = {
            'generations': args.generations,
            'pop_size': args.pop_size,
            'crossover_rate': args.crossover_rate,
            'mutation_rate': args.mutation_rate,
            'tournament_size': args.tournament_size
        }
    
    try:
        start_time = time.time()
        
        # Override the random seed for each algorithm based on start_iteration
        # This ensures no overlap between distributed runs
        import random
        random.seed(args.start_iteration)
        np.random.seed(args.start_iteration)
        
        # Run the algorithm
        results = run_func(
            args.mentors_file,
            args.mentees_file,
            num_randomizations=args.iterations,
            mentor_sample_size=args.mentor_size,
            mentee_sample_size=args.mentee_size,
            constraints=constraints,
            **kwargs
        )
        
        # Add metadata to results
        results['metadata'] = {
            'algorithm': args.algorithm,
            'mentor_size': args.mentor_size,
            'mentee_size': args.mentee_size,
            'constraints': constraints,
            'iterations': args.iterations,
            'start_iteration': args.start_iteration,
            'execution_time': time.time() - start_time
        }
        
        # Save results as pickle for later merging
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Results saved to {output_file}")
        print(f"Execution time: {time.time() - start_time:.2f} seconds")
        print(f"Validity: {results['avg_validity']:.2f}%, Similarity: {results['avg_similarity']:.2f}%")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()