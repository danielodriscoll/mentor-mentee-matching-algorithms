import os
import glob
import pickle
import pandas as pd
import numpy as np
import argparse
import json
from datetime import datetime

def merge_distributed_results(input_dir, output_dir, pattern=None):
    """
    Merge distributed algorithm results from pickle files.
    
    Args:
        input_dir: Directory containing distributed result pickle files
        output_dir: Directory to write merged results
        pattern: Glob pattern to match specific result files (default: all .pkl files)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Find all result files
    if pattern is None:
        pattern = "*.pkl"
    file_path = os.path.join(input_dir, pattern)
    result_files = glob.glob(file_path)
    
    if not result_files:
        print(f"No result files found matching pattern: {file_path}")
        return
    
    print(f"Found {len(result_files)} result files to merge")
    
    # Group files by algorithm and parameters
    grouped_files = {}
    
    for file_path in result_files:
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
        
        if 'metadata' not in results:
            print(f"Skipping file {file_path} - no metadata found")
            continue
        
        metadata = results['metadata']
        algo = metadata.get('algorithm', 'unknown')
        
        # Create group key based on algorithm and parameters
        if 'mentor_size' in metadata and 'mentee_size' in metadata:
            # This is a scalability test
            group_key = f"{algo}_scale_{metadata['mentor_size']}m_{metadata['mentee_size']}m"
        else:
            # This is a constraint test
            constraint_count = len(metadata.get('constraints', {}))
            group_key = f"{algo}_constraints_{constraint_count}"
        
        if group_key not in grouped_files:
            grouped_files[group_key] = []
        
        grouped_files[group_key].append((file_path, results))
    
    # Process each group
    for group_key, file_results in grouped_files.items():
        print(f"Processing group: {group_key} with {len(file_results)} files")
        
        # Collect all result DataFrames and metrics
        all_dfs = []
        total_valid_matches = 0
        total_matches = 0
        total_exec_time = 0
        iterations = 0
        
        # Metadata from first file (should be same for all in group)
        sample_metadata = file_results[0][1]['metadata']
        
        for _, results in file_results:
            if 'all_results' in results and isinstance(results['all_results'], pd.DataFrame):
                all_dfs.append(results['all_results'])
            
            total_valid_matches += results.get('total_valid_matches', 0)
            total_matches += results.get('total_matches', 0)
            total_exec_time += results['metadata'].get('execution_time', 0)
            iterations += results['metadata'].get('iterations', 0)
        
        # Merge all DataFrames
        if all_dfs:
            merged_df = pd.concat(all_dfs, ignore_index=True)
            
            # Calculate aggregate metrics
            avg_validity = merged_df['validity_percentage'].mean()
            avg_similarity = merged_df['avg_similarity_score'].mean() if 'avg_similarity_score' in merged_df.columns else 0
            avg_execution_time = merged_df['execution_time'].mean() if 'execution_time' in merged_df.columns else 0
            
            # Create summary dict
            summary = {
                'algorithm': sample_metadata.get('algorithm', 'unknown'),
                'iterations': iterations,
                'avg_validity': avg_validity,
                'avg_similarity': avg_similarity,
                'avg_execution_time': avg_execution_time,
                'total_valid_matches': total_valid_matches,
                'total_matches': total_matches,
                'total_execution_time': total_exec_time
            }
            
            # Add scalability or constraint specific info
            if 'mentor_size' in sample_metadata:
                summary['mentor_size'] = sample_metadata['mentor_size']
                summary['mentee_size'] = sample_metadata['mentee_size']
                summary['total_dataset_size'] = sample_metadata['mentor_size'] + sample_metadata['mentee_size']
                summary['test_type'] = 'scalability'
            else:
                summary['constraints'] = json.dumps(sample_metadata.get('constraints', {}))
                summary['num_constraints'] = len(sample_metadata.get('constraints', {}))
                summary['test_type'] = 'constraints'
            
            # Save merged results
            output_csv = os.path.join(output_dir, f"{group_key}_merged_{timestamp}.csv")
            merged_df.to_csv(output_csv, index=False)
            
            # Save summary
            summary_json = os.path.join(output_dir, f"{group_key}_summary_{timestamp}.json")
            with open(summary_json, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"Merged results saved to {output_csv}")
            print(f"Summary saved to {summary_json}")
            print(f"Metrics: Validity={avg_validity:.2f}%, Similarity={avg_similarity:.2f}%, Avg Time={avg_execution_time:.4f}s")
        else:
            print(f"No valid DataFrames found for group {group_key}")

def main():
    parser = argparse.ArgumentParser(description='Merge distributed algorithm results')
    parser.add_argument('--input-dir', type=str, default='distributed_results', 
                        help='Directory containing distributed result files')
    parser.add_argument('--output-dir', type=str, default='merged_results', 
                        help='Directory to write merged results')
    parser.add_argument('--pattern', type=str, default=None, 
                        help='Glob pattern to match specific result files')
    
    args = parser.parse_args()
    
    merge_distributed_results(args.input_dir, args.output_dir, args.pattern)

if __name__ == "__main__":
    main()