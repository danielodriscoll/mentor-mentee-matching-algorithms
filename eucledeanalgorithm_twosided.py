import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import time
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

def preprocess_data(mentors_file, mentees_file, mentor_sample_size=500, mentee_sample_size=1000):
    """
    Preprocess mentor and mentee data with consistent sample sizes
    """
    mentors_df = pd.read_csv(mentors_file)
    mentees_df = pd.read_csv(mentees_file)
    
    # Use consistent sample size across all algorithms
    SAMPLE_SIZE_MENTORS = mentor_sample_size
    SAMPLE_SIZE_MENTEES = mentee_sample_size
    
    # Limit dataset size
    if len(mentors_df) > SAMPLE_SIZE_MENTORS:
        mentors_df = mentors_df.sample(n=SAMPLE_SIZE_MENTORS, random_state=42)
    if len(mentees_df) > SAMPLE_SIZE_MENTEES:
        mentees_df = mentees_df.sample(n=SAMPLE_SIZE_MENTEES, random_state=42)
    
    features = [
        'Position', 'Experience_Level', 'Primary_Language', 'Industry',
        'Availability', 'Communication_Methods', 'Mentoring_Preferences',
        'Education', 'Secondary_Languages', 'Expertise', 'Gender', 'Location',
        'Years_of_Experience'
    ]
    
    # Keep only necessary columns
    mentors_df = mentors_df[['User_ID'] + features]
    mentees_df = mentees_df[['User_ID'] + features]
    
    # Fill missing values
    mentors_df.fillna('Unknown', inplace=True)
    mentees_df.fillna('Unknown', inplace=True)
    
    # Feature weights from paper's importance criteria
    feature_weights = {
        'Position': 1.5,
        'Experience_Level': 1.2,
        'Primary_Language': 1.3,
        'Industry': 1.4,
        'Availability': 1.0,
        'Communication_Methods': 1.0,
        'Mentoring_Preferences': 1.5,
        'Education': 1.2
    }
    
    # Encode categorical variables with weighted normalization
    encoders = {}
    encoded_mentors = pd.DataFrame()
    encoded_mentees = pd.DataFrame()
    
    # Define which features to use for similarity calculation (these have weights)
    similarity_features = [
        'Position', 'Experience_Level', 'Primary_Language', 'Industry',
        'Availability', 'Communication_Methods', 'Mentoring_Preferences',
        'Education'
    ]

    # Encode only similarity features
    for feature in similarity_features:
        encoders[feature] = LabelEncoder()
        combined_values = pd.concat([mentors_df[feature], mentees_df[feature]]).unique()
        encoders[feature].fit(combined_values)
        
        mentor_encoded = encoders[feature].transform(mentors_df[feature]) * feature_weights[feature]
        mentee_encoded = encoders[feature].transform(mentees_df[feature]) * feature_weights[feature]
        
        encoded_mentors[feature] = mentor_encoded
        encoded_mentees[feature] = mentee_encoded
    
    encoded_mentors['User_ID'] = mentors_df['User_ID']
    encoded_mentees['User_ID'] = mentees_df['User_ID']
    
    print(f"\nDataset sizes:")
    print(f"Number of mentors: {len(encoded_mentors)}")
    print(f"Number of mentees: {len(encoded_mentees)}")
    
    return encoded_mentors, encoded_mentees

def calculate_mentee_to_mentor_similarities(mentor_features, mentee_features):
    """Calculate similarity scores between mentors and mentees from mentee perspective."""
    features = [col for col in mentor_features.columns if col != 'User_ID']
    
    mentor_array = mentor_features[features].values
    mentee_array = mentee_features[features].values
    
    distances = euclidean_distances(mentee_array, mentor_array)
    max_distance = np.max(distances)
    similarity_matrix = 1 - (distances / max_distance)
    
    return similarity_matrix

def calculate_mentor_to_mentee_similarities(mentor_features, mentee_features):
    """Calculate similarity scores between mentors and mentees from mentor perspective."""
    features = [col for col in mentor_features.columns if col != 'User_ID']
    
    mentor_array = mentor_features[features].values
    mentee_array = mentee_features[features].values
    
    # Calculate mentor to mentee distances
    distances = euclidean_distances(mentor_array, mentee_array)
    max_distance = np.max(distances)
    similarity_matrix = 1 - (distances / max_distance)
    
    return similarity_matrix

def meets_all_constraints(mentors_df, mentees_df, mentor_idx, mentee_idx, constraints):
    """
    Check if a mentor-mentee pair meets all enabled constraints.
    Returns True if all enabled constraints are satisfied, False otherwise.
    """
    # I'll skip including this function since you mentioned you'll add it yourself
    # It's the same as in the other algorithms
    pass

def calculate_exact_matches(mentor_row, mentee_row, features):
    """Calculate number of exact feature matches between a mentor and mentee."""
    matches = 0
    for f in features:
        try:
            # Convert both values to strings for comparison
            mentor_val = str(mentor_row[f]) if mentor_row[f] is not None else ""
            mentee_val = str(mentee_row[f]) if mentee_row[f] is not None else ""
            
            # Count as match if both are exactly equal as strings
            if mentor_val == mentee_val and mentor_val != "" and mentor_val.lower() != "unknown":
                matches += 1
        except Exception as e:
            # If error in comparison, don't count as match
            continue
    
    return matches

def match_mentees_to_mentors(mentors_df, mentees_df, max_mentees_per_mentor=2, constraints=None):
    """
    Match mentees to mentors using a two-sided Euclidean distance approach.
    Both mentee preferences for mentors and mentor preferences for mentees are considered.
    """
    # Initialize default constraints if none provided
    if constraints is None:
        constraints = {
            'max_mentees_per_mentor': max_mentees_per_mentor,
            'min_similarities': 2
        }
    
    features = [col for col in mentors_df.columns if col != 'User_ID']
    
    # Calculate mentee→mentor similarity (mentee preferences)
    mentee_preferences = calculate_mentee_to_mentor_similarities(mentors_df, mentees_df)
    
    # Calculate mentor→mentee similarity (mentor preferences)
    mentor_preferences = calculate_mentor_to_mentee_similarities(mentors_df, mentees_df)
    
    # Combine preferences with weights (0.7 for mentee preference, 0.3 for mentor preference)
    # This prioritizes mentee satisfaction while still considering mentor preferences
    n_mentees = len(mentees_df)
    n_mentors = len(mentors_df)
    combined_preferences = np.zeros((n_mentees, n_mentors))
    
    for mentee_idx in range(n_mentees):
        for mentor_idx in range(n_mentors):
            combined_preferences[mentee_idx, mentor_idx] = (
                0.7 * mentee_preferences[mentee_idx, mentor_idx] + 
                0.3 * mentor_preferences[mentor_idx, mentee_idx]
            )
    
    matches = []
    mentor_counts = {idx: 0 for idx in range(len(mentors_df))}
    
    # For each mentee, find the best available mentor using combined preferences
    for mentee_idx in range(n_mentees):
        mentor_similarities = combined_preferences[mentee_idx]
        sorted_mentor_indices = np.argsort(-mentor_similarities)
        
        # Find best available mentor who hasn't reached their maximum mentees
        for mentor_idx in sorted_mentor_indices:
            # Check if this mentor satisfies all enabled constraints
            if not meets_all_constraints(mentors_df, mentees_df, mentor_idx, mentee_idx, constraints):
                continue
                
            if mentor_counts[mentor_idx] < constraints.get('max_mentees_per_mentor', max_mentees_per_mentor):
                mentor_counts[mentor_idx] += 1
                
                try:
                    # Calculate exact matches for validity checking
                    mentor_row = mentors_df.iloc[mentor_idx][features]
                    mentee_row = mentees_df.iloc[mentee_idx][features]
                    
                    # Use the calculate_exact_matches function
                    num_similarities = calculate_exact_matches(mentor_row, mentee_row, features)
                    
                    matches.append({
                        'Mentor_ID': mentors_df.iloc[mentor_idx]['User_ID'],
                        'Mentee_ID': mentees_df.iloc[mentee_idx]['User_ID'],
                        'Similarity_Score': combined_preferences[mentee_idx][mentor_idx],
                        'Num_Similarities': num_similarities
                    })
                except Exception as e:
                    print(f"Error creating match data: {e}")
                    # Add a default entry to prevent algorithm failure
                    matches.append({
                        'Mentor_ID': mentors_df.iloc[mentor_idx]['User_ID'],
                        'Mentee_ID': mentees_df.iloc[mentee_idx]['User_ID'],
                        'Similarity_Score': combined_preferences[mentee_idx][mentor_idx],
                        'Num_Similarities': 0  # Default to zero similarities if calculation fails
                    })
                break
    
    # Create DataFrame or handle empty matches list
    if not matches:
        # Return empty DataFrame with correct column structure
        return pd.DataFrame(columns=['Mentor_ID', 'Mentee_ID', 'Similarity_Score', 'Num_Similarities'])
    
    matches_df = pd.DataFrame(matches)
    return matches_df

def evaluate_matching(matches_df):
    """Evaluate matching results with paper's metrics."""
    # Handle empty DataFrame case
    if matches_df.empty:
        return {
            'validity_percentage': 0,
            'valid_matches': 0,
            'total_matches': 0,
            'avg_similarity_score': 0
        }
    
    total_matches = len(matches_df)
    
    # Ensure Num_Similarities exists and has valid values
    if 'Num_Similarities' not in matches_df.columns:
        matches_df['Num_Similarities'] = 0  # Add default column if missing
    
    # Convert to numeric, coercing errors to NaN, then replace NaN with 0
    matches_df['Num_Similarities'] = pd.to_numeric(matches_df['Num_Similarities'], errors='coerce').fillna(0)
    
    valid_matches = len(matches_df[matches_df['Num_Similarities'] >= 2])
    validity_percentage = (valid_matches / total_matches) * 100 if total_matches > 0 else 0
    
    # Similarly handle Similarity_Score column
    if 'Similarity_Score' not in matches_df.columns:
        matches_df['Similarity_Score'] = 0
    
    matches_df['Similarity_Score'] = pd.to_numeric(matches_df['Similarity_Score'], errors='coerce').fillna(0)
    avg_similarity_score = matches_df['Similarity_Score'].mean() * 100  # Convert to percentage
    
    results = {
        'validity_percentage': validity_percentage,
        'valid_matches': valid_matches,
        'total_matches': total_matches,
        'avg_similarity_score': avg_similarity_score
    }
    
    return results

def randomized_matching_experiment(mentors_df, mentees_df, num_randomizations=100, max_mentees_per_mentor=2, constraints=None):
    """
    Run the two-sided Euclidean mentor-mentee matching algorithm across multiple randomized datasets.
    """
    # Use default constraints if none provided
    if constraints is None:
        constraints = {
            'max_mentees_per_mentor': max_mentees_per_mentor,
            'min_similarities': 2
        }
        
    all_results = []
    execution_times = []
    
    print(f"\nRunning {num_randomizations} randomized iterations of the Two-Sided Euclidean matching algorithm...")
    print(f"Applied constraints: {', '.join(constraints.keys())}")
    
    for i in range(num_randomizations):
        # Use consistent random seeds across algorithms
        # Randomize mentors and mentees
        shuffled_mentors = mentors_df.sample(frac=1, random_state=i).reset_index(drop=True)
        shuffled_mentees = mentees_df.sample(frac=1, random_state=i+1000).reset_index(drop=True)
        
        # Measure execution time for this iteration
        start_time = time.time()
        
        # Get matches for this randomization
        matches_df = match_mentees_to_mentors(
            shuffled_mentors, 
            shuffled_mentees, 
            max_mentees_per_mentor=constraints.get('max_mentees_per_mentor', max_mentees_per_mentor),
            constraints=constraints
        )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        execution_times.append(execution_time)
        
        # Evaluate matches
        results = evaluate_matching(matches_df)
        results['execution_time'] = execution_time  # Add execution time to results
        all_results.append(results)
        
        # Print progress every 10 iterations
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_randomizations} iterations...")
    
    # Convert all_results to DataFrame for CSV export
    results_df = pd.DataFrame(all_results)
    
    # Calculate averages across all randomizations
    avg_validity = results_df['validity_percentage'].mean()
    avg_similarity = results_df['avg_similarity_score'].mean()
    avg_execution_time = results_df['execution_time'].mean()
    
    # Calculate total valid matches and total matches
    total_valid_matches = results_df['valid_matches'].sum()
    total_matches = results_df['total_matches'].sum()
    
    return {
        'avg_validity': avg_validity,
        'avg_similarity': avg_similarity,
        'avg_execution_time': avg_execution_time,
        'total_valid_matches': total_valid_matches,
        'total_matches': total_matches,
        'all_results': results_df
    }


def run_experiment(mentors_file, mentees_file, num_randomizations=100, max_mentees_per_mentor=2, 
                  mentor_sample_size=500, mentee_sample_size=1000, constraints=None):
    """
    Run the complete experiment from data loading to final results
    """
    # Use default constraints if none provided
    if constraints is None:
        constraints = {
            'max_mentees_per_mentor': max_mentees_per_mentor,
            'min_similarities': 2
        }
    
    # Start overall timing
    overall_start_time = time.time()
    
    # Preprocess data
    mentors_processed, mentees_processed = preprocess_data(
        mentors_file, mentees_file, 
        mentor_sample_size=mentor_sample_size, 
        mentee_sample_size=mentee_sample_size
    )
    
    # Run randomized experiment
    experiment_results = randomized_matching_experiment(
        mentors_processed, 
        mentees_processed, 
        num_randomizations=num_randomizations,
        max_mentees_per_mentor=constraints.get('max_mentees_per_mentor', max_mentees_per_mentor),
        constraints=constraints
    )
    
    # Overall execution time
    overall_execution_time = time.time() - overall_start_time
    
    # Calculate average matches per iteration
    avg_matches_per_iteration = experiment_results['total_matches'] / num_randomizations
    
    print("\n" + "="*50)
    print(f"TWO-SIDED EUCLIDEAN ALGORITHM RESULTS ({num_randomizations} RANDOMIZED ITERATIONS)")
    print("="*50)
    
    print(f"\nMatching Results (Averaged over {num_randomizations} iterations):")
    print(f"✅ Match validity: {experiment_results['avg_validity']:.2f}%")
    print(f"📊 On average, {experiment_results['total_valid_matches']/num_randomizations:.0f} out of {avg_matches_per_iteration:.0f} mentor-mentee pairs have at least 2 shared characteristics.")
    print(f"⭐ Average similarity score: {experiment_results['avg_similarity']:.2f}%")
    print(f"⏱️ Average execution time per iteration: {experiment_results['avg_execution_time']:.4f} seconds")
    print(f"⏱️ Total experiment execution time: {overall_execution_time:.2f} seconds")
    
    # Save results to CSV file - adjust the path based on experiment type
    output_path = 'data/twosided_euclidean_results.csv'
    if mentor_sample_size != 500 or mentee_sample_size != 1000:
        # This is a scalability test
        output_path = f'data/twosided_euclidean_{mentor_sample_size}m_{mentee_sample_size}m.csv'
    elif len(constraints) > 2:
        # This is a constraint test
        constraint_count = len(constraints)
        output_path = f'data/twosided_euclidean_constraints_{constraint_count}.csv'
    
    results_df = experiment_results['all_results']
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    return experiment_results

if __name__ == "__main__":
    try:
        print("Starting Two-Sided Euclidean mentor-mentee matching experiment...")
        experiment_results = run_experiment('mentors_dataset.csv', 'mentees_dataset.csv', num_randomizations=10)
        print("\nExperiment complete!")
        
    except Exception as e:
        print(f"Error in execution: {str(e)}")
        raise