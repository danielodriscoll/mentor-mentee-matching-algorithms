import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import euclidean_distances
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

def meets_all_constraints(mentors_df, mentees_df, mentor_idx, mentee_idx, constraints):
    """
    Check if a mentor-mentee pair meets all enabled constraints.
    Returns True if all enabled constraints are satisfied, False otherwise.
    """
    # If no constraints provided, all pairs meet constraints
    if constraints is None:
        return True
        
    mentor = mentors_df.iloc[mentor_idx]
    mentee = mentees_df.iloc[mentee_idx]
    
    # 1. Industry matching constraint
    if constraints.get('industry_match', False):
        if mentor['Industry'] != mentee['Industry']:
            return False
    
    # 2. Experience gap requirement
    if constraints.get('min_experience_gap', 0) > 0:
        min_gap = constraints.get('min_experience_gap')
        if 'Years_of_Experience' in mentor and 'Years_of_Experience' in mentee:
            mentor_exp = mentor['Years_of_Experience']
            mentee_exp = mentee['Years_of_Experience']
            # Convert to numeric if they're strings
            if isinstance(mentor_exp, str):
                try:
                    mentor_exp = float(mentor_exp)
                except:
                    mentor_exp = 0
            if isinstance(mentee_exp, str):
                try:
                    mentee_exp = float(mentee_exp)
                except:
                    mentee_exp = 0
                    
            if mentor_exp - mentee_exp < min_gap:
                return False
    
    # 3. Communication method compatibility
    if constraints.get('communication_match', False):
        mentor_methods = mentor['Communication_Methods']
        mentee_methods = mentee['Communication_Methods']
        
        # Handle different formats
        if isinstance(mentor_methods, str) and isinstance(mentee_methods, str):
            # Check for exact match first
            if mentor_methods == mentee_methods:
                pass  # This is fine, continue checking other constraints
            else:
                # Split by comma if present
                if ',' in mentor_methods:
                    mentor_methods = [m.strip() for m in mentor_methods.split(',')]
                else:
                    mentor_methods = [mentor_methods.strip()]
                    
                if ',' in mentee_methods:
                    mentee_methods = [m.strip() for m in mentee_methods.split(',')]
                else:
                    mentee_methods = [mentee_methods.strip()]
                
                # Check for any overlap
                if not any(method in mentee_methods for method in mentor_methods):
                    return False
    
    # 4. Availability matching
    if constraints.get('availability_match', False):
        if mentor['Availability'] != mentee['Availability']:
            return False
    
    # 5. Language matching (Primary or Secondary)
    if constraints.get('language_match', False):
        # Get primary languages
        mentor_primary = mentor['Primary_Language']
        mentee_primary = mentee['Primary_Language']
        
        # Check primary match only for now (simplification)
        if mentor_primary != mentee_primary and mentor_primary != 'None' and mentee_primary != 'None':
            return False
    
    # 6. Education level matching
    if constraints.get('education_level', False):
        # Define education level hierarchy
        education_levels = {
            'High School': 1,
            'Associate': 2,
            'Bachelors': 3,
            'Masters': 4,
            'PhD': 5
        }
        
        mentor_level = education_levels.get(mentor['Education'], 0)
        mentee_level = education_levels.get(mentee['Education'], 0)
        
        if mentor_level < mentee_level:
            return False
    
    # 7. Mentoring preferences match
    if constraints.get('mentoring_preferences_match', False):
        if mentor['Mentoring_Preferences'] != mentee['Mentoring_Preferences']:
            return False
        
   # 8. Gender preference matching
        # For gender_match constraint
    if constraints.get('gender_match', False):
        try:
            if 'Gender' not in mentor or 'Gender' not in mentee:
                # Skip this constraint if the field doesn't exist
                pass
            else:
                mentor_gender = str(mentor['Gender'])
                mentee_gender = str(mentee['Gender'])
                
                if mentor_gender != mentee_gender and constraints.get('gender_match_strict', False):
                    return False
                if constraints.get('gender_match_non_binary', False) and mentee_gender == 'Non-binary':
                    if mentor_gender != 'Non-binary':
                        return False
        except Exception as e:
            print(f"Error in gender constraint: {e}")
            pass
        
    # 9. Expertise overlap constraint
        # For the expertise_overlap constraint
    if constraints.get('expertise_overlap', False):
        try:
            # First ensure both values exist
            if 'Expertise' not in mentor or 'Expertise' not in mentee:
                # Skip this constraint if the field doesn't exist
                pass
            else:
                mentor_expertise = mentor['Expertise']
                mentee_expertise = mentee['Expertise']
                
                # Handle different formats with robust error handling
                try:
                    # If it's already a list, use it as is
                    if isinstance(mentor_expertise, list):
                        pass
                    # If it's a string that looks like a list, try to eval it
                    elif isinstance(mentor_expertise, str):
                        if mentor_expertise.startswith('['):
                            mentor_expertise = eval(mentor_expertise)
                        else:
                            mentor_expertise = [mentor_expertise]
                    else:
                        mentor_expertise = [str(mentor_expertise)]
                        
                    # Same for mentee
                    if isinstance(mentee_expertise, list):
                        pass
                    elif isinstance(mentee_expertise, str):
                        if mentee_expertise.startswith('['):
                            mentee_expertise = eval(mentee_expertise)
                        else:
                            mentee_expertise = [mentee_expertise]
                    else:
                        mentee_expertise = [str(mentee_expertise)]
                    
                    # Now check for overlap
                    if not any(str(exp).lower() in [str(m_exp).lower() for m_exp in mentor_expertise] for exp in mentee_expertise):
                        return False
                        
                except Exception as e:
                    print(f"Error processing expertise: {e}")
                    # Don't fail the match because of parsing error
                    pass
        except Exception as e:
            print(f"Error in expertise constraint: {e}")
            pass


    # 10. Location matching
    # For the location_match constraint in both files
    if constraints.get('location_match', False):
        # Check if location exists in both profiles
        if 'Location' not in mentor or 'Location' not in mentee:
            return False
        
        # Handle None or empty string locations
        mentor_location = str(mentor['Location']).strip() if mentor['Location'] else ""
        mentee_location = str(mentee['Location']).strip() if mentee['Location'] else ""
        
        # If both are empty/unknown, consider it a match
        if (mentor_location == "" or mentor_location.lower() == "unknown") and \
        (mentee_location == "" or mentee_location.lower() == "unknown"):
            return True
        
        # Otherwise, check for exact match
        if mentor_location != mentee_location:
            return False

    # 11. Secondary language compatibility
    if constraints.get('secondary_language_match', False):
        try:
            # Check if both have the field first
            if 'Secondary_Languages' not in mentor or 'Secondary_Languages' not in mentee:
                return False
                
            mentor_secondary = mentor['Secondary_Languages']
            mentee_secondary = mentee['Secondary_Languages']
            
            # Handle None values
            if mentor_secondary is None or mentee_secondary is None:
                return False
                
            # Handle different formats (string vs list)
            try:
                if isinstance(mentor_secondary, str):
                    mentor_secondary = eval(mentor_secondary) if mentor_secondary.startswith('[') else [mentor_secondary]
                else:
                    mentor_secondary = [str(mentor_secondary)]
                    
                if isinstance(mentee_secondary, str):
                    mentee_secondary = eval(mentee_secondary) if mentee_secondary.startswith('[') else [mentee_secondary]
                else:
                    mentee_secondary = [str(mentee_secondary)]
                
                # Check if there's any overlap in secondary languages
                if not any(str(lang1).lower() == str(lang2).lower() 
                        for lang1 in mentor_secondary if lang1 
                        for lang2 in mentee_secondary if lang2):
                    return False
            except Exception as e:
                print(f"Error processing secondary languages: {e}")
                return False
        except Exception as e:
            print(f"Error in secondary_language_match constraint: {e}")
            return False

    # 12. Workstyle compatibility
    if constraints.get('workstyle_match', False):
        # Both full-time should match, both part-time should match, both freelance should match
        if mentor['Availability'] != mentee['Availability']:
            return False

    # 13. Position level progression constraint
    if constraints.get('position_progression', False):
        position_levels = {
            'Entry': 1,
            'Junior': 2,
            'Mid': 3,
            'Senior': 4,
            'Lead': 5
        }
        
        mentor_level = position_levels.get(mentor['Experience_Level'], 0)
        mentee_level = position_levels.get(mentee['Experience_Level'], 0)
        
        min_level_gap = constraints.get('min_position_level_gap', 1)
        if mentor_level - mentee_level < min_level_gap:
            return False

    # 14. Maximum mentee-mentor distance (for multi-dimensional optimization)
    if constraints.get('max_distance', float('inf')) < float('inf'):
        max_allowed_distance = constraints.get('max_distance')
        features = [f for f in mentor.keys() if f not in ['User_ID', 'Gender', 'Location', 'Availability']]
        
        # Calculate a simple distance based on matching features
        distance = sum(1 for f in features if mentor[f] != mentee[f])
        if distance > max_allowed_distance:
            return False

    # 15. Similar but complementary roles (e.g., front-end dev with back-end dev)
    if constraints.get('complementary_roles', False):
        complementary_pairs = [
            ('Full-Stack Developer', 'DevOps Engineer'),
            ('Frontend Developer', 'Backend Developer'),
            ('Data Scientist', 'Data Engineer'),
            ('UX Designer', 'UI Developer'),
            ('Product Manager', 'Developer'),
            ('AI Researcher', 'ML Engineer'),
            ('Security Analyst', 'Network Engineer'),
            ('IT Consultant', 'Systems Architect')
        ]
        
        is_complementary = False
        for role1, role2 in complementary_pairs:
            if (mentor['Position'] == role1 and mentee['Position'] == role2) or \
            (mentor['Position'] == role2 and mentee['Position'] == role1):
                is_complementary = True
                break
        
        if not is_complementary:
            return False
        
    
    # 16. Minimum years in the industry constraint
    if constraints.get('min_industry_years', 0) > 0:
        min_years = constraints.get('min_industry_years')
        if mentor['Years_of_Experience'] < min_years:
            return False

    # 17. Maximum mentees constraint (already implemented as max_mentees_per_mentor)

    # 18. Matching based on required expertise for specific mentee needs
    if constraints.get('required_mentor_expertise', None):
        required_expertise = constraints.get('required_mentor_expertise')
        mentor_expertise = mentor['Expertise']
        
        # Convert to list if it's a string
        if isinstance(mentor_expertise, str):
            mentor_expertise = eval(mentor_expertise) if mentor_expertise.startswith('[') else [mentor_expertise]
        
        if required_expertise not in mentor_expertise:
            return False

    # 19. Communication frequency constraint
    if constraints.get('communication_frequency', None):
        # This would require additional data about communication frequency preferences
        # For this example, let's assume it's encoded in Communication_Methods
        # e.g., "Video Calls (Weekly)"
        mentor_freq = mentor['Communication_Methods']
        mentee_freq = mentee['Communication_Methods']
        
        required_frequency = constraints.get('communication_frequency')
        if required_frequency not in mentor_freq or required_frequency not in mentee_freq:
            return False

    # 20. Project-based vs. career guidance match
    if constraints.get('mentoring_style_match', False):
        if 'Project Collaboration' in mentor['Mentoring_Preferences'] and \
        'Career Guidance' in mentee['Mentoring_Preferences']:
            return False
        if 'Career Guidance' in mentor['Mentoring_Preferences'] and \
        'Project Collaboration' in mentee['Mentoring_Preferences']:
            return False
        
    # All enabled constraints are satisfied
    return True


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

def modified_kmeans_matching(mentors_df, mentees_df, max_mentees_per_mentor=2, max_iterations=100, constraints=None):
    """
    Implements modified K-means clustering where mentors are fixed centroids
    and mentees are assigned to closest mentor clusters
    """
    # Initialize default constraints if none provided
    if constraints is None:
        constraints = {
            'max_mentees_per_mentor': max_mentees_per_mentor,
            'min_similarities': 2
        }
    
    features = [col for col in mentors_df.columns if col != 'User_ID']
    
    # Initialize cluster assignments
    n_mentees = len(mentees_df)
    n_mentors = len(mentors_df)
    
    # Get feature arrays
    mentor_features = mentors_df[features].values
    mentee_features = mentees_df[features].values
    
    # Initialize mentee assignments and cluster sizes
    mentee_assignments = np.full(n_mentees, -1)  # -1 means unassigned
    cluster_sizes = np.zeros(n_mentors, dtype=int)
    
    # Keep track of matches for validity checking
    matches = []
    
    # Calculate distances between all mentees and mentors
    distances = euclidean_distances(mentee_features, mentor_features)
    max_distance = np.max(distances)
    
    # Modified K-means iterations
    for iteration in range(max_iterations):
        changes_made = False
        
        # Sort mentors by distance for each mentee
        mentor_preferences = np.argsort(distances, axis=1)
        
        # Try to assign each unassigned mentee to their preferred mentor
        for mentee_idx in range(n_mentees):
            if mentee_assignments[mentee_idx] == -1:  # Only process unassigned mentees
                for preferred_mentor in mentor_preferences[mentee_idx]:
                    # Check if this mentor satisfies all enabled constraints
                    if not meets_all_constraints(mentors_df, mentees_df, preferred_mentor, mentee_idx, constraints):
                        continue
                        
                    if cluster_sizes[preferred_mentor] < constraints.get('max_mentees_per_mentor', max_mentees_per_mentor):
                        mentee_assignments[mentee_idx] = preferred_mentor
                        cluster_sizes[preferred_mentor] += 1
                        changes_made = True
                        
                        # Calculate exact matches for validity checking
                        mentor_row = mentors_df.iloc[preferred_mentor][features]
                        mentee_row = mentees_df.iloc[mentee_idx][features]
                        num_similarities = calculate_exact_matches(mentor_row, mentee_row, features)
                        
                        
                        matches.append({
                            'Mentor_ID': mentors_df.iloc[preferred_mentor]['User_ID'],
                            'Mentee_ID': mentees_df.iloc[mentee_idx]['User_ID'],
                            'Similarity_Score': 1 - (distances[mentee_idx][preferred_mentor] / max_distance),
                            'Num_Similarities': num_similarities
                        })
                        break
        
        # If no changes were made in this iteration, we've reached convergence
        if not changes_made:
            break
    
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
        matches_df['Num_Similarities'] = 0
    
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
    Run the K-means mentor-mentee matching algorithm across multiple randomized datasets.
    """
    # Use default constraints if none provided
    if constraints is None:
        constraints = {
            'max_mentees_per_mentor': max_mentees_per_mentor,
            'min_similarities': 2
        }
        
    all_results = []
    execution_times = []
    
    print(f"\nRunning {num_randomizations} randomized iterations of the K-means matching algorithm...")
    print(f"Applied constraints: {', '.join(constraints.keys())}")
    
    for i in range(num_randomizations):
        # Use consistent random seeds across algorithms
        # Randomize mentors and mentees
        shuffled_mentors = mentors_df.sample(frac=1, random_state=i).reset_index(drop=True)
        shuffled_mentees = mentees_df.sample(frac=1, random_state=i+1000).reset_index(drop=True)
        
        # Measure execution time for this iteration
        start_time = time.time()
        
        # Get matches for this randomization using K-means approach
        matches_df = modified_kmeans_matching(
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
    print(f"K-MEANS EXPERIMENT RESULTS ({num_randomizations} RANDOMIZED ITERATIONS)")
    print("="*50)
    
    print(f"\nMatching Results (Averaged over {num_randomizations} iterations):")
    print(f"✅ Match validity: {experiment_results['avg_validity']:.2f}%")
    print(f"📊 On average, {experiment_results['total_valid_matches']/num_randomizations:.0f} out of {avg_matches_per_iteration:.0f} mentor-mentee pairs have at least 2 shared characteristics.")
    print(f"⭐ Average similarity score: {experiment_results['avg_similarity']:.2f}%")
    print(f"⏱️ Average execution time per iteration: {experiment_results['avg_execution_time']:.4f} seconds")
    print(f"⏱️ Total experiment execution time: {overall_execution_time:.2f} seconds")
    
    # Save results to CSV file - adjust the path based on experiment type
    output_path = 'data/kmeans_results.csv'
    if mentor_sample_size != 500 or mentee_sample_size != 1000:
        # This is a scalability test
        output_path = f'data/kmeans_{mentor_sample_size}m_{mentee_sample_size}m.csv'
    elif len(constraints) > 2:
        # This is a constraint test
        constraint_count = len(constraints)
        output_path = f'data/kmeans_constraints_{constraint_count}.csv'
    
    results_df = experiment_results['all_results']
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    return experiment_results

if __name__ == "__main__":
    try:
        print("Starting K-means mentor-mentee matching experiment...")
        experiment_results = run_experiment('mentors_dataset.csv', 'mentees_dataset.csv', num_randomizations=10)
        print("\nExperiment complete!")
        
    except Exception as e:
        print(f"Error in execution: {str(e)}")
        raise