import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import euclidean_distances
import time
import random
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
    
    # Calculate feature-wise equality (for checking exact matches)
    exact_match_mentors = pd.DataFrame()
    exact_match_mentees = pd.DataFrame()
    
    for feature in features:
        exact_match_mentors[feature] = mentors_df[feature]
        exact_match_mentees[feature] = mentees_df[feature]
    
    exact_match_mentors['User_ID'] = mentors_df['User_ID']
    exact_match_mentees['User_ID'] = mentees_df['User_ID']
    
    print(f"\nDataset sizes:")
    print(f"Number of mentors: {len(encoded_mentors)}")
    print(f"Number of mentees: {len(encoded_mentees)}")
    
    return encoded_mentors, encoded_mentees, exact_match_mentors, exact_match_mentees, features

def calculate_similarity_matrix(mentor_features, mentee_features):
    """Calculate similarity scores between all mentors and mentees."""
    features = [col for col in mentor_features.columns if col != 'User_ID']
    
    mentor_array = mentor_features[features].values
    mentee_array = mentee_features[features].values
    
    distances = euclidean_distances(mentee_array, mentor_array)
    max_distance = np.max(distances)
    similarity_matrix = 1 - (distances / max_distance)
    
    return similarity_matrix

def calculate_exact_matches(mentor_row, mentee_row, features):
    """Calculate number of exact feature matches between a mentor and mentee."""
    return sum(mentor_row[f] == mentee_row[f] for f in features)

def initialize_population(pop_size, mentors_df, mentees_df, similarity_matrix, max_mentees_per_mentor=2):
    """
    Initialize a population of random valid matchings.
    Each chromosome is represented as a list where the index is the mentee ID 
    and the value is the mentor ID they're assigned to (-1 if unassigned).
    """
    population = []
    n_mentors = len(mentors_df)
    n_mentees = len(mentees_df)
    
    for _ in range(pop_size):
        # Initialize an unassigned chromosome
        chromosome = np.full(n_mentees, -1)
        
        # Track how many mentees are assigned to each mentor
        mentor_counts = np.zeros(n_mentors, dtype=int)
        
        # Create a random permutation of mentees
        mentee_order = np.random.permutation(n_mentees)
        
        for mentee_idx in mentee_order:
            # Get mentor preferences for this mentee based on similarity
            mentor_similarities = similarity_matrix[mentee_idx]
            sorted_mentor_indices = np.argsort(-mentor_similarities)
            
            # Try to assign to a mentor with available capacity
            for mentor_idx in sorted_mentor_indices:
                if mentor_counts[mentor_idx] < max_mentees_per_mentor:
                    chromosome[mentee_idx] = mentor_idx
                    mentor_counts[mentor_idx] += 1
                    break
        
        population.append(chromosome)
    
    return population

def fitness_function(chromosome, similarity_matrix, exact_match_mentors, exact_match_mentees, 
                    features, constraints, min_similarities=2):
    """
    Modified fitness function with more graduated constraint penalties.
    """
    n_mentees = len(chromosome)
    total_similarity = 0
    valid_matches = 0
    total_matches = 0
    constraint_violations = 0
    
    # Track number of mentees per mentor for constraint checking
    mentor_counts = {}
    
    for mentee_idx in range(n_mentees):
        mentor_idx = chromosome[mentee_idx]
        
        # Skip unassigned mentees
        if mentor_idx == -1:
            continue
        
        # Count this mentor assignment
        if mentor_idx not in mentor_counts:
            mentor_counts[mentor_idx] = 0
        mentor_counts[mentor_idx] += 1
        
        # Check if max mentees constraint is violated
        if mentor_counts[mentor_idx] > constraints.get('max_mentees_per_mentor', 2):
            # Apply penalty proportional to the number of constraints
            # This makes the penalty more severe as constraint count increases
            constraint_violations += 10 * len(constraints)
            continue  # Skip further evaluation for this invalid match
        
        # Check all other constraints
        if not meets_all_constraints(exact_match_mentors, exact_match_mentees, mentor_idx, mentee_idx, constraints):
            # Apply penalty proportional to the number of constraints
            constraint_violations += 20 * len(constraints)
            continue  # Skip further evaluation for this invalid match
        
        # Only calculate similarity for valid matches (meets all constraints)
        similarity = similarity_matrix[mentee_idx][mentor_idx]
        total_similarity += similarity
        
        # Check for exact matches (for validity checking)
        mentor_row = exact_match_mentors.iloc[mentor_idx]
        mentee_row = exact_match_mentees.iloc[mentee_idx]
        num_similarities = calculate_exact_matches(mentor_row, mentee_row, features)
        
        # Count valid matches (matches with at least min_similarities shared characteristics)
        if num_similarities >= constraints.get('min_similarities', min_similarities):
            valid_matches += 1
        
        total_matches += 1
    
    # If no matches were made, return a very low fitness
    if total_matches == 0:
        return -1000
    
    # If there are constraint violations, apply a proportional penalty
    # This makes the GA more sensitive to higher constraint counts
    if constraint_violations > 0:
        return -constraint_violations
    
    # Calculate average similarity and validity percentage
    avg_similarity = total_similarity / total_matches if total_matches > 0 else 0
    validity_percentage = (valid_matches / total_matches) * 100 if total_matches > 0 else 0
    
    # Fitness is a combination of similarity score and validity percentage
    # Weight validity higher as constraint count increases
    validity_weight = 5 + 5 * (len(constraints) / 10)  # Increases with more constraints
    fitness = (validity_percentage * validity_weight) + (avg_similarity * 100)
    
    return fitness

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
    if constraints.get('location_match', False):
        if mentor['Location'] != mentee['Location']:
            return False

    # 11. Secondary language compatibility
    if constraints.get('secondary_language_match', False):
        mentor_secondary = mentor['Secondary_Languages']
        mentee_secondary = mentee['Secondary_Languages']
        
        # Handle different formats (string vs list)
        if isinstance(mentor_secondary, str):
            mentor_secondary = eval(mentor_secondary) if mentor_secondary.startswith('[') else [mentor_secondary]
        if isinstance(mentee_secondary, str):
            mentee_secondary = eval(mentee_secondary) if mentee_secondary.startswith('[') else [mentee_secondary]
        
        # Check if there's any overlap in secondary languages
        if not any(lang in mentor_secondary for lang in mentee_secondary):
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

def tournament_selection(population, fitness_scores, tournament_size=3):
    """
    Select an individual using tournament selection.
    """
    # Randomly select tournament_size individuals
    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    
    # Return the best individual from the tournament
    winner_idx = tournament_indices[np.argmax(tournament_fitness)]
    return population[winner_idx].copy()

def crossover(parent1, parent2, crossover_rate=0.8):
    """
    Perform crossover between two parents to create two offspring.
    Using a multi-point crossover strategy that preserves valid matchings.
    """
    # Check if crossover should occur
    if random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()
    
    n_mentees = len(parent1)
    offspring1 = np.full(n_mentees, -1)
    offspring2 = np.full(n_mentees, -1)
    
    # Create a crossover mask - True means take from parent1, False from parent2
    crossover_mask = np.random.random(n_mentees) < 0.5
    
    # Initialize mentor count dictionaries for constraint tracking
    mentor_counts1 = {}
    mentor_counts2 = {}
    
    # First pass - try to inherit directly from parents according to mask
    for mentee_idx in range(n_mentees):
        # Determine which parent to inherit from for each offspring
        # Offspring1 follows the mask, offspring2 follows the inverse
        if crossover_mask[mentee_idx]:
            # Offspring1 inherits from parent1
            mentor_idx = parent1[mentee_idx]
            if mentor_idx != -1:
                if mentor_idx not in mentor_counts1:
                    mentor_counts1[mentor_idx] = 0
                mentor_counts1[mentor_idx] += 1
                if mentor_counts1[mentor_idx] <= 2:  # Max mentees constraint
                    offspring1[mentee_idx] = mentor_idx
                
            # Offspring2 inherits from parent2
            mentor_idx = parent2[mentee_idx]
            if mentor_idx != -1:
                if mentor_idx not in mentor_counts2:
                    mentor_counts2[mentor_idx] = 0
                mentor_counts2[mentor_idx] += 1
                if mentor_counts2[mentor_idx] <= 2:  # Max mentees constraint
                    offspring2[mentee_idx] = mentor_idx
        else:
            # Offspring1 inherits from parent2
            mentor_idx = parent2[mentee_idx]
            if mentor_idx != -1:
                if mentor_idx not in mentor_counts1:
                    mentor_counts1[mentor_idx] = 0
                mentor_counts1[mentor_idx] += 1
                if mentor_counts1[mentor_idx] <= 2:  # Max mentees constraint
                    offspring1[mentee_idx] = mentor_idx
                
            # Offspring2 inherits from parent1
            mentor_idx = parent1[mentee_idx]
            if mentor_idx != -1:
                if mentor_idx not in mentor_counts2:
                    mentor_counts2[mentor_idx] = 0
                mentor_counts2[mentor_idx] += 1
                if mentor_counts2[mentor_idx] <= 2:  # Max mentees constraint
                    offspring2[mentee_idx] = mentor_idx
    
    return offspring1, offspring2

def mutation(chromosome, n_mentors, similarity_matrix, mutation_rate=0.05):
    """
    Mutate a chromosome by randomly reassigning some mentees to different mentors.
    """
    n_mentees = len(chromosome)
    mutated = chromosome.copy()
    
    # Track mentor counts to maintain max_mentees_per_mentor constraint
    mentor_counts = {}
    for mentee_idx in range(n_mentees):
        mentor_idx = mutated[mentee_idx]
        if mentor_idx != -1:
            if mentor_idx not in mentor_counts:
                mentor_counts[mentor_idx] = 0
            mentor_counts[mentor_idx] += 1
    
    # Consider each mentee for potential mutation
    for mentee_idx in range(n_mentees):
        # Apply mutation with probability mutation_rate
        if random.random() < mutation_rate:
            current_mentor = mutated[mentee_idx]
            
            # Get mentor preferences for this mentee based on similarity
            mentor_similarities = similarity_matrix[mentee_idx]
            sorted_mentor_indices = np.argsort(-mentor_similarities)
            
            # Try to reassign to one of the top mentors by similarity
            for potential_mentor in sorted_mentor_indices[:10]:  # Consider top 10 mentors
                # Skip if it's the same mentor or if this would exceed max mentees
                if potential_mentor == current_mentor:
                    continue
                
                if potential_mentor not in mentor_counts:
                    mentor_counts[potential_mentor] = 0
                
                if mentor_counts[potential_mentor] < 2:  # Max mentees constraint
                    # Update mentor counts
                    if current_mentor != -1:
                        mentor_counts[current_mentor] -= 1
                    mentor_counts[potential_mentor] += 1
                    
                    # Assign new mentor
                    mutated[mentee_idx] = potential_mentor
                    break
    
    return mutated

def convert_chromosome_to_matches(chromosome, mentors_df, mentees_df, similarity_matrix, 
                                exact_match_mentors, exact_match_mentees, features):
    """
    Convert a chromosome to a list of mentor-mentee match dictionaries.
    """
    matches = []
    n_mentees = len(chromosome)
    
    for mentee_idx in range(n_mentees):
        mentor_idx = chromosome[mentee_idx]
        
        # Skip unassigned mentees
        if mentor_idx == -1:
            continue
        
        # Calculate similarity score
        similarity = similarity_matrix[mentee_idx][mentor_idx]
        
        # Check for exact matches
        mentor_row = exact_match_mentors.iloc[mentor_idx]
        mentee_row = exact_match_mentees.iloc[mentee_idx]
        num_similarities = calculate_exact_matches(mentor_row, mentee_row, features)
        
        matches.append({
            'Mentor_ID': mentors_df.iloc[mentor_idx]['User_ID'],
            'Mentee_ID': mentees_df.iloc[mentee_idx]['User_ID'],
            'Similarity_Score': similarity,
            'Num_Similarities': num_similarities
        })
    
    return pd.DataFrame(matches)

def evaluate_matching(matches_df):
    """Evaluate matching results with paper's metrics."""
    total_matches = len(matches_df)
    valid_matches = len(matches_df[matches_df['Num_Similarities'] >= 2])
    validity_percentage = (valid_matches / total_matches) * 100 if total_matches > 0 else 0
    avg_similarity_score = matches_df['Similarity_Score'].mean() * 100  # Convert to percentage
    
    results = {
        'validity_percentage': validity_percentage,
        'valid_matches': valid_matches,
        'total_matches': total_matches,
        'avg_similarity_score': avg_similarity_score
    }
    
    return results

def genetic_algorithm(mentors_df, mentees_df, exact_match_mentors, exact_match_mentees, features,
                     similarity_matrix, constraints, pop_size=100, generations=1, 
                     crossover_rate=0.8, mutation_rate=0.05, tournament_size=3):
    """
    Run the genetic algorithm to find optimal mentor-mentee matches.
    """
    n_mentors = len(mentors_df)
    n_mentees = len(mentees_df)
    
    # Initialize population
    print(f"Initializing population with {pop_size} individuals...")
    print(f"Applied constraints: {', '.join(constraints.keys())}")
    
    population = initialize_population(
        pop_size, mentors_df, mentees_df, similarity_matrix, 
        max_mentees_per_mentor=constraints.get('max_mentees_per_mentor', 2)
    )
    
    best_fitness = float('-inf')
    best_chromosome = None
    
    # Run for specified number of generations
    for generation in range(generations):
        gen_start_time = time.time()
        
        # Evaluate fitness of each individual
        fitness_scores = [
            fitness_function(
                chrom, similarity_matrix, exact_match_mentors, exact_match_mentees, 
                features, constraints, min_similarities=constraints.get('min_similarities', 2)
            ) 
            for chrom in population
        ]
        
        # Find the best individual in this generation
        gen_best_idx = np.argmax(fitness_scores)
        gen_best_fitness = fitness_scores[gen_best_idx]
        
        # Update overall best if this generation has a better solution
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_chromosome = population[gen_best_idx].copy()
        
        # Create new population through selection, crossover, and mutation
        new_population = []
        
        while len(new_population) < pop_size:
            # Selection
            parent1 = tournament_selection(population, fitness_scores, tournament_size)
            parent2 = tournament_selection(population, fitness_scores, tournament_size)
            
            # Crossover
            offspring1, offspring2 = crossover(parent1, parent2, crossover_rate)
            
            # Mutation
            offspring1 = mutation(offspring1, n_mentors, similarity_matrix, mutation_rate)
            offspring2 = mutation(offspring2, n_mentors, similarity_matrix, mutation_rate)
            
            # Add to new population
            new_population.append(offspring1)
            if len(new_population) < pop_size:
                new_population.append(offspring2)
        
        # Replace old population with new population
        population = new_population
        
        gen_end_time = time.time()
        print(f"Generation {generation + 1}: Best fitness = {gen_best_fitness:.2f}, Time: {gen_end_time - gen_start_time:.2f}s")
    
    # Convert best chromosome to matches dataframe
    best_matches = convert_chromosome_to_matches(
        best_chromosome, mentors_df, mentees_df, similarity_matrix, 
        exact_match_mentors, exact_match_mentees, features
    )
    
    return best_matches

def run_experiment(mentors_file, mentees_file, num_randomizations=10, generations=50, pop_size=100,
                 crossover_rate=0.9, mutation_rate=0.01, tournament_size=5, 
                 mentor_sample_size=500, mentee_sample_size=1000, constraints=None):
    """
    Run the complete experiment from data loading to final results.
    Runs the genetic algorithm multiple times with different random seeds,
    similar to the randomizations in the other algorithms.
    """
    # Start overall timing
    overall_start_time = time.time()
    
    # Define constraints if none provided
    if constraints is None:
        constraints = {
            'max_mentees_per_mentor': 2,
            'min_similarities': 2
        }
    
    # Preprocess data
    mentors_processed, mentees_processed, exact_match_mentors, exact_match_mentees, features = preprocess_data(
        mentors_file, mentees_file,
        mentor_sample_size=mentor_sample_size,
        mentee_sample_size=mentee_sample_size
    )
    
    # Calculate similarity matrix (used for both initialization and fitness evaluation)
    print("Calculating similarity matrix...")
    similarity_matrix = calculate_similarity_matrix(mentors_processed, mentees_processed)
    
    # Store results from all randomizations
    all_results = []
    execution_times = []
    
    print(f"\nRunning {num_randomizations} randomized iterations of the genetic algorithm...")
    print(f"Each iteration runs for {generations} generations")
    print(f"GA parameters: pop_size={pop_size}, crossover_rate={crossover_rate}, mutation_rate={mutation_rate}, tournament_size={tournament_size}")
    
    for i in range(num_randomizations):
        # Set different random seed for each run
        np.random.seed(i)
        random.seed(i)
        
        # Measure execution time for this iteration
        start_time = time.time()
        
        # Run genetic algorithm
        matches_df = genetic_algorithm(
            mentors_processed, mentees_processed, exact_match_mentors, exact_match_mentees, features,
            similarity_matrix, constraints, pop_size=pop_size, generations=generations,
            crossover_rate=crossover_rate, mutation_rate=mutation_rate, tournament_size=tournament_size
        )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        execution_times.append(execution_time)
        
        # Evaluate matching
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
    
    # Overall execution time
    overall_execution_time = time.time() - overall_start_time
    
    # Calculate average matches per iteration
    avg_matches_per_iteration = total_matches / num_randomizations
    
    print("\n" + "="*50)
    print(f"GENETIC ALGORITHM RESULTS ({num_randomizations} RANDOMIZED ITERATIONS)")
    print("="*50)
    
    print(f"\nMatching Results (Averaged over {num_randomizations} iterations):")
    print(f"✅ Match validity: {avg_validity:.2f}%")
    print(f"📊 On average, {total_valid_matches/num_randomizations:.0f} out of {avg_matches_per_iteration:.0f} mentor-mentee pairs have at least 2 shared characteristics.")
    print(f"⭐ Average similarity score: {avg_similarity:.2f}%")
    print(f"⏱️ Average execution time per iteration: {avg_execution_time:.4f} seconds")
    print(f"⏱️ Total experiment execution time: {overall_execution_time:.2f} seconds")
    
    # Save results to CSV file - adjust the path based on experiment type
    output_path = 'data/ga_results.csv'
    if mentor_sample_size != 500 or mentee_sample_size != 1000:
        # This is a scalability test
        output_path = f'data/ga_{mentor_sample_size}m_{mentee_sample_size}m.csv'
    elif len(constraints) > 2:
        # This is a constraint test
        constraint_count = len(constraints)
        output_path = f'data/ga_constraints_{constraint_count}.csv'
    
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    return {
        'avg_validity': avg_validity,
        'avg_similarity': avg_similarity,
        'avg_execution_time': avg_execution_time,
        'total_valid_matches': total_valid_matches,
        'total_matches': total_matches,
        'all_results': results_df
    }

if __name__ == "__main__":
    try:
        '''
        print("Starting genetic algorithm mentor-mentee matching experiment...")
        
        # First, try with a single run and just 1 generation to test functionality
        print("Running quick test with 1 generation...")
        experiment_results = run_experiment(
            'mentors_dataset.csv', 'mentees_dataset.csv', 
            num_randomizations=1,
            generations=1, 
            pop_size=100,
            crossover_rate=0.8, 
            mutation_rate=0.05, 
            tournament_size=3
        )
        print("\nInitial test complete!")
        '''
        # If successful, run the full experiment
        print("\nRunning full experiment with multiple randomizations...")
        experiment_results = run_experiment(
            'mentors_dataset.csv', 'mentees_dataset.csv', 
            num_randomizations=10,  # Same as other algorithms for fair comparison
            generations=50,          # Each run performs 50 generations of evolution
            pop_size=100,
            crossover_rate=0.9, 
            mutation_rate=0.01, 
            tournament_size=5
        )
        print("\nExperiment complete!")
        
    except Exception as e:
        print(f"Error in execution: {str(e)}")
        raise