#!/bin/bash

# Script to distribute algorithm runs across multiple terminals or processes
# Usage: ./distribute_runs.sh [options]

# Default parameters
ALGORITHM="Euclidean"
ITERATIONS=1000
NUM_PROCESSES=8
TEST_TYPE="constraints"  # constraints or scalability
OUTPUT_DIR="distributed_results"
MENTORS_FILE="mentors_dataset.csv"
MENTEES_FILE="mentees_dataset.csv"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --algorithm)
      ALGORITHM="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --processes)
      NUM_PROCESSES="$2"
      shift 2
      ;;
    --test-type)
      TEST_TYPE="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --mentors-file)
      MENTORS_FILE="$2"
      shift 2
      ;;
    --mentees-file)
      MENTEES_FILE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Validate inputs
if [[ ! "$ALGORITHM" =~ ^(Euclidean|KMeans|GA)$ ]]; then
  echo "Error: Algorithm must be one of: Euclidean, KMeans, GA"
  exit 1
fi

if [[ ! "$TEST_TYPE" =~ ^(constraints|scalability)$ ]]; then
  echo "Error: Test type must be either 'constraints' or 'scalability'"
  exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Calculate iterations per process
ITER_PER_PROCESS=$((ITERATIONS / NUM_PROCESSES))
REMAINDER=$((ITERATIONS % NUM_PROCESSES))

echo "Distributing $ITERATIONS iterations across $NUM_PROCESSES processes"
echo "Algorithm: $ALGORITHM"
echo "Test type: $TEST_TYPE"

# Define constraint configurations
if [[ "$TEST_TYPE" == "constraints" ]]; then
  declare -a CONSTRAINTS=(
    "max_mentees_per_mentor=2,min_similarities=2"
    "max_mentees_per_mentor=2,min_similarities=2,industry_match=true"
    "max_mentees_per_mentor=2,min_similarities=2,industry_match=true,min_experience_gap=2,education_level=true"
    "max_mentees_per_mentor=2,min_similarities=2,industry_match=true,min_experience_gap=2,education_level=true,communication_match=true,availability_match=true"
    "max_mentees_per_mentor=2,min_similarities=2,industry_match=true,min_experience_gap=2,education_level=true,communication_match=true,availability_match=true,language_match=true,secondary_language_match=true,mentoring_preferences_match=true"
    "max_mentees_per_mentor=2,min_similarities=2,industry_match=true,min_experience_gap=2,education_level=true,communication_match=true,availability_match=true,language_match=true,secondary_language_match=true,mentoring_preferences_match=true,expertise_overlap=true,position_progression=true,min_position_level_gap=1,complementary_roles=true,required_mentor_expertise=Machine Learning"
  )
  
  # Create constraint test commands
  for constraints in "${CONSTRAINTS[@]}"; do
    CONSTRAINT_COUNT=$(echo "$constraints" | tr -cd ',' | wc -c)
    CONSTRAINT_COUNT=$((CONSTRAINT_COUNT + 1))
    
    echo "Creating commands for $ALGORITHM with $CONSTRAINT_COUNT constraints"
    
    # Create command file for this constraint
    COMMAND_FILE="$OUTPUT_DIR/${ALGORITHM}_constraints_${CONSTRAINT_COUNT}_commands.sh"
    echo "#!/bin/bash" > "$COMMAND_FILE"
    
    start_iter=0
    for ((i=0; i<NUM_PROCESSES; i++)); do
      # Calculate iterations for this process
      this_iter=$ITER_PER_PROCESS
      if [[ $i -lt $REMAINDER ]]; then
        this_iter=$((this_iter + 1))
      fi
      
      # Skip if no iterations for this process
      if [[ $this_iter -eq 0 ]]; then
        continue
      fi
      
      # Create command
      cmd="python run_single_algorithm.py --algorithm $ALGORITHM --iterations $this_iter --start-iteration $start_iter --constraints \"$constraints\" --output-dir $OUTPUT_DIR --output-prefix ${ALGORITHM}_constraints_${CONSTRAINT_COUNT} --mentors-file $MENTORS_FILE --mentees-file $MENTEES_FILE"
      
      # Add GA-specific parameters if needed
      if [[ "$ALGORITHM" == "GA" ]]; then
        cmd="$cmd --generations 50 --pop-size 100 --crossover-rate 0.9 --mutation-rate 0.01 --tournament-size 5"
      fi
      
      # Add command to file
      echo "$cmd" >> "$COMMAND_FILE"
      
      # Update start_iter for next process
      start_iter=$((start_iter + this_iter))
    done
    
    # Make command file executable
    chmod +x "$COMMAND_FILE"
    echo "Created $COMMAND_FILE"
  done

else
  # Define dataset sizes for scalability test
  declare -a SIZES=(
    "500:1000:Small"
    "1000:2000:Medium"
    "2000:4000:Large"
    "4000:4000:MaxEqual"
  )
  
  # Create scalability test commands
  for size_config in "${SIZES[@]}"; do
    IFS=':' read -r mentor_size mentee_size label <<< "$size_config"
    
    echo "Creating commands for $ALGORITHM with dataset size $label ($mentor_size mentors, $mentee_size mentees)"
    
    # Create command file for this size
    COMMAND_FILE="$OUTPUT_DIR/${ALGORITHM}_size_${mentor_size}m_${mentee_size}m_commands.sh"
    echo "#!/bin/bash" > "$COMMAND_FILE"
    
    start_iter=0
    for ((i=0; i<NUM_PROCESSES; i++)); do
      # Calculate iterations for this process
      this_iter=$ITER_PER_PROCESS
      if [[ $i -lt $REMAINDER ]]; then
        this_iter=$((this_iter + 1))
      fi
      
      # Skip if no iterations for this process
      if [[ $this_iter -eq 0 ]]; then
        continue
      fi
      
      # Create command
      cmd="python run_single_algorithm.py --algorithm $ALGORITHM --iterations $this_iter --start-iteration $start_iter --mentor-size $mentor_size --mentee-size $mentee_size --output-dir $OUTPUT_DIR --output-prefix ${ALGORITHM}_size_${mentor_size}m_${mentee_size}m --mentors-file $MENTORS_FILE --mentees-file $MENTEES_FILE"
      
      # Add GA-specific parameters if needed
      if [[ "$ALGORITHM" == "GA" ]]; then
        cmd="$cmd --generations 50 --pop-size 100 --crossover-rate 0.9 --mutation-rate 0.01 --tournament-size 5"
      fi
      
      # Add command to file
      echo "$cmd" >> "$COMMAND_FILE"
      
      # Update start_iter for next process
      start_iter=$((start_iter + this_iter))
    done
    
    # Make command file executable
    chmod +x "$COMMAND_FILE"
    echo "Created $COMMAND_FILE"
  done
fi

echo "Done! Command files have been generated in $OUTPUT_DIR."
echo "You can run each process in a separate terminal using the generated script files."