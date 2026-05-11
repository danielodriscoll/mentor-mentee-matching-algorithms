import matplotlib.pyplot as plt
import pandas as pd
from testing_framework_parallelized import run_parallel_algorithm

# ========== CONFIGURATION ==========
MENTORS_FILE = "mentors_dataset.csv"
MENTEES_FILE = "mentees_dataset.csv"
CONSTRAINTS = {
    'max_mentees_per_mentor': 2,
    'min_similarities': 2,
    'industry_match': True,
    'education_level': True,
    'communication_match': True
}
GENERATION_VALUES = [2, 4, 6, 10, 20, 30, 40, 50]
ITERATIONS_PER_SETTING = 5
MENTOR_SAMPLE_SIZE = 500
MENTEE_SAMPLE_SIZE = 1000
# ====================================

def run_generation_tests():
    results = []

    for gen in GENERATION_VALUES:
        print(f"\n--- Testing GA with {gen} generations ---")

        result = run_parallel_algorithm(
            algo_type='GA',
            mentors_file=MENTORS_FILE,
            mentees_file=MENTEES_FILE,
            iterations=ITERATIONS_PER_SETTING,
            mentor_sample_size=MENTOR_SAMPLE_SIZE,
            mentee_sample_size=MENTEE_SAMPLE_SIZE,
            constraints=CONSTRAINTS,
            generations=gen,
            pop_size=100,
            crossover_rate=0.9,
            mutation_rate=0.01,
            tournament_size=5,
            num_processes=4  # Adjust as needed
        )

        results.append({
            'generations': gen,
            'avg_similarity': result['avg_similarity'],
            'avg_validity': result['avg_validity'],
            'avg_execution_time': result['avg_execution_time']
        })

    return pd.DataFrame(results)

def plot_results(df):
    print("\n=== Performance by Generation Count ===")
    print(df.round(2))

    plt.figure(figsize=(12, 6))
    plt.plot(df['generations'], df['avg_similarity'], label='Similarity (%)', marker='o')
    plt.plot(df['generations'], df['avg_validity'], label='Validity (%)', marker='s')
    plt.xlabel('Generations')
    plt.ylabel('Score')
    plt.title('Genetic Algorithm: Effect of Generations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ga_generation_performance.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    df = run_generation_tests()
    plot_results(df)
