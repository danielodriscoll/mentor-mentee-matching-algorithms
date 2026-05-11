# generate_heatmaps.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from glob import glob

# === CONFIG ===
DATA_FOLDER = r"C:\Users\danie\OneDrive - mycit.ie\Year 4\FYP2\500_constraints_onesided\500_constarints_results\constraint_analysis\constraints_20250405_144040"
OUTPUT_FOLDER = os.path.join(DATA_FOLDER, "charts")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === ALGORITHM NAME MAPPING ===
algorithm_name_map = {
    "ga": "genetic algorithm",
    "kmeans": "k_clustering",
    "dacceptance": "deferred_acceptance",
    "euclidean": "eucledean_distance"
}

# === LOAD & COMBINE CSV FILES ===
csv_files = glob(os.path.join(DATA_FOLDER, "*.csv"))
summary_data = []

for filepath in csv_files:
    filename = os.path.basename(filepath)

    # Skip files that are not constraint result CSVs
    if not re.search(r'\(\d+_constraints\)', filename):
        continue

    df = pd.read_csv(filepath)
    raw_algo = filename.split('_')[0]  # e.g., 'ga', 'kmeans'
    algo = algorithm_name_map.get(raw_algo, raw_algo)
    constraint_type = '_'.join(filename.split('_')[1:-1])

    sim_cols = [col for col in df.columns if "similarity" in col.lower()]
    val_cols = [col for col in df.columns if "validity" in col.lower()]

    avg_similarity = df[sim_cols[0]].mean() if sim_cols else 0
    avg_validity = df[val_cols[0]].mean() if val_cols else 0

    summary_data.append({
        "Algorithm": algo,
        "Constraint Type": constraint_type,
        "Avg Similarity": avg_similarity,
        "Avg Validity": avg_validity
    })

summary_df = pd.DataFrame(summary_data)

# === EXTRACT CONSTRAINT COUNT ===
def extract_constraint_count(name):
    match = re.search(r'\((\d+)', name)  # e.g., (2_constraints)
    return int(match.group(1)) if match else None

summary_df["Constraint Count"] = summary_df["Constraint Type"].apply(extract_constraint_count)

# === PREPARE FULL COMBINATION FOR HEATMAPS ===
all_algorithms = summary_df["Algorithm"].unique()
all_constraints = summary_df[["Constraint Type", "Constraint Count"]].drop_duplicates()

full_index = pd.MultiIndex.from_product(
    [all_constraints["Constraint Type"], all_algorithms],
    names=["Constraint Type", "Algorithm"]
)

# === VALIDITY HEATMAP ===
validity_data = pd.DataFrame(index=full_index).reset_index()
validity_data = validity_data.merge(summary_df[["Constraint Type", "Algorithm", "Avg Validity"]],
                                    on=["Constraint Type", "Algorithm"], how="left")
validity_data["Avg Validity"] = validity_data["Avg Validity"].fillna(0)
validity_data = validity_data.merge(all_constraints, on="Constraint Type", how="left")
validity_pivot = validity_data.pivot(index="Constraint Count", columns="Algorithm", values="Avg Validity").sort_index()

plt.figure(figsize=(12, 6))
sns.heatmap(validity_pivot, annot=True, fmt=".2f", cmap="YlOrBr", cbar_kws={"label": "Avg Validity"})
plt.title("Heatmap: Avg Validity by Constraint Count and Algorithm")
plt.ylabel("Constraint Count")
plt.xlabel("Algorithm")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "validity_heatmap.png"), dpi=300)
plt.close()

# === SIMILARITY HEATMAP ===
similarity_data = pd.DataFrame(index=full_index).reset_index()
similarity_data = similarity_data.merge(summary_df[["Constraint Type", "Algorithm", "Avg Similarity"]],
                                        on=["Constraint Type", "Algorithm"], how="left")
similarity_data["Avg Similarity"] = similarity_data["Avg Similarity"].fillna(0)
similarity_data = similarity_data.merge(all_constraints, on="Constraint Type", how="left")
similarity_pivot = similarity_data.pivot(index="Constraint Count", columns="Algorithm", values="Avg Similarity").sort_index()

plt.figure(figsize=(12, 6))
sns.heatmap(similarity_pivot, annot=True, fmt=".2f", cmap="PuBuGn", cbar_kws={"label": "Avg Similarity"})
plt.title("Heatmap: Avg Similarity by Constraint Count and Algorithm")
plt.ylabel("Constraint Count")
plt.xlabel("Algorithm")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "similarity_heatmap.png"), dpi=300)
plt.close()

print("✅ Heatmaps generated and saved in:", OUTPUT_FOLDER)