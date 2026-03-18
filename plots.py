import json
import os
import pandas as pd
import matplotlib.pyplot as plt

path = "golden_dataset.jsonl"

rows = []
bad_lines = []

with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            bad_lines.append((i, str(e), line[:200]))

print("Loaded records:", len(rows))
print("Bad lines:", len(bad_lines))

if bad_lines:
    print("\nFirst few bad lines:")
    for x in bad_lines[:5]:
        print("Line", x[0], "| Error:", x[1])
        print("Snippet:", x[2])
        print("-" * 60)

df = pd.DataFrame(rows)
print("\nColumns:", df.columns.tolist())


def pick_col(candidates, columns):
    for c in candidates:
        if c in columns:
            return c
    return None


label_col = pick_col(["label", "drift_type", "category"], df.columns)
repo_col = pick_col(["repo", "repository", "project"], df.columns)
symbol_col = pick_col(["symbol_type", "type", "symbol_kind"], df.columns)

print("label_col =", label_col)
print("repo_col =", repo_col)
print("symbol_col =", symbol_col)

# Plot 1: label distribution
if label_col:
    label_counts = df[label_col].value_counts()
    ax = label_counts.plot(kind="bar", figsize=(8, 5))
    ax.set_title("Distribution of Drift Labels in the Golden Dataset")
    ax.set_xlabel("Drift Label")
    ax.set_ylabel("Count")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("plot_1_label_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

# Plot 2: repository distribution
if repo_col:
    repo_counts = df[repo_col].value_counts()
    ax = repo_counts.sort_values().plot(kind="barh", figsize=(8, 5))
    ax.set_title("Distribution of Records Across Repositories")
    ax.set_xlabel("Count")
    ax.set_ylabel("Repository")
    plt.tight_layout()
    plt.savefig("plot_2_repo_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

# Plot 3: symbol type distribution
if symbol_col:
    symbol_counts = df[symbol_col].value_counts()
    ax = symbol_counts.plot(kind="bar", figsize=(7, 5))
    ax.set_title("Distribution of Records by Symbol Type")
    ax.set_xlabel("Symbol Type")
    ax.set_ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("plot_3_symbol_type_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

# Plot 4: drift type by repository
if repo_col and label_col:
    cross_tab = pd.crosstab(df[repo_col], df[label_col])
    ax = cross_tab.plot(kind="bar", stacked=True, figsize=(10, 6))
    ax.set_title("Drift Type Distribution Across Repositories")
    ax.set_xlabel("Repository")
    ax.set_ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("plot_4_drift_type_by_repo.png", dpi=300, bbox_inches="tight")
    plt.close()

print("\nSaved PNG files:")
print([f for f in os.listdir(".") if f.endswith(".png")])