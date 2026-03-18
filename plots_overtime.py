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
            bad_lines.append((i, str(e)))

print("Loaded records:", len(rows))
print("Bad lines skipped:", len(bad_lines))

df = pd.DataFrame(rows)
print("Columns:", df.columns.tolist())

# pick date column automatically
date_col = None
for c in ["commit_date", "date", "timestamp", "created_at"]:
    if c in df.columns:
        date_col = c
        break

print("date_col =", date_col)

if date_col is None:
    print("No usable date column found.")
else:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    time_df = df.dropna(subset=[date_col]).copy()

    if len(time_df) == 0:
        print("Date column exists, but no valid dates could be parsed.")
    else:
        monthly_counts = (
            time_df.set_index(date_col)
                   .resample("M")
                   .size()
        )

        ax = monthly_counts.plot(figsize=(10, 5), marker="o")
        ax.set_title("Drift Events Over Time (Monthly)")
        ax.set_xlabel("Month")
        ax.set_ylabel("Number of Drift Events")
        plt.tight_layout()
        plt.savefig("plot_5_drift_over_time.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

        print("Saved: plot_5_drift_over_time.png")