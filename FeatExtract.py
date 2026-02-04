from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def extract_features(df, row, col):
    grid = pd.to_numeric(df.iloc[0:row, 0:col].values.flatten(), errors='coerce').reshape(-1, col)
    # Remove NaN values and reshape if needed
    grid_clean = grid[~np.isnan(grid)]
    if len(grid_clean) == 0:
        return None
    max = np.nanmax(grid)
    mean = np.nanmean(grid)
    std = np.nanstd(grid)
    uniformity = np.std(grid_clean) / np.mean(grid_clean) if np.mean(grid_clean) != 0 else 0
    stability = 1/uniformity
    return {
        "Max_photocurrent": max,
        "Mean_photocurrent": mean,
        "Std_photocurrent": std,
        "Uniformity": uniformity,

        "Stability":  stability,
        "Reproducibility": mean,
        "Activity": max
    }

folder_path = Path(r"C:\Users\User\OneDrive\Dokumenty")
catalyst_files = list(folder_path.glob("*.xls*"))

rows = []
for filename in catalyst_files:
    raw_df = pd.read_excel(filename, header=None)

    blank_row_indices = raw_df.index[raw_df.isnull().all(axis=1)].tolist()
    split_rows = [0] + blank_row_indices + [len(raw_df)]

    for i in range(len(split_rows) - 1):
        start = split_rows[i]
        end = split_rows[i+1]
        df_slice = raw_df.iloc[start:end].dropna(how='all').reset_index(drop=True)

        if not df_slice.empty:
            catalyst_name = df_slice.iloc[0, 1]
            body = df_slice.iloc[1:].reset_index(drop=True)
            features = extract_features(body, 8, 8)
            if features is not None:
                features["Catalysts"] = catalyst_name
                rows.append(features)
dataset = pd.DataFrame(rows)
print(dataset.head())
"""
A dominates B if:
1. A is at least as good as B in every objective
2. A is strictly better in at least one objective

If that happens → B is dominated, throw it away.
If no other catalyst dominates A, then A is Pareto-optimal.
"""
def dominates(a, b):
#Returns true if point a dominates point b. With both a and b bieng arrays of [Stability, Reproducability, Activity]
    return all(a >= b) and any(a>b)
def pareto_front(points):
# The parameter 'points' is a numpy array of certain shape.
# Returns a boolean array indicating Pareto optimal points.
    n = points.shape[0]
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if(i != j and dominates(points.iloc[j], points.iloc[i])):
                is_pareto[i] = False
                break
    return is_pareto

objectives = dataset[["Activity", "Stability", "Reproducibility"]]
pareto_mask = pareto_front(objectives)
pareto_set = dataset[pareto_mask]
print(pareto_set)
"""
EXPLAINATION:
Because catalyst performance depends on multiple competing objectives, there is no single 
'best' material. Here pareto optimization is used to identifu catalysts that are not outperformed
across all metrics simultaneously. This allows for a selction of candidates thatg represent optimal
tradeoffs between activity, stability and reproducibility.

"""


#2D Visualization:
# the red points cannot be improved in one metric without sacrificing one another
plt.scatter(dataset["Activity"], dataset["Stability"], label = "All Catalysts")
plt.scatter(pareto_set["Activity"], pareto_set["Stability"], color = "red", label = "Pareto Front")
plt.xlabel("Activity (max photocurrent)")
plt.ylabel("Stability(1/uniformity)")
plt.legend()
# plt.show()

#3D Visualization:
plt.figure()
ax = plt.axes(projection = '3d')
fg = ax.scatter3D(dataset["Activity"], dataset["Stability"], dataset["Reproducibility"])
ax.set_xlabel("Activity")
ax.set_ylabel("Stability")
ax.set_zlabel("Reproducibility")
plt.show()