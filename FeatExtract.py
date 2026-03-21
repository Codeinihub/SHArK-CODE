from pathlib import Path
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def extract_features(df, row, col):
    grid = df.iloc[0:row, 0:col].apply(pd.to_numeric, errors='coerce').values
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
        "Activity": max,
        "Grid" : grid
    }

# folder_path = Path(r"C:\Users\User\OneDrive\Dokumenty")
# catalyst_files = list(folder_path.glob("*.xls*"))

rows = []
cell_rows = []
def plate_to_cell_conversion(grid, catalyst_name):
    cell_rows = []
    for r in range(8):
        for c in range(8):
            Light = r * 10 
            Charge = c * 10 
            Structural = 90 - Light - Charge

            if Structural < 0:
                continue
            PhotoCurrent = grid[r][c]

            # if np.isnan(PhotoCurrent):
            #     continue

            L_Frac = Light/90
            C_Frac = Charge/90
            S_Frac = Structural/90
            cell_rows.append({
                "Catalyst Name": catalyst_name,
                "L_Frac": L_Frac,
                "C_Frac": C_Frac,
                "S_Frac": S_Frac,
                "PhotoCurrent": PhotoCurrent
            })
    return cell_rows
            
        
# for filename in catalyst_files:
raw_df = pd.read_excel(r"C:\Users\User\OneDrive\Dokumenty\Copy of shark data.xlsx")

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
            grid = features.pop("Grid")
            features["Catalysts"] = catalyst_name
            rows.append(features)
            cell_data = plate_to_cell_conversion(grid, catalyst_name)
            cell_rows.extend(cell_data)
dataset = pd.DataFrame(rows)
cell_dataset = pd.DataFrame(cell_rows)
print(cell_dataset.head())
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
'best' material. Here pareto optimization is used to identify catalysts that are not outperformed
across all metrics simultaneously. This allows for a selction of candidates thatg represent optimal
tradeoffs between activity, stability and reproducibility.

"""


# 2D Visualization:
# the red points cannot be improved in one metric without sacrificing one another
plt.scatter(dataset["Activity"], dataset["Stability"], label = "All Catalysts")
plt.scatter(pareto_set["Activity"], pareto_set["Stability"], color = "red", label = "Pareto Front")
plt.xlabel("Activity (max photocurrent)")
plt.ylabel("Stability(1/uniformity)")
plt.legend()
#plt.show()

# 3D Visualization:
# fig = plt.figure()
# ax = fig.add_subplot(projection = '3d')
# ax.scatter(dataset["Activity"], dataset["Stability"], dataset["Reproducibility"], c = "blue", alpha = 0.4, s = 40)
# ax.scatter(pareto_set["Activity"], pareto_set["Stability"], pareto_set["Reproducibility"], c = "red", s = 80)
# ax.set_xlabel("Activity")
# ax.set_ylabel("Stability")
# ax.set_zlabel("Reproducibility")
# legend_elements = [
#     Line2D([0], [0], marker = 'o', color = 'w', label = "All Catalysts", markersize = 8, markerfacecolor = "blue", alpha = 0.4),
#     Line2D([0], [0], marker = 'o', color = 'w', label = "Pareto Front", markersize = 10, markerfacecolor = "red")
# ]
# ax.legend(handles = legend_elements)
# ax.view_init(elev = 20, azim = 45)
# plt.show()

#  Utopian Point:
"""
Concept: The utopian point represents a hypothetical catalyst that simultaneously achieves the best observed 
activity, stability, and reproducibility. No real catalyst reaches this point, but it serves as a reference for optimal performance.
"""
norm_dataset = dataset.copy()
for col in objectives:
    norm_dataset[col] = (
        (dataset[col] - dataset[col].min())/
        (dataset[col].max() - dataset[col].min())
    )
norm_pareto = norm_dataset.loc[pareto_set.index]
utopian = np.array([1.0, 1.0, 1.0])
objectives = ["Activity", "Stability", "Reproducibility"]
pareto_points = norm_pareto.loc[:, objectives].to_numpy()
# We compute Euclidean distance in objective space:
distances = np.linalg.norm(pareto_points - utopian, axis = 1)
# Utopian Visualization
# fig = plt.figure(figsize=(9, 7))
# ax = fig.add_subplot(111, projection='3d')

# # All catalysts (background)
# ax.scatter(
#     dataset["Activity"],
#     dataset["Stability"],
#     dataset["Reproducibility"],
#     alpha=0.3,
#     label="All Catalysts"
# )

# # Pareto front (colored by distance)
# sc = ax.scatter(
#     pareto_set["Activity"],
#     pareto_set["Stability"],
#     pareto_set["Reproducibility"],
#     c=distances,
#     cmap="Reds_r",  # red = closer
#     s=80,
#     label="Pareto-optimal Catalysts"
# )

# # Utopian point
# ax.scatter(
#     utopian[0],
#     utopian[1],
#     utopian[2],
#     c="black",
#     marker="*",
#     s=200,
#     label="Utopian Point"
# )

# ax.set_xlabel("Activity (Max Photocurrent)")
# ax.set_ylabel("Stability (1 / Uniformity)")
# ax.set_zlabel("Reproducibility (Mean Response)")

# fig.colorbar(sc, ax=ax, label="Distance to Utopian Point")
# ax.legend()
# plt.tight_layout()
# plt.show()

# Label top 3
top3_index = np.argsort(distances)[:3]
plt.figure(figsize=(9, 7))
ax = plt.axes(projection="3d")

# All catalysts
ax.scatter3D(
    dataset["Activity"],
    dataset["Stability"],
    dataset["Reproducibility"],
    alpha=0.3,
    label="All Catalysts"
)

# Pareto front
sc = ax.scatter3D(
    pareto_set["Activity"],
    pareto_set["Stability"],
    pareto_set["Reproducibility"],
    c=distances,
    cmap="Reds_r",
    s=60,
    label="Pareto Front"
)

# Utopian point
ax.scatter3D(
    1, 1, 1,
    c="blue",
    s=100,
    marker="*",
    label="Utopian Point"
)

# Label top 3
for i in top3_index:
    row = pareto_set.iloc[i]
    ax.text(
        row["Activity"],
        row["Stability"],
        row["Reproducibility"],
        row["Catalysts"],
        fontsize=9,
        color="black"
    )

ax.set_xlabel("Activity")
ax.set_ylabel("Stability")
ax.set_zlabel("Reproducibility")
ax.legend()
plt.colorbar(sc, label="Distance to Utopian Point")
# plt.show()


# Ternary Map:
ternary_data = cell_dataset.dropna(subset = ["PhotoCurrent"])
L = ternary_data["L_Frac"].values
C = ternary_data["C_Frac"].values
S = ternary_data["S_Frac"].values
Z = ternary_data["PhotoCurrent"].values
x = C + 0.5*L
y = np.sqrt(3)/2 * L


fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(x, y, c= Z ,cmap= 'Spectral')
plt.colorbar(sc, label= 'PhotoCurrent')
plt.title("Ternary_Composition Map")
plt.xlabel("Charge/Structural Axis")
plt.ylabel("Light Axis")
print(cell_dataset.shape)
print(cell_dataset["PhotoCurrent"].isna().sum())
print(dataset["Catalysts"].unique())
plt.show()
# Ternary Map Visualization: