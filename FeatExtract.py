from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut, KFold
from sklearn.metrics import mean_absolute_error, r2_score
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

def get_standard_values(grid):
    """Extract Fe2O3 and CuO standard values from known plate positions."""
    fe2o3_positions = [(3, 7), (6, 4), (7, 6)]
    cuo_positions   = [(4, 6), (6, 7), (7, 3)]
    
    fe2o3_vals = [grid[r][c] for r, c in fe2o3_positions 
                  if not np.isnan(grid[r][c])]
    cuo_vals   = [grid[r][c] for r, c in cuo_positions 
                  if not np.isnan(grid[r][c])]
    
    fe2o3_mean = np.mean(fe2o3_vals) if fe2o3_vals else np.nan
    cuo_mean   = np.mean(cuo_vals)   if cuo_vals   else np.nan
    
    return fe2o3_mean, cuo_mean
def normalize_grid(grid):
    """Normalize entire grid by Fe2O3 standard value."""
    fe2o3_mean, _ = get_standard_values(grid)
    
    if np.isnan(fe2o3_mean) or fe2o3_mean < 0.05:
        print(f"  WARNING: Fe2O3 reference too low ({fe2o3_mean:.4f}), skipping normalization")
        return grid, np.nan
    
    return grid / fe2o3_mean, fe2o3_mean

def parse_catalyst(catalyst):
    elements = []
    new_letter = ""
    for letter in catalyst:
        if letter.isupper() and new_letter != "":
            elements.append(new_letter)
            new_letter = letter
        else:
            new_letter += letter
    if new_letter != "":
        elements.append(new_letter)
        
    return elements

# Chemical Charactristics:
element_properties = {
    # Light absorbers
    # Band gaps are oxide forms (e.g. Fe → Fe₂O₃) since these are oxide catalysts
    # Ionization energies converted to eV (1 eV = 96.485 kJ/mol)
    # Light absorbers
    "V":  {"electronegativity": 1.63, "radius": 134, "ionization_eV": 6.74, "oxidation": 5, "oxide_bandgap_eV": 2.2},  # V₂O₅
    "Cr": {"electronegativity": 1.66, "radius": 128, "ionization_eV": 6.77, "oxidation": 3, "oxide_bandgap_eV": 3.4},  # Cr₂O₃
    "Mn": {"electronegativity": 1.55, "radius": 127, "ionization_eV": 7.43, "oxidation": 2, "oxide_bandgap_eV": 1.2},  # MnO
    "Fe": {"electronegativity": 1.83, "radius": 126, "ionization_eV": 7.90, "oxidation": 3, "oxide_bandgap_eV": 2.1},  # Fe₂O₃ — oxidative standard
    "Co": {"electronegativity": 1.88, "radius": 125, "ionization_eV": 7.88, "oxidation": 2, "oxide_bandgap_eV": 1.7},  # Co₃O₄
    "Ni": {"electronegativity": 1.91, "radius": 124, "ionization_eV": 7.64, "oxidation": 2, "oxide_bandgap_eV": 3.6},  # NiO
    "Cu": {"electronegativity": 1.90, "radius": 128, "ionization_eV": 7.72, "oxidation": 2, "oxide_bandgap_eV": 1.5},  # CuO — reductive standard

    # Structural
    "Al": {"electronegativity": 1.61, "radius": 143, "ionization_eV": 5.98, "oxidation": 3, "oxide_bandgap_eV": 8.8},  # Al₂O₃ — wide gap insulator
    "Si": {"electronegativity": 1.90, "radius": 111, "ionization_eV": 8.15, "oxidation": 4, "oxide_bandgap_eV": 9.0},  # SiO₂ — insulator
    "Ti": {"electronegativity": 1.54, "radius": 147, "ionization_eV": 6.82, "oxidation": 4, "oxide_bandgap_eV": 3.1},  # TiO₂
    "Ga": {"electronegativity": 1.81, "radius": 122, "ionization_eV": 6.00, "oxidation": 3, "oxide_bandgap_eV": 4.8},  # Ga₂O₃
    "Se": {"electronegativity": 2.55, "radius": 120, "ionization_eV": 9.75, "oxidation": 4, "oxide_bandgap_eV": 3.6},  # SeO₂
    "Y":  {"electronegativity": 1.22, "radius": 180, "ionization_eV": 6.22, "oxidation": 3, "oxide_bandgap_eV": 5.8},  # Y₂O₃
    "Zr": {"electronegativity": 1.33, "radius": 160, "ionization_eV": 6.63, "oxidation": 4, "oxide_bandgap_eV": 5.0},  # ZrO₂
    "Nb": {"electronegativity": 1.60, "radius": 146, "ionization_eV": 6.76, "oxidation": 5, "oxide_bandgap_eV": 3.4},  # Nb₂O₅
    "Mo": {"electronegativity": 2.16, "radius": 139, "ionization_eV": 7.09, "oxidation": 6, "oxide_bandgap_eV": 2.9},  # MoO₃
    "In": {"electronegativity": 1.78, "radius": 167, "ionization_eV": 5.78, "oxidation": 3, "oxide_bandgap_eV": 2.9},  # In₂O₃
    "Sn": {"electronegativity": 1.96, "radius": 141, "ionization_eV": 7.35, "oxidation": 4, "oxide_bandgap_eV": 3.6},  # SnO₂
    "Hf": {"electronegativity": 1.30, "radius": 159, "ionization_eV": 6.83, "oxidation": 4, "oxide_bandgap_eV": 5.8},  # HfO₂
    "Ta": {"electronegativity": 1.50, "radius": 146, "ionization_eV": 7.89, "oxidation": 5, "oxide_bandgap_eV": 3.9},  # Ta₂O₅
    "W":  {"electronegativity": 2.36, "radius": 139, "ionization_eV": 7.98, "oxidation": 6, "oxide_bandgap_eV": 2.7},  # WO₃

    # Charge compensators
    "Li": {"electronegativity": 0.98, "radius": 152, "ionization_eV": 5.39, "oxidation": 1, "oxide_bandgap_eV": 8.0},  # Li₂O
    "Na": {"electronegativity": 0.93, "radius": 186, "ionization_eV": 5.14, "oxidation": 1, "oxide_bandgap_eV": 5.0},  # Na₂O
    "Mg": {"electronegativity": 1.31, "radius": 160, "ionization_eV": 7.65, "oxidation": 2, "oxide_bandgap_eV": 7.8},  # MgO
    "K":  {"electronegativity": 0.82, "radius": 227, "ionization_eV": 4.34, "oxidation": 1, "oxide_bandgap_eV": 7.0},  # K₂O
    "Ca": {"electronegativity": 1.00, "radius": 197, "ionization_eV": 6.11, "oxidation": 2, "oxide_bandgap_eV": 7.1},  # CaO
    "Zn": {"electronegativity": 1.65, "radius": 134, "ionization_eV": 9.39, "oxidation": 2, "oxide_bandgap_eV": 3.4},  # ZnO
    "Rb": {"electronegativity": 0.82, "radius": 248, "ionization_eV": 4.18, "oxidation": 1, "oxide_bandgap_eV": 7.0},  # Rb₂O
    "Sr": {"electronegativity": 0.95, "radius": 215, "ionization_eV": 5.70, "oxidation": 2, "oxide_bandgap_eV": 5.3},  # SrO
    "Cd": {"electronegativity": 1.69, "radius": 151, "ionization_eV": 9.00, "oxidation": 2, "oxide_bandgap_eV": 2.2},  # CdO
    "Cs": {"electronegativity": 0.79, "radius": 265, "ionization_eV": 3.90, "oxidation": 1, "oxide_bandgap_eV": 7.0},  # Cs₂O
    "Ba": {"electronegativity": 0.89, "radius": 222, "ionization_eV": 5.21, "oxidation": 2, "oxide_bandgap_eV": 3.8},  # BaO
}

rows = []
cell_rows = []
def plate_to_cell_conversion(grid, catalyst_name):
    cell_rows = []
    if pd.isna(catalyst_name):
        return cell_rows
    elements = parse_catalyst(catalyst_name)
    if(len(elements) < 3):
        return cell_rows

    light_prop = element_properties[elements[0]]
    struct_prop = element_properties[elements[1]]
    charge_prop = element_properties[elements[2]]
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
                "L_Electro": light_prop["electronegativity"],
                "C_Electro": charge_prop["electronegativity"],
                "S_Electro": struct_prop["electronegativity"],
                "L_Radius": light_prop["radius"],
                "C_Radius": charge_prop["radius"],
                "S_Radius": struct_prop["radius"],
                "L_Ionization_eV": light_prop["ionization_eV"],
                "C_Ionization_eV": charge_prop["ionization_eV"],
                "S_Ionization_eV": struct_prop["ionization_eV"],
                "L_Oxidation": light_prop["oxidation"],
                "C_Oxidation": charge_prop["oxidation"],
                "S_Oxidation": struct_prop["oxidation"],
                "L_BandGap": light_prop["oxide_bandgap_eV"],
                "C_BandGap": charge_prop["oxide_bandgap_eV"],
                "S_BandGap": struct_prop["oxide_bandgap_eV"],
                "PhotoCurrent": PhotoCurrent
            })
    return cell_rows
            
'''READING DATA'''       
# for filename in catalyst_files:
raw_df = pd.read_excel(r"C:\Users\User\OneDrive\Dokumenty\Copy of shark data.xlsx")

blank_row_indices = raw_df.index[raw_df.iloc[:, 0:9].isnull().all(axis=1)].tolist()
split_rows = [0] + blank_row_indices + [len(raw_df)]
data_starts = raw_df[raw_df.iloc[:, 1].notnull()].index.tolist()

# Filter data_starts to only include the first row of each 8x8 block 
# (assuming blocks are separated by at least one row)
actual_starts = []
if data_starts:
    actual_starts.append(data_starts[0])
    for i in range(1, len(data_starts)):
        if data_starts[i] > data_starts[i-1] + 1:
            actual_starts.append(data_starts[i])

for start in actual_starts:
    # Each block is 8x8, so take the start row + 8 more
    df_slice = raw_df.iloc[start : start + 9].reset_index(drop=True)
    
    catalyst_name = df_slice.iloc[0, 1]
    
    # If the name is 'nan' or empty, skip
    if pd.isna(catalyst_name) or str(catalyst_name).lower() == 'nan':
        continue
        
   
    body = df_slice.iloc[1:9].reset_index(drop=True) 
    
    print(f"Processing Catalyst: {catalyst_name}, Shape: {body.shape}")
    # print(f"Rows {start}-{end}: {catalyst_name}, body shape: {body.shape}")
    features = extract_features(body, 8, 8)
    if features is None:
        print(f"  → extract_features returned None for {catalyst_name}")
        continue
    if features is not None:
        grid = features.pop("Grid")
        grid_normalized, fe2o3_ref = normalize_grid(grid)
        features["Catalysts"] = catalyst_name
        features["Fe2O3_reference"] = fe2o3_ref
        rows.append(features)
        cell_data = plate_to_cell_conversion(grid_normalized, catalyst_name)
        cell_rows.extend(cell_data)
print(f"Total rows: {len(raw_df)}")
print(f"Blank rows found at: {blank_row_indices}")
print(f"Number of splits: {len(split_rows)-1}")
# print(raw_df.iloc[0:99].to_string())
dataset = pd.DataFrame(rows)
dataset = dataset[dataset["Catalysts"].apply(
    lambda x: not pd.isna(x) and len(parse_catalyst(x)) == 3
)].reset_index(drop=True)
cell_dataset = pd.DataFrame(cell_rows)
cell_dataset = cell_dataset[cell_dataset['PhotoCurrent'] < 5.0] 
print(cell_dataset)
print(dataset)

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
# plt.show()
excel_data = cell_dataset[cell_dataset["Catalyst Name"].isin([
    'MnNbCd', 'MnNbNa', 'CoZrK'  # known Excel catalysts
])]["PhotoCurrent"]

image_data = cell_dataset[cell_dataset["Catalyst Name"].isin([
    'TaZnFe', 'YKCr', 'SeAlNi'  # known image catalysts
])]["PhotoCurrent"]

print("Excel photocurrent stats:")
print(excel_data.describe())
print("\nImage photocurrent stats:")
print(image_data.describe())
print(dataset[["Catalysts", "Fe2O3_reference"]].to_string())

# 2. Flag suspicious extractions
suspicious = dataset[dataset["Max_photocurrent"] > 3.0]
print("\nCatalysts with suspiciously high max values:")
print(suspicious[["Catalysts", "Max_photocurrent"]])
print(cell_dataset[cell_dataset["PhotoCurrent"] > 5.0][["Catalyst Name", "PhotoCurrent"]].drop_duplicates())


X = cell_dataset.drop(columns=["Catalyst Name", "S_Frac", "PhotoCurrent"])
y = cell_dataset["PhotoCurrent"]

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scores = []
r2_scores = []
all_true = []
all_predicted = []

for train_idx, test_idx in kfold.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae_scores.append(mae)
    r2_scores.append(r2)
    all_true.extend(y_test)
    all_predicted.extend(y_pred)
    
    fold_num = len(mae_scores)
    print(f"Fold {fold_num} | MAE: {mae:.4f} | R²: {r2:.4f} | "
          f"y_mean: {y_test.mean():.4f} | y_pred_mean: {y_pred.mean():.4f}")

print(f"\nMean MAE: {np.mean(mae_scores):.4f}")
print(f"Mean R²:  {np.mean(r2_scores):.4f}")

full_model = RandomForestRegressor(n_estimators=100, random_state=42)
full_model.fit(X, y)

importances = pd.Series(
    full_model.feature_importances_, 
    index=X.columns
).sort_values(ascending=False)

print(importances)
print(f"Median MAE: {np.median(mae_scores):.4f}")
print(f"Median R²:  {np.median(r2_scores):.4f}")


# logo = LeaveOneGroupOut()
# gpr_mae_scores = []
# gpr_r2_scores = []
# gpr_all_true = []
# gpr_all_predicted = []
# gpr_all_uncertainty = []
# catalyst_names = groups.unique()

# kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

# for (train_idx, test_idx), catalyst in zip(logo.split(X, y, groups), catalyst_names):
#     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
#     # Scale features
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     # Train GPR
#     gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, random_state=42)
#     gpr.fit(X_train_scaled, y_train)
    
#     # Predict with uncertainty
#     y_pred, y_std = gpr.predict(X_test_scaled, return_std=True)
    
#     mae = mean_absolute_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     gpr_mae_scores.append(mae)
#     gpr_r2_scores.append(r2)
#     gpr_all_true.extend(y_test)
#     gpr_all_predicted.extend(y_pred)
#     gpr_all_uncertainty.extend(y_std)
    
#     print(f"{catalyst:12} | MAE: {mae:.4f} | R²: {r2:.4f} | "
#           f"y_mean: {y_test.mean():.4f} | y_pred_mean: {y_pred.mean():.4f} | "
#           f"avg_uncertainty: {y_std.mean():.4f}")

# print(f"\nMean MAE:   {np.mean(gpr_mae_scores):.4f}")
# print(f"Median R²:  {np.median(gpr_r2_scores):.4f}")
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ax1.scatter(all_true, all_predicted, alpha=0.4)
# ax1.plot([0, max(all_true)], [0, max(all_true)], 'r--', label='Perfect prediction')
# ax1.set_xlabel("Measured Photocurrent")
# ax1.set_ylabel("Predicted Photocurrent")
# ax1.set_title("Random Forest")
# ax1.legend()

# ax2.scatter(gpr_all_true, gpr_all_predicted, alpha=0.4)
# ax2.plot([0, max(gpr_all_true)], [0, max(gpr_all_true)], 'r--', label='Perfect prediction')
# ax2.set_xlabel("Measured Photocurrent")
# ax2.set_ylabel("Predicted Photocurrent")
# ax2.set_title("Gaussian Process")
# ax2.legend()

# plt.tight_layout()
# plt.savefig("model_comparison.png", dpi=150, bbox_inches='tight')
# plt.close()
plt.show()