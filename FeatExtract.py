from pathlib import Path

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
        "Reproducability": mean,
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
# def ParetoSelection():
