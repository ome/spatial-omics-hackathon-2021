from pathlib import Path
import re

import loompy
import pandas as pd

csv_path = Path("../spots_reduced.csv")
df = pd.read_csv(csv_path)
point_cols = ["z", "y", "x"]

points_np = df[point_cols].values
non_points_df = df[[c for c in df.columns if c not in point_cols]]

# A pandas DataFrame holding row metadata
row_metadata_df = non_points_df
# A pandas DataFrame holding column metadata
# col_metadata_df = ...
# A numpy ndarray holding the main dataset
data = points_np

loompy.create(
    filename="spots_reduced.loom",
    layers=data,
    row_attrs=row_metadata_df.to_dict("list"),
    col_attrs={},  # col_metadata_df.to_dict("list")
)
