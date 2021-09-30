from pathlib import Path
import re

import loompy
import pandas as pd

csv_path = Path("../spots_reduced.csv")
df = pd.read_csv(csv_path)
DENSE_COLUMNS = ["zc", "yc", "xc"]

points_np = df[DENSE_COLUMNS].values
obs = df.drop(DENSE_COLUMNS, axis="columns")

# A pandas DataFrame holding row metadata
row_metadata_df = obs
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
