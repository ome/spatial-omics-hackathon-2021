from pathlib import Path
import re

import loompy
import pandas as pd


def is_ion_formula(string: str) -> bool:
    # At least two chemical elements with number or no number, optional positive or negative adduct,
    # optional numbered suffix (to match also annotation_id)
    return (
        re.match(r"^(?:[A-Z][a-z]?\d*){2,}(?:[+-][A-Z][a-z]?)?(_\d+)?$", string)
        is not None
    )


csv_path = Path("../spatiomolecular_matrix.csv")
df = pd.read_csv(csv_path)
point_cols = ["center_y", "center_x"]
ion_intensity_cols = [c for c in df.columns if is_ion_formula(c)]

points_np = df[point_cols].values
non_points_df = df[[c for c in df.columns if c not in point_cols]]

# A pandas DataFrame holding row metadata
row_metadata_df = non_points_df
# A pandas DataFrame holding column metadata
# col_metadata_df = ...
# A numpy ndarray holding the main dataset
data = points_np

loompy.create(
    filename="spatiomolecular_matrix.loom",
    layers=data,
    row_attrs=row_metadata_df.to_dict("list"),
    col_attrs={},  # col_metadata_df.to_dict("list")
)

with loompy.connect("spatiomolecular_matrix.loom") as ds:
    ds.row_attrs["area"] # int64 column
    # Use scan to iterate over views of "chunks"
    # or create a view of the whole dataset:
    view = ds.view[:, :]
    data = view[:]
    row_attributes = view.ra # AttributeManager
    # Each row attribute is a numpy array
    # Storage is a dict of attribute names (str) and values (np.array)
    row_attributes_df = pd.DataFrame(view.ra.storage)
    row_filter_np = (view.ra.area < 3000) & (view.ra.area >= 2000)
    filtered_points_np = view[row_filter_np, :]

