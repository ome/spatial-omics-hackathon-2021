import re

import loompy
import pandas as pd

from loom_table.loom_utils import csv_to_loom


def is_ion_formula(string: str) -> bool:
    # At least two chemical elements with number or no number, optional positive or negative adduct,
    # optional numbered suffix (to match also annotation_id)
    return (
        re.match(r"^(?:[A-Z][a-z]?\d*){2,}(?:[+-][A-Z][a-z]?)?(_\d+)?$", string)
        is not None
    )


point_cols = ["center_y", "center_x"]
ion_intensity_cols = [
    c for c in pd.read_csv("../spatiomolecular_matrix.csv").columns if is_ion_formula(c)
]

csv_to_loom(
    csv_path="../spatiomolecular_matrix.csv",
    dense_columns=point_cols,
    out_path="spatiomolecular_matrix.zarr.loom",
)

with loompy.connect("spatiomolecular_matrix.zarr.loom") as ds:
    ds.row_attrs["area"]  # int64 column
    # Use scan to iterate over views of "chunks"
    # or create a view of the whole dataset:
    view = ds.view[:, :]
    data = view[:]
    row_attributes = view.ra  # AttributeManager
    # Each row attribute is a numpy array
    # Storage is a dict of attribute names (str) and values (np.array)
    row_attributes_df = pd.DataFrame(view.ra.storage)
    row_filter_np = (view.ra.area < 3000) & (view.ra.area >= 2000)
    filtered_points_np = view[row_filter_np, :]
