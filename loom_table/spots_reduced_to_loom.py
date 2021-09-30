from loom_table.loom_utils import csv_to_loom


point_cols = ["zc", "yc", "xc"]

csv_to_loom(
    csv_path="../spots_reduced.csv",
    dense_columns=point_cols,
    out_path="spots_reduced.zarr.loom",
)
