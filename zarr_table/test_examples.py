import pandas as pd
import zarr

from zarr_table.reader import to_dataframe
from zarr_table.writer import write_table


def test_spots_reduced():
    group = zarr.open("spots_reduced.zarr", mode="w")
    table = pd.read_csv("../spots_reduced.csv")
    # Not a good example, since some contained columns collision with dimension axes.
    table = table.drop(["z", "y", "x"], axis="columns")
    # No masking supported yet, so missing values (nan) must have same dtype as column.
    table["target"] = table["target"].astype(str)
    write_table(
        group=group,
        table=table,
        coordinates_columns={"z": "zc", "y": "yc", "x": "xc"},
        overwrite=True,
    )
    expected_restored_table = table.rename(columns={"zc": "z", "yc": "y", "xc": "x"})
    restored_table = to_dataframe(group)
    assert set(expected_restored_table.columns) <= set(restored_table.columns)
    # It may contain more dimension axes
    pd.testing.assert_frame_equal(
        restored_table[expected_restored_table.columns], expected_restored_table
    )


def test_spatiomolecular_matrix():
    group = zarr.open("spatiomolecular_matrix.zarr", mode="w")
    table = pd.read_csv("../spatiomolecular_matrix.csv")
    write_table(
        group=group,
        table=table,
        coordinates_columns={"y": "center_y", "x": "center_x"},
        overwrite=True,
    )
    expected_restored_table = table.rename(columns={"center_y": "y", "center_x": "x"})
    restored_table = to_dataframe(group)
    assert set(expected_restored_table.columns) <= set(restored_table.columns)
    # It may contain more dimension axes
    pd.testing.assert_frame_equal(
        restored_table[expected_restored_table.columns], expected_restored_table
    )
