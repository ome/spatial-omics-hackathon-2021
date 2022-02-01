import anndata.tests.helpers
import numpy as np
import pandas as pd
import zarr
from anndata import AnnData

from zarr_table.reader import table_to_anndata, table_to_dataframe
from zarr_table.writer import DIMENSION_AXES, write_points_dataframe


def test_spots_reduced():
    df = pd.read_csv("data/spots_reduced.csv")
    index_column = "ID"
    df[index_column] = df[index_column].astype(str)
    df.set_index("ID", inplace=True)
    # Not a good example, since some contained columns collision with dimension axes.
    df = df.drop(["z", "y", "x"], axis="columns")
    # No masking supported yet, so missing values (nan) must have same dtype as column.
    df["target"] = df["target"].astype(str)
    point_columns = {"z": "zc", "y": "yc", "x": "xc"}
    table = write_points_dataframe(
        parent_group=zarr.open("spots_reduced.zarr", mode="w"),
        dataframe=df,
        point_columns=point_columns,
        overwrite=True,
    )

    actual_dataframe = table_to_dataframe(table)
    # It may contain more dimension axes
    expected_dataframe = df.rename(
        columns={col: dim for dim, col in point_columns.items()}
    )
    expected_dataframe.insert(0, "t", np.nan)
    expected_dataframe.insert(1, "c", np.nan)
    expected_dataframe.insert(2, "z", expected_dataframe.pop("z"))
    expected_dataframe.insert(3, "y", expected_dataframe.pop("y"))
    expected_dataframe.insert(4, "x", expected_dataframe.pop("x"))
    assert set(expected_dataframe.columns) == set(actual_dataframe.columns)
    pd.testing.assert_frame_equal(actual_dataframe, expected_dataframe)

    actual_adata = table_to_anndata(table)
    expected_adata = AnnData(
        X=expected_dataframe[list(DIMENSION_AXES)],
        obs=expected_dataframe.drop(list(DIMENSION_AXES), axis="columns"),
        var=pd.DataFrame([], index=pd.Index(DIMENSION_AXES, name="index")),
    )
    anndata.tests.helpers.assert_adata_equal(actual_adata, expected_adata)


def test_spatiomolecular_matrix():
    df = pd.read_csv("data/spatiomolecular_matrix.csv")
    index_column = "cell_id"
    df[index_column] = df[index_column].astype(str)
    df.set_index(index_column, inplace=True)
    table = write_points_dataframe(
        parent_group=zarr.open("spatiomolecular_matrix.zarr", mode="w"),
        dataframe=df,
        point_columns={"y": "center_y", "x": "center_x"},
        overwrite=True,
    )

    actual_dataframe = table_to_dataframe(table)
    # It may contain more dimension axes
    expected_dataframe = df.rename(columns={"center_y": "y", "center_x": "x"})
    expected_dataframe.insert(0, "t", np.nan)
    expected_dataframe.insert(1, "c", np.nan)
    expected_dataframe.insert(2, "z", np.nan)
    expected_dataframe.insert(3, "y", expected_dataframe.pop("y"))
    expected_dataframe.insert(4, "x", expected_dataframe.pop("x"))
    assert set(expected_dataframe.columns) == set(actual_dataframe.columns)
    pd.testing.assert_frame_equal(actual_dataframe, expected_dataframe)

    actual_adata = table_to_anndata(table)
    expected_adata = AnnData(
        X=expected_dataframe[list(DIMENSION_AXES)],
        obs=expected_dataframe.drop(list(DIMENSION_AXES), axis="columns"),
        var=pd.DataFrame([], index=pd.Index(DIMENSION_AXES, name="index")),
    )
    anndata.tests.helpers.assert_adata_equal(actual_adata, expected_adata)
