from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import zarr

GROUP_TABLE = "table"
GROUP_X = "x"
GROUP_OBS = "obs"
GROUP_VAR = "var"
ATTR_INDEX = "_index"
ATTR_COL_ORD = "column-order"
ATTR_ENC_TYP = "encoding-type"
ATTR_ENC_VER = "encoding-version"
ENCODING_TYPE_DATAFRAME = "dataframe"
ENCODING_VERSION = "0.1.0"
DIMENSION_AXES = ("t", "c", "z", "y", "x")
DimensionAxisType = Literal["t", "c", "z", "y", "x"]
DIMENSION_SEPARATOR = "/"
DimensionSeparatorType = Literal["/", "."]


def write_points_dataframe(
    parent_group: zarr.Group,
    dataframe: pd.DataFrame,
    point_columns: Dict[DimensionAxisType, str],
    column_chunksize: int = 100,
    overwrite: bool = False,
) -> zarr.Group:
    """
    Adds a points dataframe as an NGFF table to a Zarr dataset

    Args:
        parent_group: The Zarr group in which to add the table group container
        dataframe: A dataframe containing point coordinates and annotations
        point_columns: A mapping from column names in the dataframe to dimension axes names
        column_chunksize: The vertical chunk size for the X array and obs arrays
        overwrite: If True, replace any existing array or group with the given name.

    Returns:
        The table group container
    """
    coordinates_dtypes = dataframe[point_columns.values()].dtypes
    assert len(set(coordinates_dtypes)) == 1

    # Create the coordinates as one Zarr array
    points = np.full(
        shape=(len(dataframe), len(DIMENSION_AXES)),
        dtype=coordinates_dtypes[0],
        fill_value=np.nan,
    )
    for i, d in enumerate(DIMENSION_AXES):
        if d in point_columns.keys():
            points[:, i] = dataframe[point_columns[d]].values

    obs = dataframe.drop(point_columns.values(), axis="columns")
    var = pd.DataFrame([], index=DIMENSION_AXES)

    table = add_table(
        parent_group=parent_group,
        x=points,
        obs=obs,
        var=var,
        chunks=(points.shape[1], column_chunksize),
        overwrite=overwrite,
    )

    return table


def add_table(
    parent_group: zarr.Group,
    x: Optional[np.ndarray] = None,
    obs: Optional[pd.DataFrame] = None,
    var: Optional[pd.DataFrame] = None,
    chunks: Tuple[int, int] = (100, 100),
    dimension_separator: DimensionSeparatorType = DIMENSION_SEPARATOR,
    overwrite: bool = False,
) -> zarr.Group:
    """
    Adds an NGFF table to a Zarr dataset

    Args:
        parent_group: The Zarr group in which to add the table group container
        x: Main array of a single type
        obs: Table of annotations on the rows in X. The rows in obs MUST be index-matched to the
            rows in X (if present).
        var: Table of annotations on the columns in X. The rows in var MUST be index-matched to the
            columns in X (if present).
        chunks: A tuple of vertical and horizontal chunk size for the X array
        overwrite: If True, replace any existing array or group with the given name.
        dimension_separator: Separator placed between the dimensions of a chunk.

    Returns:
        The table group container
    """
    group = parent_group.require_group(GROUP_TABLE, overwrite=overwrite)
    if x is not None:
        group.create_dataset(
            name=GROUP_X,
            data=x,
            chunks=chunks,
            dimension_separator=dimension_separator,
            overwrite=overwrite,
        )

    if obs is not None:
        _add_dataframe(
            parent_group=group,
            name=GROUP_OBS,
            dataframe=obs,
            column_chunksize=chunks[0],
            dimension_separator=dimension_separator,
            overwrite=overwrite,
        )

    if var is not None:
        _add_dataframe(
            parent_group=group,
            name=GROUP_VAR,
            dataframe=var,
            column_chunksize=chunks[1],
            dimension_separator=dimension_separator,
            overwrite=overwrite,
        )

    return group


def _add_dataframe(
    parent_group: zarr.Group,
    name: str,
    dataframe: pd.DataFrame,
    column_chunksize: int,
    index_column: Optional[str] = None,
    dimension_separator: DimensionSeparatorType = DIMENSION_SEPARATOR,
    overwrite: bool = False,
) -> zarr.Group:
    group = parent_group.require_group(name)
    if index_column is None:
        index_column = dataframe.index.name or "index"
    dataframe = dataframe.reset_index()
    # AnnData only support string index columns
    dataframe[index_column] = dataframe[index_column].astype(str)
    for column in dataframe.columns:
        series = dataframe[column]
        # TODO: If there are restrictions on Zarr group (directory) names, we need to normalize the
        #  series name (or use generic names like c1, c2) and have an ordered mapping to the
        #  original Unicode name.
        # TODO: Avoid collision Zarr array names (columns) with Zarr attributes
        dtype = series.dtype
        if dtype == object:
            # FIXME: Not sure what to do with other non-string object dtypes
            #  Here the case was a column of strings with missing values as float nan.
            dtype = str
        array = group.create_dataset(
            name=series.name,
            data=series.values.astype(dtype),
            dtype=dtype,
            chunks=column_chunksize,
            dimension_separator=dimension_separator,
            overwrite=overwrite,
        )
    group.attrs[ATTR_INDEX] = index_column
    group.attrs[ATTR_COL_ORD] = dataframe.columns.to_list()
    group.attrs[ATTR_ENC_TYP] = ENCODING_TYPE_DATAFRAME
    group.attrs[ATTR_ENC_VER] = ENCODING_VERSION
    return group
