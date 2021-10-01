from typing import Dict, Literal

import numpy as np
import pandas as pd
import zarr

DIMENSION_AXES = ("t", "c", "z", "y", "x")
DimensionAxisType = Literal["t", "c", "z", "y", "x"]


def write_table(
    group: zarr.Group,
    table: pd.DataFrame,
    coordinates_columns: Dict[DimensionAxisType, str],
    column_chunksize: int = 100,
    overwrite: bool = False,
):
    # TODO: Maybe it is more flexible if we don't require coordinates to be contained as columns in
    #  the table, but can be provide as a separate array. Then coordinates_columns don't need to be
    #  sliced from the table.
    coordinates_dtypes = table[coordinates_columns.values()].dtypes
    assert len(set(coordinates_dtypes)) == 1

    # Create the coordinates as one Zarr array
    coordinates = np.full(
        shape=(len(table), len(DIMENSION_AXES)),
        dtype=coordinates_dtypes[0],
        fill_value=0.0,
    )
    # TODO: Or fill_value=np.nan for not provided dimensions?
    for i, d in enumerate(DIMENSION_AXES):
        if d in coordinates_columns.keys():
            coordinates[:, i] = table[coordinates_columns[d]].values

    group.create_dataset(
        name="coordinates",
        data=coordinates,
        chunks=(column_chunksize, len(DIMENSION_AXES)),
        dimension_separator="/",
        overwrite=overwrite,
    )

    # Create for each column individually one Zarr array
    table_rest = table.drop(coordinates_columns.values(), axis="columns")
    for column in table_rest.columns:
        series = table_rest[column]
        # TODO: If there are restrictions on Zarr group (directory) names, we need to normalize the
        #  series name (or use generic names like c1, c2) and have an ordered mapping to the
        #  original Unicode name.
        # TODO: avoid collision of Zarr array names for columns in table_rest with "coordinates"
        dtype = series.dtype
        if dtype==object:
            # FIXME: Not sure what to do with other non-string object dtypes
            #  Here the case was a column of strings with missing values as float nan.
            dtype = str
        array = group.create_dataset(
            name=series.name,
            data=series.values.astype(dtype),
            dtype=dtype,
            chunks=column_chunksize,
            dimension_separator="/",
            overwrite=overwrite,
        )
        array.attrs["table_column"] = {
            "name": series.name,
        }

    group.attrs["table"] = {
        "axes": DIMENSION_AXES,
        "column_names": table_rest.columns.tolist(),
        "column_chunksize": column_chunksize,
    }
