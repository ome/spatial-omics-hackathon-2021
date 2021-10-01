import warnings
from typing import Dict, Literal
import numpy as np
import pandas as pd
import zarr

DIMENSION_AXES = ("t", "c", "z", "y", "x")
DimensionAxisType = Literal["t", "c", "z", "y", "x"]


def to_dataframe(
    group: zarr.Group
) -> pd.DataFrame:
    assert "table" in group.attrs
    table_attrs = group.attrs["table"]
    assert "axes" in table_attrs
    assert "column_names" in table_attrs
    # column_names = list(table_attrs["axes"]) + list(table_attrs["columns_names"])
    coordinates = pd.DataFrame(group["coordinates"], columns=table_attrs["axes"])
    table_rest = pd.DataFrame({
        column: group[column] for column in table_attrs["column_names"]
    })
    if set(table_attrs["axes"]) & set(table_rest.columns):
        warnings.warn("Table contains column names reserved for dimension axes. "
                      "Dropping them from returned dataframe.")
        table_rest = table_rest.drop(table_rest["axes"], axis="columns", errors="ignore")
    return pd.concat([coordinates, table_rest], axis=1)
