from typing import Tuple

import pandas as pd
import zarr
from anndata import AnnData

from zarr_table.writer import (
    ATTR_COL_ORD,
    ATTR_INDEX,
    GROUP_OBS,
    GROUP_TABLE,
    GROUP_VAR,
    GROUP_X,
)


def table_to_dataframe(group: zarr.Group) -> pd.DataFrame:
    """
    Converts a Zarr table (X and obs) to a Pandas dataframe.

    Args:
        group: The Zarr table group container

    Returns:
        A dataframe with point coordinates and annotations
    """
    assert group.basename == GROUP_TABLE
    x, obs, var = _table_to_x_obs_var(group)
    return pd.concat([x, obs], axis=1)


def table_to_anndata(group: zarr.Group) -> AnnData:
    """
    Converts a Zarr table (X, obs and var) to an AnnData object.

    Args:
        group: The Zarr table group container

    Returns:
        An AnnData object
    """
    assert group.basename == GROUP_TABLE
    x, obs, var = _table_to_x_obs_var(group)
    return AnnData(X=x, obs=obs, var=var)


def _group_to_dataframe(group: zarr.Group):
    dct = {}
    columns = group.attrs[ATTR_COL_ORD]
    for col in columns:
        dct[col] = group[col]
    df = pd.DataFrame(dct)
    df.set_index(group.attrs[ATTR_INDEX], inplace=True)
    return df


def _table_to_x_obs_var(
    group: zarr.Group,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert group.basename == GROUP_TABLE
    x = None
    obs = None
    var = None
    if GROUP_X in group:
        x = pd.DataFrame(group[GROUP_X])
    if GROUP_OBS in group:
        obs = _group_to_dataframe(group[GROUP_OBS])
        if GROUP_X in group:
            x.index = obs.index
    if GROUP_VAR in group:
        var = _group_to_dataframe(group[GROUP_VAR])
        if GROUP_X in group:
            x.columns = var.index
    return x, obs, var
