import os
from typing import Dict, Tuple, Union

import anndata
import pandas as pd


DENSE_COLUMNS = [
    'zc',
    'yc',
    'xc'
]


def setup_anndata(
        fpath: os.PathLike,
        out_dir: os.PathLike
) -> Tuple[Tuple[()], Dict[str, Union[anndata.AnnData, os.PathLike]]]:
    """Create the anndata object from the example csv

    Parameters
    ----------
    fpath : os.PathLike
        The filepath to the csv file.

    Returns
    -------
    benchmark_args : Dict[str, Union[anndata.AnnData, os.PathLike]]
        The input arguments for ann_to_zarr. 'ann_obj' is the
        AnnData object to write, 'out_path' is the file path
        to write the zarr to.
    """
    df = pd.read_csv(fpath)

    # get the dense array
    dense_array = df[DENSE_COLUMNS].to_numpy()

    # drop the dense array from the table
    obs = df.drop(DENSE_COLUMNS, axis='columns')

    # create the AnnData object
    ann_obj = anndata.AnnData(X=dense_array, obs=obs)

    # make the filepath
    out_path = os.path.join(out_dir, 'test.zarr')

    return (), {'ann_obj': ann_obj, 'out_path': out_path}


def ann_to_zarr(ann_obj: anndata.AnnData, out_path: os.PathLike):
    ann_obj.write_zarr(out_path)
