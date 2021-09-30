import os

import anndata


def create_anndata_from_csv(fpath: os.PathLike) -> anndata.AnnData:
    """Create the anndata object from the example csv

    Parameters
    ----------
    fpath : os.PathLike
        The filepath to the csv file.

    Returns
    -------
    ann : anndata.AnnData
    """