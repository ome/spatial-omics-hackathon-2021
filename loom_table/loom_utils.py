import os
import tempfile
from pathlib import Path
from typing import List

import h5py
import numpy as np
import pandas as pd
import zarr
from loompy import loompy


# from loompy.loompy_to_zarr import hdf5_to_zarr

def hdf5_to_zarr(hdf5_file, zarr_group=None):
    try:
        unicode
    except NameError:
        unicode = str

    opened = False
    if isinstance(hdf5_file, (bytes, unicode)):
        hdf5_filename = hdf5_file
        hdf5_file = h5py.File(hdf5_file, "r")
        opened = True
    else:
        hdf5_filename = hdf5_file.filename

    if zarr_group is None:
        zarr_name = os.path.splitext(hdf5_filename)[0] + os.extsep + "zarr.loom"
        zarr_group = zarr.open_group(zarr_name, mode="w")

    def copy(name, obj):
        if isinstance(obj, h5py.Group):
            zarr_obj = zarr_group.create_group(name)
        elif isinstance(obj, h5py.Dataset):
            if obj.dtype == "|O":
                # string
                try:
                    # string array
                    zarr_obj = zarr_group.create_dataset(name, data=obj[:].astype(np.string_), chunks=obj.chunks)
                except:
                    # string scalar
                    obj = str(obj[()])
                    zarr_obj = zarr_group.create_dataset(name, data=obj)
                print(name, obj)
            else:
                zarr_obj = zarr_group.create_dataset(name, data=obj, chunks=obj.chunks)
        else:
            assert False, "Unsupport HDF5 type."

        try:
            zarr_obj.attrs.update(obj.attrs)
        except AttributeError:
            pass

    hdf5_file.visititems(copy)

    if opened:
        hdf5_file.close()

    return zarr_group


def csv_to_loom(csv_path: os.PathLike, dense_columns: List[str], out_path: os.PathLike):
    """Convert an CSV file to loom

    Parameters
    ----------
    csv_path : os.PathLike
        The filepath to the csv file.
    dense_columns : list
        List of column names from the CSV table to include in the loom main matrix. Other columns
        are stored as row attributes.
    out_path : os.PathLike
        The filepath where to save the loom file.

    Returns
    -------
    Nothing
    """
    df = pd.read_csv(csv_path)
    data = df[dense_columns].values
    obs = df.drop(dense_columns, axis="columns")

    # With proper zarr support according to https://github.com/pjb7687/loompy-zarr
    # this would work, but it doesn't:
    # with loompy.new(filename=out_path, file_attrs={"mode": "r+"}, backend="zarr") as ds:
    #     ds.add_columns(
    #         layers=data,
    #         row_attrs=obs.to_dict("list"),
    #         col_attrs={},
    #     )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = str(Path(temp_dir) / Path(out_path).with_suffix(".loom").name)
        loompy.create(
            filename=temp_file,
            layers=data,
            row_attrs=obs.to_dict("list"),
            col_attrs={},
        )
        store = zarr.DirectoryStore(path=out_path, dimension_separator="/")
        zarr_group = zarr.open_group(store, mode="w")
        hdf5_to_zarr(temp_file, zarr_group=zarr_group)
