import pandas as pd

import numpy as np
import zarr

from numcodecs import GZip


def fixed_length():

    tab = pd.read_csv("../spots_reduced.csv")
    col = np.array([str(v) for v in tab["target"].values])

    f = zarr.open("table.zarr")
    f.create_dataset("fixed-length", data=col, chunks=(2048,), compressor=GZip())


def float_reference():
    data = np.random.rand(2048).astype("float32")
    f = zarr.open("table.zarr")
    f.create_dataset("float", data=data, chunks=(512,), compressor=GZip())


float_reference()
