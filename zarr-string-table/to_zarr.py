import pandas as pd

import numpy as np
import zarr

from numcodecs import GZip


tab = pd.read_csv("../spots_reduced.csv")
col = np.array([str(v) for v in tab["target"].values])

f = zarr.open("table.zarr")
f.create_dataset("fixed-length", data=col, chunks=(2048,), compressor=GZip())
