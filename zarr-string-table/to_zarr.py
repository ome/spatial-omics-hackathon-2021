import pandas as pd

import numpy as np
import zarr


tab = pd.read_csv("../spots_reduced.csv")
col = np.array([str(v) for v in tab["target"].values])
print(np.unique(col))

f = zarr.open("table.zarr")
f.create_dataset("fixed-length", data=col, chunks=(2048,))
