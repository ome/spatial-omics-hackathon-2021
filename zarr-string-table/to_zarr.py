import random
import string
import lorem

import pandas as pd

import numpy as np
import zarr

from numcodecs import GZip


def fixed_length():
    tab = pd.read_csv("../spots_reduced.csv")
    col = np.array([str(v) for v in tab["target"].values])

    f = zarr.open("table.zarr")
    f.create_dataset("fixed-length", data=col, chunks=(2048,), compressor=GZip())


def fixed_length_random():

    def rand_string(mlen):
        letters = string.ascii_lowercase
        slen = np.random.randint(2, 10)
        return "".join(random.choice(letters) for _ in range(slen))

    col = np.array(
        [rand_string(10) for _ in range(2048)]
    )

    f = zarr.open("table.zarr")
    f.create_dataset("fixed-length-random", data=col, chunks=(512,), compressor=GZip())


def fixed_length_hallo():

    col = np.array(
        [f"hallo-{i}" for i in range(8)]
    )
    print(col)

    f = zarr.open("table.zarr")
    f.create_dataset("fixed-length-hallo", data=col, chunks=(4,), compressor=GZip())


def float_reference():
    data = np.random.rand(2048).astype("float32")
    f = zarr.open("table.zarr")
    f.create_dataset("float", data=data, chunks=(512,), compressor=GZip())



# float_reference()
# fixed_length_random()
# fixed_length_lorem()
fixed_length_hallo()
