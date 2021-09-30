#!/usr/bin/env python

import numpy as np
import pandas as pd
import tables as tb
import os
import time


def write_pandas(input, output):
    df = pd.read_csv(input)
    df.to_hdf(
        output,
        key="spots",
        format="table",
        mode="w",
        data_columns=["ID", "z", "y", "x", "target"],
    )


def write_tables(input, output):
    h5 = tb.open_file(output, "w")
    data = np.genfromtxt(
        input,
        dtype=(
            [
                ("ID", "i8"),
                ("intensity", "f4"),
                ("z", "i8"),
                ("y", "i8"),
                ("x", "i8"),
                ("radius", "f4"),
                ("spot_id", "i8"),
                ("z_min", "f4"),
                ("z_max", "f4"),
                ("y_min", "f4"),
                ("y_max", "f4"),
                ("x_min", "f4"),
                ("x_max", "f4"),
                ("features", "f4"),
                ("xc", "f4"),
                ("yc", "f4"),
                ("zc", "f4"),
                ("target", "f4"),
                ("distance", "f4"),
                ("passes_thresholds", "?"),
            ]
        ),
        comments="#",
        delimiter=",",
        skip_header=1,
    )
    group = h5.create_group(h5.root, "spots")
    table = h5.create_table(
        group, description=data, name="table", title="table", expectedrows=len(data)
    )
    table.cols.ID.create_index()
    table.cols.z.create_index()
    table.cols.y.create_index()
    table.cols.x.create_index()
    table.cols.target.create_index()
    h5.close()


def query_hdf5(input, query):
    t = tb.open_file(input)
    try:
        return list(t.root.spots.table.where(query))
    finally:
        t.close()
