import os
import pandas as pd
from typing import List

from loompy import loompy


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

    loompy.create(
        filename=out_path,
        layers=data,
        row_attrs=obs.to_dict("list"),
        col_attrs={},
    )
