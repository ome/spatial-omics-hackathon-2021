from functools import partial
import os

import pytest

from zarr_anndata.pytables_util import write_pandas, write_tables, query_hdf5


query = "(y > 2000) & (y < 3000) & (x > 1000) & (x < 2000)"


def test_csv_to_hdf5_pandas(benchmark, tmp_path, request):
    """Test writing csv to hdf5 with pandas"""

    csv = f"{request.config.invocation_dir}/spots_reduced.csv"
    filename = tmp_path / "pandas.h5"
    setup_func = partial(write_pandas, input=csv, output=filename)
    read_func = partial(query_hdf5, input=filename, query=query)
    benchmark.pedantic(
        read_func,
        setup=setup_func,
        rounds=20
    )


def test_csv_to_hdf5_tables(benchmark, tmp_path, request):
    """Test writing csv to hdf5 with pandas"""

    filename = tmp_path / "tables.h5"
    csv = f"{request.config.invocation_dir}/spots_reduced.csv"
    setup_func = partial(write_tables, input=csv, output=filename)
    read_func = partial(query_hdf5, input=filename, query=query)
    benchmark.pedantic(
        read_func,
        setup=setup_func,
        rounds=20
    )
