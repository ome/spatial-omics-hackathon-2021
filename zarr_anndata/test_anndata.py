from functools import partial
import os

import anndata

from zarr_anndata.anndata_utils import setup_anndata, ann_to_zarr


def test_anndata_to_zarr(benchmark, tmp_path):
    """Test writing anndata to zarr"""
    setup_func = partial(
        setup_anndata, fpath="data/spots_reduced.csv", out_dir=tmp_path
    )
    benchmark.pedantic(ann_to_zarr, setup=setup_func, rounds=20)


def test_zarr_to_anndata(benchmark, tmp_path):
    """Test loading anndata from zarr"""
    _, setup_output = setup_anndata(fpath="data/spots_reduced.csv", out_dir=tmp_path)
    ann_obj = setup_output["ann_obj"]
    tmp_out_path = setup_output["out_path"]
    ann_obj.write_zarr(tmp_out_path)

    read_func = partial(anndata.read_zarr, tmp_out_path)
    benchmark.pedantic(read_func, rounds=20)
