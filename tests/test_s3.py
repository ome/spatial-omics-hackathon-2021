import s3fs
import os
import pytest_benchmark
import pandas as pd
import pytest


@pytest.fixture
def s3_filesystem():
    os.environ["AWS_ACCESS_KEY_ID"] = "weak_access_key"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "weak_secret_key"

    return s3fs.S3FileSystem(client_kwargs={"endpoint_url": "http://localhost:9090"})


def test_s3_filesystem(s3_filesystem):
    assert s3_filesystem.exists("csv/eiffel-tower-smlm.csv")


def test_read_csv(s3_filesystem):
    with s3_filesystem.open("csv/eiffel-tower-smlm.csv") as f:
        df = pd.read_csv(f)

    assert df.shape == (31464, 3)
