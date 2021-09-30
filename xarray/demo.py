# %% Short demo of using xarray for mixed datasets
import zarr
import xarray as xr
import pandas as pd
import numpy as np
from numcodecs import GZip
import skimage
import matplotlib.pyplot as plt

spatiomolecular_matrix = pd.read_csv("../spatiomolecular_matrix.csv")
spots_reduced = pd.read_csv("../spots_reduced.csv")
string_list = spots_reduced["target"]
spots_reduced_indexed = spots_reduced.set_index(["x","y"])

data = skimage.data.astronaut()
data = np.random.rand(2048).astype("float32")

data = skimage.data.binary_blobs(
    length=2048, blob_size_fraction=0.1, n_dim=2, volume_fraction=0.5, seed=None)

xr_data = xr.DataArray(data,coords={"x":np.arange(0,2048), "y":np.arange(0,2048)})
# %%
flat_image = xr_data.to_pandas().unstack().rename("Intensity")
#  %%
df = spots_reduced_indexed.join(flat_image,how="right")
xr_df = df.to_xarray()
# %% Save zarr
# xr_df.to_zarr("temp2",mode="w",compute=False)
#  %% Get the full image back

image_plane = xr_df.isel(x=slice(0, None), y=slice(0, None))["Intensity"]
plt.imshow(image_plane)
plt.show()

# %% Get tabular data
spot_ids = xr_df["spot_id"].to_series().dropna()

# %%
