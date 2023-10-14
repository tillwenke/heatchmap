import rasterio
from matplotlib import pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np

points = gpd.read_file("../waiting_time_per_point.csv")
points.wait = points.wait.astype(float)
points.lat = points.lat.astype(float)
points.lon = points.lon.astype(float)

points = gpd.read_file("../waiting_time_per_point.csv")
points.wait = points.wait.astype(float)
points.lat = points.lat.astype(float)
points.lon = points.lon.astype(float)
points = points[points["wait"] <= 100]

x = np.linspace(-180.0, 180.0, 3600)
y = np.linspace(-90.0, 90.0, 1800)
X, Y = np.meshgrid(x, y)


def get_dist(lat, lon, time):
    return time * np.exp(-2 * np.log(2) * ((X - lon) ** 2 + (Y - lat) ** 2) / 1**2)


Zn = [
    get_dist(lat, lon, time)
    for lat, lon, time in zip(points.lat, points.lon, points.wait)
]
Z = np.sum(Zn, axis=0) / sum(points.wait)
p = plt.contour(X, Y, Z)
plt.clabel(p, inline=1, fontsize=10)
plt.show()

with rasterio.open(
    "german.tif",
    "w",
    driver="GTiff",
    height=Z.shape[0],
    width=Z.shape[1],
    count=1,
    dtype=Z.dtype,
) as dst:
    dst.write(Z, 1)
src = rasterio.open("german.tif")
plt.imshow(src.read(1), cmap="pink")
plt.show()
