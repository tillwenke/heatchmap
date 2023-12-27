# heatchmap

`map/map_gpd_rasterio.ipynb` is relevant

### Modelling

- do not want to color where we have no certainty/ data
- in europe normal distribution with 50 km stdv is a good asumption

### Problems

For low density of points far away averaging them works poorly because calculation are on really small numbers (tail of normal distribution). We get hard edges:

![1703636597008](image/README/1703636597008.png)

Instabilities between color scale sections:

![1703679266533](image/README/1703679266533.png)
