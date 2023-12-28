# notebook

# print full np arrays
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

# import libraries

import rasterio
import rasterio.plot
from rasterio.crs import CRS
from matplotlib import pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm
import rasterio.mask
from shapely.geometry import Point
from geopandas import GeoDataFrame
import matplotlib.colors as colors
from matplotlib import cm
from shapely.geometry import Polygon
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint as GCP

plt.rcParams.update({'font.size': 20})
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# set vars here!

here = "/home/till/projects"
regions = ["south_america", "europe", "asia", "africa", "australia"]

ROADS = False
CITIES = False
CIRCLE = 50000
WAIT_MAX = 100
# resolution 40 for world -> 6800x14400 image -> lets calculating distribution crash -> too many values in np array to calculate?
RESOLUTION = 10 
THRESHOLD = 0.00000001 # quite arbitrary
RECOMPUTE = True

def run(region):
    def get_points(path):
        points = gpd.read_file(path)
        points.wait = points.wait.astype(float)
        points.lat = points.lat.astype(float)
        points.lon = points.lon.astype(float)
        # threshold - assuming that values above that are skewed due to angriness of the hiker
        points = points[points['wait'] <= WAIT_MAX]

        # use epsg 3857 as default as it gives coordinates in meters
        points.geometry = gpd.points_from_xy(points.lon, points.lat)
        points.crs = CRS.from_epsg(4326)
        points = points.to_crs(epsg=3857)

        return points
        
    # read from the hitchmap dump
    points = get_points(f"{here}/heatchmap/waiting_time_per_point.csv")

    all_points = points

    # to get the same example as here https://abelblogja.wordpress.com/average-waiting-times-in-europe/
    artificial_points = get_points(f"{here}/heatchmap/artificial_points.csv")
    def get_points_in_region(points, region):
            # set lat long boundaries of different scopes of the map

            maps = {
                    "germany": [5.0, 48.0, 15.0, 55.0],
                    "europe": [-12.0, 35.0, 45.0, 71.0],
                    "world": [-180.0, -85.0, 180.0, 85.0], # 85 lat bc of 3857
                    "small": [12.0, 52.0, 15.0, 54.0],
                    "africa": [-20.0, -35.0, 60.0, 40.0],
                    "asia": [40.0, 0.0, 180.0, 85.0],
                    "north_america": [-180.0, 0.0, -20.0, 85.0],
                    "south_america": [-90.0, -60.0, -30.0, 15.0],
                    "australia": [100.0, -50.0, 180.0, 0.0],
                    "middle_africa": [-10.0, -35.0, 60.0, 20.0],
                    "artificial": [8.0, -10.0, 30.0, 10.0],
                    "greenland": [-80.0, 60.0, -10.0, 85.0],
                    }
            map = maps[region]

            # create boundary polygon
            polygon = Polygon([(map[0], map[1]), (map[0], map[3]), (map[2], map[3]), (map[2], map[1]), (map[0], map[1])])
            polygon = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[polygon])  
            polygon = polygon.to_crs(epsg=3857)

            # extract points within polygon
            points = points[points.geometry.within(polygon.geometry[0])]

            return points, polygon, map

    points, polygon, map = get_points_in_region(points, region)
    if region == "artificial": 
            points = artificial_points
            print("artificial points")
    # # test our normal distribution formula

    # a = 5.0
    # res = int(a * 2 * 100)
    # x = np.linspace(-a, a, res)
    # # mind starting with upper value of y axis here
    # y = np.linspace(-a, a, res)
    # X, Y = np.meshgrid(x, y)

    # stdv = 0.5
    # fwhm = 2.355 * stdv

    # dist = np.exp(-4 * np.log(2) * ((X - 0.0) ** 2 + (Y - 0.0) ** 2) / fwhm**2)


    # hf = plt.figure()
    # ha = hf.add_subplot(111, projection='3d')

    # X, Y = numpy.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    # ha.plot_surface(X, Y, dist)

    # plt.show()

    # plt.plot(dist[int(res / 2)])
    # # example for 2 spots with distance "m" m from each other
    # # verifying that where they meet the edge is as hard as seen in the plots

    # sigma = 50000
    # mu = 0

    # fwhm = 2.355 * sigma

    # # gives the distribution in the whole raster space as X and Y are used here
    # def n(x):
    #     return np.exp(-4 * np.log(2) * ((x - mu) ** 2 + (0 - mu) ** 2) / fwhm**2)

    # a = []
    # m = 100000

    # a = [(n(i) * 45 + n(m-i) * 5) / (n(i) + n(m-i)) for i in range(0, m, 10000)]
    # plt.plot(a)

    def make_raster_map(points, polygon, map):

        # https://stackoverflow.com/questions/7687679/how-to-generate-2d-gaussian-with-python
        def makeGaussian(stdv, x0, y0):
            """Make a square gaussian kernel.
            size is the length of a side of the square
            fwhm is full-width-half-maximum, which
            can be thought of as an effective radius.
            """
            # https://en.wikipedia.org/wiki/Full_width_at_half_maximum
            # TODO why fwhm used here?
            fwhm = 2.355 * stdv

            # gives the distribution in the whole raster space as X and Y are used here
            return np.exp(-4 * np.log(2) * ((X - x0) ** 2 + (Y - y0) ** 2) / fwhm**2)
        
        def get_distribution(lat, lon):
            # standard deviation in meters -> 50 km around each spot; quite arbitrary
            # TODO adapt width of distribution to density of data points in the region
            STDV_M = CIRCLE
            return makeGaussian(STDV_M, lon, lat)    

        # create pixel grid for map

        xx, yy = polygon.geometry[0].exterior.coords.xy

        # Note above return values are of type `array.array` 
        xx = xx.tolist()
        yy = yy.tolist()

        # set resulution here
        resoltions_factor = RESOLUTION

        degree_width = int(map[2] - map[0])
        degree_height = int(map[3] - map[1])
        pixel_width = degree_width * resoltions_factor
        pixel_height = degree_height * resoltions_factor
        x = np.linspace(xx[0], xx[2], pixel_width)
        # mind starting with upper value of y axis here
        y = np.linspace(yy[2], yy[0], pixel_height)
        X, Y = np.meshgrid(x, y)
        X = np.longdouble(X)
        Y = np.longdouble(Y)


        # sum of distributions
        Zn = None
        # weighted sum of distributions
        Zn_weighted = None

        try:
            if not RECOMPUTE:
                Z = np.loadtxt(f'{here}/heatchmap/map/map_{region}.txt', dtype=float)
            else:
                raise Exception("recompute")
        except:
            # create a raster map - resulution is defined above
            # https://stackoverflow.com/questions/56677267/tqdm-extract-time-passed-time-remaining
            with tqdm(zip(points.geometry.y, points.geometry.x, points.wait)) as t:

                # TODO find out how to speed up and parallelize this
                for lat, lon, wait in t:
                    # distribution inserted by a single point
                    Zi = get_distribution(lat, lon)
                    # add the new distribution to the sum of existing distributions
                    # write them to Zn_weighted and wait every single point/ distribution by the waiting time
                    # => it matters where a distribiton is inserted (areas with more distributions have a higher certainty)
                    # and which waiting time weight is associated with it
                    if Zn is None:
                        Zn = Zi
                        Zn_weighted = Zi * wait
                    else:
                        Zn = np.sum([Zn, Zi], axis=0)
                        Zn_weighted = np.sum([Zn_weighted, Zi * wait], axis=0)

                elapsed = t.format_dict['elapsed']
                elapsed_str = t.format_interval(elapsed)
                df = pd.DataFrame({"region": region, 'elapsed time': [elapsed_str]})

                tracker_name = f'{here}/heatchmap/map/time_tracker.csv'
                try:
                    full_df = pd.read_csv(tracker_name, index_col=0)
                    full_df = pd.concat([full_df, df])
                    full_df.to_csv(tracker_name, sep=',')
                except:
                    df.to_csv(tracker_name)


            # normalize the weighted sum by the sum of all distributions -> so we see the actual waiting times in the raster
            # Z = Zn_weighted / Zn
            Z = np.divide(Zn_weighted, Zn, out=np.zeros_like(Zn_weighted), where=Zn!=0)

            Z = (Zn > THRESHOLD) * Z

        
            # save the underlying raster data of the heatmap for later use
            np.savetxt(f'{here}/heatchmap/map/map_{region}.txt', Z)

            # read
            # Z = np.loadtxt(f'{here}/heatchmap/map/map_{region}.txt', dtype=float)

        # https://gis.stackexchange.com/questions/425903/getting-rasterio-transform-affine-from-lat-and-long-array

        # lower/upper - left/right
        ll = (xx[0], yy[0])
        ul = (xx[1], yy[1])  # in lon, lat / x, y order
        ur = (xx[2], yy[2])
        lr = (xx[3], yy[3])
        cols, rows = pixel_width, pixel_height

        # ground control points
        gcps = [
            GCP(0, 0, *ul),
            GCP(0, cols, *ur),
            GCP(rows, 0, *ll),
            GCP(rows, cols, *lr),
        ]

        # seems to need the vertices of the map polygon
        transform = from_gcps(gcps)

        # higher precision prevents pixels far away from the points to be 0/ nan
        Z = np.double(Z)

        

        # save the colored raster using the above transform
        # TODO find out why raster is getting smaller in x direction when stored as tif (e.g. 393x700 -> 425x700)
        with rasterio.open(
            map_path,
            "w",
            driver="GTiff",
            height=Z.shape[0],
            width=Z.shape[1],
            count=1,
            crs=CRS.from_epsg(3857),
            transform=transform,
            dtype=Z.dtype,
        ) as destination:
            destination.write(Z, 1)

        return X, Y, Z, Zn, Zn_weighted

    X, Y, Z_raw, Zn, Zn_w = make_raster_map(points, polygon, map)
    Z = Z_raw
    # with rasterio.open(map_path) as heatmap:
    #     A = heatmap.read(1)
    #     print(A.shape)
    #     # modify
    #     A = (Zn > 0.0000000000001) * A
    #     heatmap.write(A, 1)
    # Zn indicator for density of spots in the raster; 1.0 at the max of a single spot
    # Z = (Zn > 0.0000000000001) * Z_raw
    # see where the single distributions are placed
    p = plt.contourf(X, Y, Z)
    plt.show()
    # get borders of all countries
    # download https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip
    # from https://www.naturalearthdata.com/downloads/110m-cultural-vectors/
    countries = gpd.datasets.get_path("naturalearth_lowres")
    countries = gpd.read_file(countries)
    countries = countries.to_crs(epsg=3857)
    countries = countries[countries.name != "Antarctica"]
    # TODO so far does not work as in final map the raster is not applied to the whole region
    # countries = countries[countries.geometry.within(polygon.geometry[0])]
    country_shapes = countries.geometry
    # limit heatmap to landmass
    with rasterio.open(map_path) as heatmap:
        out_image, out_transform = rasterio.mask.mask(heatmap, country_shapes, crop=True, filled=False)
        out_meta = heatmap.meta

    # grey out countries with no hitchhiking spots
        
    out_meta.update({"driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform})

    with rasterio.open(map_path, "w", **out_meta) as destination:
        destination.write(out_image)

    # select the sice of (important/ populated) cities to show on the map

    # CITY_MAX = 100000

    # cities = pd.read_csv("../data/worldcities.csv")
    # cities.population = cities.population.replace("", np.nan)
    # cities = cities.dropna()
    # cities.population = cities.population.astype(int)
    # cities = cities[cities.population > CITY_MAX]

    # geometry = cities.apply(lambda x: Point(x.lng, x.lat), axis=1)
    # cities = GeoDataFrame(cities, crs=CRS.from_epsg(4326), geometry=geometry)
    # cities = cities.to_crs(epsg=3857)
    # cities = cities[cities.geometry.within(polygon.geometry[0])]
    # TODO takes more time than expected

    # use a pre-compiled list of important cities
    # download https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_populated_places.zip
    # from https://www.naturalearthdata.com/downloads/10m-cultural-vectors/
    # cities = gpd.read_file("cities/ne_10m_populated_places.shp", bbox=polygon.geometry[0]) should work but does not
    if CITIES:
        cities = gpd.read_file("cities/ne_10m_populated_places.shp") # takes most time
        cities = cities.to_crs(epsg=3857)
        cities = cities[cities.geometry.within(polygon.geometry[0])]

    # use a pre-compiles list of important roads
    # download https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_roads.zip
    # from https://www.naturalearthdata.com/downloads/10m-cultural-vectors/
    if ROADS:
        roads = gpd.read_file("roads/ne_10m_roads.shp")
        roads = roads.to_crs(epsg=3857)
        roads = roads[roads.geometry.within(polygon.geometry[0])]
    # define the heatmap color scale
    cmap = colors.ListedColormap(["grey", '#008200', '#00c800','#00ff00', '#c8ff00',\
                                '#ffff00', '#ffc800', '#ff8200', '#ff0000',\
                                    '#c80000', '#820000', "blue"])
    boundaries = [-1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 100, 110]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    # plot the heatmap
    raster = rasterio.open(map_path)
    fig, ax = plt.subplots(figsize=(100, 100))
    rasterio.plot.show(raster, ax=ax, cmap=cmap, norm=norm)
    countries.plot(ax=ax, facecolor='none', edgecolor='black')
    if CITIES: cities.plot(ax=ax, markersize=1, color='black')
    if ROADS: roads.plot(ax=ax, markersize=1, color='black')

    all_points.plot(ax=ax, markersize=10, color='red')

    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    plt.savefig(f"{here}/heatchmap/map/map_{region}.png", bbox_inches='tight')
    # extra

    # # 75 km = 1 degree at 48 lat

    # cut_line = 125

    # fig = plt.figure(figsize=(20, 2))
    # ax = fig.add_subplot(111)
    # ax.plot(Z[cut_line])
    # plt.plot()

    # # TODO check fwhm -> have to set it to 70km to make it look like on website


for region in regions:
    map_path = f"{here}/heatchmap/map/map_{region}.tif"
    run(region)
    print(f"done with {region}")