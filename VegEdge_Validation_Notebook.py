#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 10:45:22 2022

@author: fmuir
"""
#!/usr/bin/env python
# coding: utf-8

# # VegEdge
# 

#%% Imports and Initialisation


import os
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
plt.ion()
from datetime import datetime, timezone, timedelta
from Toolshed import Download, Image_Processing, Shoreline, Toolbox, Transects, VegetationLine
import mpl_toolkits as mpl
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns; sns.set()
import math
import geemap
import ee
import pprint
from shapely import geometry
from shapely.geometry import Point, LineString
import pandas as pd
import geopandas as gpd
import matplotlib.cm as cm
import pyproj
from IPython.display import clear_output
import scipy
from scipy import optimize
import csv
import math

ee.Initialize()


#%% Define ROI using map


"""
OPTION 1: Generate a map. Use the polygon drawing tool on the left-hand side to 
draw out the region of coast you're interested in.
"""

Map = geemap.Map(center=[0,0],zoom=2)
Map.add_basemap('HYBRID')
Map


#%% 

# Run this after hand digitising to capture the coordinates of the ref shore
roi = Map.user_roi.geometries().getInfo()[0]['coordinates']
polygon = [[roi[0][0],roi[0][3],roi[0][1],roi[0][2]]]
point = ee.Geometry.Point(roi[0][0])



#%% Image Settings


# it's recommended to convert the polygon to the smallest rectangle (sides parallel to coordinate axes)       
polygon = Toolbox.smallest_rectangle(polygon)

# directory where the data will be stored
filepath = os.path.join(os.getcwd(), 'Data')

# date range
#dates = ['2021-05-01', '2021-07-02']

# date range for valiation
vegsurveyshp = './Validation/StAndrews_Veg_Edge_combined_singlepart.shp'
vegsurvey = gpd.read_file(vegsurveyshp)
vegdatemin = vegsurvey.Date.min()
vegdatemax = vegsurvey.Date.max()
# vegdatemin = datetime.strftime(datetime.strptime(vegsurvey.Date.min(), '%Y-%m-%d') - timedelta(weeks=4),'%Y-%m-%d')
# vegdatemax = datetime.strftime(datetime.strptime(vegsurvey.Date.max(), '%Y-%m-%d') + timedelta(weeks=4),'%Y-%m-%d')
dates = [vegdatemin, vegdatemax]
#dates = list(vegsurvey.Date.unique())
#dates = ['2011-03-04','2016-01-20']

print(dates)

if len(dates)>2:
    daterange='no'
else:
    daterange='yes'


years = list(Toolbox.daterange(datetime.strptime(dates[0],'%Y-%m-%d'), datetime.strptime(dates[-1],'%Y-%m-%d')))

# satellite missions
# Input a list of containing any/all of 'L5', 'L8', 'S2'
sat_list = ['L5','L8','S2']

projection_epsg = 27700
image_epsg = 32630


# put all the inputs into a dictionnary
inputs = {'polygon': polygon, 'dates': dates, 'daterange':daterange, 'sat_list': sat_list, 'sitename': sitename, 'filepath':filepath}

direc = os.path.join(filepath, sitename)

if os.path.isdir(direc) is False:
    os.mkdir(direc)
 
    
#%% Image Retrieval

# before downloading the images, check how many images are available for your inputs

Download.check_images_available(inputs)


#%% Image Download

"""
OPTION 1: Populate metadata using image names pulled from server.
"""
Sat = Toolbox.image_retrieval(inputs)
metadata = Toolbox.metadata_collection(sat_list, Sat, filepath, sitename)



#%% Image Load-In
"""
OPTION 2: Populate metadata using pre-existing metadata.
"""

filepath = os.path.join(inputs['filepath'], sitename)
with open(os.path.join(filepath, sitename + '_metadata.pkl'), 'rb') as f:
    metadata = pickle.load(f)


#%% Vegetation Edge Settings

BasePath = 'Data/' + sitename + '/Veglines'

if os.path.isdir(BasePath) is False:
    os.mkdir(BasePath)

settings = {
    # general parameters:
    'cloud_thresh': 0.5,        # threshold on maximum cloud cover
    'output_epsg': image_epsg,     # epsg code of spatial reference system desired for the output   
    # quality control:
    'check_detection': True,    # if True, shows each shoreline detection to the user for validation
    'adjust_detection': True,  # if True, allows user to adjust the postion of each shoreline by changing the threhold
    'save_figure': True,        # if True, saves a figure showing the mapped shoreline for each image
    # [ONLY FOR ADVANCED USERS] shoreline detection parameters:
    'min_beach_area': 200,     # minimum area (in metres^2) for an object to be labelled as a beach
    'buffer_size': 250,         # radius (in metres) for buffer around sandy pixels considered in the shoreline detection
    'min_length_sl': 500,       # minimum length (in metres) of shoreline perimeter to be valid
    'cloud_mask_issue': False,  # switch this parameter to True if sand pixels are masked (in black) on many images  
    'sand_color': 'bright',    # 'default', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
    # add the inputs defined previously
    'inputs': inputs,
    'projection_epsg': projection_epsg,
    'year_list': years,
    'hausdorff_threshold':3*(10**50)
}



#%% Vegetation Edge Reference Line Digitisation

"""
OPTION 1: Generate a map. Use the line drawing tool on the left-hand side to 
trace along the reference vegetation edge.
"""
#Draw reference line onto the map then run the next cell

Map = geemap.Map(center=[0,0],zoom=2)
Map.add_basemap('HYBRID')
Map

#%%
referenceLine = Map.user_roi.geometries().getInfo()[0]['coordinates']

for i in range(len(referenceLine)):
    #referenceLine[i][0], referenceLine[i][1] = referenceLine[i][1], referenceLine[i][0]
    referenceLine[i] = list(referenceLine[i])

ref_epsg = 4326
referenceLine = Toolbox.convert_epsg(np.array(referenceLine),ref_epsg,image_epsg)
referenceLine = Toolbox.spaced_vertices(referenceLine)

settings['reference_shoreline'] = referenceLine
settings['ref_epsg'] = ref_epsg
settings['max_dist_ref'] = 500


#%% Vegetation Line Extraction

"""
OPTION 1: Run extraction tool and return output dates, lines, filenames and 
image properties.
"""
#get_ipython().run_line_magic('matplotlib', 'qt')
output, output_latlon, output_proj = VegetationLine.extract_veglines(metadata, settings, polygon, dates)

# L5: 44 images, 2:13 (133s) = 0.33 im/s OR 3 s/im
# L8: 20 images (10% of 198), 4:23 (263s) = 0.08 im/s OR 13 s/im
# S2: 335 images, 5:54 (354s) = 0.096 im/s OR 10.4 s/im


#%% Vegetation Line Extraction Load-In

"""
OPTION 2: Load in pre-existing output dates, lines, filenames and image properties.
"""

filepath = os.path.join(inputs['filepath'], sitename)
with open(os.path.join(filepath, sitename + '_output.pkl'), 'rb') as f:
    output = pickle.load(f)
with open(os.path.join(filepath, sitename + '_output_latlon.pkl'), 'rb') as f:
    output_latlon = pickle.load(f)
with open(os.path.join(filepath, sitename + '_output_proj.pkl'), 'rb') as f:
    output_proj = pickle.load(f)


#%% remove duplicate date lines 

output = Toolbox.remove_duplicates(output) # removes duplicates (images taken on the same date by the same satellite)
output_latlon = Toolbox.remove_duplicates(output_latlon)
output_proj = Toolbox.remove_duplicates(output_proj)


#%% Save output veglines


#Saves the veglines as shapefiles locally under Veglines.
direc = os.path.join(filepath, 'veglines')
geomtype = 'lines'
name_prefix = 'Data/' + sitename + '/veglines/'

if os.path.isdir(direc) is False:
    os.mkdir(direc)

Toolbox.save_shapefiles(output_proj, name_prefix, sitename, projection_epsg)

# initialise the ref variable for storing line info in
#ref_line = np.delete(settings['reference_shoreline'],2,1)
#ref = {'dates':['3000-12-30'], 'shorelines':[ref_line], 'filename':[0], 'cloud_cover':[0], 'geoaccuracy':[0], 'idx':[0], 'Otsu_threshold':[0], 'satname':[0]}
#Toolbox.save_shapefiles(ref, geomtype, name_prefix, sitename)

#%% Produces Transects and Coast shape-files for the reference line

SmoothingWindowSize = 21
NoSmooths = 100
TransectSpacing = 10
DistanceInland = 350
DistanceOffshore = 350
BasePath = 'Data/' + sitename + '/Veglines'


Transects.produce_transects(SmoothingWindowSize, NoSmooths, TransectSpacing, DistanceInland, DistanceOffshore, image_epsg, sitename, BasePath, referenceLineShp)

#(Optional) Produces transects for all produced lines
#Transects.produce_transects_all(SmoothingWindowSize, NoSmooths, TransectSpacing, DistanceInland, DistanceOffshore, projection_epsg, BasePath)


#%% **Option 1**: Defines all transects in a library.

TransectSpec =  os.path.join(BasePath, 'Transect.shp')
geo = gpd.read_file(TransectSpec)

transect_latlon, transect_proj = Transects.stuffIntoLibrary(geo, image_epsg, projection_epsg, filepath, sitename)


#%% **Option 2**: Or just load them if already produced


with open(os.path.join(filepath, sitename + '_transect_proj' + '.pkl'), 'rb') as f:
    transect_proj = pickle.load(f)
with open(os.path.join(filepath, sitename + '_transect_latlon' + '.pkl'), 'rb') as f:
    transect_latlon = pickle.load(f)



