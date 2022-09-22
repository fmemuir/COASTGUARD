#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 13:08:30 2022

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
plt.ion()
from datetime import datetime
from Toolshed import Download, Toolbox, VegetationLine
import seaborn as sns; sns.set()
import ee
import geopandas as gpd

ee.Initialize()



#%% Define AOI using coordinates of a rectangle


##ST ANDREWS EAST
# The points represent the corners of a bounding box that go around your site
sitename = 'SITENAME'
lonmin, lonmax = -2.84869, -2.79878
latmin, latmax = 56.32641, 56.39814


polygon, point = Toolbox.AOI(lonmin, lonmax, latmin, latmax)
# it's recommended to convert the polygon to the smallest rectangle (sides parallel to coordinate axes)       
polygon = Toolbox.smallest_rectangle(polygon)

#%% Image Settings

# directory where the data will be stored
filepath = os.path.join(os.getcwd(), 'Data')
if os.path.isdir(filepath) is False:
    os.mkdir(filepath)

# date range
dates = ['2021-05-01', '2021-07-02']
if len(dates)>2:
    daterange='no'
else:
    daterange='yes'
years = list(Toolbox.daterange(datetime.strptime(dates[0],'%Y-%m-%d'), datetime.strptime(dates[-1],'%Y-%m-%d')))

# satellite missions
# Input a list of containing any/all of 'L5', 'L8', 'S2'
sat_list = ['L5','L8','S2']

projection_epsg = 27700 # OSGB 1936
image_epsg = 32630 # UTM Zone 30N

# put all the inputs into a dictionnary
inputs = {'polygon': polygon, 'dates': dates, 'daterange':daterange, 'sat_list': sat_list, 'sitename': sitename, 'filepath':filepath}

direc = os.path.join(filepath, sitename)

if os.path.isdir(direc) is False:
    os.mkdir(direc)
 
    
#%% Image Retrieval

# before downloading the images, check how many images are available for your inputs
Download.check_images_available(inputs)


#%% Image Download

Sat = Toolbox.image_retrieval(inputs)
metadata = Toolbox.metadata_collection(sat_list, Sat, filepath, sitename)


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
    'adjust_detection': False,  # if True, allows user to adjust the postion of each shoreline by changing the threhold
    'save_figure': True,        # if True, saves a figure showing the mapped shoreline for each image
    # [ONLY FOR ADVANCED USERS] shoreline detection parameters:
    'min_beach_area': 200,     # minimum area (in metres^2) for an object to be labelled as a beach
    'buffer_size': 250,         # radius (in metres) for buffer around sandy pixels considered in the shoreline detection
    'min_length_sl': 500,       # minimum length (in metres) of shoreline perimeter to be valid
    'cloud_mask_issue': False,  # switch this parameter to True if sand pixels are masked (in black) on many images  
    # add the inputs defined previously
    'inputs': inputs,
    'projection_epsg': projection_epsg,
    'year_list': years,
    'hausdorff_threshold':3*(10**50)
}


#%% Vegetation Edge Reference Line Load-In

referenceLineShp = os.path.join(inputs['filepath'], 'SITE_reference_line.shp')
referenceLineDF = gpd.read_file(referenceLineShp)
refLinex,refLiney = referenceLineDF.geometry[0].coords.xy
# format latlon coords into list
#referenceLineList = list([refLinex[i],refLiney[i]] for i in range(len(refLinex)))
referenceLineList = list([refLiney[i],refLinex[i]] for i in range(len(refLinex)))
# convert to UTM zone for use with the satellite images
ref_epsg = 4326
referenceLine = Toolbox.convert_epsg(np.array(referenceLineList),ref_epsg,image_epsg)
referenceLine = Toolbox.spaced_vertices(referenceLine)

settings['reference_shoreline'] = referenceLine
settings['ref_epsg'] = ref_epsg
settings['max_dist_ref'] = 100


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

#%%

Plotting.SatGIF(metadata,settings,output)

