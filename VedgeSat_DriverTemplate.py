#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#%% Imports and Initialisation


import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.ion()
from datetime import datetime
from Toolshed import Download, Toolbox, VegetationLine
import ee
import geopandas as gpd

ee.Initialize()



#%% EDIT ME: Requirements

# Name of site to save directory and files under
sitename = 'SITENAME'

# Date range
dates = ['2021-05-01', '2021-07-02']

# Satellite missions
# Input a list of containing any/all of 'L5', 'L7', 'L8', 'L9', 'S2', 'PSScene4Band'
# L5: 1984-2013; L7: 1999-2017 (SLC error from 2003); L8: 2013-present; S2: 2014-present; L9: 2021-present
sat_list = ['L5','L8','S2']

# Cloud threshold for screening out cloudy imagery (0.5 or 50% recommended)
cloud_thresh = 0.5

# Extract shoreline (wet-dry boundary) as well as veg edge
wetdry = True

# Reference shoreline/veg line shapefile name (should be stored in a folder called referenceLines in Data)
# Line should be ONE CONTINUOUS linestring along the shore, stored as a shapefile in WGS84 coord system
referenceLineShp = 'SITENAME_refLine.shp'
# Maximum amount in metres by which to buffer the reference line for capturing veg edges within
max_dist_ref = 150

#%% Set Up Site Directory
# Directory where the data will be stored
filepath = Toolbox.CreateFileStructure(sitename, sat_list)

# Return AOI from reference line bounding box and save AOI folium map HTML in sitename directory
referenceLinePath = os.path.join(filepath, 'referenceLines', referenceLineShp)
referenceLineDF = gpd.read_file(referenceLinePath)
polygon, point, lonmin, lonmax, latmin, latmax = Toolbox.AOIfromLine(referenceLinePath, max_dist_ref, sitename)

# It's recommended to convert the polygon to the smallest rectangle (sides parallel to coordinate axes)       
polygon = Toolbox.smallest_rectangle(polygon)

#%% Compile Input Settings for Imagery

if len(dates)>2:
    daterange='no'
else:
    daterange='yes'
years = list(Toolbox.daterange(datetime.strptime(dates[0],'%Y-%m-%d'), datetime.strptime(dates[-1],'%Y-%m-%d')))

# Put all the inputs into a dictionary
inputs = {'polygon': polygon, 'dates': dates, 'daterange':daterange, 'sat_list': sat_list, 'sitename': sitename, 'filepath':filepath}


#%% Image Retrieval

# Before downloading the images, check how many images are available for your inputs
inputs = Download.check_images_available(inputs)


#%% Image Download

# Make the metadata
Sat = Toolbox.image_retrieval(inputs)
metadata = Toolbox.metadata_collection(inputs, Sat)


#%% Local Image Retrieval (Planet)
# For the time being, Planet API is very slow at downloading files (depending on the size of your area).
# You may want to use the Planet online data download portal and move the files to local folders that will
# have been created at the Toolbox.CreateFileStructure() step. 
# Move the image TIFFs into 'Data/YOURSITENAME/local_images/PlanetScope';
# and the respective cloud masks into 'Data/YOURSITENAME/local_images/PlanetScope/cloudmasks'.
# You can move any leftover extra files into 'Data/YOURSITENAME/AuxiliaryImages'

# If you want to include Landsat 7 but DON'T want to include Scan Line Corrector affected images, set SLC=False
Sat = Download.RetrieveImages(inputs, SLC=True)
metadata = Download.CollectMetadata(inputs, Sat)


#%% Vegetation Edge Settings
# ONLY EDIT IF ADJUSTMENTS ARE NEEDED

BasePath = 'Data/' + sitename + '/lines'

if os.path.isdir(BasePath) is False:
    os.mkdir(BasePath)

projection_epsg, _ = Toolbox.FindUTM(polygon[0][0][1],polygon[0][0][0])

settings = {
    # general parameters:
    'cloud_thresh': cloud_thresh,        # threshold on maximum cloud cover
    'output_epsg': projection_epsg,     # epsg code of spatial reference system desired for the output   
    'wetdry': wetdry,              # extract wet-dry boundary as well as veg
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
    'year_list': years
}


#%% Compute Tides from FES2014 
# Compute tides from FES2014 or FES2022 data which is downloaded from the pyFES server. Only relevant for shoreline processing 
# (as it is used to correct for the effects of tides on the cross-shore waterline position). 
# (ONLY RUN IF YOU HAVE `pyfes` INSTALLED AND WANT TIDAL INFO SAVED. If you do, change `tidepath` to the path to your `aviso-fes` folder, see the README for details)
# Note: FES2022 is more accurate than FES2014 but takes several minutes longer to compute.
tidepath = "../aviso-fes/data/fes2014"
tideoutpath = os.path.join(settings['inputs']['filepath'],'tides',
                           settings['inputs']['sitename']+'_tides_'+
                           settings['inputs']['dates'][0]+'_'+settings['inputs']['dates'][1]+'.csv')
daterange = dates
tidelatlon = [(latmin+latmax)/2, (lonmin+lonmax)/2] # centre of bounding box
Toolbox.ComputeTides(settings,tidepath,tideoutpath,daterange,tidelatlon) 
    
#%% Vegetation Edge Reference Line Load-In
referenceLine, ref_epsg = Toolbox.ProcessRefline(referenceLinePath,settings)

settings['reference_shoreline'] = referenceLine
settings['ref_epsg'] = ref_epsg
# Distance to buffer reference line by (this is in metres)
settings['max_dist_ref'] = max_dist_ref

#%% Reference Image for Coregistration
# You can now coregister your satellite images using AROSICS. 
# If you want to try coregistering your images to improve timeseries accuracy, provide a filepath to a reference RGB image
settings['reference_coreg_im'] = None # leave as None if no coregistration is to be performed

#%% Vegetation Line Extraction
"""
OPTION 1: Run extraction tool and return output veg edges as a dictionary of lines
"""

output, output_latlon, output_proj = VegetationLine.extract_veglines(metadata, settings, polygon, dates)


#%% Vegetation Line Extraction Load-In
"""
OPTION 2: Load in pre-existing outputs
"""

output, output_latlon, output_proj = Toolbox.ReadOutput(inputs)
    

#%% Remove Duplicate Lines
# For images taken on the same date by the same satellite, keep only the longest line

output = Toolbox.RemoveDuplicates(output) 


#%% Save Veglines as Local Shapefiles

# Save output veglines 
Toolbox.SaveConvShapefiles(output, BasePath, sitename, settings['output_epsg'])
# Save output shorelines if they were generated
if settings['wetdry'] == True:
    Toolbox.SaveConvShapefiles_Water(output, BasePath, sitename, settings['output_epsg'])


    
