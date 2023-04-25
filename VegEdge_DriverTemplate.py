#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 13:08:30 2022

@author: fmuir
"""
#!/usr/bin/env python
# coding: utf-8

#%% Imports and Initialisation


import os
import glob
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.ion()
from datetime import datetime
from Toolshed import Download, Toolbox, VegetationLine, Plotting, PlottingSeaborn, Transects
import ee
import geopandas as gpd

ee.Initialize()



#%% Define AOI using coordinates of a rectangle

projection_epsg = 27700 # OSGB 1936
image_epsg = 32630 # UTM Zone 30N

# The points represent the corners of a bounding box that go around your site
sitename = 'SITENAME'
lonmin, lonmax = -2.84869, -2.79878
latmin, latmax = 56.32641, 56.39814

# Return AOI after checking coords and saving folium map HTML in sitename directory
polygon, point = Toolbox.AOI(lonmin, lonmax, latmin, latmax, sitename, image_epsg)

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
# Input a list of containing any/all of 'L5', 'L7', 'L8', 'L9', 'S2', 'PSScene4Band'
# L5: 1984-2013; L7: 1999-2017 (SLC error from 2003); L8: 2013-present; S2: 2014-present; L9: 2021-present
sat_list = ['L5','L8','S2']

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
    'wetdry':True,              # extract wet-dry boundary as well as veg
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
tidepath = "/path/to/FES2014/files/aviso-fes/data/fes2014"
daterange = dates
tidelatlon = [-2.79878,latmax-(2/latmin)] # seaward edge, halfway between S and N
Toolbox.ComputeTides(settings,tidepath,daterange,tidelatlon) 
    
#%% Vegetation Edge Reference Line Load-In

"""
OPTION 2: Load in coordinates of reference line shapefile and format for use in
the veg extraction.
"""

#referenceLineShp = os.path.join(inputs['filepath'], sitename,'StAndrews_refLine.shp')
referenceLineShp = os.path.join(inputs['filepath'], 'SITE_refLine.shp')
referenceLine, ref_epsg = Toolbox.ProcessRefline(referenceLineShp,settings)

settings['reference_shoreline'] = referenceLine
settings['ref_epsg'] = ref_epsg
# Distance to buffer reference line by (this is in metres)
settings['max_dist_ref'] = 150


#%% Vegetation Line Extraction

"""
OPTION 1: Run extraction tool and return output dates, lines, filenames and 
image properties.
"""
#get_ipython().run_line_magic('matplotlib', 'qt')
clf_model = 'Aberdeen_MLPClassifier_Veg_S2.pkl'
output, output_latlon, output_proj = VegetationLine.extract_veglines(metadata, settings, polygon, dates)


#%% Vegetation Line Extraction Load-In

"""
OPTION 2: Load in pre-existing output dates, lines, filenames and image properties.
"""

SiteFilepath = os.path.join(inputs['filepath'], sitename)
with open(os.path.join(SiteFilepath, sitename + '_output.pkl'), 'rb') as f:
    output = pickle.load(f)
with open(os.path.join(SiteFilepath, sitename + '_output_latlon.pkl'), 'rb') as f:
    output_latlon = pickle.load(f)
with open(os.path.join(SiteFilepath, sitename + '_output_proj.pkl'), 'rb') as f:
    output_proj = pickle.load(f)
    

#%% remove duplicate date lines 

output = Toolbox.remove_duplicates(output) # removes duplicates (images taken on the same date by the same satellite)
output_latlon = Toolbox.remove_duplicates(output_latlon)
output_proj = Toolbox.remove_duplicates(output_proj)

#%% Save the veglines as shapefiles locally
# Save output veglines 
Toolbox.SaveConvShapefiles(output, BasePath, sitename, settings['projection_epsg'])
if settings['wetdry'] == True:
    Toolbox.SaveConvShapefiles_Water(output, BasePath, sitename, settings['projection_epsg'])

#%% Create GIF of satellite images and related shorelines

Plotting.SatGIF(metadata,settings,output)

#%% Create Transects
SmoothingWindowSize = 21 
NoSmooths = 100
TransectSpacing = 10
DistanceInland = 100
DistanceOffshore = 350

BasePath = 'Data/' + sitename + '/veglines'
VeglineShp = glob.glob(BasePath+'/*veglines.shp')
VeglineGDF = gpd.read_file(VeglineShp[0])
WaterlineShp = glob.glob(BasePath+'/*waterlines.shp')
WaterlineGDF = gpd.read_file(WaterlineShp[0])
# Produces Transects for the reference line
TransectSpec =  os.path.join(BasePath, sitename+'_Transects.shp')

if os.path.isfile(TransectSpec) is False:
    TransectGDF = Transects.ProduceTransects(SmoothingWindowSize, NoSmooths, TransectSpacing, DistanceInland, DistanceOffshore, settings['output_epsg'], sitename, BasePath, referenceLineShp)
else:
    print('Transects already exist and were loaded')
    TransectGDF = gpd.read_file(TransectSpec)

#%% Create (or load) intersections with sat and validation lines per transect

if os.path.isfile(os.path.join(filepath, sitename, sitename + '_transect_intersects.pkl')):
    print('TransectDict exists and was loaded')
    with open(os.path.join(filepath , sitename, sitename + '_transect_intersects.pkl'), 'rb') as f:
        TransectDict, TransectInterGDF = pickle.load(f)
else:
    # Get intersections
    TransectDict = Transects.GetIntersections(BasePath, TransectGDF, VeglineGDF)
    # Save newly intersected transects as shapefile
    TransectInterGDF = Transects.SaveIntersections(TransectDict, VeglineGDF, BasePath, sitename, settings['projection_epsg'])
    # Repopulate dict with intersection distances along transects normalised to transect midpoints
    TransectDict = Transects.CalculateChanges(TransectDict,TransectInterGDF)
    if settings['wetdry'] == True:
        beachslope = 0.02 # provide average beach slope for site
        TransectDict = Transects.GetBeachWidth(BasePath, TransectGDF, TransectDict, WaterlineGDF, settings, output, beachslope)  
        TransectInterGDF = Transects.SaveWaterIntersections(TransectDict, WaterlineGDF, TransectInterGDF, BasePath, sitename, settings['projection_epsg'])
    
    with open(os.path.join(filepath , sitename, sitename + '_transect_intersects.pkl'), 'wb') as f:
        pickle.dump([TransectDict,TransectInterGDF], f)


#%% VALIDATION

# Name of date column in validation shapefile (case sensitive!) 
DatesCol = 'Date'
ValidationShp = './Validation/StAndrews_Veg_Edge_combined_2007_2022_singlepart.shp'
validpath = os.path.join(os.getcwd(), 'Data', sitename, 'validation')

if os.path.isfile(os.path.join(validpath, sitename + '_valid_dict.pkl')):
    print('ValidDict exists and was loaded')
    with open(os.path.join(validpath, sitename + '_valid_dict.pkl'), 'rb') as f:
        ValidDict = pickle.load(f)
else:
    ValidDict = Transects.ValidateSatIntersects(sitename, ValidationShp, DatesCol, TransectGDF, TransectDict)
    with open(os.path.join(validpath, sitename + '_valid_dict.pkl'), 'wb') as f:
        pickle.dump(ValidDict, f)

#%%
ValidDict = Transects.ValidateSatIntersects(ValidationShp, DatesCol, TransectGDF, TransectDict)
#%% Validation Plots
TransectIDList= [(0,1741)]

for TransectIDs in TransectIDList:
    PlotTitle = 'Accuracy of Transects ' + str(TransectIDs[0]) + ' to ' + str(TransectIDs[1])
    PlottingSeaborn.SatViolin(sitename,VeglineShp[0],'dates',ValidDict,TransectIDs, PlotTitle)
    
#%% Quantify errors between validation and satellite derived lines
TransectIDList = [(595,711),(726,889),(972,1140),(1141,1297)]
for TransectIDs in TransectIDList:
    Toolbox.QuantifyErrors(sitename, VeglineShp[0],'dates',ValidDict,TransectIDs)
    
    
    
#%% TIMESERIES PLOTTING

# Select transect ID to plot
TransectIDs = [1575]
for TransectID in TransectIDs:
    DateRange = [0,len(TransectDict['dates'][TransectID])]
    # Plot timeseries of cross-shore veg position
    Plotting.VegTimeseries(sitename, TransectDict, TransectID, DateRange)
    
#%%

# Select transect ID to plot
TransectIDs = [180,1650]
for TransectID in TransectIDs:
    # Plot timeseries of cross-shore width between water edge and veg edge 
    DateRange = [0,len(TransectDict['dates'][TransectID])]
    Plotting.WidthTimeseries(sitename, TransectDict, TransectID, DateRange)