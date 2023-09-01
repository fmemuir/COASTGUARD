#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 10:40:13 2023

@author: fmuir
"""

#%% Imports and Initialisation

import os
import glob
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

"""
OPTION 2: AOI polygon is defined using coordinates of a bounding box (in WGS84).
"""
projection_epsg = 27700
image_epsg = 32630

#%%
sitename = 'StAndrewsEastS2Full'
lonmin, lonmax = -2.84869, -2.79878
latmin, latmax = 56.32641, 56.39814
sat_list = ['S2']
dates = ['2015-06-28', '2023-06-13']


#%%

# directory where the data will be stored
filepath = Toolbox.CreateFileStructure(sitename, sat_list)

# Return AOI after checking coords and saving folium map HTML in sitename directory
polygon, point = Toolbox.AOI(lonmin, lonmax, latmin, latmax, sitename, image_epsg)

# it's recommended to convert the polygon to the smallest rectangle (sides parallel to coordinate axes)       
polygon = Toolbox.smallest_rectangle(polygon)

#%% Image Settings

if len(dates)>2:
    daterange='no'
else:
    daterange='yes'

years = list(Toolbox.daterange(datetime.strptime(dates[0],'%Y-%m-%d'), datetime.strptime(dates[-1],'%Y-%m-%d')))

# put all the inputs into a dictionnary
inputs = {'polygon': polygon, 'dates': dates, 'daterange':daterange, 'sat_list': sat_list, 'sitename': sitename, 'filepath':filepath}

    
#%% Image Retrieval

# before downloading the images, check how many images are available for your inputs

Download.check_images_available(inputs)


#%% Image Download

"""
OPTION 1: Populate metadata using image names pulled from server.
"""
Sat = Toolbox.image_retrieval(inputs)
metadata = Toolbox.metadata_collection(inputs, Sat)


#%% Vegetation Edge Settings

BasePath = 'Data/' + sitename + '/veglines'

if os.path.isdir(BasePath) is False:
    os.mkdir(BasePath)

settings = {
    # general parameters:
    'cloud_thresh': 0.5,        # threshold on maximum cloud cover
    'output_epsg': image_epsg,  # epsg code of spatial reference system desired for the output  
    'wetdry':True,              # extract wet-dry boundary as well as veg
    # quality control:
    'check_detection': False,    # if True, shows each shoreline detection to the user for validation
    'adjust_detection': False,  # if True, allows user to adjust the postion of each shoreline by changing the threhold
    'save_figure': True,        # if True, saves a figure showing the mapped shoreline for each image
    # [ONLY FOR ADVANCED USERS] shoreline detection parameters:
    'min_beach_area': 200,      # minimum area (in metres^2) for an object to be labelled as a beach
    'buffer_size': 250,         # radius (in metres) for buffer around sandy pixels considered in the shoreline detection
    'min_length_sl': 500,       # minimum length (in metres) of shoreline perimeter to be valid
    'cloud_mask_issue': False,  # switch this parameter to True if sand pixels are masked (in black) on many images  
    # add the inputs defined previously
    'inputs': inputs,
    'projection_epsg': projection_epsg,
    'year_list': years
}


#%% Vegetation Edge Reference Line Load-In

"""
OPTION 2: Load in coordinates of reference line shapefile and format for use in
the veg extraction.
"""

#referenceLineShp = os.path.join(inputs['filepath'], sitename,'StAndrews_refLine.shp')
referenceLineShp = os.path.join(inputs['filepath'], 'StAndrews_refLine.shp')
referenceLine, ref_epsg = Toolbox.ProcessRefline(referenceLineShp,settings)

settings['reference_shoreline'] = referenceLine
settings['ref_epsg'] = ref_epsg
# Distance to buffer reference line by (this is in metres)
settings['max_dist_ref'] = 150


#%% Compute Tides from FES2014
tidepath = "/media/14TB_RAID_Array/User_Homes/Freya_Muir/PhD/Year2/ModelsFrameworks/aviso-fes/data/fes2014"
daterange = dates
tidelatlon = [lonmax,latmax-(2/latmin)] # seaward edge, halfway between S and N
if os.path.isfile(os.path.join('./Data/tides',sitename+'_tides.csv')) == False:
    Toolbox.ComputeTides(settings,tidepath,daterange,tidelatlon)


#%% Vegetation Line Extraction

"""
OPTION 1: Run extraction tool and return output dates, lines, filenames and 
image properties.
"""
#get_ipython().run_line_magic('matplotlib', 'qt')
# clf_model = 'MLPClassifier_Veg_L8S2.pkl'
# clf_model = 'MLPClassifier_Veg_PSScene.pkl'
clf_model = 'MLPClassifier_Veg_L5L8S2.pkl' 
output, output_latlon, output_proj = VegetationLine.extract_veglines(metadata, settings, polygon, dates, clf_model)

# L5: 44 images, 2:13 (133s) = 0.33 im/s OR 3 s/im
# L8: 20 images (10% of 198), 4:23 (263s) = 0.08 im/s OR 13 s/im
# S2: 335 images, 5:54 (354s) = 0.096 im/s OR 10.4 s/im


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


#%% Create Transects
SmoothingWindowSize = 21 
NoSmooths = 100
TransectSpacing = 10
DistanceInland = 140
DistanceOffshore = 900
# DistanceInland = 150 # East
# DistanceOffshore = 700 # East
BasePath = 'Data/' + sitename + '/veglines'
VeglineShp = glob.glob(BasePath+'/*veglines.shp')
VeglineGDF = gpd.read_file(VeglineShp[0])
WaterlineShp = glob.glob(BasePath+'/*waterlines.shp')
WaterlineGDF = gpd.read_file(WaterlineShp[0])
# Produces Transects for the reference line
TransectPath =  os.path.join(BasePath, sitename+'_Transects.shp')

if os.path.isfile(TransectPath) is False:
    TransectGDF = Transects.ProduceTransects(SmoothingWindowSize, NoSmooths, TransectSpacing, DistanceInland, DistanceOffshore, settings['output_epsg'], sitename, BasePath, referenceLineShp, projection_epsg)
else:
    print('Transects already exist and were loaded')
    TransectGDF = gpd.read_file(TransectPath)

# make new transect intersections folder
if os.path.isdir(os.path.join(filepath, sitename, 'intersections')) is False:
    os.mkdir(os.path.join(filepath, sitename, 'intersections'))

#%% Create (or load) intersections with sat and validation lines per transect

if os.path.isfile(os.path.join(filepath, sitename, 'intersections', sitename + '_transect_intersects.pkl')):
    print('Transect Intersect GDF exists and was loaded')
    with open(os.path.join
              (filepath , sitename, 'intersections', sitename + '_transect_intersects.pkl'), 'rb') as f:
        TransectInterGDF = pickle.load(f)
else:
    # Get intersections
    TransectInterGDF = Transects.GetIntersections(BasePath, TransectGDF, VeglineGDF)
    # Save newly intersected transects as shapefile
    TransectInterGDF = Transects.SaveIntersections(TransectInterGDF, VeglineGDF, BasePath, sitename)
    # Repopulate dict with intersection distances along transects normalised to transect midpoints
    TransectInterGDF = Transects.CalculateChanges(TransectInterGDF)
    
    with open(os.path.join(filepath, sitename, 'intersections', sitename + '_transect_intersects.pkl'), 'wb') as f:
        pickle.dump(TransectInterGDF, f)
        
#%% Instantaneous waterline intersect info
if os.path.isfile(os.path.join(filepath, sitename, 'intersections', sitename + '_transect_water_intersects.pkl')):
    print('Transect Intersect + Water GDF exists and was loaded')
    with open(os.path.join
              (filepath , sitename, 'intersections', sitename + '_transect_water_intersects.pkl'), 'rb') as f:
        TransectInterGDFWater = pickle.load(f)
else:        
    if settings['wetdry'] == True:
        # beachslope = 0.006 # tanBeta StAnd W
        beachslope = 0.04 # tanBeta StAnE
        TransectInterGDFWater = Transects.GetBeachWidth(BasePath, TransectGDF, TransectInterGDF, WaterlineGDF, settings, output, beachslope)  
        TransectInterGDFWater = Transects.SaveWaterIntersections(TransectInterGDFWater, WaterlineGDF,  BasePath, sitename, settings['projection_epsg'])
    
    with open(os.path.join(filepath, sitename, 'intersections', sitename + '_transect_water_intersects.pkl'), 'wb') as f:
        pickle.dump(TransectInterGDFWater, f)

#%% Topography and TZ intersect info

if os.path.isfile(os.path.join(filepath, sitename, 'intersections', sitename + '_transect_topo_intersects.pkl')):
    print('Transect Intersect + Topo GDF exists and was loaded')
    with open(os.path.join
              (filepath , sitename, 'intersections', sitename + '_transect_topo_intersects.pkl'), 'rb') as f:
        TransectInterGDFTopo = pickle.load(f)
else:
    DTM = '/media/14TB_RAID_Array/User_Homes/Freya_Muir/PhD/Year2/ModelsFrameworks/CoastLearn-main/Validation/StAndrews_20201120_Phase5DTM_1m_Slope.tif'
               
    # Update Transects with Transition Zone widths and slope if available
    TransectInterGDFTopo = Transects.TZIntersect(settings, TransectInterGDF, VeglineGDF, BasePath)
    TransectInterGDFTopo = Transects.SlopeIntersect(settings, TransectInterGDFTopo, VeglineGDF, BasePath, DTM)
    
    with open(os.path.join(filepath, sitename, 'intersections', sitename + '_transect_topo_intersects.pkl'), 'wb') as f:
        pickle.dump(TransectInterGDFTopo, f)
        
#%% Wave hindcast intersect info
if os.path.isfile(os.path.join(filepath, sitename, 'intersections', sitename + '_transect_wave_intersects.pkl')):
    print('Transect Intersect + Wave GDF exists and was loaded')
    with open(os.path.join
              (filepath , sitename, 'intersections', sitename + '_transect_wave_intersects.pkl'), 'rb') as f:
        TransectInterGDFWave = pickle.load(f)
else:
    TransectInterGDFWave = Transects.WavesIntersect(settings, TransectInterGDF, output, lonmin, lonmax, latmin, latmax)
    
    with open(os.path.join(filepath, sitename, 'intersections', sitename + '_transect_wave_intersects.pkl'), 'wb') as f:
        pickle.dump(TransectInterGDFWave, f)



#%% PLOTTING

# Veg water timeseries plot
TransectIDs = [[1575,309]]
for TransectID in TransectIDs:
    
    # Plotting.VegTimeseries(sitename, TransectInterGDF, TransectID, DateRange)
    Plotting.VegWaterTimeseries(sitename, TransectInterGDF, TransectID, Hemisphere='N')
    
#%% Beach width timeseries plot
TransectIDs = [180,1650]
for TransectID in TransectIDs:
    DateRange = [0,len(TransectInterGDF['dates'].iloc[TransectID])]
    Plotting.WidthTimeseries(sitename, TransectInterGDF, TransectID, DateRange)

#%% Plot multivariate matrix of different variables per transect subsets
# Subsets should have same number of transects

# Plotting.ClusterRates(sitename, TransectInterGDF, [232,290], [1661,1719])
Plotting.MultivariateMatrix(sitename, TransectInterGDF, TransectInterGDFWater, TransectInterGDFTopo,TransectInterGDFWave, [232,290], [1661,1719])

#%% WP Errors plot
CSVpath = '/media/14TB_RAID_Array/User_Homes/Freya_Muir/PhD/Year2/Outputs/Spreadsheets/StAndrews_VegIntersect_WeightedPeaks_Errors_Planet.csv'
figpath = '/media/14TB_RAID_Array/User_Homes/Freya_Muir/PhD/Year2/Outputs/Figures/VegPaperFigs'
Plotting.WPErrors(figpath, sitename, CSVpath)

#%% Storms timeline plot
figpath = '/media/14TB_RAID_Array/User_Homes/Freya_Muir/PhD/Year2/Outputs/Figures/VegPaperFigs'
CSVpath = '/media/14TB_RAID_Array/User_Homes/Freya_Muir/PhD/Year2/Outputs/Spreadsheets/UK_Storms.csv'
Plotting.StormsTimeline(figpath, 'StAndrewsEWP', CSVpath)



