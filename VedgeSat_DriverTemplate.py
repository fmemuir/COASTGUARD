#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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

Sat = Toolbox.LocalImageRetrieval(inputs)
metadata = Toolbox.LocalImageMetadata(inputs, Sat)


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
# (ONLY RUN IF YOU HAVE pyFES INSTALLED AND WANT TIDAL INFO SAVED)
tidepath = "/path/to/your/FES2014/files/aviso-fes/data/fes2014"
daterange = dates
tidelatlon = [(latmin+latmax)/2, (lonmin+lonmax)/2] # centre of bounding box
Toolbox.ComputeTides(settings,tidepath,daterange,tidelatlon) 
    
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


#%% EDIT ME: Define Settings for Cross-shore Transects

SmoothingWindowSize = 21 
NoSmooths = 100
TransectSpacing = 10
DistanceInland = 100
DistanceOffshore = 100

# Provide average beach slope (tanBeta) for site, for calculating corrected beach widths
# Set to 'None' if you want to use CoastSat.slope to calculate per-transect slopes for correcting with
beachslope = None 


#%% Create Cross-shore Transects

VegBasePath = 'Data/' + sitename + '/lines'
VeglineShp = glob.glob(BasePath+'/*veglines.shp')
VeglineGDF = gpd.read_file(VeglineShp[0])
VeglineGDF = VeglineGDF.sort_values(by='dates') # sort GDF by dates to ensure transect intersects occur in chronological order
VeglineGDF = VeglineGDF.reset_index(drop=True) # reset GDF index after date sorting
if settings['wetdry'] == True:
    WaterlineShp = glob.glob(BasePath+'/*waterlines.shp')
    WaterlineGDF = gpd.read_file(WaterlineShp[0])
    WaterlineGDF = WaterlineGDF.sort_values(by='dates') # as above with VeglineGDF date sorting
    WaterlineGDF = WaterlineGDF.reset_index(drop=True)
# Produces Transects for the reference line
TransectSpec =  os.path.join(BasePath, sitename+'_Transects.shp')

# If transects already exist, load them in
if os.path.isfile(TransectSpec[:-3]+'pkl') is False:
    TransectGDF = Transects.ProduceTransects(settings, SmoothingWindowSize, NoSmooths, TransectSpacing, DistanceInland, DistanceOffshore, VegBasePath, referenceLineShp)
else:
    print('Transects already exist and were loaded')
    with open(TransectSpec[:-3]+'pkl', 'rb') as Tfile: 
        TransectGDF = pickle.load(Tfile)
    
# make new transect intersections folder
if os.path.isdir(os.path.join(filepath, sitename, 'intersections')) is False:
    os.mkdir(os.path.join(filepath, sitename, 'intersections'))

#%% Transect-Veg Intersections
# Create (or load) intersections with all satellite lines per transect

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

#%% Transect-Waves Intersections (needs to be run before water if using runup in tidal corrections)

if os.path.isfile(os.path.join(filepath, sitename, 'intersections', sitename + '_transect_wave_intersects.pkl')):
    print('Transect Intersect + Wave GDF exists and was loaded')
    with open(os.path.join
              (filepath , sitename, 'intersections', sitename + '_transect_wave_intersects.pkl'), 'rb') as f:
        TransectInterGDFWave = pickle.load(f)
else:
    TransectInterGDFWave = Transects.WavesIntersect(settings, TransectInterGDF, output, lonmin, lonmax, latmin, latmax)
    
    with open(os.path.join(filepath, sitename, 'intersections', sitename + '_transect_wave_intersects.pkl'), 'wb') as f:
        pickle.dump(TransectInterGDFWave, f)

#%% Transect-Water Intersections
# If you want to include runup in your waterline corrections, add TransectInterGDFWave:
    # Transects.GetWaterIntersections(BasePath, TransectGDF, TransectInterGDF, WaterlineGDF, settings, output, TransectInterGDFWave, beachslope)
# If you want to include runup AND calculate slope using CoastSat.slope, exclude beachslope:
    # Transects.GetWaterIntersections(BasePath, TransectGDF, TransectInterGDF, WaterlineGDF, settings, output, TransectInterGDFWave)
if os.path.isfile(os.path.join(filepath, sitename, 'intersections', sitename + '_transect_water_intersects.pkl')):
    print('Transect Intersect + Water GDF exists and was loaded')
    with open(os.path.join
              (filepath , sitename, 'intersections', sitename + '_transect_water_intersects.pkl'), 'rb') as f:
        TransectInterGDFWater = pickle.load(f)
else:        
    if settings['wetdry'] == True:
        TransectInterGDFWater = Transects.GetWaterIntersections(BasePath, TransectGDF, TransectInterGDF, WaterlineGDF, settings, output, beachslope)  
        TransectInterGDFWater = Transects.SaveWaterIntersections(TransectInterGDFWater, WaterlineGDF,  BasePath, sitename)
    
    with open(os.path.join(filepath, sitename, 'intersections', sitename + '_transect_water_intersects.pkl'), 'wb') as f:
        pickle.dump(TransectInterGDFWater, f)

#%% Transect-Topo Intersections
# EDIT ME: Path to slope raster for extracting slope values
TIF = '/path/to/Slope_Raster.tif'

if os.path.isfile(os.path.join(filepath, sitename, 'intersections', sitename + '_transect_topo_intersects.pkl')):
    print('Transect Intersect + Topo GDF exists and was loaded')
    with open(os.path.join
              (filepath , sitename, 'intersections', sitename + '_transect_topo_intersects.pkl'), 'rb') as f:
        TransectInterGDFTopo = pickle.load(f)
else:
    # Update Transects with Transition Zone widths and slope if available
    TransectInterGDFTopo = Transects.TZIntersect(settings, TransectInterGDF, VeglineGDF, BasePath)
    TransectInterGDFTopo = Transects.SlopeIntersect(settings, TransectInterGDFTopo, VeglineGDF, BasePath, TIF)
    
    with open(os.path.join(filepath, sitename, 'intersections', sitename + '_transect_topo_intersects.pkl'), 'wb') as f:
        pickle.dump(TransectInterGDFTopo, f)


#%% Timeseries Plotting

# EDIT ME: Select transect ID to plot
# You can plot subplots within a list of plot IDs, e.g. [[sub1, sub2], plot2]
# You can also comment Line 1 out and uncomment Line 2 to create plots for ALL Transect IDs
# NOTE: If you want to plot ALL transects, it's recommended you switch ShowPlot=False

TransectIDs = [[25,30,35],50,75] # Line 1
# TransectIDs = list(TransectInterGDF['TransectID']) # Line 2

for TransectID in TransectIDs:
    # Plot timeseries of cross-shore veg position
    Plotting.VegTimeseries(sitename, TransectInterGDF, TransectID, Hemisphere='N', ShowPlot=True)
    # If plotting veg and water lines together
    if settings['wetdry']:
        Plotting.VegWaterTimeseries(sitename, TransectInterGDFWater, TransectID, Hemisphere='N', ShowPlot=True)

    
#%% Beach Width Plotting

# Select transect ID to plot
TransectIDs = [[25,30,35],50,75]
for TransectID in TransectIDs:
    # Plot timeseries of cross-shore width between water edge and veg edge 
    Plotting.WidthTimeseries(sitename, TransectInterGDFWater, TransectID, Hemisphere='N')


#%% EDIT ME: Validation Settings
# Most likely you won't need to validate your lines, but if you do, edit these parameters

# Name of date column in validation edges shapefile (case sensitive!) 
DatesCol = 'Date'
ValidationShp = './Validation/StAndrews_Veg_Edge_combined_2007_2022_singlepart.shp'

#%% Satellite Edges Validation

validpath = os.path.join(os.getcwd(), 'Data', sitename, 'validation')

if os.path.isfile(os.path.join(validpath, sitename + '_valid_intersects.pkl')):
    print('ValidDict exists and was loaded')
    with open(os.path.join(validpath, sitename + '_valid_intersects.pkl'), 'rb') as f:
        ValidInterGDF = pickle.load(f)
else:
    ValidInterGDF = Transects.ValidateSatIntersects(sitename, ValidationShp, DatesCol, TransectGDF, TransectInterGDF)
    with open(os.path.join(validpath, sitename + '_valid_intersects.pkl'), 'wb') as f:
        pickle.dump(ValidInterGDF, f)


#%% Validation Plots
# EDIT ME: List of transect ID tuples (startID, finishID)
TransectIDList = [(0,1741)]

for TransectIDs in TransectIDList:
    PlotTitle = 'Accuracy of Transects ' + str(TransectIDs[0]) + ' to ' + str(TransectIDs[1])
    PlottingSeaborn.SatViolin(sitename,VeglineGDF,'dates',ValidInterGDF,TransectIDs, PlotTitle)
    PlottingSeaborn.SatPDF(sitename,VeglineGDF,'dates',ValidInterGDF,TransectIDs, PlotTitle)
    Plotting.SatRegress(sitename,VeglineGDF,'dates',ValidInterGDF,TransectIDs, PlotTitle)

    
#%% Quantify errors between validation and satellite derived lines
# EDIT ME: List of transect ID tuples (startID, finishID
TransectIDList = [(595,711),(726,889),(972,1140),(1141,1297)]

for TransectIDs in TransectIDList:
    Toolbox.QuantifyErrors(sitename, VeglineGDF,'dates',ValidInterGDF,TransectIDs)
    

    
