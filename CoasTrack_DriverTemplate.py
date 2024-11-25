#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 11:22:08 2024

@author: fmuir
"""

import os
import glob
import pickle
import geopandas as gpd

from Toolshed import Transects, Toolbox, Plotting, PlottingSeaborn


#%% EDIT ME: Requirements

# Name of site to save directory and files under
sitename = 'SITENAME'

# Filepaths to data and veg edge/waterline shapefiles
BasePath = 'Data/' + sitename + '/lines'
filepath = os.path.join(os.getcwd(), 'Data')

# Name of reference shoreline shapefile
referenceLineShp = 'SITENAME_refLine.shp'

# Load in settings previously saved at end of run
with open(os.path.join(filepath, sitename, sitename + '_settings.pkl'), 'rb') as f:
    settings = pickle.load(f)

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

# Read in VedgeSat run outputs
output, output_latlon, output_proj = Toolbox.ReadOutput(inputs)
# Remove Duplicate Lines
output = Toolbox.RemoveDuplicates(output) 

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

##%% Transect-Water Intersections
if os.path.isfile(os.path.join(filepath, sitename, 'intersections', sitename + '_transect_water_intersects.pkl')):
    print('Transect Intersect + Water GDF exists and was loaded')
    with open(os.path.join
              (filepath , sitename, 'intersections', sitename + '_transect_water_intersects.pkl'), 'rb') as f:
        TransectInterGDFWater = pickle.load(f)
else:        
    if settings['wetdry'] == True:
        TransectInterGDFWater = Transects.GetWaterIntersections(BasePath, TransectGDF, TransectInterGDF, WaterlineGDF, settings, output)  
    
    with open(os.path.join(filepath, sitename, 'intersections', sitename + '_transect_water_intersects.pkl'), 'wb') as f:
        pickle.dump(TransectInterGDFWater, f)

#%% Transect-Waves Intersections (needs to be before tidal corrections if using runup as well)
# This is for comparing veg edge positions with nearshore wave conditions at the time the image was taken. 
# Note: this requires you to have a Copernicus Marine Service (CMEMS) account with access to their hindcast model, 
# as you will be asked for a username and password.

lonmin, lonmax, latmin, latmax = Toolbox.GetBounds(settings)

if os.path.isfile(os.path.join(filepath, sitename, 'intersections', sitename + '_transect_wave_intersects.pkl')):
    print('Transect Intersect + Wave GDF exists and was loaded')
    with open(os.path.join
              (filepath , sitename, 'intersections', sitename + '_transect_wave_intersects.pkl'), 'rb') as f:
        TransectInterGDFWave = pickle.load(f)
else:
    TransectInterGDFWave = Transects.WavesIntersect(settings, TransectInterGDF, BasePath, output, lonmin, lonmax, latmin, latmax)
    
    with open(os.path.join(filepath, sitename, 'intersections', sitename + '_transect_wave_intersects.pkl'), 'wb') as f:
        pickle.dump(TransectInterGDFWave, f)

#%% Additional wave-based WL metrics
# This is for comparing shoreline change with vegetation change, and for quantifying the beach width between the two for each image. If you would like to 
# include runup in your waterline corrections, add `TransectInterGDFWave` to `GetWaterIntersections()`:
    # TransectInterGDFWater = Transects.GetWaterIntersections(
    # BasePath, TransectGDF, TransectInterGDF, WaterlineGDF, settings, output, TransectInterGDFWave, beachslope)

# If you want to include runup AND calculate slopes using CoastSat.slope (recommended), exclude the `beachslope` variable:
    # TransectInterGDFWater = Transects.GetWaterIntersections(
    # BasePath, TransectGDF, TransectInterGDF, WaterlineGDF, settings, output, TransectInterGDFWave)

if 'wlcorrdist' not in TransectInterGDFWater.columns:
    # Tidal correction to get corrected distances along transects
    TransectInterGDFWater = Transects.WLCorrections(settings, output, TransectInterGDFWater, TransectInterGDFWave)     
    # Calculate width between VE and corrected WL
    TransectInterGDFWater = Transects.CalcBeachWidth(settings, TransectGDF, TransectInterGDFWater)
    # Calculate rates of change on corrected WL and save as Transects shapefile
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