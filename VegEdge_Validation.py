#!/usr/bin/env python
# coding: utf-8

# # VegEdge
# 

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
import pandas as pd

ee.Initialize()

#%% Define AOI using coordinates of a rectangle

"""
OPTION 2: AOI polygon is defined using coordinates of a bounding box (in WGS84).
"""
projection_epsg = 27700
image_epsg = 32630

#%%ST ANDREWS EAST
sitename = 'StAndrewsEast'
lonmin, lonmax = -2.84869, -2.79878
latmin, latmax = 56.32641, 56.39814

#%%
sitename = 'StAndrewsEastFull'
lonmin, lonmax = -2.84869, -2.79878
latmin, latmax = 56.32641, 56.39814

#%%ST ANDREWS WEST
sitename = 'StAndrewsWest'
lonmin, lonmax = -2.89087, -2.84869
latmin, latmax = 56.32641, 56.39814

#%% other sites

##ST ANDREWS
sitename = 'StAndrewsPlanet'
lonmin, lonmax = -2.89087, -2.79878
latmin, latmax = 56.32641, 56.39814
# lonmin, lonmax = -2.842023, -2.774955
# latmin, latmax = 56.338343, 56.368490

##FELIXSTOWE
#lonmin, lonmax = 1.316128, 1.370888
#latmin, latmax = 51.930771, 51.965265

##BAY OF SKAILL
#lonmin, lonmax = -3.351555, -3.332693
#latmin, latmax = 59.048456, 59.057759

##SHINGLE STREET
#lonmin, lonmax = 1.446131, 1.460008
#latmin, latmax = 52.027039, 52.037448

#%%

# Return AOI after checking coords and saving folium map HTML in sitename directory
polygon, point = Toolbox.AOI(lonmin, lonmax, latmin, latmax, sitename, image_epsg)

# it's recommended to convert the polygon to the smallest rectangle (sides parallel to coordinate axes)       
polygon = Toolbox.smallest_rectangle(polygon)

#%% Image Settings

# directory where the data will be stored
filepath = os.path.join(os.getcwd(), 'Data')

# date range
#dates = ['2021-05-01', '2021-07-02']

# date range for valiation
vegsurveyshp = './Validation/StAndrews_Veg_Edge_combined_2007_2022_singlepart.shp'
vegsurvey = gpd.read_file(vegsurveyshp)
vegdatemin = vegsurvey.Date.min()
vegdatemax = vegsurvey.Date.max()
# vegdatemin = datetime.strftime(datetime.strptime(vegsurvey.Date.min(), '%Y-%m-%d') - timedelta(weeks=4),'%Y-%m-%d')
# vegdatemax = datetime.strftime(datetime.strptime(vegsurvey.Date.max(), '%Y-%m-%d') + timedelta(weeks=4),'%Y-%m-%d')
dates = [vegdatemin, vegdatemax]
# dates = list(vegsurvey.Date.unique())
# dates = ['2007-04-04', '2011-05-05']

print(dates)

if len(dates)>2:
    daterange='no'
else:
    daterange='yes'


years = list(Toolbox.daterange(datetime.strptime(dates[0],'%Y-%m-%d'), datetime.strptime(dates[-1],'%Y-%m-%d')))

# satellite missions
# Input a list of containing any/all of 'L5', 'L7', 'L8', 'S2'
sat_list = ['L5','L7','L8','S2']
# sat_list = ['PSScene4Band']


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

#%%  Metadata filtering using validation dates only

nearestdates, nearestIDs = Toolbox.NearestDates(vegsurvey,metadata,sat_list)

#%%
L5 = dict.fromkeys(metadata['L5'].keys())
L7 = dict.fromkeys(metadata['L7'].keys())
L8 = dict.fromkeys(metadata['L8'].keys())
S2 = dict.fromkeys(metadata['S2'].keys())

#%% St Andrews East list of indices reflecting chosen validation dates
L5index = [0,20,21,45,46]
L7index = [0,1]
L8index = [117,34,118,35,42,128,129,143,144,57,73,74,153,154,159,160,169,190,198,199]
S2index = [41,42,43,44,133,134,135,136,137,138,139,140,141,142,143,144,257,258,259,260,261,262,263,411,412,413,414,415,416,417,501,502,503,504,505,506,507,508,509,510,619,620,621,622,623,624,948,949,950,951,952,953,954,1059,1060,1061,1062,1063]
#%% St Andrews West
L5index = [0,19,20,46,21]
L7index = [0,1]
L8index = [117,34,118,35,42,43,128,129,143,144,57,73,74,153,154,159,160,205,206,210,180, 211,220,212,213]
S2index = [41,42,43,44,133,134,135,136,137,138,139,140,141,142,143,144,257,258,259,260,261,262,263,411,412,413,414,415,416,417,505,506,507,508,509,834,835,836,837,838,839,840,947,948,949, 967,968,969,970,971,972,973,974,975,976,977,978,979,1003,1004,1005,1006]
#%%

for satkey in dict.fromkeys(metadata['L5'].keys()):
    L5[satkey] = [metadata['L5'][satkey][L5index[0]]]
    L7[satkey] = [metadata['L7'][satkey][L7index[0]]]
    L8[satkey] = [metadata['L8'][satkey][L8index[0]]]
    S2[satkey] = [metadata['S2'][satkey][S2index[0]]]
    for i in L5index[1:]:
        L5[satkey].append(metadata['L5'][satkey][i])
    for j in L7index[1:]:
        L7[satkey].append(metadata['L7'][satkey][j])    
    for k in L8index[1:]:
        L8[satkey].append(metadata['L8'][satkey][k])
    for l in S2index[1:]:
        S2[satkey].append(metadata['S2'][satkey][l])

            
metadata = {'L5':L5, 'L7':L7, 'L8':L8, 'S2':S2}
with open(os.path.join(filepath, sitename, sitename + '_validation_metadata.pkl'), 'wb') as f:
    pickle.dump(metadata, f)


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
tidelatlon = [-2.79878,latmax-(2/latmin)] # seaward edge, halfway between S and N
Toolbox.ComputeTides(settings,tidepath,daterange,tidelatlon)


#%% Vegetation Line Extraction

"""
OPTION 1: Run extraction tool and return output dates, lines, filenames and 
image properties.
"""
#get_ipython().run_line_magic('matplotlib', 'qt')
clf_model = 'Aberdeen_MLPClassifier_Veg_S2.pkl'
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

#%% Slice up output for validation
# east dates
slicedates = ['2007-04-17','2011-04-28','2011-10-28','2015-09-30','2017-07-24','2018-06-24','2019-02-04','2019-08-18','2021-07-01','2022-02-11']
#%% west dates
slicedates = ['2007-04-17','2011-04-28','2011-11-06','2015-09-30','2017-07-24','2018-06-24','2018-12-16','2019-08-13','2021-07-01','2022-02-11']

#%% if old otsu threshold name remains
output = {"vthreshold" if k == 'Otsu_threshold' else k:v for k,v in output.items()}
output_latlon = {"vthreshold" if k == 'Otsu_threshold' else k:v for k,v in output_latlon.items()}
output_proj = {"vthreshold" if k == 'Otsu_threshold' else k:v for k,v in output_proj.items()}

newoutputdict = output.copy()
for key in output.keys():
    newoutput = []
    for slicedate in slicedates:
        if slicedate == '2017-07-24':
            newoutput.append(output[key][[i for i, x in enumerate(output['dates']) if x == slicedate][1]])        
        else:
            newoutput.append(output[key][output['dates'].index(slicedate)])
    newoutputdict[key] = newoutput

output = newoutputdict.copy()
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
# DistanceInland = 150 # East
# DistanceOffshore = 700 # East
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
#%%
TransectInterGDF = Transects.SaveIntersections(TransectDict, VeglineGDF, BasePath, sitename, settings['projection_epsg'])
    # Repopulate dict with intersection distances along transects normalised to transect midpoints
TransectDict = Transects.CalculateChanges(TransectDict,TransectInterGDF)
    # TransectDict = Transects.GetTransitionDists(TransectDict,TransectInterGDF)
#%%    
if settings['wetdry'] == True:
    beachslope = 0.006 # tanBeta StAnd W
    # beachslope = 0.04 # tanBeta StAnE
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


# %% Validation Plots
TransectIDList = [(40,281),(312,415),(1637,1741)]#,(416,594),(1365,1462),(1463,1636),(1637,1741)] # east 
#%%    
TransectIDList = [(595,889),(972,1297)] # west

#%%
TransectIDList = [(40,281),(312,415),(416,711),(726,889),(972,1140),(1141,1297),(1365,1462),(1463,1636),(1637,1741)]# Planet
TransectIDList= [(0,1741)]
#%%
# Plotting.ValidViolin(sitename,ValidationShp,DatesCol,ValidDict,TransectIDs)
for TransectIDs in TransectIDList:
    PlotTitle = 'Accuracy of Transects ' + str(TransectIDs[0]) + ' to ' + str(TransectIDs[1])
    # PlottingSeaborn.SatViolin(sitename,VeglineShp[0],'dates',ValidDict,TransectIDs, PlotTitle)
    PlottingSeaborn.SatPDF(sitename,VeglineShp[0],'dates',ValidDict,TransectIDs, PlotTitle)

    
#%% not finished
TransectIDList = [(40,281)]
for TransectIDs in TransectIDList:
    PlotTitle = 'Accuracy of Transects ' + str(TransectIDs[0]) + ' to ' + str(TransectIDs[1])
    Plotting.ValidPDF(sitename,VeglineShp[0],'dates',ValidDict,TransectIDs, PlotTitle)
 
    
#%% Error stats
# East errors
TransectIDList = [(40,281),(312,415),(416,594),(1365,1462),(1463,1636),(1637,1741)]

# West errors
# TransectIDList = [(595,711),(726,889),(972,1140),(1141,1297)]
for TransectIDs in TransectIDList:
    Toolbox.QuantifyErrors(sitename, VeglineShp[0],'dates',ValidDict,TransectIDs)

#%% Combine East and West
with open(os.path.join(os.getcwd(), 'Data', 'StAndrewsEast', 'validation','StAndrewsEast' + '_valid_dict.pkl'), 'rb') as f:
    EastValidDict = pickle.load(f)
with open(os.path.join(os.getcwd(), 'Data', 'StAndrewsWest', 'validation', 'StAndrewsWest' + '_valid_dict.pkl'), 'rb') as f:
    WestValidDict = pickle.load(f)

FullValidDict = EastValidDict.copy()
for keyname in FullValidDict.keys():
    FullValidDict[keyname][586:1303] = WestValidDict[keyname][586:1303]

BasePath = 'Data/' + 'StAndrewsEast' + '/veglines'
EastVeglineShp = gpd.read_file(glob.glob(BasePath+'/*veglines.shp')[0])
BasePath = 'Data/' + 'StAndrewsWest' + '/veglines'
WestVeglineShp = gpd.read_file(glob.glob(BasePath+'/*veglines.shp')[0])

FullVeglineShp = gpd.pd.concat([EastVeglineShp, WestVeglineShp])

#%% Full violin and errors

ClipValidDict = dict.fromkeys(FullValidDict.keys())
for keyname in FullValidDict.keys():
    ClipValidDict[keyname] = []
    ClipValidDict[keyname].extend(FullValidDict[keyname][40:281])
    ClipValidDict[keyname].extend(FullValidDict[keyname][312:711])
    ClipValidDict[keyname].extend(FullValidDict[keyname][726:889])
    ClipValidDict[keyname].extend(FullValidDict[keyname][972:1297])
    ClipValidDict[keyname].extend(FullValidDict[keyname][1365:1741])

TransectIDs = (0,len(ClipValidDict['dates'])) # full

# PlottingSeaborn.SatViolin(sitename,FullVeglineShp,'dates',ClipValidDict,TransectIDs, 'Full Site Accuracy')
PlottingSeaborn.SatPDF(sitename,FullVeglineShp,'dates',ClipValidDict,TransectIDs, 'Full Site Accuracy')
# Plotting.SatRegress(sitename,FullVeglineShp,'dates',ClipValidDict,TransectIDs, 'Full Site Accuracy')

for TransectID in [TransectIDs]:
    Toolbox.QuantifyErrors(sitename, VeglineShp[0],'dates',ValidDict,TransectID)



#%%
PlottingSeaborn.PlatformViolin(sitename,FullVeglineShp,'satname',ClipValidDict,TransectIDs)


#%% Theshold plotting


sites = ['StAndrewsWest', 'StAndrewsEast']
PlottingSeaborn.ThresholdViolin(filepath, sites)


#%% Validation vs satellite cross-shore distance through time

Plotting.ValidTimeseries(sitename, ValidDict, 1575)

#%%
TransectIDs = [[1575,309],[120]]
for TransectID in TransectIDs:
    
    # Plotting.VegTimeseries(sitename, TransectDict, TransectID, DateRange)
    Plotting.VegWaterTimeseries(sitename, TransectDict, TransectID)
    
#%%
TransectIDs = [180,1650]
for TransectID in TransectIDs:
    DateRange = [0,len(TransectDict['dates'][TransectID])]
    Plotting.WidthTimeseries(sitename, TransectDict, TransectID, DateRange)

#%% 

PlottingSeaborn.PlatformViolin(sitename,FullVeglineShp,'satname',ClipValidDict,TransectIDs, 'Full Site Accuracy')



