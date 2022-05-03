#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 08:31:29 2022

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
from Elves import Download, Image_Processing, Shoreline, Toolbox, Transects, VegetationLine
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
from shapely.geometry import Point, LineString, Polygon
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


#%% Define ROI using coordinates of a rectangle

"""
OPTION 2: ROI polygon is defined using coordinates of a bounding box (in WGS84).
"""

##ST ANDREWS WEST
# sitename = 'StAndrewsWest'
# lonmin, lonmax = -2.89087, -2.84869
# latmin, latmax = 56.32641, 56.39814


##ST ANDREWS EAST
# sitename = 'StAndrewsEast'
# lonmin, lonmax = -2.84869, -2.79878
# latmin, latmax = 56.32641, 56.39814

##ST ANDREWS
sitename = 'StAndrewsPlanet'
lonmin, lonmax = -2.89087, -2.79878
latmin, latmax = 56.32641, 56.39814

##FELIXSTOWE
#lonmin, lonmax = 1.316128, 1.370888
#latmin, latmax = 51.930771, 51.965265

##BAY OF SKAILL
#lonmin, lonmax = -3.351555, -3.332693
#latmin, latmax = 59.048456, 59.057759

##SHINGLE STREET
#lonmin, lonmax = 1.446131, 1.460008
#latmin, latmax = 52.027039, 52.037448

#point = ee.Geometry.Point([lonmin, latmin]) 
if latmin > latmax:
    print('Check your latitude min and max bounding box values!')
    oldlatmin = latmin
    oldlatmax = latmax
    latmin = oldlatmax
    latmax = oldlatmin
if lonmin > lonmax:
    print('Check your longitude min and max bounding box values!')
    oldlonmin = lonmin
    oldlonmax = lonmax
    lonmin = oldlonmax
    lonmax = oldlonmin
    
polygon = [[[lonmin, latmin],[lonmax, latmin],[lonmin, latmax],[lonmax, latmax]]]
point = ee.Geometry.Point(polygon[0][0]) 

#%% Image Settings


# it's recommended to convert the polygon to the smallest rectangle (sides parallel to coordinate axes)       
polygon = Toolbox.smallest_rectangle(polygon)

# directory where the data will be stored
filepath = os.path.join(os.getcwd(), 'Data')

# date range
#dates = ['2021-05-01', '2021-07-02']

# date range for valiation
vegsurveyshp = './Validation/StAndrews_Veg_Edge_combined_2019_2022_singlepart.shp'
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
sat_list = ['PSScene4Band']

cloudthresh = 0.5

projection_epsg = 27700
image_epsg = 32630


# put all the inputs into a dictionnary
inputs = {'polygon': polygon, 'dates': dates, 'daterange':daterange, 'sat_list': sat_list, 'sitename': sitename, 'filepath':filepath, 'cloudthresh':cloudthresh}

direc = os.path.join(filepath, sitename)

if os.path.isdir(direc) is False:
    os.mkdir(direc)
 
    
#%% Image Retrieval

# before downloading the images, check how many images are available for your inputs

print('Enter your Planet API key: ')
os.environ['PL_API_KEY'] = input()
print('Your API key is: ')
Sat = Toolbox.PlanetImageRetrieval(inputs)

Subset1 = Sat[0][45:63]
Subset2 = Sat[0][243:]
Sat[0] = Subset1
Sat[0].extend(Subset2)

# idURLs = Toolbox.PlanetDownload(Sat,filepath,sitename)
# metadata = Toolbox.PlanetMetadata(Sat, filepath, sitename)

#%% Image Download

"""
OPTION 1: Populate metadata using image names pulled from server.
"""
Sat = Toolbox.image_retrieval(inputs)
metadata = Toolbox.metadata_collection(sat_list, Sat, filepath, sitename)

#%% Load In Local Imagery
Sat = Toolbox.LocalImageRetrieval(inputs)
metadata = Toolbox.LocalImageMetadata(inputs, Sat)

#%%  Metadata filtering using validation dates only
veridates = list(vegsurvey.Date.unique())
def neardate(satdates,veridate):
    return min(satdates, key=lambda x: abs(x - veridate))

nearestdates = dict.fromkeys(sat_list)
nearestIDs = dict.fromkeys(sat_list)

for sat in sat_list:
    print(sat,'sat')
    satdates=[]
    nearestdate = []
    nearestID = []
    for veridate in veridates:
        print('verification:\t',veridate)
        veridate = datetime.strptime(veridate,'%Y-%m-%d')
        for satdate in metadata[sat]['dates']:
            satdates.append(datetime.strptime(satdate,'%Y-%m-%d'))
        nearestdate.append(datetime.strftime(neardate(satdates,veridate),'%Y-%m-%d'))
        nearestID.append(metadata[sat]['dates'].index(datetime.strftime(neardate(satdates,veridate),'%Y-%m-%d')))
        print('nearest:\t\t',neardate(satdates,veridate))
    nearestdates[sat] = nearestdate
    nearestIDs[sat] = nearestID


#%%
L5 = dict.fromkeys(metadata['L5'].keys())
L8 = dict.fromkeys(metadata['L5'].keys())
S2 = dict.fromkeys(metadata['L5'].keys())

#must use extend() instead of append() for ranges of values
for satkey in dict.fromkeys(metadata['L5'].keys()):
    L5[satkey] = [metadata['L5'][satkey][0]]
    L5[satkey].extend(metadata['L5'][satkey][20:22])
    L5[satkey].extend(metadata['L5'][satkey][45:47])
    L8[satkey] = [metadata['L8'][satkey][41]]
    L8[satkey].append(metadata['L8'][satkey][42])
    L8[satkey].extend(metadata['L8'][satkey][143:145])
    L8[satkey].append(metadata['L8'][satkey][153])
    L8[satkey].append(metadata['L8'][satkey][159])       
    S2[satkey] = metadata['S2'][satkey][127:148]
    S2[satkey].extend(metadata['S2'][satkey][255:267])
    S2[satkey].extend(metadata['S2'][satkey][405:424])
    S2[satkey].extend(metadata['S2'][satkey][490:507])
    
    

metadata = {'L5':L5,'L8':L8,'S2':S2}
with open(os.path.join(filepath, sitename, sitename + '_validation_metadata.pkl'), 'wb') as f:
    pickle.dump(metadata, f)

#for nearestd in nearestdate:
#    print([i for i, x in enumerate(metadata['S2']['dates']) if x == nearestd])

# L5: 12 images (28% of 44), 1:27 (87s) = 0.138 im/s OR 7.25 im/s
# L8: 198 images, 16:42 (1002s) = 0.198 im/s OR 5 s/im
# S2: 34 images (10% of 335), 5:54 (354s) = 0.096 im/s OR 10.4 s/im

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
    'cloud_thresh': cloudthresh,        # threshold on maximum cloud cover
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

#%% Vegetation Edge Reference Line Load-In

"""
OPTION 2: Load in coordinates of reference line shapefile and format for use in
the veg extraction.
"""

#referenceLineShp = os.path.join(inputs['filepath'], sitename,'StAndrews_refLine.shp')
referenceLineShp = os.path.join(inputs['filepath'], 'StAndrews_refLine.shp')
referenceLineDF = gpd.read_file(referenceLineShp)
refLinex,refLiney = referenceLineDF.geometry[0].coords.xy
# swap latlon coordinates around and format into list
referenceLineList = list([refLinex[i],refLiney[i]] for i in range(len(refLinex)))
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


#%% Save output veglines


#Saves the veglines as shapefiles locally under Veglines.
direc = os.path.join(filepath, 'Veglines')
geomtype = 'lines'
name_prefix = 'Data/' + sitename + '/Veglines/'

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


#%% Option 1: Compute distances of shorelines along transects


settings['along_dist'] = 50
cross_distance = Transects.compute_intersection(output_proj, transect_proj, settings, 'vegetation_') 


#%% Option 2: Load distances in if they already exist

cross_distance = dict([])

with open('Data/'+sitename+'/vegetation_transect_time_series.csv', newline='') as csvfile:
    spamreader = csv.DictReader(csvfile, delimiter=',', quotechar='|')
    for lines in spamreader:
        for i in range(len(lines)-2):
            cross_distance['Transect_'+str(i+1)] = []

with open('Data/'+sitename+'/vegetation_transect_time_series.csv', newline='') as csvfile:
    spamreader = csv.DictReader(csvfile, delimiter=',', quotechar='|')
    for lines in spamreader:
        for i in range(len(lines)-2):
            transect_name = 'Transect Transect_' + str(i+1)
            try:
                cross_distance['Transect_'+str(i+1)].append(float(lines[transect_name]))
            except:
                cross_distance['Transect_'+str(i+1)].append(np.nan)


#%% Validation Data compilation into dict
vegsurveyshp = './Validation/StAndrews_Veg_Edge_combined_singlepart.shp'
vegsurvey = gpd.read_file(vegsurveyshp)

settings['along_dist'] = 50

# define disctionary of same structure as output_proj
vegsurvey_proj = dict.fromkeys(output_proj.keys())
vegsurvey = vegsurvey.sort_values(by=['Date'])
# fill dates field from geodataframe of survey lines
vegsurvey_proj['dates'] = list(vegsurvey['Date'])
vegsurvey_proj['shorelines'] = []

for i in range(len(vegsurvey)):
    # get x and y coords of each survey line (singlepart!)
    vegxs,vegys = vegsurvey.geometry[i].coords.xy
    vegx_points = np.array([])
    vegy_points = np.array([])
    for j in range(len(vegxs)):
        # populate separate arrays of x and y values
        vegx_points = np.append(vegx_points,vegxs[j])
        vegy_points = np.append(vegy_points,vegys[j])
    # concatenate x and y coords together as two columns in array
    vegsurvey_proj['shorelines'].append(np.column_stack([vegx_points,vegy_points])) 

#%%Validation Data intersections
# perform intersection calculations for each transect       
veg_cross_distance = Transects.compute_intersection(vegsurvey_proj, transect_proj, settings, 'vegsurveys_') 

#%% Option 2: Load veg survey distances in if they already exist

veg_cross_distance = dict([])

with open('Data/'+sitename+'/vegsurveys_transect_time_series.csv', newline='') as csvfile:
    spamreader = csv.DictReader(csvfile, delimiter=',', quotechar='|')
    for lines in spamreader:
        for i in range(len(lines)-2):
            veg_cross_distance['Transect_'+str(i+1)] = []

with open('Data/'+sitename+'/vegsurveys_transect_time_series.csv', newline='') as csvfile:
    spamreader = csv.DictReader(csvfile, delimiter=',', quotechar='|')
    for lines in spamreader:
        for i in range(len(lines)-2):
            transect_name = 'Transect Transect_' + str(i+1)
            try:
                veg_cross_distance['Transect_'+str(i+1)].append(float(lines[transect_name]))
            except:
                veg_cross_distance['Transect_'+str(i+1)].append(np.nan)

#%% Validation statistics
'''compare distances along transects for each veg survey date matched with its closest satellite date.
cross_distance is in m'''
cross_distance['dates'] = output_proj['dates']
veg_cross_distance['dates'] = vegsurvey_proj['dates']

# Survey dates:
 # '2007-04-04'
 # '2011-05-01'
 # '2012-03-27'
 # '2016-08-01'
 # '2017-07-17'
 # '2018-06-28'
 # '2018-12-11'

# define dict of same structure as cross_distance
veg_dist = dict.fromkeys(transect_proj.keys())

for trno in range(len(transect_proj)):
    # for each transect, capture single cross_distances to fix date duplicates from singlepart
     veg_dist['Transect_'+str(trno+1)] = []
     for vegdate in list(dict.fromkeys(veg_cross_distance['dates'])):
         # get matching indices for each unique survey date
         indices = [i for i, x in enumerate(veg_cross_distance['dates']) if x == vegdate]
         # repopulate each transect list with the maximum cross distance value for each list of same dates
         try:
             veg_dist['Transect_'+str(trno+1)].append(np.nanmax(veg_cross_distance['Transect_'+str(trno+1)][indices[0]:indices[-1]+1]))
         except ValueError:
             veg_dist['Transect_'+str(trno+1)].append(np.nanmax(veg_cross_distance['Transect_'+str(trno+1)][indices[0]]))
veg_dist['dates'] = list(dict.fromkeys(veg_cross_distance['dates']))
 

#%% Export Transect Intersection Data


E_cross_distance = dict([])
W_cross_distance = dict([])
sitename = 'StAndrewsEast'
with open('Data/'+sitename+'/vegetation_transect_time_series.csv', newline='') as csvfile:
    spamreader = csv.DictReader(csvfile, delimiter=',', quotechar='|')
    for lines in spamreader:
        for i in range(len(lines)-2):
            E_cross_distance['Transect_'+str(i+1)] = []

with open('Data/'+sitename+'/vegetation_transect_time_series.csv', newline='') as csvfile:
    spamreader = csv.DictReader(csvfile, delimiter=',', quotechar='|')
    for lines in spamreader:
        for i in range(len(lines)-2):
            transect_name = 'Transect Transect_' + str(i+1)
            try:
                E_cross_distance['Transect_'+str(i+1)].append(float(lines[transect_name]))
            except:
                E_cross_distance['Transect_'+str(i+1)].append(np.nan)
                
sitename = 'StAndrewsWest'
with open('Data/'+sitename+'/vegetation_transect_time_series.csv', newline='') as csvfile:
    spamreader = csv.DictReader(csvfile, delimiter=',', quotechar='|')
    for lines in spamreader:
        for i in range(len(lines)-2):
            W_cross_distance['Transect_'+str(i+1)] = []

with open('Data/'+sitename+'/vegetation_transect_time_series.csv', newline='') as csvfile:
    spamreader = csv.DictReader(csvfile, delimiter=',', quotechar='|')
    for lines in spamreader:
        for i in range(len(lines)-2):
            transect_name = 'Transect Transect_' + str(i+1)
            try:
                W_cross_distance['Transect_'+str(i+1)].append(float(lines[transect_name]))
            except:
                W_cross_distance['Transect_'+str(i+1)].append(np.nan)

# parse out transect numbers and linestrings
parsed_transects = [[trno, LineString(transect_proj[trno])] for trno in transect_proj.keys()]
transect_df = pd.DataFrame(data=parsed_transects,columns=['TrName','geometry'])
transect_df = transect_df.set_index(transect_df['TrName'])
transect_gdf = gpd.GeoDataFrame(transect_df, geometry=transect_df['geometry'])
transect_gdf.index = range(1,transect_gdf.shape[0]+1)

# reformat/transpose to dataframes where index is Transect_x and cols are dates, formatted as 'yyyymmdd'
W_crossdist_df = pd.DataFrame(W_cross_distance, index=['s'+date.replace('-','') for date in cross_distance['dates']]).T

vegdist_df = pd.DataFrame(veg_dist, index=['v'+date.replace('-','') for date in veg_dist['dates']]).T
vegdist_df = vegdist_df.drop(vegdist_df.index[-1])
bothdists_df = pd.concat([crossdist_df,vegdist_df], axis=1)
bothdists_df.index = range(1,bothdists_df.shape[0]+1)
fulldist_df = pd.concat([bothdists_df, transect_gdf['geometry']],axis=1)
fulldist_gdf = gpd.GeoDataFrame(fulldist_df,geometry=fulldist_df['geometry'])

fulldist_gdf.to_file(os.path.join(os.getcwd()+'/Data/StAndrews_VegSat_TransectDistances.shp'))
        

#%% Plotting - Validation Statistics
#St Andrews West plotting

#Inner estuary south side
fig = plt.figure(figsize=[10,8], tight_layout=True)
#plt.axis('equal')
plt.grid(linestyle=':', color='0.5')
plt.title('Edenside, South Side')



for i,j,c in zip(range(7),[0,1,4,6,10,18,30],['#FCFFA1','#FBB314','#ED641F','#BA3251','#75176A','#2F0B5B','#07070A']):
    veg_x = list([])
    sat_y = list([])
    for trno in range(570,924):
        datelabels = 'Survey date: '+veg_dist['dates'][i]+'; Sat image date: '+cross_distance['dates'][j]
        plt.plot(veg_dist['Transect_'+str(trno+1)][i],cross_distance['Transect_'+str(trno+1)][j],color=c, marker='o', alpha=.5, label=datelabels if trno == 570 else "") #2007
        veg_x.append(veg_dist['Transect_'+str(trno+1)][i])
        sat_y.append(cross_distance['Transect_'+str(trno+1)][j])
    idx = np.isfinite(veg_x) & np.isfinite(sat_y)
    veg_x = np.array(veg_x)[idx]
    sat_y = np.array(sat_y)[idx]
    try:
        m, b = np.polyfit(veg_x, sat_y, 1)
        print(cross_distance['dates'][j]+' RMSE: '+str(mean_squared_error(veg_x, sat_y, squared=False)))
        print(cross_distance['dates'][j]+' R squared: '+str(r2_score(veg_x, sat_y)))
        plt.plot(veg_x,m*veg_x+b, color=c)
    except:
        continue
plt.plot(range(1000), range(1000), color=(0.3,0.3,0.3,0.5), linestyle='--', label='Expected trend')
plt.xlabel('Validation edge distance (m)')
plt.ylabel('Satellite derived edge distance (m)')
plt.xlim((0,1000))
plt.ylim((0,1000))
plt.legend()
plt.savefig('/media/14TB_RAID_Array/User_Homes/Freya_Muir/PhD/Year2/Outputs/Figures/VegSat_StAndrews_Errors_SEdenside.png')
plt.show()


#Inner estuary north side
fig = plt.figure(figsize=[10,8], tight_layout=True)
#plt.axis('equal')
plt.grid(linestyle=':', color='0.5')
plt.title('Edenside, North Side')
for i,j,c in zip(range(7),[0,1,4,6,10,18,30],['#FCFFA1','#FBB314','#ED641F','#BA3251','#75176A','#2F0B5B','#07070A']):
    veg_x = list([])
    sat_y = list([])
    for trno in range(925,1290):
        datelabels = 'Survey date: '+veg_dist['dates'][i]+'; Sat image date: '+cross_distance['dates'][j]
        plt.plot(veg_dist['Transect_'+str(trno+1)][i],cross_distance['Transect_'+str(trno+1)][j],color=c, marker='^', alpha=.5, label=datelabels if trno == 925 else "") #2007
        veg_x.append(veg_dist['Transect_'+str(trno+1)][i])
        sat_y.append(cross_distance['Transect_'+str(trno+1)][j])
    idx = np.isfinite(veg_x) & np.isfinite(sat_y)
    veg_x = np.array(veg_x)[idx]
    sat_y = np.array(sat_y)[idx]
    try:
        m, b = np.polyfit(veg_x, sat_y, 1)
        print(cross_distance['dates'][j]+' RMSE: '+str(mean_squared_error(veg_x, sat_y, squared=False)))
        print(cross_distance['dates'][j]+' R squared: '+str(r2_score(veg_x, sat_y)))
        plt.plot(veg_x,m*veg_x+b,color=c)
    except:
        continue
plt.plot(range(1000), range(1000), color=(0.3,0.3,0.3,0.5), linestyle='--', label='Expected trend')
plt.xlabel('Validation edge distance (m)')
plt.ylabel('Satellite derived edge distance (m)')
plt.xlim((0,600))
plt.ylim((0,600))
plt.legend()
plt.savefig('/media/14TB_RAID_Array/User_Homes/Freya_Muir/PhD/Year2/Outputs/Figures/VegSat_StAndrews_Errors_NEdenside.png')
plt.show()

#%% St Andrews East plotting

#St Andrews Peninsula
fig, ax = plt.subplots(figsize=[10,10], tight_layout=True)
#plt.axis('equal')
ax.grid(linestyle=':', color='0.5')
plt.title('St Andrews Peninsula')
axins = ax.inset_axes([0.05, 0.55, 0.4, 0.4])
axins.grid(linestyle=':', color='0.5')

for i,j,c in zip(range(7),[0,1,3,6,9,13,18],['#FCFFA1','#FBB314','#ED641F','#BA3251','#75176A','#2F0B5B','#07070A']):
    veg_x = []
    sat_y = []
    for trno in range(0,600):
        datelabels = 'Survey date: '+veg_dist['dates'][i]+'; Sat image date: '+cross_distance['dates'][j]
        ax.plot(veg_dist['Transect_'+str(trno+1)][i],cross_distance['Transect_'+str(trno+1)][j],color=c, marker='o', alpha=.5, label=datelabels if trno == 0 else "") #2007
        axins.plot(veg_dist['Transect_'+str(trno+1)][i],cross_distance['Transect_'+str(trno+1)][j],color=c, marker='o', alpha=.5, label=datelabels if trno == 0 else "") #2007
        veg_x.append(veg_dist['Transect_'+str(trno+1)][i])
        sat_y.append(cross_distance['Transect_'+str(trno+1)][j])
    idx = np.isfinite(veg_x) & np.isfinite(sat_y)
    veg_x = np.array(veg_x)[idx]
    sat_y = np.array(sat_y)[idx]
    try:
        m, b = np.polyfit(veg_x, sat_y, 1)
        print(cross_distance['dates'][j]+' RMSE: '+str(mean_squared_error(veg_x, sat_y, squared=False)))
        print(cross_distance['dates'][j]+' R squared: '+str(r2_score(veg_x, sat_y)))
        ax.plot(veg_x,m*veg_x+b,color=c)
        axins.plot(veg_x,m*veg_x+b,color=c)
    except:
        continue
ax.plot(range(1000), range(1000), color=(0.3,0.3,0.3,0.5), linestyle='--', label='Expected trend')
axins.plot(range(1000), range(1000), color=(0.3,0.3,0.3,0.5), linestyle='--', label='Expected trend')
plt.xlabel('Validation edge distance (m)')
plt.ylabel('Satellite derived edge distance (m)')
ax.set_xlim((0,1000))
ax.set_ylim((0,1000))
axins.set_xlim(250,450)
axins.set_ylim(250,450)
plt.legend()
plt.savefig('/media/14TB_RAID_Array/User_Homes/Freya_Muir/PhD/Year2/Outputs/Figures/VegSat_StAndrews_Errors_Peninsula.png')
plt.show()

# Tentsmuir
fig = plt.figure(figsize=[10,10], tight_layout=True)
#plt.axis('equal')
plt.grid(linestyle=':', color='0.5')
plt.title('Tentsmuir')
for i,j,c in zip(range(7),[0,1,3,6,9,13,18],['#FCFFA1','#FBB314','#ED641F','#BA3251','#75176A','#2F0B5B','#07070A']):
    veg_x = list([])
    sat_y = list([])
    for trno in range(1291,1712):
        datelabels = 'Survey date: '+veg_dist['dates'][i]+'; Sat image date: '+cross_distance['dates'][j]
        plt.plot(veg_dist['Transect_'+str(trno+1)][i],cross_distance['Transect_'+str(trno+1)][j],color=c, marker='^', alpha=.5, label=datelabels if trno == 1291 else "") #2007
        veg_x.append(veg_dist['Transect_'+str(trno+1)][i])
        sat_y.append(cross_distance['Transect_'+str(trno+1)][j])
    idx = np.isfinite(veg_x) & np.isfinite(sat_y)
    veg_x = np.array(veg_x)[idx]
    sat_y = np.array(sat_y)[idx]
    try:
        m, b = np.polyfit(veg_x, sat_y, 1)
        print(cross_distance['dates'][j]+' RMSE: '+str(mean_squared_error(veg_x, sat_y, squared=False)))
        print(cross_distance['dates'][j]+' R squared: '+str(r2_score(veg_x, sat_y)))
        plt.plot(veg_x,m*veg_x+b, color=c)
    except:
        continue
plt.plot(range(1000), range(1000), color=(0.3,0.3,0.3,0.5), linestyle='--', label='Expected trend')
plt.xlabel('Validation edge distance (m)')
plt.ylabel('Satellite derived edge distance (m)')
plt.xlim((200,500))
plt.ylim((200,500))
plt.legend()
plt.savefig('/media/14TB_RAID_Array/User_Homes/Freya_Muir/PhD/Year2/Outputs/Figures/VegSat_StAndrews_Errors_Tentsmuir.png')
plt.show()
 
#%% Plotting - Otsu threshold amounts
from matplotlib import rcParams
rcParams['font.sans-serif'] = 'Arial'



sitename = 'StAndrewsWest'
with open(os.path.join('/media/14TB_RAID_Array/User_Homes/Freya_Muir/PhD/Year2/ModelsFrameworks/CoastWatch-main/Data/',sitename ,sitename+ '_output_proj.pkl'), 'rb') as f:
    output_proj_West = pickle.load(f)

sitename = 'StAndrewsEast'
with open(os.path.join('/media/14TB_RAID_Array/User_Homes/Freya_Muir/PhD/Year2/ModelsFrameworks/CoastWatch-main/Data/',sitename ,sitename+ '_output_proj.pkl'), 'rb') as f:
    output_proj_East = pickle.load(f)

output_proj_East['dates_dt'] = [datetime.strptime(date, '%Y-%m-%d') for date in output_proj_East['dates']]

output_proj_West['dates_dt'] = [datetime.strptime(date, '%Y-%m-%d') for date in output_proj_West['dates']]

colors = ['#21A790','#1D37FB'] #West = Teal, East = Blue
fig = plt.figure(figsize=[16,6], tight_layout=True)

plt.plot(output_proj_West['dates_dt'],output_proj_West['Otsu_threshold'], 'o', color=colors[0],  label='West/Inner estuarine')

plt.plot(output_proj_East['dates_dt'],output_proj_East['Otsu_threshold'], 'o', color=colors[1], label='East/Open coast')

plt.xlabel('Date (yyyy-mm-dd)')
plt.ylabel('Otsu threshold value (1)')
plt.legend(loc='upper left')
plt.gca().xaxis.set_major_locator(matplotlib.dates.YearLocator())
plt.xticks(rotation=270)
plt.savefig('/media/14TB_RAID_Array/User_Homes/Freya_Muir/PhD/Year2/Outputs/Figures/VegSat_StAndrews_OtsuThresholds.png')
plt.show()

# combine these different collections into a list
East_West_Otsu = [output_proj_West['Otsu_threshold'], output_proj_East['Otsu_threshold']]

fig, ax = plt.subplots(figsize=[8,8], tight_layout=True)
violin = ax.violinplot(East_West_Otsu)

for patch, color in zip(violin['bodies'], colors):
    patch.set_color(color)
    for partname in list(violin.keys())[1:]:
        vp = violin[partname]
        vp.set_edgecolor(colors)
        #vp.set_linewidth(1)

plt.xticks([1,2], ['West/Inner estuarine','East/Open coast'])    
plt.ylabel('NDVI threshold')
plt.savefig('/media/14TB_RAID_Array/User_Homes/Freya_Muir/PhD/Year2/Outputs/Figures/VegSat_StAndrews_OtsuThresholdsViolin.png')
plt.show()

#%% Validation MSE

      


#%%Plotting - Vegetation Edge

#Displays produced lines/transects

fig = plt.figure(figsize=[15,8], tight_layout=True)
plt.axis('equal')
plt.xlabel('Eastings')
plt.ylabel('Northings')
#plt.xlim(509000,513000)
#plt.ylim(6244400,6247250)
plt.grid(linestyle=':', color='0.5')
for i in range(len(output_proj['shorelines'])):
    sl = output_proj['shorelines'][i]
    date = output_proj['dates'][i]
    plt.plot(sl[:,0], sl[:,1], '.')#, label=date.strptime('%d-%m-%Y'))
 
for i,key in enumerate(list(transect_proj.keys())):
    plt.plot(transect_proj[key][0,0],transect_proj[key][0,1], 'bo', ms=5)
    plt.plot(transect_proj[key][:,0],transect_proj[key][:,1],'k-',lw=1)
    #plt.text(transects_proj[key][0,0]-100, transects_proj[key][0,1]+100, key, va='center', ha='right', bbox=dict(boxstyle="square", ec='k',fc='w'))
plt.show()

#%% Mapping of Results

"""
Creates map object centred at ROI + adds compiled satellite image as base-layer
"""
#Map = geemap.Map(center=[polygon[0][0][1],polygon[0][0][0]],zoom=12)
#Map.add_basemap('HYBRID')

#Generates colours for lines to be drawn in. Check out https://seaborn.pydata.org/tutorial/color_palettes.html for colour options...
palette = sns.color_palette("bright", len(output['shorelines']))
palette = palette.as_hex()

#Choose 'points' or 'lines' for the layer geometry
geomtype = 'points'

for i in range(len(output['shorelines'])):
    shore = dict([])
    if len(output_latlon['shorelines'][i])==0:
        continue
    shore = {'dates':[output_latlon['dates'][i]], 'shorelines':[output_latlon['shorelines'][i]], 'filename':[output_latlon['filename'][i]], 'cloud_cover':[output_latlon['cloud_cover'][i]], 'idx':[output_latlon['idx'][i]], 'Otsu_threshold':[output_latlon['Otsu_threshold'][i]], 'satname':[output_latlon['satname'][i]]}
    gdf = Toolbox.output_to_gdf(shore, geomtype)
    Line = geemap.geopandas_to_ee(gdf, geodesic=True)
    Map.addLayer(Line,{'color': str(palette[i])},'coast'+str(i))

Map

# In[ ]:


#Displays the transects

for i,key in enumerate(list(transect_proj.keys())):
    plt.plot(transect_proj[key][0,0],transect_proj[key][0,1], 'bo', ms=5)
    plt.plot(transect_proj[key][:,0],transect_proj[key][:,1],'k-',lw=1)
    #plt.text(transects_proj[key][0,0]-100, transects_proj[key][0,1]+100, key, va='center', ha='right', bbox=dict(boxstyle="square", ec='k',fc='w'))
plt.show()


# In[ ]:


#Displays the lines

fig = plt.figure(figsize=[15,8])
plt.axis('equal')
plt.xlabel('Eastings')
plt.ylabel('Northings')
plt.grid(linestyle=':', color='0.5')
for i in range(len(output_proj['shorelines'])):
    sl = output_proj['shorelines'][i]
    date = output_proj['dates'][i]
    plt.plot(sl[:,0], sl[:,1], '.')#, label=date.strftime('%d-%m-%Y'))
plt.legend()
plt.show()


# In[ ]:


#Cross-distance plots for ALL transects (do not bother if you are considering a LOT of transects)

fig = plt.figure(figsize=[15,12], tight_layout=True)
gs = gridspec.GridSpec(len(cross_distance),2, wspace=0.035, width_ratios=[3,1])
gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.2)
for i,key in enumerate(cross_distance.keys()):
    if np.all(np.isnan(cross_distance[key])):
        continue
    ax = fig.add_subplot(gs[i,0])
    ax.grid(linestyle=':', color='0.5')
    ax.set_ylim([-100,110])
    ax.plot(output['dates'], cross_distance[key]- np.nanmedian(cross_distance[key]), '-o', ms=6, mfc='w')
    #ax.set_ylabel('distance [m]', fontsize=12)
    ax.text(0.5,0.95, key, bbox=dict(boxstyle="square", ec='k',fc='w'), ha='center',va='top', transform=ax.transAxes, fontsize=14)
    if i!= len(cross_distance.keys())-1:
        ax.set_xticklabels('')
    ax = fig.add_subplot(gs[i,1])
    #ax.set_xlim([-50,50])
    ax.set_xlim([0,0.015])
    sns.distplot(cross_distance[key]- np.nanmedian(cross_distance[key]), bins=10, color="b", ax=ax, vertical=True)
    ax.set_yticklabels('')
    if i!= len(cross_distance.keys())-1:
        ax.set_xticklabels('')
fig.text(0.01, 0.5, 'Cross-Shore Distance / m', va='center', rotation='vertical', fontsize=12)


# In[ ]:


transect_range = [[0, 50],[51,110],[111,180],[181,240],[241,len(output['dates'])-1]]
#transect_colour = sns.color_palette("bright", len(transect_range))
colours = ['#ff0000','#0084ff','#ff00f7','#00fa0c', '#ffb300', '#00ffcc','#7b00ff']
transect_colour = colours


# In[ ]:


#In this cell, you can iterate on transect range (we will use these ranges to analyse specific regions of the edge)

fig = plt.figure(figsize=[15,8], tight_layout=True)
plt.axis('equal')
plt.xlabel('Eastings')
plt.ylabel('Northings')
#plt.xlim(509000,513000)
#plt.ylim(6244400,6247250)
plt.grid(linestyle=':', color='0.5')
for i in range(len(output_proj['shorelines'])):
    sl = output_proj['shorelines'][i]
    date = output_proj['dates'][i]
    plt.plot(sl[:,0], sl[:,1], '.')#, label=date.strptime('%d-%m-%Y'))

if transect_range == 'full':
    transect_range = [[0,len(transect_proj.keys())]]   

for i,key in enumerate(list(transect_proj.keys())):
    for j in range(len(transect_range)):
        if transect_range[j][0] <= i <= transect_range[j][1]:
            plt.plot(transect_proj[key][0,0],transect_proj[key][0,1], 'bo', ms=5,color=transect_colour[j])
            plt.plot(transect_proj[key][:,0],transect_proj[key][:,1],'k-',lw=1,color=transect_colour[j])
    #plt.text(transects_proj[key][0,0]-100, transects_proj[key][0,1]+100, key, va='center', ha='right', bbox=dict(boxstyle="square", ec='k',fc='w'))

plt.savefig('Data/' + sitename + '/jpg_files/transectsFull', bbox_inches='tight')
  
plt.show()


# In[ ]:


#Year by Year

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

fig, axs = plt.subplots(len(transect_range),sharex=True,figsize=(10, 12))
fig.text(0.005, 0.5, "Average Yearly Vegetation Cross-Edge Distance / m", va='center', rotation='vertical', fontsize=12)

for i in range(len(transect_range)):
    axs[i].set_title("Transects:"+str(transect_range[i][0])+"-"+str(transect_range[i][1]),backgroundcolor=transect_colour[i],color='white')
    if i != len(transect_range)-1:
        axs[i].xaxis.set_visible(False)
    if i == len(transect_range)-1:
        axs[i].set_xlabel("Year", fontsize=12)
    for j in range(transect_range[i][0],transect_range[i][1]):
        KEY = 'Transect_'+str(j+1)
        try:
            a, b, c, d, e = Toolbox.Separate_TimeSeries_year(cross_distance, output_proj, KEY)
            NaN_mask = np.isfinite(e)
            axs[i].plot(np.array(d)[NaN_mask],np.array(e)[NaN_mask])
        except:
            continue
            
plt.savefig('Data/' + sitename + '/jpg_files/avgYearlyVegPosition', bbox_inches='tight')


# In[ ]:


#Good at looking at seasonal patterns. Takes a while.

#plt.figure(figsize=[15,12])

months = ["Jan", "Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sept", "Oct", "Nov", "Dec"]
Month_dict = {"Jan":[], "Feb":[], "Mar":[], "Apr":[], "May":[], "June":[], "July":[], "Aug":[], "Sept":[], "Oct":[], "Nov":[], "Dec":[]}

Total_Month_Arr = []
test1 = []
test2 = []

fig, axs = plt.subplots(len(transect_range),sharex=True,figsize=(10, 12))

for l in range(len(transect_range)):

    for i in range(transect_range[l][0],transect_range[l][1]):
        KEY = 'Transect_'+str(i+1)
        try:
            a, b, c, d, e = Toolbox.Separate_TimeSeries_month(cross_distance, output_proj,KEY)

            zipped_lists = zip(d,e)
            s = sorted(zipped_lists)
            tuples = zip(s)

            new_d = []
            new_e = []

            sortedList = [list(tuple) for tuple in  tuples]

            for v in range(len(sortedList)):
                new_d.append(sortedList[v][0][0])
                new_e.append(sortedList[v][0][1])

            month_arr = []
            for j in range(len(d)):
                a = datetime.strptime(str(new_d[j]),'%m')
                month_arr.append(a.strftime('%b'))

            axs[l].scatter(month_arr,new_e,label=KEY)
            test1.append(new_d)
            test2.append(new_e)
        except:
            continue

    avg = []
    st_err = []
    Total_organised = []
    temp = []

    for k in range(len(test2[0])):
        for h in range(len(test2)):
            temp.append(test2[h][k])
        Total_organised.append(temp)
        avg.append(np.nanmean(temp))
        st_err.append(np.nanstd(temp)/(len(temp)**0.5))
        temp = []
    
    Total_Month_Arr.append(Total_organised)
    
    #plt.errorbar(month_arr,avg, yerr=st_err, color='k')
    axs[l].scatter(month_arr,avg, color='k', s=50, marker='x')

    #plt.legend()
    axs[l].set_title("Transects:"+str(transect_range[l][0])+"-"+str(transect_range[l][1]),backgroundcolor=transect_colour[l],color='white')

fig.text(0.01,0.5,"Averaged Monthly Vegetation Cross-Edge Distance / m", va='center', rotation='vertical')
plt.xlabel("Month")

plt.savefig('Data/' + sitename + '/jpg_files/monthScatter', bbox_inches='tight')


# In[ ]:


fig, axs = plt.subplots(len(transect_range),sharex=True,figsize=(10, 12))

for j in range(len(Total_Month_Arr)):
    
    axs[j].set_title("Transects:"+str(transect_range[j][0])+"-"+str(transect_range[j][1]),backgroundcolor=transect_colour[j],color='white')
    axs[j].boxplot(Total_Month_Arr[j],notch=True, flierprops = dict(marker='o', markersize=8, linestyle='none', markeredgecolor='r'))

fig.text(0.01,0.5,"Averaged Monthly Vegetation Cross-Edge Distance / m", va='center', rotation='vertical')
plt.xticks(new_d, month_arr)

plt.savefig('Data/' + sitename + '/jpg_files/monthBox', bbox_inches='tight')

plt.show()


# In[ ]:


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    #ax.set_xlabel('Sample name')

fig, axs = plt.subplots(len(transect_range),sharex=True,figsize=(10, 12))

for j in range(len(Total_Month_Arr)):
    
    axs[j].set_title("Transects:"+str(transect_range[j][0])+"-"+str(transect_range[j][1]),backgroundcolor=transect_colour[j],color='white')
    parts = axs[j].violinplot(Total_Month_Arr[j], showmeans=False, showmedians=False, showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor(transect_colour[j])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)

    quartile1, medians, quartile3 = np.percentile(Total_Month_Arr[j], [25, 50, 75], axis=1)
    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(Total_Month_Arr[j], quartile1, quartile3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    axs[j].scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    axs[j].vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    axs[j].vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
    
    if j == len(Total_Month_Arr):
        set_axis_style(axs[j], month_arr)

fig.text(0.005,0.5,"Averaged Monthly Vegetation Cross-Edge Distance / m", va='center', rotation='vertical')
plt.xticks(new_d, month_arr)

plt.savefig('Data/' + sitename + '/jpg_files/monthViolin', bbox_inches='tight')

plt.show()


# In[ ]:


#array of colours for each of the averaged transect-analysis (add more if need be)
colours = ['#ff0000','#0084ff','#ff00f7','#00fa0c', '#ffb300', '#00ffcc','#7b00ff']

Rows = []

with open('Data/'+sitename+'/vegetation_transect_time_series.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        Rows.append(row[2:])

cross_distance_condensed, standard_err_condensed, transect_condensed, Dates = Transects.transect_compiler(Rows, transect_proj, 100, output)


# In[ ]:


fig = plt.figure(figsize=[15,12], tight_layout=True)
gs = gridspec.GridSpec(len(cross_distance_condensed),2, wspace=0.035, width_ratios=[4,1])
gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.05)

x = np.arange(datetime(1984,1,1), datetime(2022,1,1), timedelta(days=100)).astype(str)
y = [0]*139

for i,key in enumerate(cross_distance_condensed.keys()):
    
    if np.all(np.isnan(cross_distance_condensed[key])):
        continue
        
    ax = fig.add_subplot(gs[i,0])
    ax.grid(linestyle=':', color='0.5')
    ax.set_ylim([min(cross_distance_condensed[key]- np.nanmedian(cross_distance_condensed[key]))-5,max(cross_distance_condensed[key]- np.nanmedian(cross_distance_condensed[key]))+5])
    dates = matplotlib.dates.date2num(Dates[key])
    ax.errorbar(dates, cross_distance_condensed[key]- np.nanmedian(cross_distance_condensed[key]), yerr = standard_err_condensed[key],fmt='-o',ecolor= 'k', color= colours[i], ms=6, mfc='w')

    ax.fill_between(dates, 0, cross_distance_condensed[key]- np.nanmedian(cross_distance_condensed[key]),alpha=0.5,color=colours[i])
    ax.set_title("Transects:"+str(transect_range[i][0])+"-"+str(transect_range[i][1]),backgroundcolor=transect_colour[i],color='white')

    ax.set_xticklabels(['1982','1986','1992','1998','2004','2010','2016','2020','2014','2018','2022'])

    if i!= len(cross_distance_condensed.keys())-1:
        ax.set_xticklabels('')

    ax = fig.add_subplot(gs[i,1])
    ax.set_xlim([0,0.020])
    sns.distplot(cross_distance_condensed[key]- np.nanmedian(cross_distance_condensed[key]), bins=10, color=colours[i], ax=ax, vertical=True)
    ax.set_yticklabels('')
    
    if i!= len(cross_distance_condensed.keys())-1:
        ax.set_xticklabels('')
        ax.set_xlabel('')
        
fig.text(0.01, 0.5, 'Cross Vegetation-Edge Distance / m', va='center', rotation='vertical', fontsize=13.8)

plt.savefig('Data/' + sitename + '/jpg_files/crossEdgeDistances', bbox_inches='tight')


# In[ ]:


ref_sl_conv = Toolbox.convert_epsg(settings['reference_shoreline'], 32630, 27700)[:,:-1]

vv = dict([])
vv['1'] = [ref_sl_conv]

#Displays produced lines/transects

fig = plt.figure()#figsize=[15,8], tight_layout=True)
plt.axis('equal')
#plt.xlabel('Eastings')
#plt.ylabel('Northings')
plt.xlim(min(vv['1'][0][:,0]),max(vv['1'][0][:,0]))
plt.xticks('')
plt.yticks('')
plt.ylim(min(vv['1'][0][:,1])-50,max(vv['1'][0][:,1])+50)
plt.grid(linestyle=':', color='0.5')
for i in range(len(vv['1'])):
    sl = vv['1'][i]
    date = vv['1'][i]
    plt.plot(sl[:,0], sl[:,1], '.', color='k')#, label=date.strptime('%d-%m-%Y'))
 
for i,key in enumerate(list(transect_condensed.keys())):
    plt.plot(transect_condensed[key][0,0],transect_condensed[key][0,1], 'bo', color= colours[i], ms=5)
    plt.plot(transect_condensed[key][:,0],transect_condensed[key][:,1],'k-', color= colours[i], lw=1)
    plt.text(transect_condensed[key][1][0],transect_condensed[key][1][1], key, va='bottom', ha='right', bbox=dict(boxstyle="round", ec='k',fc='w'), fontsize=10)

plt.savefig('Data/' + sitename + '/jpg_files/refEdge_Transects', bbox_inches='tight')
plt.show()


# In[ ]:


Big_percent = []
for i,key in enumerate(cross_distance_condensed.keys()):
    cross = cross_distance_condensed[key]- np.nanmedian(cross_distance_condensed[key])
    percent_diff = []
    for j in range(len(cross)):
        percent_diff.append(100*(cross[j]-cross[0])/cross[0])
        
    Big_percent.append(percent_diff)


# In[ ]:


Big_arr = []
Big_datearr = []

Year = [[]]*(2021-1984)

for i in range(len(transect_range)):
    percent_diff = []
    dist_arr = []
    date_arr = []
    for j in range(transect_range[i][0],transect_range[i][1]):
        KEY = 'Transect_'+str(j+1)
        try:
            a, b, c, d, e = Toolbox.Separate_TimeSeries_year(cross_distance, output_proj, KEY)
            NaN_mask = np.isfinite(e)
            dist_arr.append(list(np.array(e)[NaN_mask]))
            date_arr.append(list(np.array(d)[NaN_mask]))
            #percent_diff.append()
        except:
            continue
    Big_arr.append(dist_arr)
    Big_datearr.append(date_arr)


# In[ ]:


Big_Percent = []

for j in range(len(Big_arr)):
    Medium_Percent_TransectRange = []
    Year = dict([])
    for i in range(len(Big_arr[j])):
        for k in range(len(Big_arr[j][i])):
            index = Big_datearr[j][i][k]-1984
            if Year.get(str(index)) == None:
                Year[str(index)] = []
            Year[str(index)].append(Big_arr[j][i][index-1])
            #print(len(Year[index-1]))
    List_year = []
    for v, key in enumerate(Year):
        List_year.append(np.mean(Year[key]))
    Big_Percent.append(List_year[1:])
    #print(List_year)


# In[ ]:


Barz = []

for i in range(len(Big_Percent)):
    temp = []
    for j in range(len(Big_Percent[i])):
        temp.append(100*(Big_Percent[i][j]-Big_Percent[i][0])/Big_Percent[i][0])
    Barz.append(temp)


# In[ ]:


fig, axs = plt.subplots(figsize=(10, 12))
for i in range(len(Barz)):
    axs.barh(np.arange(len(Barz[i]))+(i/5), Barz[i], align='center',height= 0.2,color=colours[i],label='Transects: '+str(transect_range[i][0])+"-"+str(transect_range[i][1]) )
axs.plot([0]*100,np.arange(0,37,0.37),'-.',color='k')
axs.set_xlabel("% Change Since 1984")
#axs.set_xlim(-500,500)
axs.set_yticks(np.arange(0,37,1))
axs.set_yticklabels(list(np.array(d)[NaN_mask]))
fig.text(0.25,0.85,"Accretion (Relative to 1984)")
fig.text(0.58,0.85,"Erosion (Relative to 1984)")
axs.legend(loc='lower right')
for i in range(37):
    axs.plot(np.arange(-500,500,10),[i-0.1]*100,'-.',color='k',alpha=0.7,linewidth=0.45)
fig.savefig(os.path.join('Data/' + sitename + '/jpg_files/barBreakdown.jpg'), dpi=150)
plt.show()


# ## Analysis - Comparison with Field Data

# In[ ]:


# extract subset of veg lines that match the dates in the veg survey 

for surveydate in unique(vegsurvey.Date):
    matchdate = min(output['dates'], key=lambda x: abs(x - surveydate))
    

