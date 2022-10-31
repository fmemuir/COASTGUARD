"""
This module contains utilities to work with satellite images
    
Author: Kilian Vos, Water Research Laboratory, University of New South Wales
"""

# load modules
import os
import numpy as np
import matplotlib.pyplot as plt
import pdb

# other modules
from osgeo import gdal, osr
import pandas as pd
import geopandas as gpd
from shapely import geometry
from shapely.geometry import Point, Polygon, LineString, MultiLineString, MultiPoint
import skimage.transform as transform
from astropy.convolution import convolve
from datetime import datetime
from IPython.display import clear_output
import ee
import pickle
import math
import requests
from requests.auth import HTTPBasicAuth
import glob
import rasterio
from rasterio.plot import show
import time

np.seterr(all='ignore') # raise/ignore divisions by 0 and nans

###################################################################################################
# COORDINATES CONVERSION FUNCTIONS
###################################################################################################

def convert_pix2world(points, georef):
    """
    Converts pixel coordinates (pixel row and column) to world projected 
    coordinates performing an affine transformation.
    
    KV WRL 2018

    Arguments:
    -----------
    points: np.array or list of np.array
        array with 2 columns (row first and column second)
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
                
    Returns:    
    -----------
    points_converted: np.array or list of np.array 
        converted coordinates, first columns with X and second column with Y
        
    """
    
    
    
    # make affine transformation matrix
    aff_mat = np.array([[georef[1], georef[2], georef[0]],
                       [georef[4], georef[5], georef[3]],
                       [0, 0, 1]])

    # create affine transformation
    tform = transform.AffineTransform(aff_mat)

    # if list of arrays
    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points): 
            tmp = arr[:,[1,0]]
            points_converted.append(tform(tmp))
          
    # if single array
    elif type(points) is np.ndarray:
        tmp = points[:,[1,0]]
        points_converted = tform(tmp)
        
    else:
        raise Exception('invalid input type')
        
    return points_converted

def convert_world2pix(points, georef):
    """
    Converts world projected coordinates (X,Y) to image coordinates 
    (pixel row and column) performing an affine transformation.
    
    KV WRL 2018

    Arguments:
    -----------
    points: np.array or list of np.array
        array with 2 columns (X,Y)
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
        Xtr = xoff
        Ytr = yoff
                
    Returns:    
    -----------
    points_converted: np.array or list of np.array 
        converted coordinates (pixel row and column)
    
    """
    # make affine transformation matrix
    aff_mat = np.array([[georef[1], georef[2], georef[0]],[georef[4], georef[5], georef[3]],[0, 0, 1]])
    # create affine transformation
    tform = transform.AffineTransform(aff_mat)
    
    # if list of arrays
    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points): 
            points_converted.append(tform.inverse(points))
            
    # if single array    
    elif type(points) is np.ndarray:
        points_converted = tform.inverse(points)
        
    else:
        print('invalid input type')
        raise
    return points_converted


def convert_epsg(points, epsg_in, epsg_out):
    """
    Converts from one spatial reference to another using the epsg codes
    
    KV WRL 2018

    Arguments:
    -----------
    points: np.array or list of np.ndarray
        array with 2 columns (rows first and columns second)
    epsg_in: int
        epsg code of the spatial reference in which the input is
    epsg_out: int
        epsg code of the spatial reference in which the output will be            
                
    Returns:    
    -----------
    points_converted: np.array or list of np.array 
        converted coordinates from epsg_in to epsg_out
        
    """

    # define input and output spatial references
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(epsg_in)
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(epsg_out)
    # create a coordinates transform
    coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    # if list of arrays
    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points): 
            points_converted.append(np.array(coordTransform.TransformPoints(arr)))
    # if single array
    elif type(points) is np.ndarray:
        points_converted = np.array(coordTransform.TransformPoints(points))  
    else:
        raise Exception('invalid input type')

    return points_converted

###################################################################################################
# IMAGE ANALYSIS FUNCTIONS
###################################################################################################
    
def nd_index(im1, im2, cloud_mask):
    """
    Computes normalised difference index on 2 images (2D), given a cloud mask (2D).

    KV WRL 2018

    Arguments:
    -----------
    im1: np.array
        first image (2D) with which to calculate the ND index
    im2: np.array
        second image (2D) with which to calculate the ND index
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are

    Returns:    
    -----------
    im_nd: np.array
        Image (2D) containing the ND index
        
    """

    # reshape the cloud mask
    vec_mask = cloud_mask.reshape(im1.shape[0] * im1.shape[1])
    # initialise with NaNs
    vec_nd = np.ones(len(vec_mask)) * np.nan
    # reshape the two images
    vec1 = im1.reshape(im1.shape[0] * im1.shape[1])
    vec2 = im2.reshape(im2.shape[0] * im2.shape[1])
    # compute the normalised difference index
    temp = np.divide(vec1[~vec_mask] - vec2[~vec_mask],
                     vec1[~vec_mask] + vec2[~vec_mask])
    vec_nd[~vec_mask] = temp
    # reshape into image
    im_nd = vec_nd.reshape(im1.shape[0], im1.shape[1])
    return im_nd
    
def image_std(image, radius):
    """
    Calculates the standard deviation of an image, using a moving window of 
    specified radius. Uses astropy's convolution library'
    
    Arguments:
    -----------
    image: np.array
        2D array containing the pixel intensities of a single-band image
    radius: int
        radius defining the moving window used to calculate the standard deviation. 
        For example, radius = 1 will produce a 3x3 moving window.
        
    Returns:    
    -----------
    win_std: np.array
        2D array containing the standard deviation of the image
        
    """  
    
    # convert to float
    image = image.astype(float)
    # first pad the image
    image_padded = np.pad(image, radius, 'reflect')
    # window size
    win_rows, win_cols = radius*2 + 1, radius*2 + 1
    # calculate std with uniform filters
    win_mean = convolve(image_padded, np.ones((win_rows,win_cols)), boundary='extend',
                        normalize_kernel=True, nan_treatment='interpolate', preserve_nan=True)
    win_sqr_mean = convolve(image_padded**2, np.ones((win_rows,win_cols)), boundary='extend',
                        normalize_kernel=True, nan_treatment='interpolate', preserve_nan=True)
    win_var = win_sqr_mean - win_mean**2
    win_std = np.sqrt(win_var)
    # remove padding
    win_std = win_std[radius:-radius, radius:-radius]

    return win_std

def mask_raster(fn, mask):
    """
    Masks a .tif raster using GDAL.
    
    Arguments:
    -----------
    fn: str
        filepath + filename of the .tif raster
    mask: np.array
        array of boolean where True indicates the pixels that are to be masked
        
    Returns:    
    -----------
    Overwrites the .tif file directly
        
    """ 
    
    # open raster
    raster = gdal.Open(fn, gdal.GA_Update)
    # mask raster
    for i in range(raster.RasterCount):
        out_band = raster.GetRasterBand(i+1)
        out_data = out_band.ReadAsArray()
        out_band.SetNoDataValue(0)
        no_data_value = out_band.GetNoDataValue()
        out_data[mask] = no_data_value
        out_band.WriteArray(out_data)
    # close dataset and flush cache
    raster = None


###################################################################################################
# UTILITIES
###################################################################################################
    
def get_filepath(inputs,satname):
    """
    Create filepath to the different folders containing the satellite images.
    
    KV WRL 2018

    Arguments:
    -----------
    inputs: dict with the following keys
        'sitename': str
            name of the site
        'polygon': list
            polygon containing the lon/lat coordinates to be extracted,
            longitudes in the first column and latitudes in the second column,
            there are 5 pairs of lat/lon with the fifth point equal to the first point:
            ```
            polygon = [[[151.3, -33.7],[151.4, -33.7],[151.4, -33.8],[151.3, -33.8],
            [151.3, -33.7]]]
            ```
        'dates': list of str
            list that contains 2 strings with the initial and final dates in 
            format 'yyyy-mm-dd':
            ```
            dates = ['1987-01-01', '2018-01-01']
            ```
        'sat_list': list of str
            list that contains the names of the satellite missions to include: 
            ```
            sat_list = ['L5', 'L7', 'L8', 'S2']
            ```
        'filepath_data': str
            filepath to the directory where the images are downloaded
    satname: str
        short name of the satellite mission ('L5','L7','L8','S2')
                
    Returns:    
    -----------
    filepath: str or list of str
        contains the filepath(s) to the folder(s) containing the satellite images
    
    """     
    
    sitename = inputs['sitename']
    filepath_data = inputs['filepath']
    # access the images
    if satname == 'L5':
        # access downloaded Landsat 5 images
        filepath = os.path.join(filepath_data, sitename, satname, '30m')
    elif satname == 'L7':
        # access downloaded Landsat 7 images
        filepath_pan = os.path.join(filepath_data, sitename, 'L7', 'pan')
        filepath_ms = os.path.join(filepath_data, sitename, 'L7', 'ms')
        filepath = [filepath_pan, filepath_ms]
    elif satname == 'L8':
        # access downloaded Landsat 8 images
        filepath_pan = os.path.join(filepath_data, sitename, 'L8', 'pan')
        filepath_ms = os.path.join(filepath_data, sitename, 'L8', 'ms')
        filepath = [filepath_pan, filepath_ms]
    elif satname == 'S2':
        # access downloaded Sentinel 2 images
        filepath10 = os.path.join(filepath_data, sitename, satname, '10m')
        filepath20 = os.path.join(filepath_data, sitename, satname, '20m')
        filepath60 = os.path.join(filepath_data, sitename, satname, '60m')
        filepath = [filepath10, filepath20, filepath60]
            
    return filepath
    
def get_filenames(filename, filepath, satname):
    """
    Creates filepath + filename for all the bands belonging to the same image.
    
    KV WRL 2018

    Arguments:
    -----------
    filename: str
        name of the downloaded satellite image as found in the metadata
    filepath: str or list of str
        contains the filepath(s) to the folder(s) containing the satellite images
    satname: str
        short name of the satellite mission       
        
    Returns:    
    -----------
    fn: str or list of str
        contains the filepath + filenames to access the satellite image
        
    """     
    
    if satname == 'L5':
        fn = os.path.join(filepath, filename)
    if satname == 'L7' or satname == 'L8':
        filename_ms = filename.replace('pan','ms')
        fn = [os.path.join(filepath[0], filename),
              os.path.join(filepath[1], filename_ms)]
    if satname == 'S2':
        filename20 = filename.replace('10m','20m')
        filename60 = filename.replace('10m','60m')
        fn = [os.path.join(filepath[0], filename),
              os.path.join(filepath[1], filename20),
              os.path.join(filepath[2], filename60)]
        
    return fn

def merge_output(output):
    """
    Function to merge the output dictionnary, which has one key per satellite mission
    into a dictionnary containing all the shorelines and dates ordered chronologically.
    
    Arguments:
    -----------
    output: dict
        contains the extracted shorelines and corresponding dates, organised by 
        satellite mission
    
    Returns:    
    -----------
    output_all: dict
        contains the extracted shorelines in a single list sorted by date
    
    """     
    
    # initialize output dict
    output_all = dict([])
    satnames = list(output.keys())
    for key in output[satnames[0]].keys():
        output_all[key] = []
    # create extra key for the satellite name
    output_all['satname'] = []
    # fill the output dict
    for satname in list(output.keys()):
        for key in output[satnames[0]].keys():
            output_all[key] = output_all[key] + output[satname][key]
        output_all['satname'] = output_all['satname'] + [_ for _ in np.tile(satname,
                  len(output[satname]['dates']))]
    # sort chronologically
    idx_sorted = sorted(range(len(output_all['dates'])), key=output_all['dates'].__getitem__)
    for key in output_all.keys():
        output_all[key] = [output_all[key][i] for i in idx_sorted]

    return output_all


def remove_duplicates(output):
    """
    Function to remove from the output dictionnary entries containing shorelines for 
    the same date and satellite mission. This happens when there is an overlap between 
    adjacent satellite images.

    Arguments:
    -----------
        output: dict
            contains output dict with shoreline and metadata

    Returns:
    -----------
        output_no_duplicates: dict
            contains the updated dict where duplicates have been removed

    """

    # nested function
    def duplicates_dict(lst):
        "return duplicates and indices"
        def duplicates(lst, item):
                return [i for i, x in enumerate(lst) if x == item]
        return dict((x, duplicates(lst, x)) for x in set(lst) if lst.count(x) > 1)

    dates = output['dates']
    
    # make a list with year/month/day
    dates_str = [datetime.strptime(_,'%Y-%m-%d').strftime('%Y-%m-%d') for _ in dates]
    # create a dictionnary with the duplicates
    dupl = duplicates_dict(dates_str)
    # if there are duplicates, only keep the first element
    if dupl:
        output_no_duplicates = dict([])
        idx_remove = []
        for k,v in dupl.items():
            idx_remove.append(v[0])
        idx_remove = sorted(idx_remove)
        idx_all = np.linspace(0, len(dates_str)-1, len(dates_str))
        idx_keep = list(np.where(~np.isin(idx_all,idx_remove))[0])
        for key in output.keys():
            output_no_duplicates[key] = [output[key][i] for i in idx_keep]
        print('%d duplicates' % len(idx_remove))
        return output_no_duplicates
    else:
        print('0 duplicates')
        return output

def get_closest_datapoint(dates, dates_ts, values_ts):
    """
    Extremely efficient script to get closest data point to a set of dates from a very
    long time-series (e.g., 15-minutes tide data, or hourly wave data)
    
    Make sure that dates and dates_ts are in the same timezone (also aware or naive)
    
    KV WRL 2020

    Arguments:
    -----------
    dates: list of datetimes
        dates at which the closest point from the time-series should be extracted
    dates_ts: list of datetimes
        dates of the long time-series
    values_ts: np.array
        array with the values of the long time-series (tides, waves, etc...)
        
    Returns:    
    -----------
    values: np.array
        values corresponding to the input dates
        
    """
    
    # check if the time-series cover the dates
    if dates[0] < dates_ts[0] or dates[-1] > dates_ts[-1]: 
        raise Exception('Time-series do not cover the range of your input dates')
    
    # get closest point to each date (no interpolation)
    temp = []
    def find(item, lst):
        start = 0
        start = lst.index(item, start)
        return start
    for i,date in enumerate(dates):
        print('\rExtracting closest points: %d%%' % int((i+1)*100/len(dates)), end='')
        temp.append(values_ts[find(min(item for item in dates_ts if item > date), dates_ts)])
    values = np.array(temp)
    
    return values

###################################################################################################
# CONVERSIONS FROM DICT TO GEODATAFRAME AND READ/WRITE GEOJSON
###################################################################################################
    
def polygon_from_kml(fn):
    """
    Extracts coordinates from a .kml file.
    
    KV WRL 2018

    Arguments:
    -----------
    fn: str
        filepath + filename of the kml file to be read          
                
    Returns:    
    -----------
    polygon: list
        coordinates extracted from the .kml file
        
    """    
    
    # read .kml file
    with open(fn) as kmlFile:
        doc = kmlFile.read() 
    # parse to find coordinates field
    str1 = '<coordinates>'
    str2 = '</coordinates>'
    subdoc = doc[doc.find(str1)+len(str1):doc.find(str2)]
    coordlist = subdoc.split('\n')
    # read coordinates
    polygon = []
    for i in range(1,len(coordlist)-1):
        polygon.append([float(coordlist[i].split(',')[0]), float(coordlist[i].split(',')[1])])
        
    return [polygon]

def transects_from_geojson(filename):
    """
    Reads transect coordinates from a .geojson file.
    
    Arguments:
    -----------
    filename: str
        contains the path and filename of the geojson file to be loaded
        
    Returns:    
    -----------
    transects: dict
        contains the X and Y coordinates of each transect
        
    """  
    
    gdf = gpd.read_file(filename)
    transects = dict([])
    for i in gdf.index:
        transects[gdf.loc[i,'name']] = np.array(gdf.loc[i,'geometry'].coords)
        
    print('%d transects have been loaded' % len(transects.keys()))

    return transects

def output_to_gdf(output, geomtype):
    """
    Saves the mapped shorelines as a gpd.GeoDataFrame    
    
    KV WRL 2018

    Arguments:
    -----------
    output: dict
        contains the coordinates of the mapped shorelines + attributes
    geomtype: str
        'lines' for LineString and 'points' for Multipoint geometry      
                
    Returns:    
    -----------
    gdf_all: gpd.GeoDataFrame
        contains the shorelines + attirbutes
  
    """    
     
    # loop through the mapped shorelines
    counter = 0
    for i in range(len(output['shorelines'])):
        # skip if there shoreline is empty 
        if len(output['shorelines'][i]) == 0:
            continue
        else:
            # save the geometry depending on the linestyle
            if geomtype == 'lines':
                for j in range (len(output['shorelines'][i])):
                    abbba = []
                    abbba.append(output['shorelines'][i][j][1])
                    abbba.append(output['shorelines'][i][j][0])
                    output['shorelines'][i][j] = abbba
                geom = geometry.LineString(output['shorelines'][i])
            elif geomtype == 'points':
                coords = output['shorelines'][i]
                geom = geometry.MultiPoint([(coords[_,1], coords[_,0]) for _ in range(coords.shape[0])])
            else:
                raise Exception('geomtype %s is not an option, choose between lines or points'%geomtype)
            # save into geodataframe with attributes
            gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geom))
            gdf.index = [i]
            gdf.loc[i,'date'] = datetime.strftime(datetime.strptime(output['dates'][0],'%Y-%m-%d'),'%Y-%m-%d %H:%M:%S')
            gdf.loc[i,'satname'] = output['satname'][i]
            gdf.loc[i,'cloud_cover'] = output['cloud_cover'][i]
            # store into geodataframe
            if counter == 0:
                gdf_all = gdf
            else:
                gdf_all = gdf_all.append(gdf)
            counter = counter + 1
            
    return gdf_all

def transects_to_gdf(transects):
    """
    Saves the shore-normal transects as a gpd.GeoDataFrame    
    
    KV WRL 2018

    Arguments:
    -----------
    transects: dict
        contains the coordinates of the transects          
                
    Returns:    
    -----------
    gdf_all: gpd.GeoDataFrame

        
    """  
       
    # loop through the mapped shorelines
    for i,key in enumerate(list(transects.keys())):
        # save the geometry + attributes
        geom = geometry.LineString(transects[key])
        gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geom))
        gdf.index = [i]
        gdf.loc[i,'name'] = key
        # store into geodataframe
        if i == 0:
            gdf_all = gdf
        else:
            gdf_all = gdf_all.append(gdf)
            
    return gdf_all

def get_image_bounds(fn):
    """
    Returns a polygon with the bounds of the image in the .tif file
     
    KV WRL 2020

    Arguments:
    -----------
    fn: str
        path to the image (.tif file)         
                
    Returns:    
    -----------
    bounds_polygon: shapely.geometry.Polygon
        polygon with the image bounds
        
    """
    
    # nested functions to get the extent 
    # copied from https://gis.stackexchange.com/questions/57834/how-to-get-raster-corner-coordinates-using-python-gdal-bindings
    def GetExtent(gt,cols,rows):
        'Return list of corner coordinates from a geotransform'
        ext=[]
        xarr=[0,cols]
        yarr=[0,rows]
        for px in xarr:
            for py in yarr:
                x=gt[0]+(px*gt[1])+(py*gt[2])
                y=gt[3]+(px*gt[4])+(py*gt[5])
                ext.append([x,y])
            yarr.reverse()
        return ext
    
    # load .tif file and get bounds
    data = gdal.Open(fn, gdal.GA_ReadOnly)
    gt = data.GetGeoTransform()
    cols = data.RasterXSize
    rows = data.RasterYSize
    ext = GetExtent(gt,cols,rows)
    
    return geometry.Polygon(ext)

def smallest_rectangle(polygon):
    """
    Converts a polygon to the smallest rectangle polygon with sides parallel
    to coordinate axes.
     
    KV WRL 2020

    Arguments:
    -----------
    polygon: list of coordinates 
        pair of coordinates for 5 vertices, in clockwise order,
        first and last points must match     
                
    Returns:    
    -----------
    polygon: list of coordinates
        smallest rectangle polygon
        
    """
    
    multipoints = geometry.Polygon(polygon[0])
    polygon_geom = multipoints.envelope
    coords_polygon = np.array(polygon_geom.exterior.coords)
    polygon_rect = [[[_[0], _[1]] for _ in coords_polygon]]
    return polygon_rect

def metadata_collection(sat_list, Sat, filepath_data, sitename):
    
    filename = sitename + '_metadata.pkl'
    filepath = os.path.join(filepath_data, sitename)
    
    if filename in os.listdir(filepath):
        print('Metadata already exists and was loaded')
        with open(os.path.join(filepath, filename), 'rb') as f:
            metadata = pickle.load(f)
        return metadata
    
    metadata = dict([])

    for i in range(len(sat_list)):
        metadata[sat_list[i]] = {'filenames':[], 'acc_georef':[], 'epsg':[], 'dates':[]}

    for i in range(len(Sat)):
        for j in range(len(Sat[i].getInfo().get('features'))):
            clear_output(wait=True)
            if sat_list[i] != 'S2':
                metadata[sat_list[i]]['filenames'].append(Sat[i].getInfo().get('features')[j]['id'])
                metadata[sat_list[i]]['acc_georef'].append(Sat[i].getInfo().get('features')[j]['properties']['GEOMETRIC_RMSE_MODEL'])
                metadata[sat_list[i]]['epsg'].append(int(Sat[i].getInfo().get('features')[j]['bands'][0]['crs'].lstrip('EPSG:')))
                metadata[sat_list[i]]['dates'].append(Sat[i].getInfo().get('features')[j]['properties']['DATE_ACQUIRED'])
            else:
                metadata[sat_list[i]]['filenames'].append(Sat[i].getInfo().get('features')[j]['id'])
                metadata[sat_list[i]]['acc_georef'].append(Sat[i].getInfo().get('features')[j]['bands'][0]['crs_transform'])
                metadata[sat_list[i]]['epsg'].append(int(Sat[i].getInfo().get('features')[j]['bands'][0]['crs'].lstrip('EPSG:')))
                d = datetime.strptime(Sat[i].getInfo().get('features')[j]['properties']['DATATAKE_IDENTIFIER'][5:13],'%Y%m%d')
                metadata[sat_list[i]]['dates'].append(str(d.strftime('%Y-%m-%d')))
            
            print(sat_list[i],": ",(100*j/len(Sat[i].getInfo().get('features'))),'%', end='')
    
    with open(os.path.join(filepath, sitename + '_metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
        
    return metadata


        
def image_retrieval(inputs):
    
    point = ee.Geometry.Point(inputs['polygon'][0][0])
    
    Sat = []

    if 'cloud_thresh' in inputs.keys():
        cloud_thresh = int(inputs['cloud_thresh']*100)
    else:
        cloud_thresh = 90
        
    if 'L5' in inputs['sat_list']:
        Landsat5 = ee.ImageCollection("LANDSAT/LT05/C01/T1_TOA").filterBounds(point).filterDate(inputs['dates'][0], inputs['dates'][1])
        Sat.append(Landsat5)
    if 'L7' in inputs['sat_list']:
        Landsat7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_TOA').filterBounds(point).filterDate(inputs['dates'][0], inputs['dates'][1]).filter(ee.Filter.lt('CLOUD_COVER', cloud_thresh))
        Sat.append(Landsat7)
    if 'L8' in inputs['sat_list']:
        Landsat8 = ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA').filterBounds(point).filterDate(inputs['dates'][0], inputs['dates'][1]).filter(ee.Filter.lt('CLOUD_COVER', cloud_thresh))
        Sat.append(Landsat8)
    if 'S2' in inputs['sat_list']:
        Sentinel2 = ee.ImageCollection("COPERNICUS/S2").filterBounds(point).filterDate(inputs['dates'][0], inputs['dates'][1]).filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloud_thresh))
        Sat.append(Sentinel2)
        
    return Sat

def LocalImageRetrieval(inputs):
    """
    Finds imagery stored locally and derives Sat information from the image filepaths.
    (different list for each sat platform sorted into respective folders).
    FM Apr 2022
    
    """
    Sat = []
    for imdir in [x[0] for x in os.walk('./Data/'+inputs['sitename']+'/local_images/')][1:]:
        imageIDs = sorted(glob.glob(imdir+'/*.tif'))
    
        Sat.append(imageIDs)
    
    return Sat

def LocalImageMetadata(inputs, Sat):
    """
    Extracts metadata info from local image filepaths and creates dict of info.
    FM Apr 2022
    
    filenames: filepaths and filenames from Sat list 
    acc_georef: affine matrix for georeferencing transformations
    
    """
    
    filename = inputs['sitename'] + '_metadata.pkl'
    filepath = os.path.join(inputs['filepath'], inputs['sitename'])
    
    if filename in os.listdir(filepath):
        print('Metadata already exists and was loaded')
        with open(os.path.join(filepath, filename), 'rb') as f:
            metadata = pickle.load(f)
        return metadata
    
    metadata = dict([])

    for i in range(len(inputs['sat_list'])):
        metadata[inputs['sat_list'][i]] = {'filenames':[], 'acc_georef':[], 'epsg':[], 'dates':[]}

    for i in range(len(Sat)):
        for j in range(len(Sat[i])):
            
            imdata = rasterio.open(Sat[i][j])
            
            metadata[inputs['sat_list'][i]]['filenames'].append(Sat[i][j])
            
            metadata[inputs['sat_list'][i]]['acc_georef'].append(list(imdata.transform)[0:6])
            
            metadata[inputs['sat_list'][i]]['epsg'].append(str(imdata.crs).lstrip('EPSG:'))
            
            date = datetime.strptime(os.path.basename(Sat[i][j])[0:8],'%Y%m%d') #relies on date being YYYYMMDD first in filename
            metadata[inputs['sat_list'][i]]['dates'].append(str(date.strftime('%Y-%m-%d')))
            
            
            print(inputs['sat_list'][i],": ",(100*(j+1)/len(Sat[i])),'%', end='')
    
    with open(os.path.join(filepath, inputs['sitename'] + '_metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
        
    return metadata

def PlanetImageRetrieval(inputs):
    '''
    Finds Planet imagery using a valid API key and parameters to constrain the dataset.
    FM Apr 2022

    Returns
    -------
    testdata.

    '''
    AOI = {
        "type": "Polygon",
        "coordinates": inputs['polygon']
        }
    
    Sat = []
    # API Key stored as an env variable
    PLANET_API_KEY = os.environ['PL_API_KEY']
    
    # # Setup Planet Data API base URL
    # URL = "https://api.planet.com/data/v1"
    # # Setup the session
    # session = requests.Session()
    # # Authenticate
    # session.auth = (PLANET_API_KEY, "")

    # get images that overlap with our AOI 
    geomFilter = {
        "type": "GeometryFilter",
        "field_name": "geometry",
        "config": AOI
        }
    
    # get images acquired within a date range
    dateFilter = {
        "type": "DateRangeFilter",
        "field_name": "acquired",
        "config": {
    "gte": inputs['dates'][0]+"T00:00:00.000Z",
    "lte": "2022-04-01T00:00:00.000Z"
        }
    }

    # only get images which have <50% cloud coverage
    cloudFilter = {
        "type": "RangeFilter",
        "field_name": "cloud_cover",
        "config": {
            "lte": 0.5
            }
        }

    # combine our geo, date, cloud filters
    combinedFilter = {
        "type": "AndFilter",
        "config": [geomFilter, dateFilter, cloudFilter]
        }
    
    item_type = "PSScene4Band"

    # API request object
    searchRequest = {
        "item_types": [item_type], 
        "filter": combinedFilter
        }

    # fire off the POST request
    searchResult = \
        requests.post(
            'https://api.planet.com/data/v1/quick-search',
            auth=HTTPBasicAuth(PLANET_API_KEY, ''),
            json=searchRequest)
    
    # extract image IDs only
    try:
        imageIDs = [feature['id'] for feature in searchResult.json()['features']]
        print('Number of images returned: '+str(len(imageIDs)))
        Sat.append(imageIDs)
    except:
        print('ERROR: \n'+searchResult.text)
    
    return Sat
    
def PlanetDownload(Sat):
    
    idURLs = []
    for ID in Sat[0]:
        idURL = 'https://api.planet.com/data/v1/item-types/{}/items/{}/assets'.format('PSScene4Band', ID)

        # Returns JSON metadata for assets in this ID. Learn more: planet.com/docs/reference/data-api/items-assets/#asset
        result = \
            requests.get(
                idURL,
                auth=HTTPBasicAuth(os.environ['PL_API_KEY'], '')
                )
        
        # List of asset types available for this particular satellite image
        #print(result.json().keys())
        idURLs.append(idURL)
        # Parse out useful links
        links = result.json()[u'analytic']['_links']
        selflink = links['_self']
        activationlink = links['activate']        
        # Activate analytic dataset
        activateResult = \
            requests.get(
                activationlink,
                auth=HTTPBasicAuth(os.environ['PL_API_KEY'], '')
                )
        activationStatResult = \
            requests.get(
                selflink,
                auth=HTTPBasicAuth(os.environ['PL_API_KEY'], '')
                )
        print(activationStatResult.json()['status'])
        while activationStatResult.json()['status'] != 'active':
            print(ID+' is still '+activationStatResult.json()['status'])
            time.sleep(10)
        else:
            break
        # Download imagery
        downloadlink = activationStatResult.json()["location"]
        print(downloadlink)
        
    return idURLs

def PlanetMetadata(sat_list, Sat, filepath_data, sitename):
    
    filename = sitename + '_metadata.pkl'
    filepath = os.path.join(filepath_data, sitename)
    
    if filename in os.listdir(filepath):
        print('Metadata already exists and was loaded')
        with open(os.path.join(filepath, filename), 'rb') as f:
            metadata = pickle.load(f)
        return metadata
    
    metadata = dict([])

    for i in range(len(sat_list)):
        metadata[sat_list[i]] = {'filenames':[], 'acc_georef':[], 'epsg':[], 'dates':[]}

    for i in range(len(Sat)):
        for j in range(len(Sat[i])):
            metadata[sat_list[i]]['filenames']
            metadata[sat_list[i]]['acc_georef']
            metadata[sat_list[i]]['epsg']
            metadata[sat_list[i]]['dates']
            
            
            # if sat_list[i] != 'S2':
            #     metadata[sat_list[i]]['filenames'].append(Sat[i].getInfo().get('features')[j]['id'])
            #     metadata[sat_list[i]]['acc_georef'].append(Sat[i].getInfo().get('features')[j]['properties']['GEOMETRIC_RMSE_MODEL'])
            #     metadata[sat_list[i]]['epsg'].append(int(Sat[i].getInfo().get('features')[j]['bands'][0]['crs'].lstrip('EPSG:')))
            #     metadata[sat_list[i]]['dates'].append(Sat[i].getInfo().get('features')[j]['properties']['DATE_ACQUIRED'])
            # else:
            #     metadata[sat_list[i]]['filenames'].append(Sat[i].getInfo().get('features')[j]['id'])
            #     metadata[sat_list[i]]['acc_georef'].append(Sat[i].getInfo().get('features')[j]['bands'][0]['crs_transform'])
            #     metadata[sat_list[i]]['epsg'].append(int(Sat[i].getInfo().get('features')[j]['bands'][0]['crs'].lstrip('EPSG:')))
            #     d = datetime.strptime(Sat[i].getInfo().get('features')[j]['properties']['DATATAKE_IDENTIFIER'][5:13],'%Y%m%d')
            #     metadata[sat_list[i]]['dates'].append(str(d.strftime('%Y-%m-%d')))
            
            print(sat_list[i],": ",(100*j/len(Sat[i].getInfo().get('features'))),'%', end='')
    
    with open(os.path.join(filepath, sitename + '_metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
        
    return metadata

def SaveShapefiles(output, name_prefix, sitename, epsg):

    '''
    FM Apr 2022
    '''
    
    # for shores stored as array of coords; export as mulitpoint
    if type(output['shorelines'][0]) == np.ndarray:
        # map to multipoint
        output_geom = gpd.GeoSeries(map(MultiPoint,output['shorelines']))
        # create geodataframe with geometry from output multipoints
        outputGDF = gpd.GeoDataFrame(output, crs='EPSG:'+str(epsg), geometry=output_geom)
        # drop duplicate shorelines column
        outputsGDF = outputGDF.drop('shorelines', axis=1)
    else:    
        DFlist = []
        for i in range(len(output['shorelines'])): # for each image + associated metadata
            # create geodataframe of individual features from each geoseries (i.e. feature collection)
            outputGDF = gpd.GeoDataFrame(geometry=output['shorelines'][i])
            for key in output.keys(): # for each column
                # add column to geodataframe with repeated metadata
                outputGDF[key] = output[key][i]
            # add formatted geodataframe to list of all geodataframes
            DFlist.append(outputGDF)
            # concatenate to one GDF with individual lines exploded out
            outputsGDF = gpd.GeoDataFrame( pd.concat( DFlist, ignore_index=True), crs=DFlist[0].crs)
            outputsGDF = outputsGDF.drop('shorelines', axis=1)
            
    outputsGDF.to_file(os.path.join(name_prefix, sitename + '_' + str(min(output['dates'])) + '_' + str(max(output['dates'])) + '_veglines.shp'))
    
    
    return

def SaveConvShapefiles(output, name_prefix, sitename, epsg):

    '''
    FM Apr 2022
    '''
    
    # for shores stored as array of coords; export as mulitpoint
    if type(output['shorelines'][0]) == np.ndarray:
        # map to multipoint
        output_geom = gpd.GeoSeries(map(MultiPoint,output['shorelines']))
        # create geodataframe with geometry from output multipoints
        outputGDF = gpd.GeoDataFrame(output, crs='EPSG:'+str(epsg), geometry=output_geom)
        # drop duplicate shorelines column
        outputsGDF = outputGDF.drop('shorelines', axis=1)
    else:    
        DFlist = []
        for i in range(len(output['shorelines'])): # for each image + associated metadata
            # create geodataframe of individual features from each geoseries (i.e. feature collection)
            convlines = output['shorelines'][i].to_crs(str(epsg))
            outputGDF = gpd.GeoDataFrame(geometry=convlines, crs=str(epsg))
            for key in output.keys(): # for each column
                # add column to geodataframe with repeated metadata
                outputGDF[key] = output[key][i]
            # add formatted geodataframe to list of all geodataframes
            DFlist.append(outputGDF)
            # concatenate to one GDF with individual lines exploded out
            outputsGDF = gpd.GeoDataFrame( pd.concat( DFlist, ignore_index=True), crs=str(epsg))
            outputsGDF = outputsGDF.drop('shorelines', axis=1)
            
    outputsGDF.to_file(os.path.join(name_prefix, sitename + '_' + str(min(output['dates'])) + '_' + str(max(output['dates'])) + '_veglines.shp'))
    
    
    return

def Separate_TimeSeries_year(cross_distance, output, key):

    Date_Organised = [[datetime.strptime(min(output['dates']),'%Y-%m-%d').year]]
    Distance_Organised = [[]]
    Month_Organised = [[]]

    for i in range(len(output['dates'])):
        appended = False
        for j in range(len(Date_Organised)):
            if datetime.strptime(output['dates'][i],'%Y-%M-%d').year == Date_Organised[j][0]:
                Date_Organised[j].append(datetime.strptime(output['dates'][i],'%Y-%m-%d').year)
                Month_Organised[j].append(datetime.strptime(output['dates'][i],'%Y-%m-%d').month)
                Distance_Organised[j].append((cross_distance[key]- np.nanmedian(cross_distance[key]))[i])
                appended = True
        if appended==False:
            Date_Organised.append([datetime.strptime(output['dates'][i],'%Y-%m-%d').year])
            Month_Organised.append([datetime.strptime(output['dates'][i],'%Y-%m-%d').month])
            Distance_Organised.append([(cross_distance[key]- np.nanmedian(cross_distance[key]))[i]])
            
    DateArr = []
    DistanceAvgArr = []
    for i in range(len(Date_Organised)):
        DateArr.append(Date_Organised[i][0])
        DistanceAvgArr.append(np.nanmean(Distance_Organised[i]))#sum(Distance_Organised[i])/len(Distance_Organised[i]))
    
    return Date_Organised, Month_Organised, Distance_Organised, DateArr, DistanceAvgArr

def Separate_TimeSeries_month(cross_distance, output, key):

    Date_Organised = [[]]
    Distance_Organised = [[]]
    Month_Organised = [[1]]

    for i in range(len(output['dates'])):

        appended = False
        
        for j in range(len(Month_Organised)):
            
            if datetime.strptime(output['dates'][i],'%Y-%m-%d').month == Month_Organised[j][0]:
                Date_Organised[j].append(datetime.strptime(output['dates'][i],'%Y-%m-%d').year)
                Month_Organised[j].append(datetime.strptime(output['dates'][i],'%Y-%m-%d').month)
                Distance_Organised[j].append((cross_distance[key]- np.nanmedian(cross_distance[key]))[i])
                appended = True
                
        if appended==False:
            Date_Organised.append([datetime.strptime(output['dates'][i],'%Y-%m-%d').year])
            Month_Organised.append([datetime.strptime(output['dates'][i],'%Y-%m-%d').month])
            #print(Month_Organised)
            Distance_Organised.append([(cross_distance[key]- np.nanmedian(cross_distance[key]))[i]])
            
    DateArr = []
    DistanceAvgArr = []
    for i in range(len(Distance_Organised)):
        DateArr.append(Month_Organised[i][0])
        temp_list = Distance_Organised[i]
        newlist = [x for x in temp_list if math.isnan(x) == False]
        DistanceAvgArr.append(np.nanmean(newlist))#sum(newlist)/len(newlist))
    
    return Date_Organised, Month_Organised, Distance_Organised, DateArr, DistanceAvgArr

def ProcessRefline(referenceLineShp,settings):
    """
    

    Parameters
    ----------
    referenceLineShp : str
        Filepath to refline shapefile.
    settings : dict
        settings dictionary.

    Returns
    -------
    referenceLine : 
        reference line coordinate array
    ref_epsg : int
        reference line EPSG ID

    """
    referenceLineDF = gpd.read_file(referenceLineShp)
    refLinex,refLiney = referenceLineDF.geometry[0].coords.xy
    # swap latlon coordinates (or don't? check this) around and format into list
    #referenceLineList = list([refLinex[i],refLiney[i]] for i in range(len(refLinex)))
    referenceLineList = list([refLiney[i],refLinex[i]] for i in range(len(refLinex)))
    # convert to UTM zone for use with the satellite images
    ref_epsg = int(str(referenceLineDF.crs)[5:])
    image_epsg = settings['output_epsg']
    referenceLine = convert_epsg(np.array(referenceLineList),ref_epsg,image_epsg)
    referenceLine = spaced_vertices(referenceLine)
    
    return referenceLine, ref_epsg

def daterange(date1, date2):
    """
    Get formatted date range for two datetime dates.
    FM Apr 2022
    """
    for n in range(int(date2.year) - int(date1.year)+1):
        yield int(date1.year) + n

    

def NearestDates(surveys,metadata,sat_list):
    """
    
    Print image dates from full sat metadata that fall nearest to validation dates.
    
    surveys : GeoDataFrame
        Shapefile of validation lines read in using geopandas
    metadata : dict
        Satellite image metadata
    sat_list : list
        List of desired satellite platforms as strings
    
        
    FM Sept 2022
    """
    
    veridates = sorted(list(surveys.Date.unique()))
    
    nearestdates = dict.fromkeys(sat_list)
    nearestIDs = dict.fromkeys(sat_list)

    for sat in sat_list:
        print(sat,'SATELLITE')
        satdates=[]
        nearestdate = []
        nearestID = []
        for veridate in veridates:
            print('verification:\t',veridate)
            veridate = datetime.strptime(veridate,'%Y-%m-%d')
            for satdate in metadata[sat]['dates']:
                satdates.append(datetime.strptime(satdate,'%Y-%m-%d'))
            
            print('nearest:\t\t',NearDate(veridate,satdates))
            if NearDate(veridate,satdates) == False:
                print('no image near in time.')
            else:
                nearestdate.append(datetime.strftime(NearDate(veridate,satdates),'%Y-%m-%d'))
                nearestID.append(metadata[sat]['dates'].index(datetime.strftime(NearDate(veridate,satdates),'%Y-%m-%d')))

        nearestdates[sat] = nearestdate
        nearestIDs[sat] = nearestID     
    
    return nearestdates, nearestIDs
        
    
        
def spaced_vertices(referenceLine):
    """
    Equally space vertices of reference line to avoid gaps in image buffer.
    FM Apr 2022

    Parameters
    ----------
    referenceLine : array
        Reference line coordinate array

    Returns
    -------
    newreferenceLine : array
        New reference line coordinate array with equally spaced vertices.

    """
    referenceLineString = LineString(referenceLine)
    vertexdist = 10
    vertexdists = np.arange(0, referenceLineString.length, vertexdist)
    newverts = [referenceLineString.interpolate(dist) for dist in vertexdists] + [referenceLineString.boundary[1]]
    newreferenceLineString = LineString(newverts)
    newreferenceLine = np.asarray(newreferenceLineString)
    
    return newreferenceLine


def AOI(lonmin, lonmax, latmin, latmax):
    # Check if lat and long min and max are around the right way
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
    # Create bounding box and convert it to a geodataframe
    BBox = Polygon([[lonmin, latmin],
                    [lonmax,latmin],
                    [lonmax,latmax],
                    [lonmin, latmax]])
    BBoxGDF = gpd.GeoDataFrame(geometry=[BBox], crs = {'init':'epsg:4326'})
    # UK conversion only
    BBoxGDF = BBoxGDF.to_crs('epsg:27700')
    # Check if AOI could exceed the 262144 (512x512) pixel limit on ee requests
    if (int(BBoxGDF.area)/(10*10))>262144:
        print('Your bounding box is too big for Sentinel2 (%s too big)' % int((BBoxGDF.area/(10*10))-262144))
    
    # Export as polygon and ee point for use in clipping satellite image requests
    polygon = [[[lonmin, latmin],[lonmax, latmin],[lonmin, latmax],[lonmax, latmax]]]
    point = ee.Geometry.Point(polygon[0][0]) 
    
    return polygon, point

def GStoArr(shoreline):
    """
    Parameters
    ----------
    shoreline : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    
    FM Aug 2022
    """

    shorelineList = [np.array(line.coords) for line in shoreline.geometry]
    shorelineArrList = [coord for line in shorelineList for coord in line]
    shorelineArr = np.array(shorelineArrList)
    return shorelineArr

def ArrtoGS(refline,georef):
    """
    
    FM Sept 2022


    """
    coords = []
    ref_sl = refline[:,:2]
    ref_sl_pix = convert_world2pix(ref_sl, georef)
    
    ref_sl_pix_rounded = np.round(ref_sl_pix).astype(int)
    
    for i in range(len(ref_sl_pix_rounded)):
        coords.append((ref_sl_pix_rounded[i][0],ref_sl_pix_rounded[i][1]))
    refLS = LineString(coords)
    refGS = gpd.GeoSeries(refLS)
    
    return refGS


def NearDate(target,items):
    """
    Find nearest date to target in list of datetimes. Returns matching date 
    only if match is within 3 months.
    
    FM Oct 2022

    Parameters
    ----------
    target : datetime
        Target datetime to find nearest date to.
    items : list
        List of datetimes.

    """
    nearestDate = min(items, key=lambda x: abs(x - target))
    
    # # if difference is longer than 3 months, no match exists  
    if abs((target - nearestDate).days) > 91: 
        return False
    else:
        return nearestDate
    # return nearestDate


def TZValuesSTDV(int_veg, int_nonveg):
    """
    Generate bounds for transition zone plot by using the 3rd standard deviations of the two classes.
    FM Oct 2022

    Parameters
    ----------
    int_veg : array
        NDVI pixel values classed as veg.
    int_nonveg : array
        NDVI pixel values classed as nonveg.

    Returns
    -------
    [minval,maxval] : list
        minimum and maximum transition zone bounds.

    """
    
    bins = np.arange(-1, 1, 0.01) # start, stop, bin width
    # create histogram but don't plot, use to access frequencies
    nvcounts, nvbins = np.histogram(int_nonveg,bins=bins)
    vcounts, vbins = np.histogram(int_veg,bins=bins)
    
    # first veg value that sits above significant frequency
    minval = np.mean(int_veg)-(3*np.std(int_veg)) # 3 stdevs to the left
           
   # get ID of first point where difference in frequencies rise above statistically significant threshold 
    maxval = np.mean(int_nonveg)+(3*np.std(int_nonveg)) # 3 stdevs to the left
    
    return [minval,maxval]


def TZValues(int_veg, int_nonveg):
    """
    Generate bounds for transition zone plot by finding the minimum NDVI value which could be veg,
    and the NDVI value which is definitely veg (the point where the veg hist. freq. is greater than the nonveg hist. freq.)

    Parameters
    ----------
    int_veg : array
        NDVI pixel values classed as veg.
    int_nonveg : array
        NDVI pixel values classed as nonveg.

    Returns
    -------
    [minval,maxval] : list
        minimum and maximum transition zone bounds.

    """
    
    bins = np.arange(-1, 1, 0.01) # start, stop, bin width
    # create histogram but don't plot, use to access frequencies
    nvcounts, nvbins = np.histogram(int_nonveg,bins=bins)
    vcounts, vbins = np.histogram(int_veg,bins=bins)
    
    ## minimum transition zone value is first point where veg is defined
    ## first veg value that sits above significant frequency (5%)
    # countthresh = int(np.nanmax(vcounts)*0.05)
    # minvalID = [idx for idx, element in enumerate(vcounts) if element>countthresh][0]
    # minval = vbins[minvalID]
    
    # 3rd standard dev to the left
    minval = np.mean(int_veg) - (3*np.std(int_veg))
        
    # calculate differences between counts to find where veg rises above nonveg
    countdiff = vcounts-nvcounts
    # get ID of first point where difference in frequencies rise above statistically significant threshold (10%)
    countthresh = int(np.nanmax(vcounts)*0.1)
    countIDs = [idx for idx, element in enumerate(countdiff) if element>countthresh]
    if countIDs == []: # for when veg never surpasses nonveg freq.
        countthresh = int(np.nanmax(vcounts)*0.1) # take first veg freq that surpasses 10% of max
        countID = [idx for idx, element in enumerate(vcounts) if element>countthresh][0]
    else:
        countID = countIDs[0] # take first index
        
    # maximum value is 
    maxval = vbins[countID]
    
    return [minval,maxval]


def CalcDistance(Geom1,Geom2):
    """
    Calculate distance between two shapely geoms, either using a point and line
    (endpoint coordinate of line is used), or two points. 
    FM Oct 2022

    Parameters
    ----------
    Geom1 : Point or LineString
        First location.
    Geom2 : Point or LineString
        Second location.

    Returns
    -------
    geom1geom2dist : float64
        Distance between two points (in metres).

    """
    if Geom2.geom_type == 'LineString': # point distance from end of transect
        geom1geom2dist = np.sqrt( (Geom1.x - Geom2.coords[0][0])**2 + 
                                 (Geom1.y - Geom2.coords[0][1])**2 )
    else: # end of transect distance from point
        geom1geom2dist = np.sqrt( (Geom1.coords[0][0] - Geom2.x)**2 + 
                                 (Geom1.coords[0][1] - Geom2.y)**2 )
        
    return geom1geom2dist
