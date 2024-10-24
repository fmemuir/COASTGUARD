"""
This module contains utilities to work with satellite images
    
Author: Kilian Vos, Water Research Laboratory, University of New South Wales

COASTGUARD edits and updates: Freya Muir, University of Glasgow
"""

# load modules
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pdb
import glob

# other modules
from osgeo import gdal, osr
import pandas as pd
import geopandas as gpd
import utm
from shapely import geometry, affinity
from shapely.geometry import Point, Polygon, LineString, MultiLineString, MultiPoint
from shapely.ops import linemerge
import folium

import skimage.transform as transform
import sklearn
import scipy
from scipy import interpolate
from scipy.stats import circmean, circstd, skew, kurtosis
from statsmodels.tsa.seasonal import seasonal_decompose
from astropy.convolution import convolve
from datetime import datetime, timedelta
from IPython.display import clear_output
import ee
import geemap
import copernicusmarine as cms

import pickle
import math

import rasterio
from rasterio.plot import show
import pyfes
import xarray as xr
import netCDF4

from Toolshed import Image_Processing

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


def get_UTMepsg_from_wgs(lat, lon):
    """
    Retrieves an epsg code from lat lon in wgs84
    see https://stackoverflow.com/a/40140326/4556479
    
    added by MDH 2023

    Arguments:
    -----------
    lat: latitude in WGS84
    lon: longitude in WGS84         
                
    Returns:    
    -----------
    epsg_code: epsg code for best UTM zone 
        
    """
    utm_band = str((math.floor((lon + 180) / 6 ) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0'+utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
        return epsg_code
    epsg_code = '327' + utm_band
    return epsg_code
    
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
    

def savi_index(im1, im2, cloud_mask):
    """
    Computes soil adjusted vegetation index on 2 bands (2D), given a cloud mask (2D).

    FM 2022

    Arguments:
    -----------
    im1: np.array
        first image (2D) with which to calculate the ND index (should be NIR band)
    im2: np.array
        second image (2D) with which to calculate the ND index (should be Red band)
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
    nir = im1.reshape(im1.shape[0] * im1.shape[1])
    red = im2.reshape(im2.shape[0] * im2.shape[1])
    # compute the normalised difference index
    temp = np.divide(nir[~vec_mask] - red[~vec_mask],
                     nir[~vec_mask] + red[~vec_mask] + 0.5) * (1 + 0.5)
    vec_nd[~vec_mask] = temp
    # reshape into image
    im_nd = vec_nd.reshape(im1.shape[0], im1.shape[1])
    return im_nd


def rbnd_index(im1, im2, im3, cloud_mask):
    """
    Computes soil adjusted vegetation index on 2 bands (2D), given a cloud mask (2D).

    FM 2022

    Arguments:
    -----------
    im1: np.array
        first image (2D) with which to calculate the ND index (should be NIR band)
    im2: np.array
        second image (2D) with which to calculate the ND index (should be Red band)
    im3: np.array
        third image (2D) with which to calculate the ND index (should be Blue band)
    
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
    # reshape the images
    nir =  im1.reshape(im1.shape[0] * im1.shape[1])
    red =  im2.reshape(im2.shape[0] * im2.shape[1])
    blue = im3.reshape(im3.shape[0] * im3.shape[1])
    # compute the normalised difference index
    temp = np.divide(nir[~vec_mask] - (red[~vec_mask] + blue[~vec_mask]),
                     nir[~vec_mask] + (red[~vec_mask] + blue[~vec_mask]))
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
    
def find(item, lst):
    """
    Iterate through list to find matching item.
    FM 2022

    Parameters
    ----------
    item : TYPE
        DESCRIPTION.
    lst : TYPE
        DESCRIPTION.

    Returns
    -------
    start : TYPE
        DESCRIPTION.

    """
    start = 0
    start = lst.index(item, start)
    return start


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
    Function to merge the output dictionary, which has one key per satellite mission
    into a dictionary containing all the shorelines and dates ordered chronologically.
    
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
        output_all['satname'] = output_all['satname'] + [str(_) for _ in np.tile(satname,
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


def RemoveDuplicates(output):
    """
    Function to remove entries containing shorelines for the same date and satellite mission. 
    This happens when there is an overlap between adjacent satellite images.
    If there is an overlap, this function keeps the longest line from the duplicates.
    FM Oct 2023

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
    def duplicates_dict(dates_str):
        "return duplicates and indices"
        def duplicateIDs(dates_str, date_str):
                # need to return output['idx'] and not just enumerate ID
                # return [i for i, x in enumerate(dates_str) if x == date_str]
                return [output['idx'][i] for i, x in enumerate(dates_str) if x == date_str]

        # return dict((date_str, duplicates(dates_str, date_str)) for date_str in set(dates_str) if dates_str.count(date_str) > 1)
        dupl = {}
        for date_str in set(dates_str):
            if dates_str.count(date_str) > 1:
                dupl[date_str] = duplicateIDs(dates_str, date_str)
        return dupl
        
    def update_dupl(output):
        dates = output['dates']
        # make a list with year/month/day
        dates_str = [datetime.strptime(_,'%Y-%m-%d').strftime('%Y-%m-%d') for _ in dates]
        # create a dictionary with the duplicates
        dupl = dict(sorted(duplicates_dict(dates_str).items(), key=lambda x:x[1]))
        
        if dupl:
            # Keep duplicates with different satnames
            for key,IDval in list(dupl.items()): # has to be list so as not to change dict while looping
                if len(set(output['satname'][IDval[0]:IDval[1]+1])) > 1:
                    del(dupl[key])
        return dupl

    dupl = update_dupl(output)
      
    # if there are duplicates, only keep the element with the longest line
    while dupl:
        
        dupl = update_dupl(output)
            
        dupcount = []
        for key,IDval in list(dupl.items()):
            dupcount.append(IDval[0])
            lengths = []
            # calculate lengths of duplicated line features
            for v in IDval:
                if len(output['veglines'][output['idx'].index(v)]) > 1: # multiline feature; take sum of line lengths
                    lengths.append(sum(output['veglines'][output['idx'].index(v)].length))
                else:
                    if len(output['veglines'][output['idx'].index(v)].length) == 0: # empty geoms
                        lengths.append(0)
                    else:
                        lengths.append(output['veglines'][output['idx'].index(v)].length.iloc[0])

                minlenID = lengths.index(min(lengths))
           
            # keep the longest line (i.e. remove the shortest)
            for okey in list(output.keys()):
                # delete the other keys first, leaving idx for last
                if okey != 'idx':
                    delID = output['idx'].index(IDval[minlenID])
                    del(output[okey][delID])
            delID = output['idx'].index(IDval[minlenID])
            del(output['idx'][delID])
                    
        print('%d duplicates' % len(dupcount))
        
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


def CreateFileStructure(sitename, sat_list):
    """
    Create file structure for veg edge extraction.
    FM 2022

    Parameters
    ----------
    sitename : str
        Name of site for analysis.
    sat_list : list
        List of strings corresponding to satellite platforms to use.

    Returns
    -------
    filepath : str
        Filepath to project folder for desired sitename.

    """
    filepath = os.path.join(os.getcwd(), 'Data')

    direc = os.path.join(filepath, sitename)

    if os.path.isdir(direc) is False:
        os.mkdir(direc)
        
    
    if 'PSScene4Band' in sat_list:
        if os.path.isdir(direc+'/local_images') is False:
            os.mkdir(direc+'/local_images')
            os.mkdir(direc+'/local_images/PlanetScope')
            os.mkdir(direc+'/AuxillaryImages')
            os.mkdir(direc+'/local_images/PlanetScope/cloudmasks')
    
    return filepath


def metadata_collection(inputs, Sat):
    """
    Compile Google Earth Engine metadata together to create a collection of image properties. 
    FM 2022

    Parameters
    ----------
    inputs : dict
        Dictionary of user inputs.
    Sat : dict
        Dictionary of returned images based on user inputs.

    Returns
    -------
    metadata : dict
        Dictionary of image metadata.

    """
    
    
    sat_list = inputs['sat_list']
    filepath_data = inputs['filepath']
    sitename = inputs['sitename']
    
    # Planet data must be loaded locally (while API is still sluggish)
    # TO DO: incorporate in so that metadata can consist of Landsat, Sentinel AND Planet
    if 'PSScene4Band' in sat_list:
        metadata = LocalImageMetadata(inputs, Sat)
    
    else: 
        filename = sitename + '_metadata.pkl'
        filepath = os.path.join(filepath_data, sitename)
        
        if filename in os.listdir(filepath):
            print('Metadata already exists and was loaded')
            with open(os.path.join(filepath, filename), 'rb') as f:
                metadata = pickle.load(f)
            return metadata
        
        print('making metadata dictionary...')
        metadata = dict([])
    
        for i in range(len(sat_list)):
            metadata[sat_list[i]] = {'filenames':[], 'acc_georef':[], 'epsg':[], 'dates':[]}
    
        for i in range(len(Sat)):
            Features = Sat[i].getInfo().get('features')
            for j in range(len(Features)):
                Feature = Features[j]
                if sat_list[i] != 'S2':
                    metadata[sat_list[i]]['filenames'].append(Feature['id'])
                    metadata[sat_list[i]]['acc_georef'].append(Feature['properties']['GEOMETRIC_RMSE_MODEL'])
                    metadata[sat_list[i]]['epsg'].append(int(Feature['bands'][0]['crs'].lstrip('EPSG:')))
                    metadata[sat_list[i]]['dates'].append(Feature['properties']['DATE_ACQUIRED'])
                else:
                    metadata[sat_list[i]]['filenames'].append(Feature['id'])
                    metadata[sat_list[i]]['acc_georef'].append(Feature['bands'][1]['crs_transform'])
                    metadata[sat_list[i]]['epsg'].append(int(Feature['bands'][1]['crs'].lstrip('EPSG:')))
                    d = datetime.strptime(Feature['properties']['DATATAKE_IDENTIFIER'][5:13],'%Y%m%d')
                    metadata[sat_list[i]]['dates'].append(str(d.strftime('%Y-%m-%d')))
                
                print('\r'+sat_list[i],": ",round(100*(j+1)/len(Features)),'%   ', end='')
            print("Done")
        
        with open(os.path.join(filepath, sitename + '_metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
    return metadata


        
def image_retrieval(inputs):
    """
    Grab image metadata from Google Earth Engine and compile into list of image names.
    FM 2022

    Parameters
    ----------
    inputs : dict
        Dictionary of input parameters from user.

    Returns
    -------
    Sat : list
        List of satellite image names per platform.

    """
    print('retrieving image metadata...')
    point = ee.Geometry.Point(inputs['polygon'][0][0])
    
    Sat = []

    if 'cloud_thresh' in inputs.keys():
        cloud_thresh = int(inputs['cloud_thresh']*100)
    else:
        cloud_thresh = 90
        
    if 'L5' in inputs['sat_list']:
        Landsat5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_TOA").filterBounds(point).filterDate(inputs['dates'][0], inputs['dates'][1])
        Sat.append(Landsat5)
    if 'L7' in inputs['sat_list']:
        Landsat7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_TOA').filterBounds(point).filterDate(inputs['dates'][0], inputs['dates'][1]).filter(ee.Filter.lt('CLOUD_COVER', cloud_thresh))
        Sat.append(Landsat7)
    if 'L8' in inputs['sat_list']:
        Landsat8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA').filterBounds(point).filterDate(inputs['dates'][0], inputs['dates'][1]).filter(ee.Filter.lt('CLOUD_COVER', cloud_thresh))
        Sat.append(Landsat8)
    if 'L9' in inputs['sat_list']:
        Landsat9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_TOA').filterBounds(point).filterDate(inputs['dates'][0], inputs['dates'][1]).filter(ee.Filter.lt('CLOUD_COVER', cloud_thresh))
        Sat.append(Landsat9)
    if 'S2' in inputs['sat_list']:
        Sentinel2 = ee.ImageCollection("COPERNICUS/S2").filterBounds(point).filterDate(inputs['dates'][0], inputs['dates'][1]).filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloud_thresh))
        Sat.append(Sentinel2)
    if 'PSScene4Band' in inputs['sat_list']:
        Sat = LocalImageRetrieval(inputs)
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

    # for i in range(len(Sat[0])):
    for i in range(1):    
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



def PlanetMetadata(sat_list, Sat, filepath_data, sitename):
    """
    Compile Planet metadata together to create a collection of image properties. 
    FM 2022

    Parameters
    ----------
    sat_list : list
        Strings of platform names requested.
    Sat : dict
        Dictionary of returned image names based on user inputs.
    filepath_data : TYPE
        DESCRIPTION.
    sitename : str
        Name of site.

    Returns
    -------
    metadata : dict
        Dictionary of image metadata.

    """
    
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


def ResumeVeglines(filepath_data, filepath_out, sitename, metadata):
    """
    Check if VedgeSat/CoastSat run alredy exists partially (e.g. if a previous
    run failed and only some of the satellite platforms have been processed). 
    If a partial run does exist (i.e. if an output.pkl file has been saved),
    load it in and only run the process from start of the platforms not yet done.
    If no output file exists, initialise the output and skipped dictionaries.
    
    This is to save time if a run fails on a specific image or for some unsolved
    reason, so users don't need to re-process images that were already successful.
    
    FM Oct 2024

    Parameters
    ----------
    filepath_data : str
        Path to 'Data' folder.
    filepath_out : str
        Path to site folder to save outputs to.
    sitename : str
        Name of site of interest.
    metadata : dict
        Dictionary of sat image filenames, georeferencing info, EPSGs and dates of capture.

    Returns
    -------
    satnames : list
        List of satellite platform names, either full list from metadata or 
        shortened to just the platforms not processed yet.
    output : dict
        Dictionary of extracted veg edges and associated info with each.
    output_latlon : dict
        Dictionary of extracted veg edges and associated info with each (in WGS84).
    output_proj : dict
        Dictionary of extracted veg edges and associated info with each (in chosen CRS).
    skipped : dict
        Global dictionary storing the reasons for each image that fails or is skipped.

    """
    
    # If output already exists, load it in
    if os.path.isfile(os.path.join(filepath_out, sitename + '_output.pkl')):
        SiteFilepath = os.path.join(filepath_data, sitename)
        with open(os.path.join(SiteFilepath, sitename + '_output.pkl'), 'rb') as f:
            output = pickle.load(f)
        with open(os.path.join(SiteFilepath, sitename + '_output_latlon.pkl'), 'rb') as f:
            output_latlon = pickle.load(f)
        with open(os.path.join(SiteFilepath, sitename + '_output_proj.pkl'), 'rb') as f:
            output_proj = pickle.load(f)
        # If platform has already been processed and saved to output,
        # redefine satnames from ones which haven't been done yet
        satnames = sorted(set(metadata.keys()) ^ set(output_proj.keys()))
        satnames_done = sorted(set(metadata.keys()) & set(output_proj.keys()))
        
        # Load in existing counter for run success rates
        with open(os.path.join(SiteFilepath, sitename + '_skip_stats.pkl'), 'rb') as f:
            skipped = pickle.load(f)
        
        print(f"Already found outputs for {', '.join([str(i) for i in satnames_done])}")
        print(f"starting from {satnames[0]}")
    
    # If not, start new run
    else:
        # use full metadata for satnames
        satnames = metadata.keys()
        # initialise output structure
        output = dict([])
        output_latlon = dict([])
        output_proj = dict([])
        
        # Initialise counter for run success rates
        skipped = {
            'empty_poor': [],
            'missing_mask':[],
            'cloudy': [],
            'no_classes': [],
            'no_contours': []}
        
    return satnames, output, output_latlon, output_proj, skipped


def ReadOutput(inputs):
    """
    Read in the saved output dictionary from a run of VedgeSat/CoastSat, and 
    convert it from the format:
        {'L5':{}, 'L7':{}, etc.}
    to the easier to use and date-sorted format:
        {'dates':[], 'satname':[], 'veglines':[], etc.}
    Should only run with new runs of VedgeSat, old output files will not be 
    affected and should be loaded in using the old method:
        with open(os.path.join(SiteFilepath, inputs['sitename'] + '_output.pkl'), 'rb') as f:
            output = pickle.load(f)
        
    FM Oct 2024

    Parameters
    ----------
    inputs : dict
        Dictionary of user requirements for VedgeSat/CoastSat run.

    Returns
    -------
    output : dict
        Dictionary of extracted veg edges and associated info with each.
    output_latlon : dict
        Dictionary of extracted veg edges and associated info with each (in WGS84).
    output_proj : dict
        Dictionary of extracted veg edges and associated info with each (in chosen CRS).

    """
    SiteFilepath = os.path.join(inputs['filepath'], inputs['sitename'])
    
    with open(os.path.join(SiteFilepath, inputs['sitename'] + '_output.pkl'), 'rb') as f:
        output = pickle.load(f)
    with open(os.path.join(SiteFilepath, inputs['sitename'] + '_output_latlon.pkl'), 'rb') as f:
        output_latlon = pickle.load(f)
    with open(os.path.join(SiteFilepath, inputs['sitename'] + '_output_proj.pkl'), 'rb') as f:
        output_proj = pickle.load(f)

    # output saved as dict{'satname1':{}, 'satname2':{}, etc.}
    # Need to convert it to dict{'dates':[], 'satname':[], 'veglines':[], etc.}
    output = merge_output(output)
    output_latlon = merge_output(output_latlon)
    output_proj = merge_output(output_proj)

    return output, output_latlon, output_proj


def SaveShapefiles(output, name_prefix, sitename, epsg):
    """
    Save shapefiles of coastal change indicator lines generated.
    FM Apr 2022

    Parameters
    ----------
    output : dict
        Dictionary of data generated by VedgeSat/CoastSat.
    name_prefix : str
        Base path to directory where shapefiles will be stored.
    sitename : str
        Name of site.
    epsg : int
        Projection to save shapefiles in.

    Returns
    -------
    None.

    """
    
    # for shores stored as array of coords; export as mulitpoint
    if type(output['veglines'][0]) == np.ndarray:
        # map to multipoint
        output_geom = gpd.GeoSeries(map(MultiPoint,output['veglines']))
        # create geodataframe with geometry from output multipoints
        outputGDF = gpd.GeoDataFrame(output, crs='EPSG:'+str(epsg), geometry=output_geom)
        # drop duplicate shorelines column
        outputsGDF = outputGDF.drop('veglines', axis=1)
    else:    
        DFlist = []
        for i in range(len(output['veglines'])): # for each image + associated metadata
            # create geodataframe of individual features from each geoseries (i.e. feature collection)
            outputGDF = gpd.GeoDataFrame(geometry=output['veglines'][i])
            for key in output.keys(): # for each column
                # add column to geodataframe with repeated metadata
                outputGDF[key] = output[key][i]
            # add formatted geodataframe to list of all geodataframes
            DFlist.append(outputGDF)
            # concatenate to one GDF with individual lines exploded out
            outputsGDF = gpd.GeoDataFrame( pd.concat( DFlist, ignore_index=True), crs=DFlist[0].crs)
            outputsGDF = outputsGDF.drop('veglines', axis=1)
            
    outputsGDF.to_file(os.path.join(name_prefix, sitename + '_' + str(min(output['dates'])) + '_' + str(max(output['dates'])) + '_veglines.shp'))
    
    
    return

def SaveConvShapefiles(outputOG, name_prefix, sitename, epsg): #, shpFileName):
    """
    Save converted shapefiles with multiple line features per date.
    FM Apr 2022
    
    Parameter sensitivity testing - change only outputsGDF.tofile and input args
    CM Sept 2023 
    
    Parameters
    ----------
    outputOG : dict
        Dictionary of data generated by VedgeSat/CoastSat.
    name_prefix : str
        Base path to directory where shapefiles will be stored.
    sitename : str
        Name of site.
    epsg : int
        Projection to save shapefiles in.

    Returns
    -------
    None.

    """

    
    output = outputOG.copy()
    # for shores stored as array of coords; export as mulitpoint
    if type(output['veglines'][0]) == np.ndarray:
        # map to multipoint
        output_geom = gpd.GeoSeries(map(MultiPoint,output['veglines']))
        # create geodataframe with geometry from output multipoints
        outputGDF = gpd.GeoDataFrame(output, crs='EPSG:'+str(epsg), geometry=output_geom)
        # drop duplicate shorelines column
        outputsGDF = outputGDF.drop('veglines', axis=1)
    else:    
        DFlist = []
        for i in range(len(output['veglines'])): # for each image + associated metadata
            # create geodataframe of individual features from each geoseries (i.e. feature collection)
            convlines = output['veglines'][i].to_crs(str(epsg))
            outputGDF = gpd.GeoDataFrame(geometry=convlines, crs=str(epsg))
            if 'waterlines' in output.keys():
                del output['waterlines']
            for key in output.keys(): # for each column
                # add column to geodataframe with repeated metadata
                outputGDF[key] = output[key][i]
            # add formatted geodataframe to list of all geodataframes
            DFlist.append(outputGDF)
            # concatenate to one GDF with individual lines exploded out
            outputsGDF = gpd.GeoDataFrame( pd.concat( DFlist, ignore_index=True), crs=str(epsg))
            outputsGDF = outputsGDF.drop('veglines', axis=1)
            
    outputsGDF.to_file(os.path.join(name_prefix, sitename + '_' + str(min(output['dates'])) + '_' + str(max(output['dates'])) + '_veglines.shp'))
    #outputsGDF.to_file(os.path.join(name_prefix, shpFileName))
    
    
    return

def SaveConvShapefiles_Water(outputOG, name_prefix, sitename, epsg):

    """
    Save converted shapefiles with multiple water line features per date.
    FM Apr 2022
    
    Parameters
    ----------
    outputOG : dict
        Dictionary of data generated by VedgeSat/CoastSat.
    name_prefix : str
        Base path to directory where shapefiles will be stored.
    sitename : str
        Name of site.
    epsg : int
        Projection to save shapefiles in.

    Returns
    -------
    None.

    """
    output = outputOG.copy()
    # for shores stored as array of coords; export as mulitpoint
    if type(output['waterlines'][0]) == np.ndarray:
        # map to multipoint
        output_geom = gpd.GeoSeries(map(MultiPoint,output['waterlines']))
        # create geodataframe with geometry from output multipoints
        outputGDF = gpd.GeoDataFrame(output, crs='EPSG:'+str(epsg), geometry=output_geom)
        # drop duplicate shorelines column
        outputsGDF = outputGDF.drop('waterlines', axis=1)
    else:    
        DFlist = []
        for i in range(len(output['waterlines'])): # for each image + associated metadata
            # create geodataframe of individual features from each geoseries (i.e. feature collection)
            convlines = output['waterlines'][i].to_crs(str(epsg))
            outputGDF = gpd.GeoDataFrame(geometry=convlines, crs=str(epsg))
            if 'veglines' in output.keys():
                del output['veglines']
            for key in output.keys(): # for each column
                # add column to geodataframe with repeated metadata
                outputGDF[key] = output[key][i]
            # add formatted geodataframe to list of all geodataframes
            DFlist.append(outputGDF)
            # concatenate to one GDF with individual lines exploded out
            outputsGDF = gpd.GeoDataFrame( pd.concat( DFlist, ignore_index=True), crs=str(epsg))
            outputsGDF = outputsGDF.drop('waterlines', axis=1)
            
    outputsGDF.to_file(os.path.join(name_prefix, sitename + '_' + str(min(output['dates'])) + '_' + str(max(output['dates'])) + '_waterlines.shp'))
    
    
    return

def Separate_TimeSeries_year(cross_distance, output, key):
    """
    IN DEVELOPMENT
    Organise output cross-shore intersections of satellite-derived 
    veg edges and waterlines into a mean value per year.
    FM June 2024

    Parameters
    ----------
    cross_distance : dict
        Lists containing cross-shore intersection values along transects.
    output : dict
        Dictionary of data generated by VedgeSat/CoastSat.
    key : str
        Specific column from cross-shore intersection data on transects.

    Returns
    -------
    Date_Organised : list
        List of lists with each sublist containing grouped values from the input sat dates.
    Month_Organised : list
        List of lists with each sublist containing months corresponding to the years from the input sat dates.
    Distance_Organised : list
        Cross-shore intersection values along transects, sorted into years.
    DateArr : array
        Unique years in timeseries.
    DistanceAvgArr : array
        Mean of adjusted cross-shore values for each year group.

    """
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
    """
    IN DEVELOPMENT
    Organise output cross-shore intersections of satellite-derived 
    veg edges and waterlines into a mean value per month.
    FM June 2024

    Parameters
    ----------
    cross_distance : dict
        Lists containing cross-shore intersection values along transects.
    output : dict
        Dictionary of data generated by VedgeSat/CoastSat.
    key : str
        Specific column from cross-shore intersection data on transects.

    Returns
    -------
    Date_Organised : list
        List of lists with each sublist containing grouped values from the input sat dates.
    Month_Organised : list
        List of lists with each sublist containing months corresponding to the months from the input sat dates.
    Distance_Organised : list
        Cross-shore intersection values along transects, sorted into months.
    DateArr : array
        Unique years in timeseries.
    DistanceAvgArr : array
        Mean of adjusted cross-shore values for each month group.

    """
    
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
    Read in reference shapefile marking shoreline and reformat for VedgeSat 
    (equally space out vertices and extract coords).
    FM 2022

    Parameters
    ----------
    referenceLineShp : str
        Filepath to refline shapefile.
    settings : dict
        Dictionary of user-defined settings used for the veg edge/waterline extraction.

    Returns
    -------
    referenceLine : 
        reference line coordinate array
    ref_epsg : int
        reference line EPSG ID

    """

    referenceLineDF = gpd.read_file(referenceLineShp)
    referenceLineDF.to_crs(epsg=4326, inplace=True) # Convert whatever CRS ref line is in to WGS84 to start off with

    # Add in here a fn to merge multilinestrings to one contiguous linestring
    # merged = linemerge([lineseg for lineseg in referenceLineDF.geometry])
    # referenceLineDF = gpd.GeoDataFrame(geometry=[merged], crs=4326)
    
    if len(referenceLineDF.geometry) > 1:
        print('More than one line object found in reference line; using just the first line found.')
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


def GetImageEPSG(inputs, metadata):
    """
    Extract the EPSG coordinate projection code from metadata info.
    Note: Landsat on GEE lists all EPSGs as N Hemisphere version (even if in S).
    FM Mar 2024

    Parameters
    ----------
    inputs : dict
        Dictionary of user requirements for VedgeSat/CoastSat run.
    metadata : dict
        Dictionary of GEE metadata (platform, georefs, filenames).

    Returns
    -------
    image_epsg : int
        Numerical EPSG code representing coordinate projection system.

    """
    # Prioritise Sentinel2 as this has southern hemisphere option
    if 'S2' in inputs['sat_list']:
        image_epsg = metadata['S2']['epsg'][0]
    else:
        image_epsg = metadata[inputs['sat_list'][0]]['epsg'][0]
    return image_epsg


def NearestDates(surveys,metadata,sat_list):
    """
    
    Print image dates from full sat metadata that fall nearest to validation dates.
    FM Sept 2022
    
    Parameters
    ----------
    surveys : GeoDataFrame
        Shapefile of validation lines read in using geopandas
    metadata : dict
        Satellite image metadata
    sat_list : list
        List of desired satellite platforms as strings
    
    Returns
    -------
    nearestdates : dict
        Dictionary of nearest image dates to requested dates
    nearestIDs : dict
        Dictionary of indexes of nearest image dates to requested dates
        
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
    newverts = [referenceLineString.interpolate(dist) for dist in vertexdists] + [referenceLineString.boundary.geoms[1]] # ERROR - not subscriptable (shapely syntax change)
    newreferenceLineString = LineString(newverts)
    newreferenceLine = np.asarray(newreferenceLineString.coords) # ERROR - new syntax to pass shapely linestring coords to numpy structure
    
    return newreferenceLine


def AOI(lonmin, lonmax, latmin, latmax, sitename):
    '''
    Creates area of interest bounding box from provided latitudes and longitudes, and
    checks to see if order is correct and size isn't too large for GEE requests.
    FM Jun 2022

    Parameters
    ----------
    lonmin : float
        Minimum longitude of bounding box.
    lonmax : float
        Maximum longitude of bounding box.
    latmin : float
        Minimum latitude of bounding box.
    latmax : float
        Maximum latitude of bounding box.

    Returns
    -------
    polygon : list of lists
        List of pairs of coordinates [x,y] representing vertices of bounding box.
    point : ee.Geometry.Point
        Corner point of bounding box.

    '''
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
    # convert crs of geodataframe to UTM to get metre measurements (not degrees)
    _, projstr = FindUTM(latmin, lonmin)
    BBoxGDF = BBoxGDF.to_crs(crs=projstr)
    # Check if AOI could exceed the 262144 (512x512) pixel limit on ee requests
    if (int(BBoxGDF.area)/(10*10))>262144:
        print('Warning: your bounding box is too big for GEE requests (%s pixels too big)' % int((BBoxGDF.area/(10*10))-262144))
    
    BBoxGDF['Area'] = BBoxGDF.area/(10*10)
    mapcentrelon = lonmin + ((lonmax - lonmin)/2)
    mapcentrelat = latmin + ((latmax - latmin)/2)
    
    m = folium.Map(location=[mapcentrelat, mapcentrelon], zoom_start = 13, tiles = 'openstreetmap')
    folium.TileLayer('MapQuest Open Aerial', attr='<a href=https://www.mapquest.com/>MapQuest</a>').add_to(m)
    folium.LayerControl().add_to(m)
    
    # gj = folium.GeoJson(geo_data=BBoxGDF['geometry'], data=BBoxGDF['Area']).add_to(m)
    gj = folium.Choropleth(geo_data=BBoxGDF['geometry'], name='AOI', data=BBoxGDF['Area'],
                            columns=['Area'], fill_color='YlGn',
                            fill_opacity=0.5)
    gj.add_to(m)
    ct = folium.Marker(location=[BBoxGDF.centroid.x,BBoxGDF.centroid.y],
                  popup=str(round(float(BBoxGDF.to_crs('epsg:32630').area)))+' sq m'
                  )
    ct.add_to(m)
    m.save("./Data/"+sitename+"/AOImap.html")
    
    # Export as polygon and ee point for use in clipping satellite image requests
    polygon = [[[lonmin, latmin],[lonmax, latmin],[lonmin, latmax],[lonmax, latmax]]]
    point = ee.Geometry.Point(polygon[0][0]) 
    
    return polygon, point


def AOIfromLeaflet(polygon, point, sitename):
    '''
    Creates area of interest bounding box from provided Leaflet AOI, and
    checks to see if order is correct and size isn't too large for GEE requests.
    Based on LR-F July 2021
    FM Jun 2022

    Parameters
    ----------
    lonmin : float
        Minimum longitude of bounding box.
    lonmax : float
        Maximum longitude of bounding box.
    latmin : float
        Minimum latitude of bounding box.
    latmax : float
        Maximum latitude of bounding box.

    Returns
    -------
    polygon : list of lists
        List of pairs of coordinates [x,y] representing vertices of bounding box.
    point : ee.Geometry.Point
        Corner point of bounding box.

    '''
    lonmin = polygon[0][0][0]
    lonmax = polygon[0][1][0]
    latmin = polygon[0][0][1]
    latmax = polygon[0][2][1]
    BBox = Polygon([[lonmin, latmin],[lonmax,latmin],[lonmax,latmax],[lonmin, latmax]])
    # Convert bounding box to a geodataframe
    BBoxGDF = gpd.GeoDataFrame(geometry=[BBox], crs = {'init':'epsg:4326'})
    # convert crs of geodataframe to UTM to get metre measurements (not degrees)
    _, projstr = FindUTM(latmin, lonmin)
    BBoxGDF = BBoxGDF.to_crs(crs=projstr)
    # Check if AOI could exceed the 262144 (512x512) pixel limit on ee requests
    if (int(BBoxGDF.area)/(10*10))>262144:
        print('Warning: your bounding box is too big for GEE requests (%s pixels too big)' % int((BBoxGDF.area/(10*10))-262144))
        response = input('Do you want us to try shrinking your bounding box? Type "yes" to run shrinking, or "no" to redraw the box yourself.')
    
        if response == 'no':
            return
        else:
            def ShrinkRect(rectangle, scale_factor=0.9):
                """
                Shrink edges of rectangle until AOI fits in BBox limit
                """
                # Get the current center of the rectangle
                center = rectangle.centroid
                # Scale the rectangle about its center
                shrunken_rectangle = affinity.scale(rectangle, xfact=scale_factor, yfact=scale_factor, origin=center)
                return shrunken_rectangle
        
            scale_factor = 0.9
            tolerance = 1e-9
            area_limit = 2000
            
            while True:
                current_area = BBoxGDF.geometry.area.iloc[0]
                if current_area <= area_limit or current_area <= tolerance:
                    break
            
                # Shrink the geometry
                shrunken_rectangle = ShrinkRect(BBoxGDF.geometry.iloc[0], scale_factor=scale_factor)
            
                # Check if the new geometry is valid and non-degenerate
                if shrunken_rectangle.is_empty or not shrunken_rectangle.is_valid:
                    print("Warning: Geometry became invalid. Stopping the loop.")
                    break
            
                BBoxGDF['geometry'] = gpd.GeoSeries([shrunken_rectangle])
            
                # Reduce the scale_factor for finer control over the shrinking process
                scale_factor *= 0.95
            
            # Output the final area
            print(f"Final area: {round(BBoxGDF.geometry.area.iloc[0],2)} m")
            lonmin = float(BBoxGDF.bounds.minx)
            lonmax = float(BBoxGDF.bounds.maxx)
            latmin = float(BBoxGDF.bounds.miny)
            latmax = float(BBoxGDF.bounds.maxy)
    
    # Export as polygon and ee point for use in clipping satellite image requests
    polygon = [[[lonmin, latmin],[lonmax, latmin],[lonmin, latmax],[lonmax, latmax]]]
    point = ee.Geometry.Point(polygon[0][0]) 
    
    return polygon, point, BBoxGDF


def AOIfromLine(referenceLinePath, max_dist_ref, sitename):
    """
    Creates area of interest bounding box from provided reference shoreline, and
    checks to see if order is correct and size isn't too large for GEE requests.
    FM Oct 2023

    Parameters
    ----------
    referenceLinePath : str
        Path to reference shoreline stored as shapefile.
    max_dist_ref : int
        Maximum distance in metres to buffer reference shoreline by (needed to add buffer to AOI).
    sitename : str
        Name of site.

    Returns
    -------
    None.

    """
    
    # Create bounding box from refline (with small buffer) and convert it to a geodataframe
    referenceLineDF = gpd.read_file(referenceLinePath)
    if len(referenceLineDF.geometry) > 1:
        print('More than one line object found in reference line; make sure you only have one continuous line!\n')
        return
    # convert crs of geodataframe to UTM to get metre measurements (not degrees)
    _, projstr = FindUTM(float(referenceLineDF.bounds.miny), float(referenceLineDF.bounds.minx))
    referenceLineDF.to_crs(crs=projstr, inplace=True)
    
    xmin, xmax, ymin, ymax = [float(referenceLineDF.bounds.minx-max_dist_ref),
                                      float(referenceLineDF.bounds.maxx+max_dist_ref),
                                      float(referenceLineDF.bounds.miny-max_dist_ref),
                                      float(referenceLineDF.bounds.maxy+max_dist_ref)]

    BBox = Polygon([[xmin, ymin],
                    [xmax,ymin],
                    [xmax,ymax],
                    [xmin, ymax]])
    
    BBoxGDF = gpd.GeoDataFrame(geometry=[BBox], crs=referenceLineDF.crs)
    
    lonmin, latmin, lonmax, latmax = [BBoxGDF.to_crs(epsg=4326).bounds.iloc[0][i] for i in range(4)]

    # Check if AOI could exceed the 262144 (512x512) pixel limit on ee requests
    if (int(BBoxGDF.area)/(10*10))>262144:
        print('Warning: your bounding box is too big for Sentinel2 (%s pixels too big)' % int((BBoxGDF.area/(10*10))-262144))
    
    BBoxGDF['Area'] = BBoxGDF.area/(10*10)
    mapcentrelon = lonmin + ((lonmax - lonmin)/2)
    mapcentrelat = latmin + ((latmax - latmin)/2)
    
    m = folium.Map(location=[mapcentrelat, mapcentrelon], zoom_start = 10, tiles = 'openstreetmap')
    folium.TileLayer('MapQuest Open Aerial', attr='<a href=https://www.mapquest.com/>MapQuest</a>').add_to(m)
    folium.LayerControl().add_to(m)
    
    # gj = folium.GeoJson(geo_data=BBoxGDF['geometry'], data=BBoxGDF['Area']).add_to(m)
    gj = folium.Choropleth(geo_data=BBoxGDF['geometry'], name='AOI', data=BBoxGDF['Area'],
                            columns=['Area'], fill_color='YlGn',
                            fill_opacity=0.5).add_to(m)
    folium.Marker(location=[BBoxGDF.centroid.x,BBoxGDF.centroid.y],
                  popup=str(round(float(BBoxGDF.to_crs('epsg:32630').area)))+' sq m'
                  ).add_to(m)
    m.save("./Data/"+sitename+"/AOImap.html")
    
    # Export as polygon and ee point for use in clipping satellite image requests
    polygon = [[[lonmin, latmin],[lonmax, latmin],[lonmin, latmax],[lonmax, latmax]]]
    point = ee.Geometry.Point(polygon[0][0]) 
    
    return polygon, point, lonmin, lonmax, latmin, latmax


def FindUTM(lat, lon):
    """
    Find UTM zone that a provided lat long pair sit in.
    FM Feb 2024

    Parameters
    ----------
    lat : float
        Latitude.
    lon : float
        Longitude.

    Returns
    -------
    projstr : str
        proj.4 string of UTM zone to be used in coordinate transformations (pyproj).

    """
    utmNum = utm.from_latlon(lat, lon)[2]
    if lat < 0:
        hemi = ' +south '
        EPSG = 32700 + utmNum
    else:
        hemi = ' '
        EPSG = 32600 + utmNum
    projstr = '+proj=utm +zone='+str(utmNum)+hemi+'+ellps=WGS84 +datum=WGS84 +units=m +no_defs'
    
    return EPSG, projstr


def GStoArr(shoreline):
    """
    Shoreline coords to array.
    FM Aug 2022
    
    Parameters
    ----------
    shoreline : GeoSeries
        Output line extracted from satellite image (as pandas GeoSeries).

    Returns
    -------
    shorelineArr : array
        Output line extracted from satellite image (as numpy array).
    
    
    """

    shorelineList = [np.array(line.coords) for line in shoreline.geometry]
    shorelineArrList = [coord for line in shorelineList for coord in line]
    shorelineArr = np.array(shorelineArrList)
    return shorelineArr

def ArrtoGS(refline,georef):
    """
    Shoreline array to coords
    FM Sept 2022

    Parameters
    ----------
    refline : array
        Output line extracted from satellite image (as numpy array).
    georef : list
        Georeferencing info [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale].

    Returns
    -------
    refGS : GeoSeries
        Output line extracted from satellite image (as pandas GeoSeries).

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


def ValCoreg(satname, georef):
    """
    Apply platform-specific mean coregistration values to each image.
    FM Jan 2024

    Parameters
    ----------
    satname : str
        Name of satellite platform.
    georef : array
        Georeferencing/affine transformation values.

    Returns
    -------
    georef : array
        Coregistered affine transformation values.

    """
    if satname == 'L5':
        georef[0] = georef[0] + (-4.5)
        georef[3] = georef[3] + (42.3)
    if satname == 'L7':
        georef[0] = georef[0] + (13.7)
        georef[3] = georef[3] + (37.4)
    if satname == 'L8':
        georef[0] = georef[0] + (13.7)
        georef[3] = georef[3] + (37.4)
    if satname == 'L9':
        georef[0] = georef[0] + (13.7)
        georef[3] = georef[3] + (37.4)
    if satname == 'S2':
        georef[0] = georef[0] + (-7.5)
        georef[3] = georef[3] + (15.5)
    # else: # Planet or local files
    #     georef[0] = georef[0] + (3.8)
    #     georef[3] = georef[3] + (6.5)
    else: # Planet or local files
        georef[0] = georef[0]
        georef[3] = georef[3]

    return georef



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
    
    # # if difference is longer than 5 months, no match exists  
    if abs((target - nearestDate).days) > 153: 
        return False
    else:
        return nearestDate


def FindWPThresh(int_veg, int_nonveg):
    """
    Find threshold normalised difference value using Weighted Peaks, based on 
    a weighting between the two probability density function peaks of 0.2:0.8.
    FM Sept 2023
    
    
    Parameters
    ----------
    int_veg : array of float64
        Array of normalised difference pixel values for the vegetation class.
    int_nonveg : array of float64
        Array of normalised difference pixel values for the 'other' class.

    Returns
    -------
    t_ndi : float64
        Threshold value with which to extract veg contour from normalised diff image.

    """
    
    # Find the peaks of veg and nonveg classes using KDE
    bins = np.arange(-1, 1, 0.01) # start, stop, bin width
    peaks = []
    for i, intdata in enumerate([int_veg, int_nonveg]):
        model = sklearn.neighbors.KernelDensity(bandwidth=0.01, kernel='gaussian')
        sample = intdata.reshape((len(intdata), 1))
        # fill nan values using pandas interpolation
        sample = np.array(pd.DataFrame(sample).interpolate(limit_direction='both'))
        model.fit(sample)
        # sample probabilities for a range of outcomes
        values = np.asarray([value for value in bins])
        values = values.reshape((len(values), 1))
        # calculate probability fns 
        probabilities = model.score_samples(values)
        probabilities = np.exp(probabilities)
        
        if i == 0: # class with weaker signal
            # take value of band index where probability is max
            peaks.append(float(values[list(probabilities).index(np.nanmax(probabilities))]))
        else:
            prom, _ = scipy.signal.find_peaks(probabilities, prominence=0.5)
            if len(prom) == 0: # for marshland where no peak above NDVI = 0 exists
                print('no peak NDVI initially found, decreasing prominence to find peak...')
                promlimit = 0.5
                # decrease prominence til peak is found
                while len(prom) == 0:
                    prom, _ = scipy.signal.find_peaks(probabilities, prominence=promlimit)
                    promlimit -= 0.05 
            if len(prom) > 1: # if more than one peak 
                if (bins[prom] < 0).any(): # if any of the peaks have an NDVI less than 0
                    # # always take peak closest to 0 (corresponds to bare land/sand in veg classification)
                    # peaks.append(min(bins[prom], key=abs))
                    # always take maximum peak (representing bare land)
                    peaks.append(max(bins[prom]))
                else:
                    peaks.append(bins[prom[0]])
            else:
                peaks.append(bins[prom[0]])
                
            
    # Calculate index value using weighted peaks
    t_ndi = float((0.2*peaks[0]) + (0.8*peaks[1]))
    
    return t_ndi, peaks



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


def TZValuesPeak(int_veg, int_nonveg):
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


def TZimage(im_ndvi,TZbuffer):
    """
    Use vegetation transition zone values to create a boolean raster from an NDVI raster.
    FM Sept 2023

    Parameters
    ----------
    im_ndvi : array
        2D array of NDVI values calculated from satellite image.
    TZbuffer : list
        Bounding values between which to designate as 'vegetation transition zone'.

    Returns
    -------
    im_TZ : array
        2D boolean array of same shape as NDVI image, with transition zone values = 1.

    """
    
    im_TZ = im_ndvi.copy()
    
    for i in range(len(im_ndvi[:,0])):
        for j in range(len(im_ndvi[0,:])):
            if im_ndvi[i,j] > TZbuffer[0] and im_ndvi[i,j] < TZbuffer[1]:
                im_TZ[i,j] = 1.0
            else:
                im_TZ[i,j] = np.nan
        
    return im_TZ
    

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
    
    minval = np.percentile(int_veg,0.5)
    maxval = np.percentile(int_veg,10)
    
    return [minval,maxval]

def QuantifyErrors(sitename, SatGDF, DatesCol,ValidInterGDF,TransectIDs):
    """
    Quantify positional uncertainty (Root Mean Squared Error, Mean Absolute Error)
    between satellite-derived lines and validation lines in cross-shore direction.
    FM 2022

    Parameters
    ----------
    sitename : str
        Name of site.
    SatGDF : GeoDataFrame
        Shapefile of satellite-derived veg lines, read in as pandas GeoDataFrame.
    DatesCol : str
        Name of field in shapefile to read dates from.
    ValidInterGDF : GeoDataFrame
        Cross-shore transects with intersection info from satellite-derived and validation lines.
    TransectIDs : list
        List of bounding transect IDs across which to calculate errors.

    Returns
    -------
    None.

    """
    
    filepath = os.path.join(os.getcwd(), 'Data', sitename, 'validation')
    if os.path.isdir(filepath) is False:
        os.mkdir(filepath)
        
    errordata = []
    errordates = []
    Sdates = SatGDF[DatesCol].unique()
    
    for Sdate in Sdates:
        valsatdist = []
        # for each transect in given range
        for Tr in range(TransectIDs[0],TransectIDs[1]): 
            if Tr > len(ValidInterGDF['dates']): # for when transect values extend beyond what transects exist
                print("check your chosen transect values!")
                return
            if Sdate in ValidInterGDF['dates'].iloc[Tr]:
                DateIndex = (ValidInterGDF['dates'].iloc[Tr].index(Sdate))
                # rare occasion where transect intersects valid line but NOT sat line (i.e. no distance between them)
                if ValidInterGDF['valsatdist'].iloc[Tr] != []:
                    valsatdist.append(ValidInterGDF['valsatdist'].iloc[Tr][DateIndex])
                else:
                    continue
            else:
                continue
        # due to way dates are used, some transects might be missing validation dates so collection will be empty
        if valsatdist != []: 
            errordata.append(valsatdist)
            errordates.append(Sdate)
    # sort both dates and list of values by date
    if len(errordates) > 1:
        errordatesrt, errorsrt = [list(d) for d in zip(*sorted(zip(errordates, errordata), key=lambda x: x[0]))]
    else:
        errordatesrt = errordates
        errorsrt = errordata
    df = pd.DataFrame(errorsrt)
    df = df.transpose()
    df.columns = errordatesrt
    
    errordict = {'Date':[],'Count':[],'MAE':[],'RMSE':[],'CountSub3m':[],'CountSub10m':[],'CountSub15m':[]}
    
    print('Transects %s to %s:' % (TransectIDs[0],TransectIDs[1]))
    totald = []
    for date in df.columns:
        d = df[date]
        for i,datum in enumerate(d):
            totald.append(datum)
        mse_f = np.mean(d**2)
        mae_f = np.mean(abs(d))
        rmse_f = np.sqrt(mse_f)
        sub3  = d.between(-3,3).sum()
        sub10 = d.between(-10,10).sum()
        sub15 = d.between(-15,15).sum()
        # r2_f = 1-(sum(d**2)/sum((y-np.mean(y))**2))
        print('For sat date %s:' % date)
        print('Count: %s' % d.count())
        print("MAE:",mae_f)
        # print("MSE:", mse_f)
        print("RMSE:", rmse_f)
        # print("R-Squared:", r2)
        print('Sub 3m-pixel Tr percent:',round(sub3/d.count()*100,1), '%')
        print('Sub 10m-pixel Tr percent:',round(sub10/d.count()*100,1), '%')
        print('Sub 15m-pixel Tr percent:',round(sub15/d.count()*100,1), '%')
        if d.count() != 0:
            errordict['Date'].append(date)
            errordict['Count'].append(d.count())
            errordict['MAE'].append(mae_f)
            errordict['RMSE'].append(rmse_f)
            errordict['CountSub3m'].append(sub3)
            errordict['CountSub10m'].append(sub10)
            errordict['CountSub15m'].append(sub15)

            
    totald = np.array(totald)
    mse = np.mean(np.power(totald[~np.isnan(totald)], 2))
    mae = np.mean(abs(totald[~np.isnan(totald)]))
    rmse = np.sqrt(mse)
    sub3  = np.logical_and(totald>=-3,totald<=3).sum()
    sub10 = np.logical_and(totald>=-10,totald<=10).sum()
    sub15 = np.logical_and(totald>=-15,totald<=15).sum()
    print('TOTAL')
    print('Count: %s' % len(totald[~np.isnan(totald)]))
    print("MAE:",mae)
    # print("MSE:", mse)
    print("RMSE:", rmse)
    print('Sub 3m-pixel Tr percent:',round(sub3/len(totald[~np.isnan(totald)])*100,1),'%')
    print('Sub 10m-pixel Tr percent:',round(sub10/len(totald[~np.isnan(totald)])*100,1),'%')
    print('Sub 15m-pixel Tr percent:',round(sub15/len(totald[~np.isnan(totald)])*100,1),'%')
    errordict['Date'].append('Total')
    errordict['Count'].append(len(totald[~np.isnan(totald)]))
    errordict['MAE'].append(mae)
    errordict['RMSE'].append(rmse)
    errordict['CountSub3m'].append(sub3)
    errordict['CountSub10m'].append(sub10)
    errordict['CountSub15m'].append(sub15)
    
    errordf = pd.DataFrame(errordict)
    savepath = os.path.join(filepath, sitename+'_Errors_Transects'+str(TransectIDs[0])+'to'+str(TransectIDs[1])+'.csv')
    print("Error stats saved to "+savepath)
    errordf.to_csv(savepath, index=False)
    
    
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


def ComputeTides(settings,tidepath,daterange,tidelatlon):
    """
    Function to compute water elevations from tidal consituents of global tide model FES2014. 
    Uses pyfes package to compute tides at specific lat long and for specified time period.
    FM Nov 2022

    Parameters
    ----------
    settings : dict
        Dictionary of user-defined settings used for the veg edge/waterline extraction.
    tidepath : str
        Path to where user has stored AVISO tide data and fns.
    daterange : list
        [start date, end date] to be computed.
    tidelatlon : list
        [longitude, latitude] of bounding box along seaward edge, halfway between S and N.

    Returns
    -------
    None.

    """
    
    tideoutpath = os.path.join(settings['inputs']['filepath'],
                               'tides',settings['inputs']['sitename']+'_tides_'+
                               settings['inputs']['dates'][0]+'_'+settings['inputs']['dates'][1]+'.csv')
    
    if os.path.isfile(tideoutpath) is True: 
        print('Tide data already compiled.')

    else:
        print('Compiling tide heights for given date range...')
        # add buffer of one day either side
        startdate = datetime.strptime(daterange[0], '%Y-%m-%d') - timedelta(days=1)
        enddate = datetime.strptime(daterange[1], '%Y-%m-%d') + timedelta(days=1)
        # create array of hourly timesteps between dates
        dates = np.arange(startdate, enddate, timedelta(hours=1)).astype(datetime) 
        
        # pass configuration files to pyfes handler to gather up tidal constituents
        config_ocean = os.path.join(tidepath,"ocean_tide_extrapolated.ini")
        ocean_tide = pyfes.Handler("ocean", "io", config_ocean)
        config_load = os.path.join(tidepath,"load_tide.ini")
        load_tide = pyfes.Handler("radial", "io", config_load)
        
        # format dates and latlon
        dates_np = np.empty((len(dates),), dtype='datetime64[us]')
        for i,date in enumerate(dates):
            dates_np[i] = datetime(date.year,date.month,date.day,date.hour,date.minute,date.second)
        lons = tidelatlon[0]*np.ones(len(dates))
        lats = tidelatlon[1]*np.ones(len(dates))
        
        # compute heights for ocean tide and loadings (both are needed for elastic tide elevations)
        ocean_short, ocean_long, min_points = ocean_tide.calculate(lons, lats, dates_np)
        load_short, load_long, min_points = load_tide.calculate(lons, lats, dates_np)
        # sum up all components and convert from cm to m
        tide_level = (ocean_short + ocean_long + load_short + load_long)/100
        
        # export as csv to tides folder
        tidesDF = pd.DataFrame([dates, tide_level]).transpose()
        tidesDF.columns = ['date', 'tide']
        # print('saving computed tides under '+os.path.join(settings['inputs']['filepath'],'tides',settings['inputs']['sitename']+'_tides.csv'))
        # tidesDF.to_csv(os.path.join(settings['inputs']['filepath'],'tides',settings['inputs']['sitename']+'_tides.csv'))
        print('saving computed tides under '+tideoutpath)
        tidesDF.to_csv(tideoutpath)
        
    return 


def GetWaterElevs(settings, Dates_Sat):
    '''
    Extracts matching water elevations from formatted CSV of tide heights and times.
    FM Jun 2023

    Parameters
    ----------
    settings : dict
        Dictionary of user-defined settings used for the veg edge/waterline extraction.
    Dates_Sat : list
        List of dates and times of images used to extract veg/water lines from.

    Returns
    -------
    tides_sat : list
        List of tidal elevations at time of image capture (with n = n(dates_sat)).

    '''

    # load tidal data
    # TideFilepath = os.path.join(settings['inputs']['filepath'],'tides',settings['inputs']['sitename']+'_tides.csv')
    TideFilepath = os.path.join(settings['inputs']['filepath'],
                                'tides',settings['inputs']['sitename']+'_tides_'+
                                settings['inputs']['dates'][0]+'_'+settings['inputs']['dates'][1]+'.csv')
    Tide_Data = pd.read_csv(TideFilepath, parse_dates=['date'])
    Dates_ts = [_.to_pydatetime() for _ in Tide_Data['date']]
    Tides_ts = np.array(Tide_Data['tide'])

    # Calculate time step used for interpolating data between
    TimeStep = (Dates_ts[1]-Dates_ts[0]).total_seconds()/(60*60)
    
    Tides_Sat = []
    
    # Previously found first following tide time, but incorrect when time is e.g. only 1min past the hour
    # for i,date in enumerate(dates_sat):
    #     tides_sat.append(tides_ts[find(min(item for item in dates_ts if item > date), dates_ts)])
    
    # Interpolate tide using number of minutes through the hour the satellite image was captured
    for i,date in enumerate(Dates_Sat):
        # find preceding and following hourly tide levels and times
        # Time_1 = Dates_ts[find(min(item for item in Dates_ts if item > date-timedelta(hours=TimeStep)), Dates_ts)]
        Tide_1 = Tides_ts[find(min(item for item in Dates_ts if item > date-timedelta(hours=TimeStep)), Dates_ts)]
        Time_2 = Dates_ts[find(min(item for item in Dates_ts if item > date), Dates_ts)]
        Tide_2 = Tides_ts[find(min(item for item in Dates_ts if item > date), Dates_ts)]
        
        # Find time difference of actual satellite timestamp (next hour minus sat timestamp)
        TimeDiff = Time_2 - date
        # Get proportion of time through the hour (e.g. 59mins past = 0.01)
        TimeProp = TimeDiff / timedelta(hours=TimeStep)
        
        # Get difference between the two tidal stages
        TideDiff = (Tide_2 - Tide_1)
        Tide_Sat = Tide_2 - (TideDiff * TimeProp)
        
        Tides_Sat.append(Tide_Sat)
    
    return Tides_Sat



def BeachTideLoc(settings, TideSeries=None):
    '''
    Create steps of water elevation based on a tidal range, which correspond to the 'lower', 'middle' and 'upper' beach.
    FM July 2023

    Parameters
    ----------
    settings : dict
        Dictionary of user-defined settings used for the veg edge/waterline extraction.

    Returns
    -------
    TideSteps : list
        Array of 4 elevations running from lowest to highest tide.
        Beach zone class is then 'lower' = TideSteps[0] to TideSteps[1], etc.

    '''
    
    if TideSeries is None:
        tidefilepath = os.path.join(settings['inputs']['filepath'],'tides',settings['inputs']['sitename']+'_tides.csv')
        tide_data = pd.read_csv(tidefilepath, parse_dates=['date'])
        tides_ts = np.array(tide_data['tide'])
    else:
        tides_ts = TideSeries
    
    MaxTide = np.max(tides_ts)
    MinTide = np.min(tides_ts)
    TideStep = (MaxTide - MinTide)/3
    
    TideSteps = [MinTide, MinTide+TideStep, MaxTide-TideStep, MaxTide]
    
    return TideSteps



def ExtendLine(LineGeom, dist):
    '''
    FM Jun 2023
    
    Parameters
    ----------
    LineGeom : LineString
        Line to be extended.
    dist : int
        distance to extend line by.

    Returns
    -------
    NewLineGeom : LineString
        new extended shapely LINESTRING.

    '''
    # extract coords
    x1, x2, y1, y2 = LineGeom.coords.xy[0][0], LineGeom.coords.xy[0][1], LineGeom.coords.xy[1][0], LineGeom.coords.xy[1][1]
    # calculate vector
    v = (x2-x1, y2-y1)
    v_ = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    # calculate normalised vector
    vnorm = v / v_
    # use norm vector to extend 
    x_1, y_1 = (x1, y1) - (dist*vnorm)
    x_2, y_2 = (x2, y2) + (dist*vnorm)
    # Extended line
    NewLineGeom = LineString([(x_1, y_1), (x_2, y_2)])
    
    return NewLineGeom
    
    
def CircMean(Array):
    """
    Calculate the mean of a list or array of values that represent circular directions/bearings.
    To be used in calculating mean wave directions especially when timeseries centres around north (0 degrees).
    FM Nov 2023

    Parameters
    ----------
    Array : list or array
        Direction values in degrees to be used in calculation.

    Returns
    -------
    MeanDeg : float
        Mean value of Array in degrees.

    """
    ArrRad = np.deg2rad(Array)
    MeanRad = circmean(ArrRad, nan_policy='omit')
    MeanDeg = np.rad2deg(MeanRad)
    
    return MeanDeg

def CircStd(Array):
    """
    Calculate the standard deviation of a list or array of values that represent circular directions/bearings.
    To be used in calculating stddev wave directions especially when timeseries centres around north (0 degrees).
    FM Nov 2023

    Parameters
    ----------
    Array : list or array
        Direction values in degrees to be used in calculation.

    Returns
    -------
    MeanDeg : float
        Std Dev value of Array in degrees.

    """
    ArrRad = np.deg2rad(Array)
    MeanRad = circstd(ArrRad, nan_policy='omit')
    MeanDeg = np.rad2deg(MeanRad)
    
    return MeanDeg


def CMSDownload(CMScmd):
    """
    Download netCDF data files containing wave hindcast or forecast data from 
    Copernicus Marine Service (CMEMS). Note: global hindcast data is offered in 
    two parts, one reanalysed and one interim; extra steps here merge these two.
    FM March 2024

    Parameters
    ----------
    CMScmd : dict
        Dictionary with params for downloading wave data from CMEMS. Should contain:
            hind_fore : hindcast or forecast data ['hind','fore']
            lonmin, lonmax, latmin, latmax : bounding box of coords (float)
            DateMin, DateMax : date range of requested data (str)
            WavePath : path to resulting downloaded file (str)
            WaveOutFile : name of resulting downloaded file (str)

    Returns
    -------
    None.

    """
    
    # Log in to Copernicus Marine Service data store using your provided username and password
    cms.login(skip_if_user_logged_in=True)
    
    # If requesting a forecast, use 'Global Ocean Waves Analysis and Forecast' data from Copernicus
    if CMScmd['hind_fore'] == 'fore':
        cms.subset(
            dataset_id = 'cmems_mod_glo_wav_anfc_0.083deg_PT3H-i', 
            variables = ['VHM0', 'VMDR','VTPK'],
            minimum_longitude = CMScmd['lonmin'], 
            maximum_longitude = CMScmd['lonmax'], 
            minimum_latitude = CMScmd['latmin'],
            maximum_latitude = CMScmd['latmax'], 
            start_datetime = CMScmd['DateMin'], 
            end_datetime = CMScmd['DateMax'],  
            output_filename = str(CMScmd['WaveOutFile']),  
            output_directory = str(CMScmd['WavePath'])
            )
    
    # If requesting a hindcast, use 'Global Ocean Waves Reanalysis' data from Copernicus
    elif CMScmd['hind_fore'] == 'hind':
        # Download multiyear data (up to ~1yr prior to present)
        MYrWavePath = cms.subset(
            dataset_id = 'cmems_mod_glo_wav_my_0.2deg_PT3H-i', 
            variables = ['VHM0', 'VMDR','VTPK'],
            minimum_longitude = CMScmd['lonmin'], 
            maximum_longitude = CMScmd['lonmax'], 
            minimum_latitude = CMScmd['latmin'],
            maximum_latitude = CMScmd['latmax'], 
            start_datetime = CMScmd['DateMin'], 
            end_datetime = CMScmd['DateMax'],  
            output_filename = str(CMScmd['WaveOutFile'][:-3]+'_my.nc'),  
            output_directory = str(CMScmd['WavePath'])
            )
        # Download interim data (up to ~4mo prior to present)
        IntWavePath = cms.subset(
            dataset_id = 'cmems_mod_glo_wav_myint_0.2deg_PT3H-i', 
            variables = ['VHM0', 'VMDR','VTPK'],
            minimum_longitude = CMScmd['lonmin'], 
            maximum_longitude = CMScmd['lonmax'], 
            minimum_latitude = CMScmd['latmin'],
            maximum_latitude = CMScmd['latmax'], 
            start_datetime = CMScmd['DateMin'], 
            end_datetime = CMScmd['DateMax'],  
            output_filename = str(CMScmd['WaveOutFile'][:-3]+'_int.nc'),  
            output_directory = str(CMScmd['WavePath'])
            )
        
        # Combine multiyear and interim data into one dataset
        CombiData = xr.open_mfdataset([MYrWavePath,IntWavePath],combine='nested', concat_dim='time')
        # Save combined dataset to new netCDF file under desired name
        CombiData.to_netcdf(os.path.join(CMScmd['WavePath'],CMScmd['WaveOutFile']))
        # Clean up folder by deleting multiyear and interim files
        for f in [MYrWavePath,IntWavePath]:
            os.remove(f)
    
    # If using specific region data (NW Atlantic Shelf)
    else:
        cms.subset(
            dataset_id = 'MetO-NWS-WAV-RAN', 
            variables = ['VHM0', 'VMDR','VTPK'],
            minimum_longitude = CMScmd['lonmin'], 
            maximum_longitude = CMScmd['lonmax'], 
            minimum_latitude = CMScmd['latmin'],
            maximum_latitude = CMScmd['latmax'], 
            start_datetime = CMScmd['DateMin'], 
            end_datetime = CMScmd['DateMax'],  
            output_filename = str(CMScmd['WaveOutFile']),  
            output_directory = str(CMScmd['WavePath'])
            )       
        

    # Grab next cells when downloaded grid cell is empty (fully masked next to coast)
    WaveData = netCDF4.Dataset(os.path.join(CMScmd['WavePath'],CMScmd['WaveOutFile']))
    
    SigWaveHeight = WaveData.variables['VHM0'][:,:,:]  # Spectral significant wave height (Hs)
    # for each grid cell in downloaded data
    emptyID = {'x':[], 'y':[]}
    for x in range(SigWaveHeight.shape[1]):
        for y in range(SigWaveHeight.shape[2]):
            # Check if grid cell contains fully masked data
            if SigWaveHeight.shape[0] == SigWaveHeight[:,x,y].mask.sum():
                # Append lat lon IDs to list for removal
                emptyID['x'].append(x)
                emptyID['y'].append(y)
    GridLen = SigWaveHeight.shape[1]*SigWaveHeight.shape[2]
    # TO DO: find way of deleting empty grid cells from netCDF to make sure they aren't used later

    # Check if all grid cells in download have been removed due to being empty
    if len(emptyID['x']) == GridLen and len(emptyID['y']) == GridLen:
        # If so, expand the bounding box requested (by 0.2deg or 22km) and repeat the download process
        CMScmd['lonmin'] = CMScmd['lonmin'] - 0.2 
        CMScmd['lonmax'] = CMScmd['lonmax'] + 0.2
        CMScmd['latmin'] = CMScmd['latmin'] - 0.2
        CMScmd['latmax'] = CMScmd['latmin'] + 0.2
    
        # If requesting a forecast, use 'Global Ocean Waves Analysis and Forecast' data from Copernicus
        if CMScmd['hind_fore'] == 'fore':
            cms.subset(
                dataset_id = 'cmems_mod_glo_wav_anfc_0.083deg_PT3H-i', 
                variables = ['VHM0', 'VMDR','VTPK'],
                minimum_longitude = CMScmd['lonmin'], 
                maximum_longitude = CMScmd['lonmax'], 
                minimum_latitude = CMScmd['latmin'],
                maximum_latitude = CMScmd['latmax'], 
                start_datetime = CMScmd['DateMin'], 
                end_datetime = CMScmd['DateMax'],  
                output_filename = str(CMScmd['WaveOutFile']),  
                output_directory = str(CMScmd['WavePath'])
                )
        
        # If requesting a hindcast, use 'Global Ocean Waves Reanalysis' data from Copernicus
        elif CMScmd['hind_fore'] == 'hind':
            # Download multiyear data (up to ~1yr prior to present)
            MYrWavePath = cms.subset(
                dataset_id = 'cmems_mod_glo_wav_my_0.2deg_PT3H-i', 
                variables = ['VHM0', 'VMDR','VTPK'],
                minimum_longitude = CMScmd['lonmin'], 
                maximum_longitude = CMScmd['lonmax'], 
                minimum_latitude = CMScmd['latmin'],
                maximum_latitude = CMScmd['latmax'], 
                start_datetime = CMScmd['DateMin'], 
                end_datetime = CMScmd['DateMax'],  
                output_filename = str(CMScmd['WaveOutFile'][:-3]+'_my.nc'),  
                output_directory = str(CMScmd['WavePath'])
                )
            # Download interim data (up to ~4mo prior to present)
            IntWavePath = cms.subset(
                dataset_id = 'cmems_mod_glo_wav_myint_0.2deg_PT3H-i', 
                variables = ['VHM0', 'VMDR','VTPK'],
                minimum_longitude = CMScmd['lonmin'], 
                maximum_longitude = CMScmd['lonmax'], 
                minimum_latitude = CMScmd['latmin'],
                maximum_latitude = CMScmd['latmax'], 
                start_datetime = CMScmd['DateMin'], 
                end_datetime = CMScmd['DateMax'],  
                output_filename = str(CMScmd['WaveOutFile'][:-3]+'_int.nc'),  
                output_directory = str(CMScmd['WavePath'])
                )
            
            # Combine multiyear and interim data into one dataset
            CombiData = xr.open_mfdataset([MYrWavePath,IntWavePath],combine='nested', concat_dim='time')
            # Save combined dataset to new netCDF file under desired name
            CombiData.to_netcdf(os.path.join(CMScmd['WavePath'],CMScmd['WaveOutFile']))
            # Clean up folder by deleting multiyear and interim files
            for f in [MYrWavePath,IntWavePath]:
                os.remove(f)
            


def MotuDownload(motuCommand):
    """
    (Legacy) download process for Copernicus Marine Service (CMEMS) data.
    FM March 2024
    
    Parameters
    ----------
    motuCommand : str
        String to send to command line to trigger Motu client download of wave data.

    Returns
    -------
    None.

    """
    # Download process sent via subprocess (to enable rolling updates from command line)
    dlprocess = subprocess.Popen(motuCommand, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    # Read and print interim status updates
    while True:
        output = dlprocess.stdout.readline().decode().strip()
        if output == '' and dlprocess.poll() is not None:
            break
        if output:
            print(output)

    # Check for any errors
    _, stderr = dlprocess.communicate()
    if stderr:
        print("Error:", stderr.decode())

    # Wait for the process to finish
    dlprocess.wait()
    
    
def GetSeasonalityIndex(x,y, P=365):
    """
    Calculate Seasonal Strength Index from a timeseries.
    FM May 2024

    Parameters
    ----------
    x : list or array
        Array of timestamps in datetime format.
    y : list or array
        Array of timeseries values (e.g. cross-shore distance of veg edge, wave conditions).
    P : int, optional
        Number of timesteps (days) to decompose seasonality over. The default is 365.

    Returns
    -------
    SSI : float
        Seasonal Strength Index (variance of seasonal signal / (variance of seasonal signal + variance of residuals)).

    """

    # Extend series and interpolate to create daily observations
    Timeseries = pd.Series(y, index=x)
    # if any duplicates exist, take the mean of those duplicates
    Timeseries = Timeseries.groupby(Timeseries.index).mean()
    Timeseries = Timeseries.resample('1D')
    Timeseries = Timeseries.interpolate(method='time')
    
    # Calculate seasonal .trend, .seasonal and .resid, using a year as the detrending period
    Seasonality = seasonal_decompose(Timeseries, model='additive', period=P)
    Season = Seasonality.seasonal
    Resid = Seasonality.resid
    # Calculate Seasonal Strength Index
    SSI = np.var(Season) / (np.var(Season) + np.var(Resid))
    
    return SSI


def PercentageCloudy(metadata, settings):
    """
    Generate boolean lists for how many satellite images should be 
    skipped vs. succeed; can be used for filtering metadata before a full run,
    or generating a post-process success rate.
    FM June 2024

    Parameters
    ----------
    metadata : dict
        Dictionary of satellite image metadata.
    settings : dict
        Dictionary of settings to be used for the veg edge extraction.

    Returns
    -------
    cloud_exceeded : list
        Populated list with n = n_filenames. Images where cloud threshold was exceeded = 1.
    cloud_score : list
        Populated list of recalculated cloud score, with n = n_filenames.
    empty_file : list
        Populated list. Empty images with no pixel values = 1.

    """
    
    cloud_exceeded = []
    cloud_score = []
    empty_file = []
    filenames = metadata['S2']['filenames']
    for i in range(len(filenames)):
        print(f"Image {i}")
        fn = int(i)
        cloud_mask_issue = settings['cloud_mask_issue']

        imgs = []
        for i in range(len(filenames)):
            imgs.append(ee.Image(filenames[i]))
        Sentinel2 = ee.ImageCollection.fromImages(imgs).filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 98.5))
        
        cloud_scoree = Sentinel2.getInfo().get('features')[fn]['properties']['CLOUDY_PIXEL_PERCENTAGE']/100
        cloud_score.append(cloud_scoree)
        
        if cloud_scoree > settings['cloud_thresh']:
            print(' - Skipped: cloud threshold exceeded (%0.1f%%)' % (cloud_scoree*100))
            cloud_exceeded.append(1)
            continue
        else:
            cloud_exceeded.append(0)
        
        img = ee.Image(Sentinel2.getInfo().get('features')[fn]['id'])
        # read 10m bands (R,G,B,NIR)        
        im10 = geemap.ee_to_numpy(img, 
                                  bands = ['B2','B3','B4','B8'], 
                                  region=ee.Geometry.Polygon(settings['inputs']['polygon']),
                                  scale=10)
        
        im20 = geemap.ee_to_numpy(img, 
                                  bands = ['B11'], 
                                  region=ee.Geometry.Polygon(settings['inputs']['polygon']),
                                  scale=20)
        
        im60 = geemap.ee_to_numpy(img, 
                                  bands = ['QA60'], 
                                  region=ee.Geometry.Polygon(settings['inputs']['polygon']),
                                  scale=60)
        
        if im10 is None or im20 is None or im60 is None:
            print(' - Skipped: empty raster')
            empty_file.append(1)
            continue
        else:
            empty_file.append(0)
        
        if sum(sum(sum(im10))) < 1:
            print(' - Skipped: empty raster (zeros)')
            empty_file.append(1)
            continue
        else:
            empty_file.append(0)
            
        im_QA = im60[:,:,0]
        
        cloud_mask = Image_Processing.create_cloud_mask(im_QA, 'S2', cloud_mask_issue)
        
        if cloud_mask is None:
            print(" - Skipped: no cloud mask available") 
            empty_file.append(1)
            continue
        else:
            empty_file.append(0)
            
    return cloud_exceeded, cloud_score, empty_file


def Moments(Arr):
    '''
    Calculate statistical moments from dataset.
    FM Jul 2024

    Parameters
    ----------
    Arr : array, list
        1D array of data over which to calculate distribution.

    Returns
    -------
    mean : float
        Mean of dataset.
    variance : float
        Variance of dataset.
    skew : float
        Skewness of the dataset.
    kurtosis : float
        Kurtosis of the dataset.

    '''
    mean_1 = np.mean(Arr)
    variance_2 = np.var(Arr)
    skew_3 = skew(Arr)
    kurtosis_4 = kurtosis(Arr)
    
    return mean_1, variance_2, skew_3, kurtosis_4


def InterpolateRaster(raster, method='nearest'):
    """
    Interpolate over empty values in a raster.
    FM July 2024

    Parameters
    ----------
    raster : array
        2D array of raster values.
    method : str, optional
        Interpolation method to be used ['linear','nearest','cubic']. Nearest
        is the only method to fill all cells in a raster.

    Returns
    -------
    interp_raster : array
        Interpolated raster.

    """
    mask = raster.mask
    validcoords = np.array(np.nonzero(~mask)).T
    invalidcoords = np.array(np.nonzero(mask)).T
    
    validvals = raster[~mask]
    interp_vals = interpolate.griddata(validcoords, validvals,
                                      invalidcoords, method=method)
    interp_raster = raster.data.copy()
    interp_raster[mask] = interp_vals
    
    return interp_raster

def InterpolateCircRaster(raster, method='nearest', nodata_value=-32767):
    '''
    Interpolate over empty values in a raster of angles in degrees between 0 and 360 (e.g. wave directions).
    FM July 2024

    Parameters
    ----------
    raster : array
        2D array of raster values.
    method : str, optional
        Interpolation method to be used ['linear','nearest','cubic']. Nearest
        is the only method to fill all cells in a raster.

    Returns
    -------
    interpolated_degrees : array
        Interpolated raster of degrees.

    '''


    # Convert nodata values (-32767) to NaN for interpolation
    raster_data = np.where(raster.data == nodata_value, np.nan, raster.data)

    # Convert degrees to radians and then to complex numbers
    radians = np.deg2rad(raster_data)
    complex_numbers = np.exp(1j * radians)

    # Extract real and imaginary parts
    real_part = np.real(complex_numbers)
    imag_part = np.imag(complex_numbers)

    # Get indices for all raster points
    x, y = np.indices(raster_data.shape)

    # Identify valid points (those that are not NaN)
    valid_mask = ~np.isnan(raster_data)
    valid_points = np.array((x[valid_mask], y[valid_mask])).T
    valid_real = real_part[valid_mask]
    valid_imag = imag_part[valid_mask]

    # Identify points that need interpolation (those that are NaN)
    masked_mask = np.isnan(raster_data)
    masked_points = np.array((x[masked_mask], y[masked_mask])).T

    # Perform linear interpolation first
    interp_real_linear = interpolate.griddata(valid_points, valid_real, masked_points, method='linear')
    interp_imag_linear = interpolate.griddata(valid_points, valid_imag, masked_points, method='linear')

    # Create a copy of the raster to store linear interpolation results
    interpolated_real = real_part.copy()
    interpolated_imag = imag_part.copy()
    interpolated_real[masked_mask] = interp_real_linear
    interpolated_imag[masked_mask] = interp_imag_linear

    # Reconstruct complex numbers from linear interpolation results
    interpolated_complex = interpolated_real + 1j * interpolated_imag
    interpolated_radians = np.angle(interpolated_complex)
    interpolated_degrees = np.rad2deg(interpolated_radians)
    interpolated_degrees = np.mod(interpolated_degrees, 360)

    # Identify any remaining NaNs after linear interpolation
    remaining_mask = np.isnan(interpolated_degrees)
    if np.any(remaining_mask):
        # Perform nearest-neighbor interpolation on remaining NaNs
        masked_points_remaining = np.array((x[remaining_mask], y[remaining_mask])).T
        interp_real_nearest = interpolate.griddata(valid_points, valid_real, masked_points_remaining, method='nearest')
        interp_imag_nearest = interpolate.griddata(valid_points, valid_imag, masked_points_remaining, method='nearest')

        # Fill remaining NaNs with nearest-neighbor results
        # Ensure the interpolation result has the correct shape before assignment
        interpolated_real[remaining_mask] = np.where(np.isnan(interp_real_nearest), interpolated_real[remaining_mask], interp_real_nearest)
        interpolated_imag[remaining_mask] = np.where(np.isnan(interp_imag_nearest), interpolated_imag[remaining_mask], interp_imag_nearest)

        # Reconstruct complex numbers for remaining values
        interpolated_complex = interpolated_real + 1j * interpolated_imag
        interpolated_radians = np.angle(interpolated_complex)
        interpolated_degrees = np.rad2deg(interpolated_radians)
        interpolated_degrees = np.mod(interpolated_degrees, 360)

    # Return the fully interpolated raster (with no NaNs)
    return interpolated_degrees


