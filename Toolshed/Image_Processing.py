"""
This module contains functions needed to preprocess satellite images before processing through line extraction.
Nov 2022.
COASTGUARD edits and updates: Freya Muir, University of Glasgow

"""

# load modules
import os
import numpy as np
import matplotlib.pyplot as plt
import pdb

# image processing modules
import skimage.transform as transform
import skimage.morphology as morphology
import sklearn.decomposition as decomposition
import skimage.exposure as exposure
from skimage.transform import resize
from geoarray import GeoArray 
import rasterio
from arosics import COREG

# other modules
from osgeo import gdal
from pylab import ginput
import pickle
import pandas as pd
import geopandas as gpd
from shapely import geometry
import ee
import geemap
import glob
from datetime import datetime

# CoastSat modules
from Toolshed import Toolbox


np.seterr(all='ignore') # raise/ignore divisions by 0 and nans

# Main function to preprocess a satellite image
def preprocess_single(ImgColl, georef, fn, datelist, filenames, satname, settings, polygon, dates, skipped):
    """
    Main function to preprocess a satellite image
    Updated FM Apr 2022
    
    FM: Problem still exists for ee_to_numpy where requests are limited to 262144 pixels (5120m for S2, 7680m for L5/7/8).
    Potential solution may be to export full image to Google Drive then convert from there?
    Example:
    bbox = img.getInfo()['properties']['system:footprint']['coordinates']
    task = ee.batch.Export.image.toDrive(img, 
        scale=10000,
        description='MOD09A1',
        fileFormat='GeoTIFF',
        region=bbox)
    task.start()

    Parameters
    ----------
    ImgColl : 
        GEE image collection, collated for each platform.
    georef : list
        List of affine transformation values (defined as the same clipped region for each image)
    fn : int
        Iteration number.
    datelist : list
        
    filenames : list
        Sat image filenames.
    satname : str
        Name of satellite platform (Landsat 5/7/8/9 = L5/L7/L8/L9, Sentinel-2 = S2, PlanetScope = PS).
    settings : dict
        Settings for running the veg edge/waterline extraction tool.
    polygon : list
        Earth Engine geom representing bounding box.
    dates : list
        Desired start date and end date in format 'YYYY-MM-DD'.
    skipped : dict
        Global dictionary storing the reasons for each image that fails or is skipped.

    Returns
    -------
    im_ms : array
        Array of image with different bands in 3rd dimension.
    georef : list
        List of affine transformation values.
    cloud_mask : array
        Mask containing cloudy pixels to be removed/ignored.
    im_extra : array
        Extra image bands.
    im_QA : array
        Mask containing QA info for image (produced by platform).
    im_nodata : array
        Mask containing nodata info for image.
    acqtime : str
        Image acquisition date and time.

    """
    
    cloud_mask_issue = settings['cloud_mask_issue']
    
            
    #=============================================================================================#
    # L5 images
    #=============================================================================================#
    if satname == 'L5':
            
        img = ee.Image(ImgColl.getInfo().get('features')[fn]['id'])
        
        acqtime = datetime.utcfromtimestamp(img.getInfo()['properties']['system:time_start']/1000).strftime('%H:%M:%S.%f')
        acqdate = datelist[fn]
        cloud_scoree = img.getInfo()['properties']['CLOUD_COVER']/100
        if cloud_scoree > settings['cloud_thresh']:
            print(' - Skipped: cloud threshold exceeded (%0.1f%%)' % (cloud_scoree*100))
            skipped['cloudy'].append([filenames[fn], satname, acqdate+' '+acqtime])
            return None, None, None, None, None, None, acqtime
        
        im_ms = geemap.ee_to_numpy(img, 
                                   bands = ['B1','B2','B3','B4','B5','QA_PIXEL'], 
                                   region=ee.Geometry.Polygon(polygon),
                                   scale=30)
        
        # Run initial QA on image
        im_ms = QAMask(im_ms, satname, settings['cloud_thresh'])
        
        if im_ms is None:
            print(' - Skipped: empty/low quality raster')
            skipped['empty_poor'].append([filenames[fn], satname, acqdate+' '+acqtime])
            return None, None, None, None, None, None, acqtime
                
        # down-sample to 15 m (half of the original pixel size)
        nrows = im_ms.shape[0]*2
        ncols = im_ms.shape[1]*2

        # create cloud mask
        im_QA = im_ms[:,:,5]
        im_ms = im_ms[:,:,:-1]
        cloud_mask = create_cloud_mask(im_QA, satname, cloud_mask_issue)

        # resize the image using bilinear interpolation (order 1)
        im_ms = transform.resize(im_ms,(nrows, ncols), order=1, preserve_range=True,
                                 mode='constant')
        # resize the image using nearest neighbour interpolation (order 0)
        cloud_mask = transform.resize(cloud_mask, (nrows, ncols), order=0, preserve_range=True,
                                      mode='constant').astype('bool_')
        
        if cloud_mask is None:
            print(" - Skipped: no cloud mask available")
            skipped['missing_mask'].append([filenames[fn], satname, acqdate+' '+acqtime])
            return None, None, None, None, None, None, acqtime
        
        # check if -inf or nan values on any band and eventually add those pixels to cloud mask        
        im_nodata = np.zeros(cloud_mask.shape).astype(bool)
        for k in range(im_ms.shape[2]):
            im_inf = np.isin(im_ms[:,:,k], -np.inf)
            im_nan = np.isnan(im_ms[:,:,k])
            im_nodata = np.logical_or(np.logical_or(im_nodata, im_inf), im_nan)
        # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
        # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
        im_zeros = np.ones(cloud_mask.shape).astype(bool)
        for k in [1,3,4]: # loop through the Green, NIR and SWIR bands
            im_zeros = np.logical_and(np.isin(im_ms[:,:,k],0), im_zeros)
        # add zeros to im nodata
        im_nodata = np.logical_or(im_zeros, im_nodata)   
        # update cloud mask with all the nodata pixels
        cloud_mask = np.logical_or(cloud_mask, im_nodata)
        
        # no extra image for Landsat 5 (they are all 30 m bands)
        im_extra = []

    #=============================================================================================#
    # L7 images
    #=============================================================================================#
    elif satname == 'L7':

        img = ee.Image(ImgColl.getInfo().get('features')[fn]['id'])
        
        acqtime = datetime.utcfromtimestamp(img.getInfo()['properties']['system:time_start']/1000).strftime('%H:%M:%S.%f')
        acqdate = datelist[fn]

        cloud_scoree = img.getInfo()['properties']['CLOUD_COVER']/100
        
        if cloud_scoree > settings['cloud_thresh']:
            print(' - Skipped: cloud threshold exceeded (%0.1f%%)' % (cloud_scoree*100))
            skipped['cloudy'].append([filenames[fn], satname, acqdate+' '+acqtime])
            return None, None, None, None, None, None, acqtime
        
        im_ms = geemap.ee_to_numpy(img, 
                                   bands = ['B1','B2','B3','B4','B5', 'B8','QA_PIXEL'], 
                                   region=ee.Geometry.Polygon(polygon),
                                   scale=30)
        
        # Run initial QA on image
        im_ms = QAMask(im_ms, satname, settings['cloud_thresh'])
        
        if im_ms is None:
            print(' - Skipped: empty/low quality raster')
            skipped['empty_poor'].append([filenames[fn], satname, acqdate+' '+acqtime])
            return None, None, None, None, None, None, acqtime
        

        cloud_scored = ee.Algorithms.Landsat.simpleCloudScore(img);

        #Create a mask from the cloud score and combine it with the image mask.
        mask = cloud_scored.select(['cloud']).lte(20);

        #Apply the mask to the image and display the result.
        masked = img.updateMask(mask);
        
        im_pan = geemap.ee_to_numpy(img, 
                                    bands = ['B8'], 
                                    region=ee.Geometry.Polygon(polygon),
                                    scale=15)
        
        # size of pan image
        nrows = im_pan.shape[0]
        ncols = im_pan.shape[1]

        # create cloud mask
        im_QA = im_ms[:,:,-1]
        cloud_mask = create_cloud_mask(im_QA, satname, cloud_mask_issue)

        # resize the image using bilinear interpolation (order 1)
        im_ms = im_ms[:,:,:5]
        im_ms = transform.resize(im_ms,(nrows, ncols), order=1, preserve_range=True,
                                 mode='constant')
        # resize the image using nearest neighbour interpolation (order 0)
        cloud_mask = transform.resize(cloud_mask, (nrows, ncols), order=0, preserve_range=True,
                                      mode='constant').astype('bool_')
        
        if cloud_mask is None:
            print(" - Skipped: no cloud mask available")
            skipped['missing_mask'].append([filenames[fn], satname, acqdate+' '+acqtime])
            return None, None, None, None, None, None, acqtime
        
        # check if -inf or nan values on any band and eventually add those pixels to cloud mask        
        im_nodata = np.zeros(cloud_mask.shape).astype(bool)
        for k in range(im_ms.shape[2]):
            im_inf = np.isin(im_ms[:,:,k], -np.inf)
            im_nan = np.isnan(im_ms[:,:,k])
            im_nodata = np.logical_or(np.logical_or(im_nodata, im_inf), im_nan)
        # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
        # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
        im_zeros = np.ones(cloud_mask.shape).astype(bool)
        for k in [1,3,4]: # loop through the Green, NIR and SWIR bands
            im_zeros = np.logical_and(np.isin(im_ms[:,:,k],0), im_zeros)
        # add zeros to im nodata
        im_nodata = np.logical_or(im_zeros, im_nodata)   
        # update cloud mask with all the nodata pixels
        cloud_mask = np.logical_or(cloud_mask, im_nodata)
        
        # pansharpen Blue, Green, Red (where there is overlapping with pan band in L8)
        try:
            im_ms_ps = pansharpen(im_ms[:,:,[0,1,2]], im_pan, cloud_mask)
        except: # if pansharpening fails, keep downsampled bands (for long runs)
            im_ms_ps = im_ms[:,:,[0,1,2]]
        # add downsampled NIR and SWIR1 bands
        im_ms_ps = np.append(im_ms_ps, im_ms[:,:,[3,4]], axis=2)

        im_ms = im_ms_ps.copy()
        # the extra image is the 15m panchromatic band
        im_extra = im_pan
        
        
    #=============================================================================================#
    # L8 images
    #=============================================================================================#
    elif satname == 'L8':
        
        img = ee.Image(ImgColl.getInfo().get('features')[fn]['id'])
        
        acqtime = datetime.utcfromtimestamp(img.getInfo()['properties']['system:time_start']/1000).strftime('%H:%M:%S.%f')      
        acqdate = datelist[fn]

        cloud_scoree = img.getInfo()['properties']['CLOUD_COVER']/100
        
        if cloud_scoree > settings['cloud_thresh']:
            print(' - Skipped: cloud threshold exceeded (%0.1f%%)' % (cloud_scoree*100))
            skipped['cloudy'].append([filenames[fn], satname, acqdate+' '+acqtime])
            return None, None, None, None, None, None, acqtime
        
        im_ms = geemap.ee_to_numpy(img, 
                                   bands = ['B2','B3','B4','B5', 'B6','B7','B10','B11','QA_PIXEL'], 
                                   region=ee.Geometry.Polygon(polygon),
                                   scale=30)
        
        # Run initial QA on image
        im_ms = QAMask(im_ms, satname, settings['cloud_thresh'])
        
        if im_ms is None:
            print(' - Skipped: empty/low quality raster')
            skipped['empty_poor'].append([filenames[fn], satname, acqdate+' '+acqtime])
            return None, None, None, None, None, None, acqtime

        cloud_scored = ee.Algorithms.Landsat.simpleCloudScore(img);

        #Create a mask from the cloud score and combine it with the image mask.
        mask = cloud_scored.select(['cloud']).lte(20);
        
        im_pan = geemap.ee_to_numpy(img, 
                                    bands = ['B8'], 
                                    region=ee.Geometry.Polygon(polygon),
                                    scale=15)
        
        # size of pan image
        nrows = im_pan.shape[0]
        ncols = im_pan.shape[1]

        # create cloud mask
        im_QA = im_ms[:,:,8]
        cloud_mask = create_cloud_mask(im_QA, satname, cloud_mask_issue)

        # resize the image using bilinear interpolation (order 1)
        im_ms = im_ms[:,:,:5]
        im_ms = transform.resize(im_ms,(nrows, ncols), order=1, preserve_range=True,
                                 mode='constant')
        # resize the image using nearest neighbour interpolation (order 0)
        cloud_mask = transform.resize(cloud_mask, (nrows, ncols), order=0, preserve_range=True,
                                      mode='constant').astype('bool_')
        
        if cloud_mask is None:
            print(" - Skipped: no cloud mask available")
            skipped['missing_mask'].append([filenames[fn], satname, acqdate+' '+acqtime])
            return None, None, None, None, None, None, acqtime
        
        # check if -inf or nan values on any band and eventually add those pixels to cloud mask        
        im_nodata = np.zeros(cloud_mask.shape).astype(bool)
        for k in range(im_ms.shape[2]):
            im_inf = np.isin(im_ms[:,:,k], -np.inf)
            im_nan = np.isnan(im_ms[:,:,k])
            im_nodata = np.logical_or(np.logical_or(im_nodata, im_inf), im_nan)
        # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
        # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
        im_zeros = np.ones(cloud_mask.shape).astype(bool)
        for k in [1,3,4]: # loop through the Green, NIR and SWIR bands
            im_zeros = np.logical_and(np.isin(im_ms[:,:,k],0), im_zeros)
        # add zeros to im nodata
        im_nodata = np.logical_or(im_zeros, im_nodata)   
        # update cloud mask with all the nodata pixels
        cloud_mask = np.logical_or(cloud_mask, im_nodata)
        
        # pansharpen Blue, Green, Red (where there is overlapping with pan band in L8)
        try:
            im_ms_ps = pansharpen(im_ms[:,:,[0,1,2]], im_pan, cloud_mask)
        except: # if pansharpening fails, keep downsampled bands (for long runs)
            im_ms_ps = im_ms[:,:,[0,1,2]]
        # add downsampled NIR and SWIR1 bands
        im_ms_ps = np.append(im_ms_ps, im_ms[:,:,[3,4]], axis=2)

        im_ms = im_ms_ps.copy()
        # the extra image is the 15m panchromatic band
        im_extra = im_pan


    #=============================================================================================#
    # L9 images
    #=============================================================================================#
    elif satname == 'L9':
        
        img = ee.Image(ImgColl.getInfo().get('features')[fn]['id'])

        acqtime = datetime.utcfromtimestamp(img.getInfo()['properties']['system:time_start']/1000).strftime('%H:%M:%S.%f')
        acqdate = datelist[fn]
        
        cloud_scoree = img.getInfo()['properties']['CLOUD_COVER']/100
        
        if cloud_scoree > settings['cloud_thresh']:
            print(' - Skipped: cloud threshold exceeded (%0.1f%%)' % (cloud_scoree*100))
            skipped['cloudy'].append([filenames[fn], satname, acqdate+' '+acqtime])
            return None, None, None, None, None, None, acqtime
        
        im_ms = geemap.ee_to_numpy(img, 
                                   bands = ['B2','B3','B4','B5', 'B6','B8','B10','B11','QA_PIXEL'], 
                                   region=ee.Geometry.Polygon(polygon),
                                   scale=30)
        
        # Run initial QA on image
        im_ms = QAMask(im_ms, satname, settings['cloud_thresh'])
        
        if im_ms is None:
            print(' - Skipped: empty/low quality raster')
            skipped['empty_poor'].append([filenames[fn], satname, acqdate+' '+acqtime])
            return None, None, None, None, None, None, acqtime

        cloud_scored = ee.Algorithms.Landsat.simpleCloudScore(img);

        #Create a mask from the cloud score and combine it with the image mask.
        mask = cloud_scored.select(['cloud']).lte(20);
  
        im_pan = geemap.ee_to_numpy(img, 
                                    bands = ['B8'], 
                                    region=ee.Geometry.Polygon(polygon),
                                    scale=15)
        
        # size of pan image
        nrows = im_pan.shape[0]
        ncols = im_pan.shape[1]

        # create cloud mask
        im_QA = im_ms[:,:,8]
        cloud_mask = create_cloud_mask(im_QA, satname, cloud_mask_issue)

        # resize the image using bilinear interpolation (order 1)
        im_ms = im_ms[:,:,:5] # B,G,R,NIR,SWIR
        im_ms = transform.resize(im_ms,(nrows, ncols), order=1, preserve_range=True,
                                 mode='constant')
        # resize the image using nearest neighbour interpolation (order 0)
        cloud_mask = transform.resize(cloud_mask, (nrows, ncols), order=0, preserve_range=True,
                                      mode='constant').astype('bool_')
        
        if cloud_mask is None:
            print(" - Skipped: no cloud mask available")
            skipped['missing_mask'].append([filenames[fn], satname, acqdate+' '+acqtime])
            return None, None, None, None, None, None, acqtime
        
        # check if -inf or nan values on any band and eventually add those pixels to cloud mask        
        im_nodata = np.zeros(cloud_mask.shape).astype(bool)
        for k in range(im_ms.shape[2]):
            im_inf = np.isin(im_ms[:,:,k], -np.inf)
            im_nan = np.isnan(im_ms[:,:,k])
            im_nodata = np.logical_or(np.logical_or(im_nodata, im_inf), im_nan)
        # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
        # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
        im_zeros = np.ones(cloud_mask.shape).astype(bool)
        for k in [1,3,4]: # loop through the Green, NIR and SWIR bands
            im_zeros = np.logical_and(np.isin(im_ms[:,:,k],0), im_zeros)
        # add zeros to im nodata
        im_nodata = np.logical_or(im_zeros, im_nodata)   
        # update cloud mask with all the nodata pixels
        cloud_mask = np.logical_or(cloud_mask, im_nodata)
        
        # pansharpen Blue, Green, Red (where there is overlapping with pan band in L8)
        try:
            im_ms_ps = pansharpen(im_ms[:,:,[0,1,2]], im_pan, cloud_mask)
        except: # if pansharpening fails, keep downsampled bands (for long runs)
            im_ms_ps = im_ms[:,:,[0,1,2]]
        # add downsampled NIR and SWIR1 bands
        im_ms_ps = np.append(im_ms_ps, im_ms[:,:,[3,4]], axis=2)

        im_ms = im_ms_ps.copy()
        # the extra image is the 15m panchromatic band
        im_extra = im_pan
        
        
    #=============================================================================================#
    # S2 images
    #=============================================================================================#
    elif satname == 'S2':
        
        img = ee.Image(ImgColl.getInfo().get('features')[fn]['id'])
        
        acqtime = datetime.utcfromtimestamp(img.getInfo()['properties']['system:time_start']/1000).strftime('%H:%M:%S.%f')
        acqdate = datelist[fn]
        
        cloud_scoree = img.getInfo()['properties']['CLOUDY_PIXEL_PERCENTAGE']/100
        
        if cloud_scoree > settings['cloud_thresh']:
            print(' - Skipped: cloud threshold exceeded (%0.1f%%)' % (cloud_scoree*100))
            skipped['cloudy'].append([filenames[fn], satname, acqdate+' '+acqtime])
            return None, None, None, None, None, None, acqtime

        # read 10m bands (R,G,B,NIR)        
        im10 = geemap.ee_to_numpy(img, 
                                  bands = ['B2','B3','B4','B8'], 
                                  region=ee.Geometry.Polygon(polygon),
                                  scale=10)
        if im10 is None:
            print(' - Skipped: empty raster')
            skipped['empty_poor'].append([filenames[fn], satname, acqdate+' '+acqtime])
            return None, None, None, None, None, None, acqtime
        
        # if image contains only zeros (can happen with S2), skip the image
        if sum(sum(sum(im10))) < 1:
            im_ms = []
            georef = []
            # skip the image by giving it a full cloud_mask
            cloud_mask = np.ones((im10.shape[0],im10.shape[1])).astype('bool')
            print(' - Skipped: only zeros in raster')
            skipped['empty_poor'].append([filenames[fn], satname, acqdate+' '+acqtime])
            return None, None, None, None, None, None, acqtime
        
        im10 = im10/10000 # TOA scaled to 10000

        # size of 10m bands
        nrows = im10.shape[0]
        ncols = im10.shape[1]

        # read 20m band (SWIR1)
        im20 = geemap.ee_to_numpy(img, 
                                  bands = ['B11'], 
                                  region=ee.Geometry.Polygon(polygon),
                                  scale=20)
        
        if im20 is None:
            print(' - Skipped: empty raster')
            skipped['empty_poor'].append([filenames[fn], satname, acqdate+' '+acqtime])
            return None, None, None, None, None, None, acqtime
        
        im20 = im20[:,:,0]
        im20 = im20/10000 # TOA scaled to 10000

        # resize the image using bilinear interpolation (order 1)
        im_swir = transform.resize(im20, (nrows, ncols), order=1, preserve_range=True,
                                   mode='constant')
        im_swir = np.expand_dims(im_swir, axis=2)
        
        # append down-sampled SWIR1 band to the other 10m bands
        im_ms = np.append(im10, im_swir, axis=2)
        
        # create cloud mask using 60m QA band (not as good as Landsat cloud cover)
        # 2024 rename of QA bands to MSK_CLASSI; implemented additional option
        try:
            im60 = geemap.ee_to_numpy(img, 
                                      bands = ['QA60'], 
                                      region=ee.Geometry.Polygon(polygon),
                                      scale=60)
        except:
            im60 = geemap.ee_to_numpy(img, 
                                      bands = ['MSK_CLASSI_OPAQUE'], 
                                      region=ee.Geometry.Polygon(polygon),
                                      scale=60)
        
        if im60 is None:
            print(' - Skipped: empty raster')
            skipped['empty_poor'].append([filenames[fn], satname, acqdate+' '+acqtime])
            return None, None, None, None, None, None, acqtime
        
        im_QA = im60[:,:,0]
        cloud_mask = create_cloud_mask(im_QA, satname, cloud_mask_issue)
        # resize the cloud mask using nearest neighbour interpolation (order 0)
        cloud_mask = transform.resize(cloud_mask,(nrows, ncols), order=0, preserve_range=True,
                                      mode='constant')
        
        if cloud_mask is None:
            print(" - Skipped: no cloud mask available")
            skipped['missing_mask'].append([filenames[fn], satname, acqdate+' '+acqtime])
            return None, None, None, None, None, None, acqtime
        
        # check if -inf or nan values on any band and create nodata image
        im_nodata = np.zeros(cloud_mask.shape).astype(bool)
        for k in range(im_ms.shape[2]):
            im_inf = np.isin(im_ms[:,:,k], -np.inf)
            im_nan = np.isnan(im_ms[:,:,k])
            im_nodata = np.logical_or(np.logical_or(im_nodata, im_inf), im_nan)
        # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
        # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
        im_zeros = np.ones(im_nodata.shape).astype(bool)
        im_zeros = np.logical_and(np.isin(im_ms[:,:,1],0), im_zeros) # Green
        im_zeros = np.logical_and(np.isin(im_ms[:,:,3],0), im_zeros) # NIR
        im_20_zeros = transform.resize(np.isin(im20,0),(nrows, ncols), order=0,
                                       preserve_range=True, mode='constant').astype(bool)
        im_zeros = np.logical_and(im_20_zeros, im_zeros) # SWIR1
        # add to im_nodata
        im_nodata = np.logical_or(im_zeros, im_nodata)
        # dilate if image was merged as there could be issues at the edges
        ##if 'merged' in fn10:
        ##    im_nodata = morphology.dilation(im_nodata,morphology.square(5))
            
        # update cloud mask with all the nodata pixels
        cloud_mask = np.logical_or(cloud_mask, im_nodata)

        # the extra image is the 20m SWIR band
        im_extra = im20
        
    #=============================================================================================#
    # Local images i.e. PlanetScope
    #=============================================================================================#
    else:
        
        # cloud cover check not relevant when using local data, set to 0%
        cloud_scoree = 0.0
        
        if cloud_scoree > settings['cloud_thresh']:
            return None, None, None, None, None, None, None
        
        # read all bands (B,G,R,NIR)        
        img = rasterio.open(filenames[fn])
        
        # TO DO: add line for cropping input image to Polygon of bounding box/AOI
        # im_cl = mask(img, [BBoxGDF_warp.geometry], crop=True)
        im_ms = img.read()
        
        # filename should be in the format 'yyyymmdd_HHMMSS_'
        acqtime = datetime.strftime(datetime.strptime(os.path.basename(filenames[fn])[9:15],'%H%M%S'),'%H:%M:%S.%f')
        
        if im_ms is None:
            return None, None, None, None, None, None, None
        
        # scaling factor to convert to floating-point reflectance
        if np.max(im_ms[:,:,0]) > 1:
            im_ms = im_ms / 10000
        
        # shape needs to be [row col band], so look for shape with small size
        if im_ms.shape[0] < 10:
            im_ms = np.transpose(im_ms, (1,2,0))
            
        # if im_ms.shape[2] < 5: # if missing SWIR, copy NIR
        #     im_ms = np.stack((im_ms[:,:,0], im_ms[:,:,1], im_ms[:,:,2], im_ms[:,:,3], im_ms[:,:,3]), axis=2)
        
        # adjust georeferencing vector to the new image size
        # ee transform: [xscale, xshear, xtrans, yshear, yscale, ytrans]
        # coastsat georef: [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
        georef = list(img.transform)[0:6] # get transform info from rasterio metadata
        georef = [round(georef[2]),georef[0],georef[1],round(georef[5]),georef[3],georef[4]] # rearrange

        datepath = os.path.basename(filenames[fn])[0:8]
        auxpath = os.path.dirname(filenames[fn])+'/cloudmasks/'
        # Use filename date to search for matching cloud mask file in cloudmasks dir
        if glob.glob(auxpath+datepath+'*') != []:
            cloud_file = glob.glob(auxpath+datepath+'*')
            with rasterio.open(cloud_file[0],'r') as ds:
                try:
                    im_QA = ds.read(6) # PlanetScope cloud mask is band 6      
                except:
                    im_QA = ds.read()
                if im_QA.shape[0] < 10:
                    im_QA = np.transpose(im_QA, (1,2,0))
                    im_QA = im_QA[:,:,0]
                cloud_mask = im_QA
        else:
            # create empty cloud mask
            im_QA = np.zeros(im_ms[:,:,0].shape).astype(bool) # needs to be one band of (nrow, ncol)
            cloud_mask = np.zeros(im_ms[:,:,0].shape).astype(bool) # needs to be one band of (nrow, ncol)
        
        # check if -inf or nan values on any band and eventually add those pixels to cloud mask        
        im_nodata = np.zeros(cloud_mask.shape).astype(bool)
        for k in range(im_ms.shape[2]): # for each band in image
            im_inf = np.isin(im_ms[:,:,k], -np.inf)
            im_nan = np.isnan(im_ms[:,:,k])
            im_nodata = np.logical_or(np.logical_or(im_nodata, im_inf), im_nan)
        # check if there are pixels with 0 intensity in any bands and add those
        # to the cloud mask as otherwise they will cause errors when calculating band indices
        im_zeros = np.ones(cloud_mask.shape).astype(bool)
        for k in range(im_ms.shape[2]): # loop through the bands
            im_zeros = np.logical_and(np.isin(im_ms[:,:,k],0), im_zeros)
        # add zeros to im nodata
        im_nodata = np.logical_or(im_zeros, im_nodata)   
        # update cloud mask with all the nodata pixels
        cloud_mask = np.logical_or(cloud_mask, im_nodata)
        
        # no extra local image
        im_extra = []
    
    return im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata, acqtime


###################################################################################################
# AUXILIARY FUNCTIONS
###################################################################################################


def InitialiseImgs(metadata, settings, satname, imgs):
    """
    Set satellite specific parameters before VedgeSat/CoastSat run.
    FM Nov 2024

    Parameters
    ----------
    metadata : dict
        Dictionary of sat image filenames, georeferencing info, EPSGs and dates of capture.
    settings : dict
        Dictionary of user-defined settings used for the veg edge extraction.
    satname : str
        Name of current satellite platform in loop (L5, L7, L8, L9, S2 or PS/other).
    imgs : list
        GEE image filenames as strings to be processed.

    Returns
    -------
    pixel_size : TYPE
        DESCRIPTION.
    clf_model : TYPE
        DESCRIPTION.
    ImgColl : ImageCollection 
        GEE ImageCollection of images to be processed.
    init_georef : list
        Initial georeferencing info (same per platform unless coregistration updates it).

    """
    if satname == 'L5':
        pixel_size = 15
        clf_model = 'MLPClassifier_Veg_L5L8S2.pkl'
        ImgColl = ee.ImageCollection.fromImages(imgs).select(['B1','B2','B3','B4','B5','QA_PIXEL'])
        # adjust georeferencing vector to the new image size
        raw_georef = ImgColl.getInfo().get('features')[0]['bands'][0]['crs_transform'] # get georef layout from first img Blue band
        init_georef = Toolbox.CalcGeoref(raw_georef, settings)
        # scale becomes pansharpened 15m and the origin is adjusted to the center of new top left pixel
        init_georef[1] = init_georef[1]/2 # xscale = 15m
        init_georef[5] = init_georef[5]/2 # yscale = -15m
        # init_georef[0] = init_georef[0] + init_georef[1]/2 # xtrans = back by half of 15m
        # init_georef[3] = init_georef[3] - init_georef[5]/2 # ytrans = up by half of 15m
        print(f"Using initial georef: {init_georef}")
        
    elif satname == 'L7':
        pixel_size = 15
        clf_model = 'MLPClassifier_Veg_L5L8S2.pkl'
        ImgColl = ee.ImageCollection.fromImages(imgs).select(['B1','B2','B3','B4','B5','B8','QA_PIXEL'])
        # adjust georeferencing vector to the new image size
        raw_georef = ImgColl.getInfo().get('features')[0]['bands'][5]['crs_transform'] # get georef info from panchromatic band (updated to Band 8)
        init_georef = Toolbox.CalcGeoref(raw_georef, settings)
        print(f"Using initial georef: {init_georef}")
        
    elif satname == 'L8':
        pixel_size = 15
        clf_model = 'MLPClassifier_Veg_L5L8S2.pkl'
        ImgColl = ee.ImageCollection.fromImages(imgs).select(['B2','B3','B4','B5','B6','B7','B8','QA_PIXEL'])
        # adjust georeferencing vector to the new image size
        raw_georef = ImgColl.getInfo().get('features')[0]['bands'][6]['crs_transform'] # get georef info from panchromatic band (updated to Band 8)
        init_georef = Toolbox.CalcGeoref(raw_georef, settings)
        print(f"Using initial georef: {init_georef}")
        
    elif satname == 'L9':
        pixel_size = 15
        clf_model = 'MLPClassifier_Veg_L5L8S2.pkl' 
        ImgColl = ee.ImageCollection.fromImages(imgs).select(['B2','B3','B4','B5','B6','B7','B8','QA_PIXEL'])
        # adjust georeferencing vector to the new image size
        raw_georef = ImgColl.getInfo().get('features')[0]['bands'][6]['crs_transform'] # get georef info from panchromatic band (updated to Band 8)
        init_georef = Toolbox.CalcGeoref(raw_georef, settings)
        print(f"Using initial georef: {init_georef}")
        
    elif satname == 'S2':
        pixel_size = 10
        clf_model = 'MLPClassifier_Veg_L5L8S2.pkl' 
        ImgColl = ee.ImageCollection.fromImages(imgs).filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 98.5))
        # adjust georeferencing vector to the new image size
        raw_georef = ImgColl.getInfo().get('features')[0]['bands'][3]['crs_transform'] # get transform info from Band4
        init_georef = Toolbox.CalcGeoref(raw_georef, settings)
        print(f"Using initial georef: {init_georef}")
        
    else: # Planet or local
        pixel_size = metadata[settings['inputs']['sat_list'][0]]['acc_georef'][0][0] #pull first image's pixel size from transform matrix
        clf_model = 'MLPClassifier_Veg_PSScene.pkl' 
        init_georef = [] # georef gets set by each local image
        
    return pixel_size, clf_model, ImgColl, init_georef
        


def Coreg(settings, im_ref_buffer, im_ms, cloud_mask, georef):
    """
    Coregister each satellite image to the first one in a list of images. Uses
    the AROSICS package for calculating phase shifts in images.
    FM Jan 2024
    Parameters
    ----------
    im_ref : array 
        Reference image for coregistering to (first in list).
    im_trg : array
        Target image to be coregistered (current im_ms in list).
    georef : list
        Affine geotransformation matrix.
    Returns
    -------
    newgeoref : list
        Updated/shifted georeference information for affine transforms.
    """
    
    newbuff = False
    coreg_stats = {'dX': np.nan, 'dY': np.nan, 'Reliability': np.nan}
    # reference image to coregister to (given as filepath)
    refArr = GeoArray(settings['reference_coreg_im'])
    
    # target sat image to register (current image in loop)
    if refArr.shape != im_ms[:,:,0:3].shape:
        # if needed, resize target to match reference image rows and cols
        trgArr_rs = resize(im_ms[:,:,0:3], (refArr.shape[0], refArr.shape[1], im_ms[:,:,0:3].shape[2]), 
                                mode='reflect', anti_aliasing=True)
        # use resized raster as array, and refArr georef to avoid any pixel grid mismatches
        trgArr = GeoArray(trgArr_rs, refArr.geotransform, 'EPSG:'+str(settings['projection_epsg']))
        # also resize reference shoreline buffer and cloud mask
        im_ref_buffer = resize(im_ref_buffer, (refArr.shape[0], refArr.shape[1]),
                               mode='reflect', anti_aliasing=False)
        cloud_mask = resize(cloud_mask, (refArr.shape[0], refArr.shape[1]),
                               mode='reflect', anti_aliasing=False)
    else:
        trgArr = GeoArray(im_ms[:,:,0:3], georef, 'EPSG:'+str(settings['projection_epsg']))
    
    # add reference shoreline buffer (and cloud mask) to region for avoiding tie point creation
    refArr.mask_baddata = im_ref_buffer
    trgArr.mask_baddata = cloud_mask + im_ref_buffer

    # wp = custom matching window position in (X,Y) in same CRS as reference image
    # ws = custom matching window size in pixels as (X,Y)
    # max_shift = maximum shift allowed in X or Y direction in pixels (default is 5px AKA 50m S2)
    CR = COREG(refArr, trgArr, max_shift=10, q=True)#, wp=(,), ws=(,)) # add align_grids=True for resampling/stretching
    try:
        CR.calculate_spatial_shifts()
    except: # RuntimeError for caculated shifts being abnormally large
        print('\nCoreg: calculated shift is too large to be valid (>100m). georef has not changed.')
        return georef, newbuff, coreg_stats
    # DOn't perform shift if calculation reliability is lower than 40%
    if CR.shift_reliability < 40:
        print('\nCoreg: calculated shift reliability is too low (%0.1f%%). georef has not changed.' % CR.shift_reliability)
        return georef, newbuff, coreg_stats
    # Correct georeferencing info based on calculated shifts
    corrCR = CR.correct_shifts()
    print('\nCoreg: X shift = %0.3fm | Y shift = %0.3fm | Reliability = %0.1f%%' % (CR.x_shift_map, CR.y_shift_map, CR.shift_reliability))
    coreg_stats = {'dX': CR.x_shift_map, 'dY':CR.y_shift_map, 'Reliability':CR.shift_reliability}
    newbuff = True
    # Reset georef info to newly shifted georef
    newgeoref = list(corrCR['updated geotransform'])
    # use original pixel sizes
    newgeoref[1] = georef[1]
    newgeoref[5] = georef[5]

    return newgeoref, newbuff, coreg_stats


def ClipIndexVec(cloud_mask, im_ndi, im_labels, im_ref_buffer):
    """
    Create classified band index value vectors and clip them to coastal buffer.
    FM Nov 2022

    Parameters
    ----------
    cloud_mask : array
        Cloud mask raster created from defined nodata pixels
    im_ndi : array
        Normalised difference raster.
    im_labels : np.array
        3D boolean raster containing an image for each class (im_classif == label)
    im_ref_buffer : np.array
        Boolean 2D array matching image dimensions, with ref shoreline buffer zone = 1.

    Returns
    -------
    int_veg : array
        Reshaped version of raster image, with vegetation pixels labelled.
    int_nonveg : array
        Reshaped version of raster image, with non-vegetation pixels labelled.

    """
    nrows = cloud_mask.shape[0]
    ncols = cloud_mask.shape[1]
    
    # reshape spectral index image to vector
    vec_ndi = im_ndi.reshape(nrows*ncols)

    # reshape labels into vectors (0 is veg, 1 is nonveg)
    vec_veg = im_labels[:,:,0].reshape(ncols*nrows)
    vec_nonveg = im_labels[:,:,1].reshape(ncols*nrows)

    # use im_ref_buffer and dilate it by 5 pixels
    se = morphology.disk(5)
    im_ref_buffer_extra = morphology.binary_dilation(im_ref_buffer, se)
    # create a buffer
    vec_buffer = im_ref_buffer_extra.reshape(nrows*ncols)
    
    # select pixels that are within the buffer
    int_veg = vec_ndi[np.logical_and(vec_buffer,vec_veg)]
    int_nonveg = vec_ndi[np.logical_and(vec_buffer,vec_nonveg)]
        
    # make sure both classes have the same number of pixels before thresholding
    if len(int_veg) > 0 and len(int_nonveg) > 0:
        if np.argmin([int_veg.shape[0],int_nonveg.shape[0]]) == 1:
            int_veg = int_veg[np.random.choice(int_veg.shape[0],int_nonveg.shape[0], replace=False)]
        else:
            int_nonveg = int_nonveg[np.random.choice(int_nonveg.shape[0],int_veg.shape[0], replace=False)]
        return int_veg, int_nonveg
    else:
        return None, None
            


def save_RGB_NDVI(im_ms, cloud_mask, georef, filenames, settings):
    """
    Saves local georeferenced versions of the RGB and NDVI images to be investigated in a GIS.
    FM March 2022

    Parameters
    ----------
    im_ms : array
        Multispectral satellite image array.
    cloud_mask : array
        2D boolean raster created from defined nodata pixels.
    georef : list
        List of affine transformation values.
    filenames : str
        Name of satellite image to save to file.
    settings : dict
        Settings for running the veg edge/waterline extraction tool.


    """
    print(' \nsaving '+filenames)
    im_NDVI = Toolbox.nd_index(im_ms[:,:,3], im_ms[:,:,2], cloud_mask) # NIR and red bands
    try: # some sentinel images with 0 axis don't get caught before this
        im_RGB = rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
    except:
        im_RGB = im_ms[:,:,:3]
    # coastsat georef: [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
    tifname = filenames.rsplit('/',1)[1] # get characters after last /
    if '.tif' in tifname: # local image has extension in filename; remove it
        tifname = tifname[:-4]
    transform = rasterio.transform.from_origin(georef[0], georef[3], georef[1], georef[1]) # use georef to get affine
    
    # 3-band RGB array and 1-band NDVI
    for imarray, imtype, bandno in zip([im_RGB, im_NDVI],['RGB.tif', 'NDVI.tif'],[3,1]):
        if bandno > 1:
            imarray_brc = np.moveaxis(imarray,2,0) # rasterio expects shape of (bands, rows, cols)
            savename = os.path.join(settings['inputs']['filepath'],settings['inputs']['sitename'],'jpg_files',tifname+'_'+imtype)
            with rasterio.open(savename,'w',
                driver='GTiff',
                height=imarray_brc.shape[1],
                width=imarray_brc.shape[2],
                count=bandno,
                dtype=imarray_brc.dtype,
                crs='EPSG:'+str(settings['output_epsg']),
                transform=transform,
            ) as tif:
                tif.write(imarray_brc)
        else:
            imarray_brc = imarray
            savename = os.path.join(settings['inputs']['filepath'],settings['inputs']['sitename'],'jpg_files',tifname+'_'+imtype)
            with rasterio.open(
                savename,'w',
                driver='GTiff',
                height=imarray_brc.shape[0],
                width=imarray_brc.shape[1],
                count=bandno,
                dtype=imarray_brc.dtype,
                crs='EPSG:'+str(settings['output_epsg']),
                transform=transform,
            ) as tif:
                tif.write(imarray_brc,1) # single band raster requires an index param
  
def save_ClassIm(im_classif, im_labels, cloud_mask, georef, filenames, settings):
    """
    Saves local georeferenced version of the classified image to be investigated in a GIS.
    FM Sept 2022

    Parameters
    ----------
    im_classif : np.array
        2D image containing pixel labels
    im_labels : np.array
        3D boolean raster containing an image for each class (im_classif == label)
    cloud_mask : array
        2D boolean raster created from defined nodata pixels
    georef : list
        List of affine transformation values.
    filenames : str
        Name of satellite image to save to file.
    settings : dict
        Settings for running the veg edge/waterline extraction tool.

    """
    print(' \nsaving classified '+filenames)

    # coastsat georef: [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
    tifname = filenames.rsplit('/',1)[1] # get characters after last /
    if '.tif' in tifname: # local image has extension in filename; remove it
        tifname = tifname[:-4]
    transform = rasterio.transform.from_origin(georef[0], georef[3], georef[1], georef[1]) # use georef to get affine
    
    # Binary classified image
    with rasterio.open(
        os.path.join(settings['inputs']['filepath'],settings['inputs']['sitename'],'jpg_files',tifname+'_'+'CLASS.tif'),
        'w',
        driver='GTiff',
        height=im_classif.shape[0],
        width=im_classif.shape[1],
        count=1,
        dtype=im_classif.dtype,
        crs='EPSG:'+str(settings['output_epsg']),
        transform=transform,
    ) as tif:
        tif.write(im_classif,1)
       
def save_TZone(im_ms, im_labels, cloud_mask, im_ref_buffer, georef, filenames, settings):
    """
    Saves local georeferenced version of the transition zone to be investigated in a GIS.
    FM Sept 2022

    Parameters
    ----------
    im_ms : array
        Multispectral satellite image array.
    im_labels : TYPE
        DESCRIPTION.
    cloud_mask : array
        2D boolean raster created from defined nodata pixels.
    im_ref_buffer : np.array
        Boolean 2D array matching image dimensions, with ref shoreline buffer zone = 1.
    georef : list
        List of affine transformation values.
    filenames : str
        Name of satellite image to save to file.
    settings : dict
        Settings for running the veg edge/waterline extraction tool.

    """
    print(' \nsaving transition zone of '+filenames)

    # coastsat georef: [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
    tifname = filenames.rsplit('/',1)[1] # get characters after last /
    if '.tif' in tifname: # local image has extension in filename; remove it
        tifname = tifname[:-4]
    transform = rasterio.transform.from_origin(georef[0], georef[3], georef[1], georef[1]) # use georef to get affine
    im_ndvi = Toolbox.nd_index(im_ms[:,:,3], im_ms[:,:,2], cloud_mask)
    # int_veg = im_ndvi[im_labels[:,:,0]]
    # int_nonveg = im_ndvi[im_labels[:,:,1]] 

    # clip down classified band index values to coastal buffer
    int_veg_clip, int_nonveg_clip = ClipIndexVec(cloud_mask, im_ndvi, im_labels, im_ref_buffer)
    # NDVI_PDF = pd.DataFrame(data={'veg_ndvi':int_veg_clip, 'nonveg_ndvi':int_nonveg_clip})
    #NDVI_PDF.to_csv(os.path.join(settings['inputs']['filepath'], 
    #                             settings['inputs']['sitename'],
    #                             'Veglines',tifname+'_NDVIvalues.csv'))
    
    # calculate TZ min and max values with which to classify the NDVI into a binary raster
    try: # 2024-07-03: for error when setting TZValues (index [-1] with axis of length 0)
        TZbuffer = Toolbox.TZValues(int_veg_clip, int_nonveg_clip)
        im_TZ = Toolbox.TZimage(im_ndvi,TZbuffer)
    except: # just set im_TZ to empty raster
        im_TZ = np.empty(im_ndvi.shape)
        im_TZ[:] = np.nan
    
    # use im_ref_buffer and dilate it by 5 pixels
    se = morphology.disk(5)
    im_ref_buffer_extra = morphology.binary_dilation(im_ref_buffer, se)
    
    # select pixels that are within the buffer
    im_TZ_cl = np.ma.masked_where(im_ref_buffer_extra==False, im_TZ)
    im_TZ_cl_fill = im_TZ_cl.filled(np.nan)
    
    # Binary classified image
    with rasterio.open(
        os.path.join(settings['inputs']['filepath'],settings['inputs']['sitename'],'jpg_files',tifname+'_'+'TZ.tif'),
        'w',
        driver='GTiff',
        height=im_TZ_cl_fill.shape[0],
        width=im_TZ_cl_fill.shape[1],
        count=1,
        dtype=im_TZ_cl_fill.dtype,
        crs='EPSG:'+str(settings['output_epsg']),
        transform=transform,
    ) as tif:
        tif.write(im_TZ_cl_fill,1)


def QAMask(im_ms, satname, cloud_thresh):
    """
    Run an additional masking check for any images that have slipped past cloud
    masking using the cloud band, by using the QA band bit-packed values instead.
    Primarily for Landsat, Sentinel-2 has new MSK_CLASSI_OPAQUE band.
    FM Oct 2024
    
    More info:
        https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files/LSDS-1618_Landsat-4-7_C2-L2-ScienceProductGuide-v4.pdf
        https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files/LSDS-1619_Landsat8-9-Collection2-Level2-Science-Product-Guide-v6.pdf

    Parameters
    ----------
    im_ms : array
        Multispectral satellite image array.
    satname : str
        Name of satellite platform (Landsat 5/7/8/9 = L5/L7/L8/L9, Sentinel-2 = S2, PlanetScope = PS).
    cloud_thresh : float
        Percentage of image covered in cloud, above which it is discarded.

    Returns
    -------
    im_ms : array or None
        Multispectral satellite image array (empty if too many pixels are poor).

    """
    
    # Different satellites have different bit-packed QA values to mean different land cover
    if satname == 'L5':
        # Landsat 4-7 Pixel Quality Assessment (QA_PIXEL)
        # HiConf cloud, HiConf cloud with shadow, HiConf cloud with shadow over water, HiConf snow/ice
        maskvals = [5896, 7960, 8088, 13664]
        QAband = 5 # index of QA band in im_ms
    if satname == 'L7':
        # Landsat 4-7 Pixel Quality Assessment (QA_PIXEL)
        # Empty (scan line error), HiConf cloud, HiConf cloud with shadow, HiConf cloud with shadow over water, HiConf snow/ice
        maskvals = [0, 5896, 7960, 8088, 13664]
        QAband = 6 # index of QA band in im_ms
    elif satname in ['L8', 'L9']:
        # HiConf cloud, HiConf cloud with shadow, HiConf cloud with shadow over water, HiConf snow/ice
        maskvals = [22280, 24344, 24472, 30048]
        QAband = 8 # index of QA band in im_ms
        
    # For each QA pixel value representing cloud of some sort
    pixcounts = []
    for maskval in maskvals:
        # Count how many pixels equal that QA value
        pixcounts.append(np.sum(im_ms[:,:,QAband] == maskval))
    # Calculate the total number of poor pixels and the % of the total image
    pixcount = np.sum(pixcounts)
    maskpct = pixcount / (im_ms.shape[0] * im_ms.shape[1])
    
    # If poor quality pixel % exceeds cloud threshold, return an emtpy multispectral raster
    if maskpct > cloud_thresh:
        im_ms = None
    
    return im_ms


###################################################################################################
# AUXILIARY COASTSAT FUNCTIONS
###################################################################################################

def create_cloud_mask(im_QA, satname, cloud_mask_issue):
    """
    Creates a cloud mask using the information contained in the QA band.

    KV WRL 2018

    Arguments:
    -----------
    im_QA: np.array
        Image containing the QA band
    satname: string
        short name for the satellite: ```'L5', 'L7', 'L8' or 'S2'```
    cloud_mask_issue: boolean
        True if there is an issue with the cloud mask and sand pixels are being
        erroneously masked on the images

    Returns:
    -----------
    cloud_mask : np.array
        boolean array with True if a pixel is cloudy and False otherwise
        
    """

    # convert QA bits (the bits allocated to cloud cover vary depending on the satellite mission)
    if satname == 'L8' or satname == 'L9':
        cloud_values = [2800, 2804, 2808, 2812, 6896, 6900, 6904, 6908]
    elif satname == 'L7' or satname == 'L5' or satname == 'L4':
        cloud_values = [752, 756, 760, 764]
    elif satname == 'S2':
        cloud_values = [1024, 2048] # 1024 = dense cloud, 2048 = cirrus clouds

    # find which pixels have bits corresponding to cloud values
    cloud_mask = np.isin(im_QA, cloud_values)

    # remove cloud pixels that form very thin features. These are beach or swash pixels that are
    # erroneously identified as clouds by the CFMASK algorithm applied to the images by the USGS.
    if sum(sum(cloud_mask)) > 0 and sum(sum(~cloud_mask)) > 0:
        cloud_mask = morphology.remove_small_objects(cloud_mask, min_size=10, connectivity=1)

        if cloud_mask_issue:
            elem = morphology.square(3) # use a square of width 3 pixels
            cloud_mask = morphology.binary_opening(cloud_mask,elem) # perform image opening
            # remove objects with less than 25 connected pixels
            cloud_mask = morphology.remove_small_objects(cloud_mask, min_size=25, connectivity=1)

    return cloud_mask

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram matches
    that of a target image.

    Arguments:
    -----------
    source: np.array
        Image to transform; the histogram is computed over the flattened
        array
    template: np.array
        Template image; can have different dimensions to source
        
    Returns:
    -----------
    matched: np.array
        The transformed output image
        
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

def pansharpen(im_ms, im_pan, cloud_mask):
    """
    Pansharpens a multispectral image, using the panchromatic band and a cloud mask.
    A PCA is applied to the image, then the 1st PC is replaced, after histogram 
    matching with the panchromatic band. Note that it is essential to match the
    histrograms of the 1st PC and the panchromatic band before replacing and 
    inverting the PCA.

    KV WRL 2018

    Arguments:
    -----------
    im_ms: np.array
        Multispectral image to pansharpen (3D)
    im_pan: np.array
        Panchromatic band (2D)
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are

    Returns:
    -----------
    im_ms_ps: np.ndarray
        Pansharpened multispectral image (3D)
        
    """

    # reshape image into vector and apply cloud mask
    vec = im_ms.reshape(im_ms.shape[0] * im_ms.shape[1], im_ms.shape[2])
    vec_mask = cloud_mask.reshape(im_ms.shape[0] * im_ms.shape[1])
    vec = vec[~vec_mask, :]
    # apply PCA to multispectral bands
    pca = decomposition.PCA()
    vec_pcs = pca.fit_transform(vec)

    # replace 1st PC with pan band (after matching histograms)
    vec_pan = im_pan.reshape(im_pan.shape[0] * im_pan.shape[1])
    vec_pan = vec_pan[~vec_mask]
    vec_pcs[:,0] = hist_match(vec_pan, vec_pcs[:,0])
    vec_ms_ps = pca.inverse_transform(vec_pcs)

    # reshape vector into image
    vec_ms_ps_full = np.ones((len(vec_mask), im_ms.shape[2])) * np.nan
    vec_ms_ps_full[~vec_mask,:] = vec_ms_ps
    im_ms_ps = vec_ms_ps_full.reshape(im_ms.shape[0], im_ms.shape[1], im_ms.shape[2])

    return im_ms_ps


def rescale_image_intensity(im, cloud_mask, prob_high):
    """
    Rescales the intensity of an image (multispectral or single band) by applying
    a cloud mask and clipping the prob_high upper percentile. This functions allows
    to stretch the contrast of an image, only for visualisation purposes.

    KV WRL 2018

    Arguments:
    -----------
    im: np.array
        Image to rescale, can be 3D (multispectral) or 2D (single band)
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    prob_high: float
        probability of exceedence used to calculate the upper percentile

    Returns:
    -----------
    im_adj: np.array
        rescaled image
    """

    # lower percentile is set to 0
    prc_low = 0

    # reshape the 2D cloud mask into a 1D vector
    vec_mask = cloud_mask.reshape(im.shape[0] * im.shape[1])

    # if image contains several bands, stretch the contrast for each band
    if len(im.shape) > 2:
        # reshape into a vector
        vec =  im.reshape(im.shape[0] * im.shape[1], im.shape[2])
        # initiliase with NaN values
        vec_adj = np.ones((len(vec_mask), im.shape[2])) * np.nan
        # loop through the bands
        for i in range(im.shape[2]):
            # find the higher percentile (based on prob)
            prc_high = np.percentile(vec[~vec_mask, i], prob_high)
            # clip the image around the 2 percentiles and rescale the contrast
            vec_rescaled = exposure.rescale_intensity(vec[~vec_mask, i],
                                                      in_range=(prc_low, prc_high))
            vec_adj[~vec_mask,i] = vec_rescaled
        # reshape into image
        im_adj = vec_adj.reshape(im.shape[0], im.shape[1], im.shape[2])

    # if image only has 1 bands (grayscale image)
    else:
        vec =  im.reshape(im.shape[0] * im.shape[1])
        vec_adj = np.ones(len(vec_mask)) * np.nan
        prc_high = np.percentile(vec[~vec_mask], prob_high)
        vec_rescaled = exposure.rescale_intensity(vec[~vec_mask], in_range=(prc_low, prc_high))
        vec_adj[~vec_mask] = vec_rescaled
        im_adj = vec_adj.reshape(im.shape[0], im.shape[1])

    return im_adj

def create_jpg(im_ms, cloud_mask, date, satname, filepath):
    """
    Saves a .jpg file with the RGB image as well as the NIR and SWIR1 grayscale images.
    This functions can be modified to obtain different visualisations of the 
    multispectral images.

    KV WRL 2018

    Arguments:
    -----------
    im_ms: np.array
        3D array containing the pansharpened/down-sampled bands (B,G,R,NIR,SWIR1)
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    date: str
        string containing the date at which the image was acquired
    satname: str
        name of the satellite mission (e.g., 'L5')

    Returns:
    -----------
        Saves a .jpg image corresponding to the preprocessed satellite image

    """

    # rescale image intensity for display purposes
    im_RGB = rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
#    im_NIR = rescale_image_intensity(im_ms[:,:,3], cloud_mask, 99.9)
#    im_SWIR = rescale_image_intensity(im_ms[:,:,4], cloud_mask, 99.9)

    # make figure (just RGB)
    fig = plt.figure()
    fig.set_size_inches([18,9])
    fig.set_tight_layout(True)
    ax1 = fig.add_subplot(111)
    ax1.axis('off')
    ax1.imshow(im_RGB)
    ax1.set_title(date + '   ' + satname, fontsize=16)

#    if im_RGB.shape[1] > 2*im_RGB.shape[0]:
#        ax1 = fig.add_subplot(311)
#        ax2 = fig.add_subplot(312)
#        ax3 = fig.add_subplot(313)
#    else:
#        ax1 = fig.add_subplot(131)
#        ax2 = fig.add_subplot(132)
#        ax3 = fig.add_subplot(133)
#    # RGB
#    ax1.axis('off')
#    ax1.imshow(im_RGB)
#    ax1.set_title(date + '   ' + satname, fontsize=16)
#    # NIR
#    ax2.axis('off')
#    ax2.imshow(im_NIR, cmap='seismic')
#    ax2.set_title('Near Infrared', fontsize=16)
#    # SWIR
#    ax3.axis('off')
#    ax3.imshow(im_SWIR, cmap='seismic')
#    ax3.set_title('Short-wave Infrared', fontsize=16)

    # save figure
    plt.rcParams['savefig.jpeg_quality'] = 100
    fig.savefig(os.path.join(filepath,
                             date + '_' + satname + '.jpg'), dpi=150)
    plt.close()


def save_jpg(metadata, settings, polygon, dates, **kwargs):
    """
    Saves a .jpg image for all the images contained in metadata.

    KV WRL 2018

    Arguments:
    -----------
    metadata: dict
        contains all the information about the satellite images that were downloaded
    settings: dict with the following keys
        'inputs': dict
            input parameters (sitename, filepath, polygon, dates, sat_list)
        'cloud_thresh': float
            value between 0 and 1 indicating the maximum cloud fraction in 
            the cropped image that is accepted
        'cloud_mask_issue': boolean
            True if there is an issue with the cloud mask and sand pixels
            are erroneously being masked on the images
            
    Returns:
    -----------
    Stores the images as .jpg in a folder named /preprocessed
    
    """
    
    sitename = settings['inputs']['sitename']
    cloud_thresh = settings['cloud_thresh']
    filepath_data = settings['inputs']['filepath']

    # create subfolder to store the jpg files
    filepath_jpg = os.path.join(filepath_data, sitename, 'jpg_files', 'preprocessed')
    if not os.path.exists(filepath_jpg):
            os.makedirs(filepath_jpg)

    # loop through satellite list
    for satname in metadata.keys():

        filepath = Toolbox.get_filepath(settings['inputs'],satname)
        filenames = metadata[satname]['filenames']

        # loop through images
        for i in range(len(filenames)):
            # image filename
            fn = Toolbox.get_filenames(filenames[i],filepath, satname)
            # read and preprocess image
            im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata = preprocess_single(fn, satname, settings, polygon, dates, savetifs=False)

            # compute cloud_cover percentage (with no data pixels)
            cloud_cover_combined = np.divide(sum(sum(cloud_mask.astype(int))),
                                    (cloud_mask.shape[0]*cloud_mask.shape[1]))
            if cloud_cover_combined > 0.99: # if 99% of cloudy pixels in image skip
                continue

            # remove no data pixels from the cloud mask (for example L7 bands of no data should not be accounted for)
            cloud_mask_adv = np.logical_xor(cloud_mask, im_nodata)
            # compute updated cloud cover percentage (without no data pixels)
            cloud_cover = np.divide(sum(sum(cloud_mask_adv.astype(int))),
                                    (sum(sum((~im_nodata).astype(int)))))
            # skip image if cloud cover is above threshold
            if cloud_cover > cloud_thresh or cloud_cover == 1:
                continue
            # save .jpg with date and satellite in the title
            date = filenames[i][:19]
            plt.ioff()  # turning interactive plotting off
            create_jpg(im_ms, cloud_mask, date, satname, filepath_jpg)

    # print the location where the images have been saved
    print('Satellite images saved as .jpg in ' + os.path.join(filepath_data, sitename,
                                                    'jpg_files', 'preprocessed'))

def get_reference_sl(metadata, settings, polygon, dates):
    """
    Allows the user to manually digitize a reference shoreline that is used seed
    the shoreline detection algorithm. The reference shoreline helps to detect 
    the outliers, making the shoreline detection more robust.

    KV WRL 2018

    Arguments:
    -----------
    metadata: dict
        contains all the information about the satellite images that were downloaded
    settings: dict with the following keys
        'inputs': dict
            input parameters (sitename, filepath, polygon, dates, sat_list)
        'cloud_thresh': float
            value between 0 and 1 indicating the maximum cloud fraction in 
            the cropped image that is accepted
        'cloud_mask_issue': boolean
            True if there is an issue with the cloud mask and sand pixels
            are erroneously being masked on the images
        'output_epsg': int
            output spatial reference system as EPSG code

    Returns:
    -----------
    reference_shoreline: np.array
        coordinates of the reference shoreline that was manually digitized. 
        This is also saved as a .pkl and .geojson file.

    """

    sitename = settings['inputs']['sitename']
    filepath_data = settings['inputs']['filepath']
    pts_coords = []
    # check if reference shoreline already exists in the corresponding folder
    filepath = os.path.join(filepath_data, sitename)
    filename = sitename + '_reference_shoreline.pkl'
    # if it exist, load it and return it
    if filename in os.listdir(filepath):
        print('Reference shoreline already exists and was loaded')
        with open(os.path.join(filepath, sitename + '_reference_shoreline.pkl'), 'rb') as f:
            refsl = pickle.load(f)
        return refsl
    
    # otherwise get the user to manually digitise a shoreline on S2, L8 or L5 images (no L7 because of scan line error)
    else:
        # first try to use S2 images (10m res for manually digitizing the reference shoreline)
        if 'S2' in metadata.keys():
            satname = 'S2'
            filepath = Toolbox.get_filepath(settings['inputs'],satname)
            filenames = metadata[satname]['filenames']
        # if no S2 images, try L8  (15m res in the RGB with pansharpening)
        elif not 'S2' in metadata.keys() and 'L8' in metadata.keys():
            satname = 'L8'
            filepath = Toolbox.get_filepath(settings['inputs'],satname)
            filenames = metadata[satname]['filenames']
        # if no S2 images and no L8, use L5 images (L7 images have black diagonal bands making it
        # hard to manually digitize a shoreline)
        elif not 'S2' in metadata.keys() and not 'L8' in metadata.keys() and 'L5' in metadata.keys():
            satname = 'L5'
            filepath = Toolbox.get_filepath(settings['inputs'],satname)
            filenames = metadata[satname]['filenames']
        else:
            raise Exception('You cannot digitize the shoreline on L7 images (because of gaps in the images), add another L8, S2 or L5 to your dataset.')
            
        # create figure
        fig, ax = plt.subplots(1,1, figsize=[18,9], tight_layout=True)
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        # loop trhough the images
        for i in range(len(filenames)):
            # read image
            fn = Toolbox.get_filenames(filenames[i],filepath, satname)
            fn=i
            im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata = preprocess_single(fn, satname, settings, polygon, dates, savetifs=False)
            
            if (im_ms is None) or (cloud_mask is None):
                continue

            # compute cloud_cover percentage (with no data pixels)
            cloud_cover_combined = np.divide(sum(sum(cloud_mask.astype(int))),
                                    (cloud_mask.shape[0]*cloud_mask.shape[1]))
            if cloud_cover_combined > 0.99: # if 99% of cloudy pixels in image skip
                continue

            # remove no data pixels from the cloud mask (for example L7 bands of no data should not be accounted for)
            cloud_mask_adv = np.logical_xor(cloud_mask, im_nodata)
            # compute updated cloud cover percentage (without no data pixels)
            cloud_cover = np.divide(sum(sum(cloud_mask_adv.astype(int))),
                                    (sum(sum((~im_nodata).astype(int)))))

            # skip image if cloud cover is above threshold
            if cloud_cover > settings['cloud_thresh']:
                continue

            # rescale image intensity for display purposes
            im_RGB = rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
            
            # plot the image RGB on a figure
            ax.axis('off')
            ax.imshow(im_RGB)

            # decide if the image if good enough for digitizing the shoreline
            ax.set_title('Press <right arrow> if image is clear enough to digitize the shoreline.\n' +
                      'If the image is cloudy press <left arrow> to get another image', fontsize=14)
            # set a key event to accept/reject the detections (see https://stackoverflow.com/a/15033071)
            # this variable needs to be immuatable so we can access it after the keypress event
            skip_image = False
            key_event = {}
            def press(event):
                # store what key was pressed in the dictionary
                key_event['pressed'] = event.key
            # let the user press a key, right arrow to keep the image, left arrow to skip it
            # to break the loop the user can press 'escape'
            while True:
                btn_keep = plt.text(1.1, 0.9, 'keep ⇨', size=12, ha="right", va="top",
                                    transform=ax.transAxes,
                                    bbox=dict(boxstyle="square", ec='k',fc='w'))
                btn_skip = plt.text(-0.1, 0.9, '⇦ skip', size=12, ha="left", va="top",
                                    transform=ax.transAxes,
                                    bbox=dict(boxstyle="square", ec='k',fc='w'))
                btn_esc = plt.text(0.5, 0, '<esc> to quit', size=12, ha="center", va="top",
                                    transform=ax.transAxes,
                                    bbox=dict(boxstyle="square", ec='k',fc='w'))
                plt.draw()
                fig.canvas.mpl_connect('key_press_event', press)
                plt.waitforbuttonpress()
                # after button is pressed, remove the buttons
                btn_skip.remove()
                btn_keep.remove()
                btn_esc.remove()
                # keep/skip image according to the pressed key, 'escape' to break the loop
                if key_event.get('pressed') == 'right':
                    skip_image = False
                    break
                elif key_event.get('pressed') == 'left':
                    skip_image = True
                    break
                elif key_event.get('pressed') == 'escape':
                    plt.close()
                    raise StopIteration('User cancelled checking shoreline detection')
                else:
                    plt.waitforbuttonpress()
                
            if skip_image:
                ax.clear()
                continue
            else:
                # create two new buttons
                add_button = plt.text(0, 0.9, 'add', size=16, ha="left", va="top",
                                       transform=plt.gca().transAxes,
                                       bbox=dict(boxstyle="square", ec='k',fc='w'))
                end_button = plt.text(1, 0.9, 'end', size=16, ha="right", va="top",
                                       transform=plt.gca().transAxes,
                                       bbox=dict(boxstyle="square", ec='k',fc='w'))
                # add multiple reference shorelines (until user clicks on <end> button)
                pts_sl = np.expand_dims(np.array([np.nan, np.nan]),axis=0)
                geoms = []
                while 1:
                    add_button.set_visible(False)
                    end_button.set_visible(False)
                    # update title (instructions)
                    ax.set_title('Click points along the shoreline (enough points to capture the beach curvature).\n' +
                              'Start at one end of the beach.\n' + 'When finished digitizing, click <ENTER>',
                              fontsize=14)
                    plt.draw()

                    # let user click on the shoreline
                    pts = ginput(n=50000, timeout=-1, show_clicks=True)
                    pts_pix = np.array(pts)
                    # convert pixel coordinates to world coordinates
                    pts_world = Toolbox.convert_pix2world(pts_pix[:,[1,0]], georef)

                    # interpolate between points clicked by the user (1m resolution)
                    pts_world_interp = np.expand_dims(np.array([np.nan, np.nan]),axis=0)
                    for k in range(len(pts_world)-1):
                        pt_dist = np.linalg.norm(pts_world[k,:]-pts_world[k+1,:])
                        xvals = np.arange(0,pt_dist)
                        yvals = np.zeros(len(xvals))
                        pt_coords = np.zeros((len(xvals),2))
                        pt_coords[:,0] = xvals
                        pt_coords[:,1] = yvals
                        phi = 0
                        deltax = pts_world[k+1,0] - pts_world[k,0]
                        deltay = pts_world[k+1,1] - pts_world[k,1]
                        phi = np.pi/2 - np.math.atan2(deltax, deltay)
                        tf = transform.EuclideanTransform(rotation=phi, translation=pts_world[k,:])
                        pts_world_interp = np.append(pts_world_interp,tf(pt_coords), axis=0)
                    pts_world_interp = np.delete(pts_world_interp,0,axis=0)

                    # save as geometry (to create .geojson file later)
                    geoms.append(geometry.LineString(pts_world_interp))

                    # convert to pixel coordinates and plot
                    pts_pix_interp = Toolbox.convert_world2pix(pts_world_interp, georef)
                    pts_sl = np.append(pts_sl, pts_world_interp, axis=0)
                    ax.plot(pts_pix_interp[:,0], pts_pix_interp[:,1], 'r--')
                    ax.plot(pts_pix_interp[0,0], pts_pix_interp[0,1],'ko')
                    ax.plot(pts_pix_interp[-1,0], pts_pix_interp[-1,1],'ko')

                    # update title and buttons
                    add_button.set_visible(True)
                    end_button.set_visible(True)
                    ax.set_title('click on <add> to digitize another shoreline or on <end> to finish and save the shoreline(s)',
                              fontsize=14)
                    plt.draw()

                    # let the user click again (<add> another shoreline or <end>)
                    pt_input = ginput(n=1, timeout=-1, show_clicks=False)
                    pt_input = np.array(pt_input)
                    # if user clicks on <end>, save the points and break the loop
                    if pt_input[0][0] > im_ms.shape[1]/2:
                        add_button.set_visible(False)
                        end_button.set_visible(False)
                        plt.title('Reference shoreline saved as ' + sitename + '_reference_shoreline.pkl and ' + sitename + '_reference_shoreline.geojson')
                        plt.draw()
                        ginput(n=1, timeout=3, show_clicks=False)
                        plt.close()
                        break
                pts_sl = np.delete(pts_sl,0,axis=0)
                # convert world image coordinates to user-defined coordinate system
                image_epsg = int(metadata[satname]['epsg'][0])
                pts_coords = Toolbox.convert_epsg(pts_sl, image_epsg, 32630)#settings['output_epsg'])
                # save the reference shoreline as .pkl
                filepath = os.path.join(filepath_data, sitename)
                with open(os.path.join(filepath, sitename + '_reference_shoreline.pkl'), 'wb') as f:
                    pickle.dump(pts_coords, f)

                return pts_coords
    # check if a shoreline was digitised
    if len(pts_coords) == 0:
        raise Exception('No cloud free images are available to digitise the reference shoreline,'+
                        'download more images and try again') 

    return pts_coords

# def preprocess_cloudfreeyearcomposite(fn, satname, settings, polygon):
#     """
#     In development
#     FM Nov 2022

#     Parameters
#     ----------
#     fn : TYPE
#         DESCRIPTION.
#     satname : TYPE
#         DESCRIPTION.
#     settings : TYPE
#         DESCRIPTION.
#     polygon : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     TYPE
#         DESCRIPTION.

#     """
    
#     cloud_mask_issue = settings['cloud_mask_issue']
    
#     point = ee.Geometry.Point(polygon[0][0]) 
    
#     years = settings['year_list']
#     dates = [str(years[fn])+'-01-01',str(years[fn])+'-12-30']
#     #=============================================================================================#
#     # L5 images
#     #=============================================================================================#
#     if satname == 'L5':
        
#         collection = ee.ImageCollection("LANDSAT/LT05/C01/T1_TOA").filterDate(dates[0], dates[-1]).filterBounds(point).select(['B1','B2','B3','B4','B5','QA_PIXEL'])
    
#         #img =  ee.Algorithms.Landsat.simpleComposite(collection)
#         img = collection.mosaic()

#         #cloud_scoree = img.getInfo().get('features')[fn]['properties']['CLOUD_COVER']/100
#         im_ms = geemap.ee_to_numpy(img, bands = ['B2','B3','B4','B5','QA_PIXEL'], region=ee.Geometry.Polygon(polygon))
        
#         # down-sample to 15 m (half of the original pixel size)
#         nrows = im_ms.shape[0]*2
#         ncols = im_ms.shape[1]*2

#         # create cloud mask
#         im_QA = im_ms[:,:,5]
#         im_ms = im_ms[:,:,:-1]
#         cloud_mask = create_cloud_mask(im_QA, satname, cloud_mask_issue)

#         # resize the image using bilinear interpolation (order 1)
#         im_ms = transform.resize(im_ms,(nrows, ncols), order=1, preserve_range=True,
#                                  mode='constant')
#         # resize the image using nearest neighbour interpolation (order 0)
#         cloud_mask = transform.resize(cloud_mask, (nrows, ncols), order=0, preserve_range=True,
#                                       mode='constant').astype('bool_')

#         # adjust georeferencing vector to the new image size
#         # scale becomes 15m and the origin is adjusted to the center of new top left pixel
        
#         x, y = polygon[0][3]
#         epsg = int(Landsat5.getInfo().get('features')[0]['bands'][0]['crs'].lstrip('EPSG:'))
#         inProj = Proj(init='epsg:4326')
#         string= 'epsg:'+str(epsg)
#         outProj = Proj(init=string)
#         eastings,northings = Transf(inProj,outProj,x,y)
        
#         georef = [eastings, 22.1, 0, northings, 0, -22.1]
        
        
#         # check if -inf or nan values on any band and eventually add those pixels to cloud mask        
#         im_nodata = np.zeros(cloud_mask.shape).astype(bool)
#         for k in range(im_ms.shape[2]):
#             im_inf = np.isin(im_ms[:,:,k], -np.inf)
#             im_nan = np.isnan(im_ms[:,:,k])
#             im_nodata = np.logical_or(np.logical_or(im_nodata, im_inf), im_nan)
#         # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
#         # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
#         im_zeros = np.ones(cloud_mask.shape).astype(bool)
#         for k in [1,3,4]: # loop through the Green, NIR and SWIR bands
#             im_zeros = np.logical_and(np.isin(im_ms[:,:,k],0), im_zeros)
#         # add zeros to im nodata
#         im_nodata = np.logical_or(im_zeros, im_nodata)   
#         # update cloud mask with all the nodata pixels
#         cloud_mask = np.logical_or(cloud_mask, im_nodata)
        
#         # no extra image for Landsat 5 (they are all 30 m bands)
#         im_extra = []

#     #=============================================================================================#
#     # L8 images
#     #=============================================================================================#
#     elif satname == 'L8':

#         collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA').filterDate(dates[0], dates[-1]).filterBounds(point).select(['B2','B3','B4','B5', 'B6','B7','B10','B11','QA_PIXEL'])
    
#         #img =  ee.Algorithms.Landsat.simpleComposite(collection)
#         img = collection.mosaic()
        
#         im_ms = geemap.ee_to_numpy(img, bands = ['B2','B3','B4','B5','QA_PIXEL'], region=ee.Geometry.Polygon(polygon))
        
        
#         if im_ms is None:
#             return None, None, None, None, None, None, None
        
#         cloud_scored = ee.Algorithms.Landsat.simpleCloudScore(img);

#         #Create a mask from the cloud score and combine it with the image mask.
#         mask = cloud_scored.select(['cloud']).lte(20);

#         #Apply the mask to the image and display the result.
#         masked = img.updateMask(mask);
        
        
#         x, y = polygon[0][3]
#         epsg = int(Landsat8.getInfo().get('features')[0]['bands'][0]['crs'].lstrip('EPSG:'))
#         inProj = Proj(init='epsg:4326')
#         string= 'epsg:'+str(epsg)
#         outProj = Proj(init=string)
#         eastings,northings = Transf(inProj,outProj,x,y)
        
#         georef = [eastings, 22.1, 0, northings, 0, -22.1]
              
#         im_pan = geemap.ee_to_numpy(img, bands = ['B8'], region=ee.Geometry.Polygon(polygon))
        
#         # size of pan image
#         nrows = im_pan.shape[0]
#         ncols = im_pan.shape[1]

#         # create cloud mask
#         im_QA = im_ms[:,:,5]
#         cloud_mask = create_cloud_mask(im_QA, satname, cloud_mask_issue)

#         # resize the image using bilinear interpolation (order 1)
#         im_ms = im_ms[:,:,:5]
#         im_ms = transform.resize(im_ms,(nrows, ncols), order=1, preserve_range=True,
#                                  mode='constant')
#         # resize the image using nearest neighbour interpolation (order 0)
#         cloud_mask = transform.resize(cloud_mask, (nrows, ncols), order=0, preserve_range=True,
#                                       mode='constant').astype('bool_')
#         # check if -inf or nan values on any band and eventually add those pixels to cloud mask        
#         im_nodata = np.zeros(cloud_mask.shape).astype(bool)
#         for k in range(im_ms.shape[2]):
#             im_inf = np.isin(im_ms[:,:,k], -np.inf)
#             im_nan = np.isnan(im_ms[:,:,k])
#             im_nodata = np.logical_or(np.logical_or(im_nodata, im_inf), im_nan)
#         # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
#         # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
#         im_zeros = np.ones(cloud_mask.shape).astype(bool)
#         for k in [1,3,4]: # loop through the Green, NIR and SWIR bands
#             im_zeros = np.logical_and(np.isin(im_ms[:,:,k],0), im_zeros)
#         # add zeros to im nodata
#         im_nodata = np.logical_or(im_zeros, im_nodata)   
#         # update cloud mask with all the nodata pixels
#         cloud_mask = np.logical_or(cloud_mask, im_nodata)
        
#         # pansharpen Blue, Green, Red (where there is overlapping with pan band in L8)
#         try:
#             im_ms_ps = pansharpen(im_ms[:,:,[0,1,2]], im_pan, cloud_mask)
#         except: # if pansharpening fails, keep downsampled bands (for long runs)
#             im_ms_ps = im_ms[:,:,[0,1,2]]
#         # add downsampled NIR and SWIR1 bands
#         im_ms_ps = np.append(im_ms_ps, im_ms[:,:,[3,4]], axis=2)

#         im_ms = im_ms_ps.copy()
#         # the extra image is the 15m panchromatic band
#         im_extra = im_pan

#     #=============================================================================================#
#     # S2 images
#     #=============================================================================================#
#     if satname == 'S2':
        
#         collection = ee.ImageCollection("COPERNICUS/S2").filterDate(dates[0], dates[-1]).filterBounds(point)
    
#         img =  collection.mosaic()
        
#         georef = Sentinel2.getInfo().get('features')[fn]['bands'][0]['crs_transform']
        
#         x, y = polygon[0][3]
#         epsg = int(Sentinel2.getInfo().get('features')[0]['bands'][0]['crs'].lstrip('EPSG:'))
#         inProj = Proj(init='epsg:4326')
#         string= 'epsg:'+str(epsg)
#         outProj = Proj(init=string)
#         eastings,northings = Transf(inProj,outProj,x,y)
        
#         georef = [eastings, 15, 0, northings, 0, -15]
        
#         img = ee.Image(Sentinel2.getInfo().get('features')[fn]['id'])
#         im10 = geemap.ee_to_numpy(img, bands = ['B2','B3','B4','B8'], region=ee.Geometry.Polygon(polygon))
        
#         if im10 is None:
#             return None, None, None, None, None, None, None
        
#         # read 10m bands (R,G,B,NIR)
#         """
#         fn10 = fn[0]
#         data = gdal.Open(fn10, gdal.GA_ReadOnly)
#         georef = np.array(data.GetGeoTransform())
#         bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
#         im10 = np.stack(bands, 2)
#         """
#         im10 = im10/10000 # TOA scaled to 10000

#         # if image contains only zeros (can happen with S2), skip the image
#         if sum(sum(sum(im10))) < 1:
#             im_ms = []
#             georef = []
#             # skip the image by giving it a full cloud_mask
#             cloud_mask = np.ones((im10.shape[0],im10.shape[1])).astype('bool')
#             return im_ms, georef, cloud_mask, [], [], []
        
#         # size of 10m bands
#         nrows = im10.shape[0]
#         ncols = im10.shape[1]

#         # read 20m band (SWIR1)
#         im20 = geemap.ee_to_numpy(img, bands = ['B11'], region=ee.Geometry.Polygon(polygon))
        
#         if im20 is None:
#             return None, None, None, None, None, None, None
        
#         im20 = im20[:,:,0]
#         im20 = im20/10000 # TOA scaled to 10000

#         # resize the image using bilinear interpolation (order 1)
#         im_swir = transform.resize(im20, (nrows, ncols), order=1, preserve_range=True,
#                                    mode='constant')
#         im_swir = np.expand_dims(im_swir, axis=2)
        
#         # append down-sampled SWIR1 band to the other 10m bands
#         im_ms = np.append(im10, im_swir, axis=2)
        
#         # create cloud mask using 60m QA band (not as good as Landsat cloud cover)
#         im60 = geemap.ee_to_numpy(img, bands = ['QA60'], region=ee.Geometry.Polygon(polygon))
        
#         if im60 is None:
#             return None, None, None, None, None, None, None
        
#         im_QA = im60[:,:,0]
#         cloud_mask = create_cloud_mask(im_QA, satname, cloud_mask_issue)
#         # resize the cloud mask using nearest neighbour interpolation (order 0)
#         cloud_mask = transform.resize(cloud_mask,(nrows, ncols), order=0, preserve_range=True,
#                                       mode='constant')
        
#         if cloud_mask is None:
#             return None, None, None, None, None, None, None
        
#         # check if -inf or nan values on any band and create nodata image
#         im_nodata = np.zeros(cloud_mask.shape).astype(bool)
#         for k in range(im_ms.shape[2]):
#             im_inf = np.isin(im_ms[:,:,k], -np.inf)
#             im_nan = np.isnan(im_ms[:,:,k])
#             im_nodata = np.logical_or(np.logical_or(im_nodata, im_inf), im_nan)
#         # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
#         # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
#         im_zeros = np.ones(im_nodata.shape).astype(bool)
#         im_zeros = np.logical_and(np.isin(im_ms[:,:,1],0), im_zeros) # Green
#         im_zeros = np.logical_and(np.isin(im_ms[:,:,3],0), im_zeros) # NIR
#         im_20_zeros = transform.resize(np.isin(im20,0),(nrows, ncols), order=0,
#                                        preserve_range=True, mode='constant').astype(bool)
#         im_zeros = np.logical_and(im_20_zeros, im_zeros) # SWIR1
#         # add to im_nodata
#         im_nodata = np.logical_or(im_zeros, im_nodata)
#         # dilate if image was merged as there could be issues at the edges
#         ##if 'merged' in fn10:
#         ##    im_nodata = morphology.dilation(im_nodata,morphology.square(5))
            
#         # update cloud mask with all the nodata pixels
#         cloud_mask = np.logical_or(cloud_mask, im_nodata)

#         # the extra image is the 20m SWIR band
#         im_extra = im20

#     return im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata