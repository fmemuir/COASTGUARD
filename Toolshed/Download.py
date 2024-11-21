"""
This module contains all the functions needed to download the satellite images
from the Google Earth Engine server

Author: Kilian Vos, Water Research Laboratory, University of New South Wales

COASTGUARD edits and updates: Freya Muir, University of Glasgow
"""

# load basic modules
import os
import numpy as np
import glob
from functools import reduce
import subprocess


# earth engine module
import ee

# modules to download, unzip and stack the images
from urllib.request import urlretrieve
import zipfile
import shutil
from osgeo import gdal
from requests.auth import HTTPBasicAuth
import requests
import time

# additional modules
from datetime import datetime, timedelta
import pytz
import pickle
from skimage import morphology, transform
from scipy import ndimage
import rasterio
import xarray as xr
import netCDF4
import copernicusmarine as cms



# CoastSat modules
from Toolshed import Image_Processing, Toolbox, gdal_merge

np.seterr(all='ignore') # raise/ignore divisions by 0 and nans


"""VedgeSat functions to obtain metadata from GEE for cloud-based processing"""

def CollectMetadata(inputs, Sat):
    """
    Compile Google Earth Engine metadata together to create a collection of image properties. 
    FM 2022 (updated Nov 2024)

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


        
def RetrieveImages(inputs, SLC=True):
    """
    Grab image metadata from Google Earth Engine and compile into list of image names.
    FM 2022 (updated Nov 2024)

    Parameters
    ----------
    inputs : dict
        Dictionary of input parameters from user.
    SLC : bool
        Decide whether to include Landsat 7 images after scan line corrector failure
        (31st May 2003).

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
        
    # If daterange is no, dates are made up of groups of date ranges
    if inputs['daterange'] == 'no':
        for satname in inputs['sat_list']:
            SatGroup = [] # resets with each satellite
            for dateset in inputs['dates']:
                if satname == 'L5':
                    Landsat5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_TOA").filterBounds(point).filterDate(dateset[0], dateset[1])
                    SatGroup.append(Landsat5)
                
                elif satname == 'L7':
                    if SLC is True: # if flag is True, include the Scan Line Correcter affected images
                        Landsat7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_TOA').filterBounds(point).filterDate(dateset[0], dateset[1]).filter(ee.Filter.lt('CLOUD_COVER', cloud_thresh))
                    else:
                        if '2003-05-31' > dateset[0]: # if first date is before SLC error date
                            Landsat7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_TOA').filterBounds(point).filterDate(dateset[0], '2003-05-31').filter(ee.Filter.lt('CLOUD_COVER', cloud_thresh))
                        else: # if dates are not within L7 error-free period
                            Landsat7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_TOA').filterBounds(point).filterDate('2003-05-30', '2003-05-31').filter(ee.Filter.lt('CLOUD_COVER', cloud_thresh))
                    SatGroup.append(Landsat7)
                    
                elif satname == 'L8':
                    Landsat8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA').filterBounds(point).filterDate(dateset[0], dateset[1]).filter(ee.Filter.lt('CLOUD_COVER', cloud_thresh))
                    SatGroup.append(Landsat8)
                    
                elif satname == 'L9':
                    Landsat9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_TOA').filterBounds(point).filterDate(dateset[0], dateset[1]).filter(ee.Filter.lt('CLOUD_COVER', cloud_thresh))
                    SatGroup.append(Landsat9)
                    
                elif satname == 'S2':
                    Sentinel2 = ee.ImageCollection("COPERNICUS/S2").filterBounds(point).filterDate(dateset[0], dateset[1]).filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloud_thresh))
                    SatGroup.append(Sentinel2)
                    
                else: # local images i.e. Planet
                    Sat = LocalImageRetrieval(inputs)
                    return Sat
                    
            # Merge each ImageCollection in list
            SatGroupMrg = reduce(lambda col1, col2: col1.merge(col2), SatGroup)
            # Check if per-satellite collection is empty (and remove satname if so)
            if SatGroupMrg.size().getInfo() == 0:
                inputs['sat_list'].remove(satname)
            else:
                Sat.append(SatGroupMrg)
        return Sat
    # otherise, dates is just list of start and end date strings
    else:
        if 'L5' in inputs['sat_list']:
            Landsat5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_TOA").filterBounds(point).filterDate(inputs['dates'][0], inputs['dates'][1])
            if Landsat5.size().getInfo() == 0:
                inputs['sat_list'].remove('L5')
            else:
                Sat.append(Landsat5)
        if 'L7' in inputs['sat_list']:
            if SLC is True: # if flag is True, include the Scan Line Correcter affected images
                Landsat7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_TOA').filterBounds(point).filterDate(inputs['dates'][0], inputs['dates'][1]).filter(ee.Filter.lt('CLOUD_COVER', cloud_thresh))
            else:
                if '2003-05-31' > inputs['dates'][0]: # if first date is before SLC error date
                    Landsat7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_TOA').filterBounds(point).filterDate(inputs['dates'][0], '2003-05-31').filter(ee.Filter.lt('CLOUD_COVER', cloud_thresh))
                else: # if dates are not within L7 error-free period
                    Landsat7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_TOA').filterBounds(point).filterDate('2003-05-30', '2003-05-31').filter(ee.Filter.lt('CLOUD_COVER', cloud_thresh))
            if Landsat7.size().getInfo() == 0:
                inputs['sat_list'].remove('L7')
            else:
                Sat.append(Landsat7)
        if 'L8' in inputs['sat_list']:
            Landsat8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA').filterBounds(point).filterDate(inputs['dates'][0], inputs['dates'][1]).filter(ee.Filter.lt('CLOUD_COVER', cloud_thresh))
            if Landsat8.size().getInfo() == 0:
                inputs['sat_list'].remove('L8')
            else:
                Sat.append(Landsat8)
        if 'L9' in inputs['sat_list']:
            Landsat9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_TOA').filterBounds(point).filterDate(inputs['dates'][0], inputs['dates'][1]).filter(ee.Filter.lt('CLOUD_COVER', cloud_thresh))
            if Landsat9.size().getInfo() == 0:
                inputs['sat_list'].remove('L9')
            else:
                Sat.append(Landsat9)
        if 'S2' in inputs['sat_list']:
            Sentinel2 = ee.ImageCollection("COPERNICUS/S2").filterBounds(point).filterDate(inputs['dates'][0], inputs['dates'][1]).filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloud_thresh))
            if Sentinel2.size().getInfo() == 0:
                inputs['sat_list'].remove('S2')
            else:
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
        # If using specific region data (NW Atlantic Shelf)
        if CMScmd['lonmin'] > -16 and CMScmd['lonmax'] < 13 and CMScmd['latmin'] > 46 and CMScmd['latmax'] < 63:
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
        else:
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
    
    


# ------------------------------------------------------------------------------------

# Main function to download images from the EarthEngine server
def retrieve_images(inputs):
    """
    Downloads all images from Landsat 5, Landsat 7, Landsat 8 and Sentinel-2
    covering the area of interest and acquired between the specified dates.
    The downloaded images are in .TIF format and organised in subfolders, divided
    by satellite mission. The bands are also subdivided by pixel resolution.

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

    Returns:
    -----------
    metadata: dict
        contains the information about the satellite images that were downloaded:
        date, filename, georeferencing accuracy and image coordinate reference system

    """
    
    # initialise connection with GEE server
    ee.Initialize()

    testdata = dict([])
    
    # check image availabiliy and retrieve list of images
    im_dict_T1, im_dict_T2 = check_images_available(inputs)

    # if user also wants to download T2 images, merge both lists
    if 'include_T2' in inputs.keys():
        for key in inputs['sat_list']:
            if key == 'S2': continue
            else: im_dict_T1[key] += im_dict_T2[key]

    # remove UTM duplicates in S2 collections (they provide several projections for same images)
    if 'S2' in inputs['sat_list'] and len(im_dict_T1['S2'])>0:
        im_dict_T1['S2'] = filter_S2_collection(im_dict_T1['S2'])

    # create a new directory for this site with the name of the site
    im_folder = os.path.join(inputs['filepath'],inputs['sitename'])
    if not os.path.exists(im_folder): os.makedirs(im_folder)

    print('\nDownloading images:')
    suffix = '.tif'
    for satname in im_dict_T1.keys():
        print('%s: %d images'%(satname,len(im_dict_T1[satname])))
        # create subfolder structure to store the different bands
        filepaths = create_folder_structure(im_folder, satname)
        # initialise variables and loop through images
        georef_accs = []; filenames = []; all_names = []; im_epsg = []
        for i in range(len(im_dict_T1[satname])):

            im_meta = im_dict_T1[satname][i]

            # get time of acquisition (UNIX time) and convert to datetime
            t = im_meta['properties']['system:time_start']
            im_timestamp = datetime.fromtimestamp(t/1000, tz=pytz.utc)
            im_date = im_timestamp.strftime('%Y-%m-%d-%H-%M-%S')

            # get epsg code
            im_epsg.append(int(im_meta['bands'][0]['crs'][5:]))

            # get geometric accuracy
            if satname in ['L5','L7','L8']:
                if 'GEOMETRIC_RMSE_MODEL' in im_meta['properties'].keys():
                    acc_georef = im_meta['properties']['GEOMETRIC_RMSE_MODEL']
                else:
                    acc_georef = 12 # default value of accuracy (RMSE = 12m)
            elif satname in ['S2']:
                # Sentinel-2 products don't provide a georeferencing accuracy (RMSE as in Landsat)
                # but they have a flag indicating if the geometric quality control was passed or failed
                # if passed a value of 1 is stored if failed a value of -1 is stored in the metadata
                # the name of the property containing the flag changes across the S2 archive
                # check which flag name is used for the image and store the 1/-1 for acc_georef
                flag_names = ['GEOMETRIC_QUALITY_FLAG', 'GEOMETRIC_QUALITY', 'quality_check', 'GENERAL_QUALITY_FLAG']
                for key in flag_names: 
                    if key in im_meta['properties'].keys(): break
                if im_meta['properties'][key] == 'PASSED': acc_georef = 1
                else: acc_georef = -1
            georef_accs.append(acc_georef)

            bands = dict([])
            im_fn = dict([])
            # first delete dimensions key from dictionnary
            # otherwise the entire image is extracted (don't know why)
            im_bands = im_meta['bands']
            for j in range(len(im_bands)): del im_bands[j]['dimensions']

            # Landsat 5 download
            if satname == 'L5':
                bands[''] = [im_bands[0], im_bands[1], im_bands[2], im_bands[3],
                             im_bands[4], im_bands[7]]
                im_fn[''] = im_date + '_' + satname + '_' + inputs['sitename'] + suffix
                # if two images taken at the same date add 'dup' to the name (duplicate)
                if any(im_fn[''] in _ for _ in all_names):
                    im_fn[''] = im_date + '_' + satname + '_' + inputs['sitename'] + '_dup' + suffix
                all_names.append(im_fn[''])
                filenames.append(im_fn[''])
                # download .tif from EE
                while True:
                    try:
                        im_ee = ee.Image(im_meta['id'])
                        testdata[satname].append(im_ee)
                        local_data = download_tif(im_ee, inputs['polygon'], bands[''], filepaths[1])
                        break
                    except:
                        continue
                # rename the file as the image is downloaded as 'data.tif'
                try:
                    os.rename(local_data, os.path.join(filepaths[1], im_fn['']))
                except: # overwrite if already exists
                    os.remove(os.path.join(filepaths[1], im_fn['']))
                    os.rename(local_data, os.path.join(filepaths[1], im_fn['']))
                # metadata for .txt file
                filename_txt = im_fn[''].replace('.tif','')
                metadict = {'filename':im_fn[''],'acc_georef':georef_accs[i],
                            'epsg':im_epsg[i]}

            # Landsat 7 and 8 download
            elif satname in ['L7', 'L8']:
                if satname == 'L7':
                    bands['pan'] = [im_bands[8]] # panchromatic band
                    bands['ms'] = [im_bands[0], im_bands[1], im_bands[2], im_bands[3],
                                   im_bands[4], im_bands[9]] # multispectral bands
                else:
                    bands['pan'] = [im_bands[7]] # panchromatic band
                    bands['ms'] = [im_bands[1], im_bands[2], im_bands[3], im_bands[4],
                                   im_bands[5], im_bands[11]] # multispectral bands
                for key in bands.keys():
                    im_fn[key] = im_date + '_' + satname + '_' + inputs['sitename'] + '_' + key + suffix
                # if two images taken at the same date add 'dup' to the name (duplicate)
                if any(im_fn['pan'] in _ for _ in all_names):
                    for key in bands.keys():
                        im_fn[key] = im_date + '_' + satname + '_' + inputs['sitename'] + '_' + key + '_dup' + suffix
                all_names.append(im_fn['pan'])
                filenames.append(im_fn['pan'])
                # download .tif from EE (panchromatic band and multispectral bands)
                while True:
                    try:
                        im_ee = ee.Image(im_meta['id'])
                        testdata[satname].append(im_ee)
                        local_data_pan = download_tif(im_ee, inputs['polygon'], bands['pan'], filepaths[1])
                        local_data_ms = download_tif(im_ee, inputs['polygon'], bands['ms'], filepaths[2])
                        break
                    except:
                        continue
                # rename the files as the image is downloaded as 'data.tif'
                try: # panchromatic
                    os.rename(local_data_pan, os.path.join(filepaths[1], im_fn['pan']))
                except: # overwrite if already exists
                    os.remove(os.path.join(filepaths[1], im_fn['pan']))
                    os.rename(local_data_pan, os.path.join(filepaths[1], im_fn['pan']))
                try: # multispectral
                    os.rename(local_data_ms, os.path.join(filepaths[2], im_fn['ms']))
                except: # overwrite if already exists
                    os.remove(os.path.join(filepaths[2], im_fn['ms']))
                    os.rename(local_data_ms, os.path.join(filepaths[2], im_fn['ms']))
                # metadata for .txt file
                filename_txt = im_fn['pan'].replace('_pan','').replace('.tif','')
                metadict = {'filename':im_fn['pan'],'acc_georef':georef_accs[i],
                            'epsg':im_epsg[i]}

            # Sentinel-2 download
            elif satname in ['S2']:
                bands['10m'] = [im_bands[1], im_bands[2], im_bands[3], im_bands[7]] # multispectral bands
                bands['20m'] = [im_bands[11]] # SWIR band
                bands['60m'] = [im_bands[15]] # QA band
                for key in bands.keys():
                    im_fn[key] = im_date + '_' + satname + '_' + inputs['sitename'] + '_' + key + suffix
                # if two images taken at the same date add 'dup' to the name (duplicate)
                if any(im_fn['10m'] in _ for _ in all_names):
                    for key in bands.keys():
                        im_fn[key] = im_date + '_' + satname + '_' + inputs['sitename'] + '_' + key + '_dup2' + suffix
                    # also check for triplicates (only on S2 imagery) and add 'tri' to the name
                    if im_fn['10m'] in all_names:
                        for key in bands.keys():
                            im_fn[key] = im_date + '_' + satname + '_' + inputs['sitename'] + '_' + key + '_dup3' + suffix
                        # also check for quadruplicates (only on S2 imagery) add 'qua' to the name
                        if im_fn['10m'] in all_names:
                            for key in bands.keys():
                                im_fn[key] = im_date + '_' + satname + '_' + inputs['sitename'] + '_' + key + '_dup4' + suffix
                all_names.append(im_fn['10m'])
                filenames.append(im_fn['10m'])
                # download .tif from EE (multispectral bands at 3 different resolutions)
                while True:
                    try:
                        im_ee = ee.Image(im_meta['id'])
                        testdata[satname].append(im_ee)
                        local_data_10m = download_tif(im_ee, inputs['polygon'], bands['10m'], filepaths[1])
                        local_data_20m = download_tif(im_ee, inputs['polygon'], bands['20m'], filepaths[2])
                        local_data_60m = download_tif(im_ee, inputs['polygon'], bands['60m'], filepaths[3])
                        break
                    except:
                        continue
                # rename the files as the image is downloaded as 'data.tif'
                try: # 10m
                    os.rename(local_data_10m, os.path.join(filepaths[1], im_fn['10m']))
                except: # overwrite if already exists
                    os.remove(os.path.join(filepaths[1], im_fn['10m']))
                    os.rename(local_data_10m, os.path.join(filepaths[1], im_fn['10m']))
                try: # 20m
                    os.rename(local_data_20m, os.path.join(filepaths[2], im_fn['20m']))
                except: # overwrite if already exists
                    os.remove(os.path.join(filepaths[2], im_fn['20m']))
                    os.rename(local_data_20m, os.path.join(filepaths[2], im_fn['20m']))
                try: # 60m
                    os.rename(local_data_60m, os.path.join(filepaths[3], im_fn['60m']))
                except: # overwrite if already exists
                    os.remove(os.path.join(filepaths[3], im_fn['60m']))
                    os.rename(local_data_60m, os.path.join(filepaths[3], im_fn['60m']))
                # metadata for .txt file
                filename_txt = im_fn['10m'].replace('_10m','').replace('.tif','')
                metadict = {'filename':im_fn['10m'],'acc_georef':georef_accs[i],
                            'epsg':im_epsg[i]}

            # write metadata
            with open(os.path.join(filepaths[0],filename_txt + '.txt'), 'w') as f:
                for key in metadict.keys():
                    f.write('%s\t%s\n'%(key,metadict[key]))
            # print percentage completion for user
            print('\r%d%%' %int((i+1)/len(im_dict_T1[satname])*100), end='')

        print('')

    # once all images have been downloaded, load metadata from .txt files
    metadata = get_metadata(inputs)

    # merge overlapping images (necessary only if the polygon is at the boundary of an image)
    if 'S2' in metadata.keys():
        try:
            metadata = merge_overlapping_images(metadata,inputs)
        except:
            print('WARNING: there was an error while merging overlapping S2 images,'+
                  ' please open an issue on Github at https://github.com/kvos/CoastSat/issues'+
                  ' and include your script so we can find out what happened.')

    # save metadata dict
    with open(os.path.join(im_folder, inputs['sitename'] + '_metadata' + '.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    return testdata



# function to load the metadata if images have already been downloaded
def get_metadata(inputs):
    """
    Gets the metadata from the downloaded images by parsing .txt files located
    in the \meta subfolder.

    KV WRL 2018

    Arguments:
    -----------
    inputs: dict with the following fields
        'sitename': str
            name of the site
        'filepath_data': str
            filepath to the directory where the images are downloaded

    Returns:
    -----------
    metadata: dict
        contains the information about the satellite images that were downloaded:
        date, filename, georeferencing accuracy and image coordinate reference system

    """
    # directory containing the images
    filepath = os.path.join(inputs['filepath'],inputs['sitename'])
    # initialize metadata dict
    metadata = dict([])
    # loop through the satellite missions
    for satname in ['L5','L7','L8','S2']:
        # if a folder has been created for the given satellite mission
        if satname in os.listdir(filepath):
            # update the metadata dict
            metadata[satname] = {'filenames':[], 'acc_georef':[], 'epsg':[], 'dates':[]}
            # directory where the metadata .txt files are stored
            filepath_meta = os.path.join(filepath, satname, 'meta')
            # get the list of filenames and sort it chronologically
            filenames_meta = os.listdir(filepath_meta)
            filenames_meta.sort()
            # loop through the .txt files
            for im_meta in filenames_meta:
                # read them and extract the metadata info: filename, georeferencing accuracy
                # epsg code and date
                with open(os.path.join(filepath_meta, im_meta), 'r') as f:
                    filename = f.readline().split('\t')[1].replace('\n','')
                    acc_georef = float(f.readline().split('\t')[1].replace('\n',''))
                    epsg = int(f.readline().split('\t')[1].replace('\n',''))
                date_str = filename[0:19]
                date = pytz.utc.localize(datetime(int(date_str[:4]),int(date_str[5:7]),
                                                  int(date_str[8:10]),int(date_str[11:13]),
                                                  int(date_str[14:16]),int(date_str[17:19])))
                # store the information in the metadata dict
                metadata[satname]['filenames'].append(filename)
                metadata[satname]['acc_georef'].append(acc_georef)
                metadata[satname]['epsg'].append(epsg)
                metadata[satname]['dates'].append(date)

    # save a .pkl file containing the metadata dict
    with open(os.path.join(filepath, inputs['sitename'] + '_metadata' + '.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    return metadata
###################################################################################################
# AUXILIARY FUNCTIONS
###################################################################################################

def check_images_available(inputs):
    """
    Create the structure of subfolders for each satellite mission

    KV WRL 2018
    Updated FM Apr 2023

    Arguments:
    -----------
    inputs: dict
        inputs dictionnary

    Returns:
    -----------
    im_dict_T1: list of dict
        list of images in Tier 1 and Level-1C
    im_dict_T2: list of dict
        list of images in Tier 2 (Landsat only)
    """
    
    point = ee.Geometry.Point(inputs['polygon'][0][0])
    
    # check if dates are in correct order
    dates = [datetime.strptime(_,'%Y-%m-%d') for _ in inputs['dates']]
    if  dates[-1] <= dates[0]:
        raise Exception('Verify that your dates are in the correct order')

    # check if EE was initialised or not
    try:
        ee.ImageCollection('LANDSAT/LT05/C02/T1_TOA')
    except:
        ee.Initialize()

    print('Images available between %s and %s:'%(inputs['dates'][0],inputs['dates'][-1]), end='\n')
    # check how many images are available in Tier 1 and Sentinel Level-1C
    # col_names_T1 = {'L5':'LANDSAT/LT05/C01/T1_TOA',
    #              'L7':'LANDSAT/LE07/C01/T1_TOA',
    #              'L8':'LANDSAT/LC08/C01/T1_TOA',
    #              'L9':'LANDSAT/LC09/C02/T1_TOA',
    #              'S2':'COPERNICUS/S2'}
    # New Collection 2 names
    col_names_T1 = {'L5':'LANDSAT/LT05/C02/T1_TOA',
                    'L7':'LANDSAT/LE07/C02/T1_TOA',
                    'L8':'LANDSAT/LC08/C02/T1_TOA',
                    'L9':'LANDSAT/LC09/C02/T1_TOA',
                    'S2':'COPERNICUS/S2'}
    

    print('- In Landsat Tier 1 & Sentinel-2 Level-1C:')
    im_dict_T1 = dict([])
    sum_img = 0
    for satname in inputs['sat_list']:

        # get list of images in EE collection
        while True:
            try:
                ee_col = ee.ImageCollection(col_names_T1[satname])
                if inputs['daterange'] == 'no': #FM: list of unique dates instead
                #NEED TO FIX THIS due to date props having specific format
                    ee_col_2 = ee_col.map(ee_col.set('simpleTime',ee.Date(ee_col.get('system:time_start')).format('YYYY-MM-dd')))
                    col = ee_col_2.filterBounds(point).filter(ee.Filter.inList("simpleTime", inputs['dates']))
                else:
                    col = ee_col.filterBounds(point)\
                                .filterDate(inputs['dates'][0], inputs['dates'][-1])
                                #.filter(ee.Filter.eq('CLOUD_COVER',inputs['cloud_thresh']))
                    
                im_list = col.getInfo().get('features')
                break
            except:
                continue
        # remove very cloudy images (>95% cloud cover)
        im_list_upt = remove_cloudy_images(im_list, satname)
        sum_img = sum_img + len(im_list_upt)
        print('  %s: %d images'%(satname,len(im_list_upt)))
        im_dict_T1[satname] = im_list_upt

    print('  Total: %d images'%sum_img)

    # in only S2 is in sat_list, stop here
    if len(inputs['sat_list']) == 1 and inputs['sat_list'][0] == 'S2':
        return inputs

    # otherwise check how many images are available in Landsat Tier 2
    col_names_T2 = {'L5':'LANDSAT/LT05/C02/T2_TOA',
                    'L7':'LANDSAT/LE07/C02/T2_TOA',
                    'L8':'LANDSAT/LC08/C02/T2_TOA',
                    'L9':'LANDSAT/LC09/C02/T2_TOA'}
    print('- In Landsat Tier 2:', end='\n')
    im_dict_T2 = dict([])
    sum_img = 0
    for satname in inputs['sat_list']:
        if satname == 'S2': continue
        # get list of images in EE collection
        while True:
            try:
                ee_col = ee.ImageCollection(col_names_T2[satname])
                col = ee_col.filterBounds(ee.Geometry.Polygon(inputs['polygon']))\
                            .filterDate(inputs['dates'][0],inputs['dates'][-1])
                            #.filter(ee.Filter.eq('CLOUD_COVER',inputs['cloud_thresh']))
                im_list = col.getInfo().get('features')
                break
            except:
                continue
        # remove very cloudy images (>95% cloud cover)
        im_list_upt = remove_cloudy_images(im_list, satname)
        sum_img = sum_img + len(im_list_upt)
        print('  %s: %d images'%(satname,len(im_list_upt)))
        im_dict_T2[satname] = im_list_upt

    print('  Total: %d images'%sum_img)
    
    # If any satnames returned 0 images, remove them from inputs
    for satdict in im_dict_T1.keys():
        if len(im_dict_T1[satdict]) == 0:
            inputs['sat_list'].remove(satdict)

    return inputs


def download_tif(image, polygon, bandsId, filepath):
    """
    Downloads a .TIF image from the ee server. The image is downloaded as a
    zip file then moved to the working directory, unzipped and stacked into a
    single .TIF file.

    Two different codes based on which version of the earth-engine-api is being
    used.

    KV WRL 2018

    Arguments:
    -----------
    image: ee.Image
        Image object to be downloaded
    polygon: list
        polygon containing the lon/lat coordinates to be extracted
        longitudes in the first column and latitudes in the second column
    bandsId: list of dict
        list of bands to be downloaded
    filepath: location where the temporary file should be saved

    Returns:
    -----------
    Downloads an image in a file named data.tif

    """

    # for the old version of ee only
    if int(ee.__version__[-3:]) <= 201:
        url = ee.data.makeDownloadUrl(ee.data.getDownloadId({
            'image': image.serialize(),
            'region': polygon,
            'bands': bandsId,
            'filePerBand': 'false',
            'name': 'data',
            }))
        local_zip, headers = urlretrieve(url)
        with zipfile.ZipFile(local_zip) as local_zipfile:
            return local_zipfile.extract('data.tif', filepath)
    # for the newer versions of ee
    else:
        # crop image on the server and create url to download
        url = ee.data.makeDownloadUrl(ee.data.getDownloadId({
            'image': image,
            'region': polygon,
            'bands': bandsId,
            'filePerBand': 'false',
            'name': 'data',
            }))
        # download zipfile with the cropped bands
        local_zip, headers = urlretrieve(url)
        # move zipfile from temp folder to data folder
        dest_file = os.path.join(filepath, 'imagezip')
        shutil.move(local_zip,dest_file)
        # unzip file
        with zipfile.ZipFile(dest_file) as local_zipfile:
            for fn in local_zipfile.namelist():
                local_zipfile.extract(fn, filepath)
            # filepath + filename to single bands
            fn_tifs = [os.path.join(filepath,_) for _ in local_zipfile.namelist()]
        # stack bands into single .tif
        outds = gdal.BuildVRT(os.path.join(filepath,'stacked.vrt'), fn_tifs, separate=True)
        outds = gdal.Translate(os.path.join(filepath,'data.tif'), outds)
        # delete single-band files
        for fn in fn_tifs: os.remove(fn)
        # delete .vrt file
        os.remove(os.path.join(filepath,'stacked.vrt'))
        # delete zipfile
        os.remove(dest_file)
        # delete data.tif.aux (not sure why this is created)
        if os.path.exists(os.path.join(filepath,'data.tif.aux')):
            os.remove(os.path.join(filepath,'data.tif.aux'))
        # return filepath to stacked file called data.tif
        return os.path.join(filepath,'data.tif')


def create_folder_structure(im_folder, satname):
    """
    Create the structure of subfolders for each satellite mission

    KV WRL 2018

    Arguments:
    -----------
    im_folder: str
        folder where the images are to be downloaded
    satname:
        name of the satellite mission

    Returns:
    -----------
    filepaths: list of str
        filepaths of the folders that were created
    """

    # one folder for the metadata (common to all satellites)
    filepaths = [os.path.join(im_folder, satname, 'meta')]
    # subfolders depending on satellite mission
    if satname == 'L5':
        filepaths.append(os.path.join(im_folder, satname, '30m'))
    elif satname in ['L7','L8']:
        filepaths.append(os.path.join(im_folder, satname, 'pan'))
        filepaths.append(os.path.join(im_folder, satname, 'ms'))
    elif satname in ['S2']:
        filepaths.append(os.path.join(im_folder, satname, '10m'))
        filepaths.append(os.path.join(im_folder, satname, '20m'))
        filepaths.append(os.path.join(im_folder, satname, '60m'))
    # create the subfolders if they don't exist already
    for fp in filepaths:
        if not os.path.exists(fp): os.makedirs(fp)

    return filepaths


def remove_cloudy_images(im_list, satname, prc_cloud_cover=95):
    """
    Removes from the EE collection very cloudy images (>95% cloud cover)

    KV WRL 2018
    Updated for L9 by FM Oct 2024

    Arguments:
    -----------
    im_list: list
        list of images in the collection
    satname:
        name of the satellite mission
    prc_cloud_cover: int
        percentage of cloud cover acceptable on the images

    Returns:
    -----------
    im_list_upt: list
        updated list of images
    """

    # remove very cloudy images from the collection (>95% cloud)
    if satname in ['L5','L7','L8','L9']:
        cloud_property = 'CLOUD_COVER'
    elif satname in ['S2']:
        cloud_property = 'CLOUDY_PIXEL_PERCENTAGE'
    cloud_cover = [_['properties'][cloud_property] for _ in im_list]
    if np.any([_ > prc_cloud_cover for _ in cloud_cover]):
        idx_delete = np.where([_ > prc_cloud_cover for _ in cloud_cover])[0]
        im_list_upt = [x for k,x in enumerate(im_list) if k not in idx_delete]
    else:
        im_list_upt = im_list

    return im_list_upt


def filter_S2_collection(im_list):
    """
    Removes duplicates from the EE collection of Sentinel-2 images (many duplicates)
    Finds the images that were acquired at the same time but have different utm zones.

    KV WRL 2018

    Arguments:
    -----------
    im_list: list
        list of images in the collection

    Returns:
    -----------
    im_list_flt: list
        filtered list of images
    """

    # get datetimes
    timestamps = [datetime.fromtimestamp(_['properties']['system:time_start']/1000,
                                         tz=pytz.utc) for _ in im_list]
    # get utm zone projections
    utm_zones = np.array([int(_['bands'][0]['crs'][5:]) for _ in im_list])
    if len(np.unique(utm_zones)) == 1:
        return im_list
    else:
        utm_zone_selected =  np.max(np.unique(utm_zones))
        # find the images that were acquired at the same time but have different utm zones
        idx_all = np.arange(0,len(im_list),1)
        idx_covered = np.ones(len(im_list)).astype(bool)
        idx_delete = []
        i = 0
        while 1:
            same_time = np.abs([(timestamps[i]-_).total_seconds() for _ in timestamps]) < 60*60*24
            idx_same_time = np.where(same_time)[0]
            same_utm = utm_zones == utm_zone_selected
            # get indices that have the same time (less than 24h apart) but not the same utm zone
            idx_temp = np.where([same_time[j] == True and same_utm[j] == False for j in idx_all])[0]
            idx_keep = idx_same_time[[_ not in idx_temp for _ in idx_same_time]]
            # if more than 2 images with same date and same utm, drop the last ones
            if len(idx_keep) > 2:
               idx_temp = np.append(idx_temp,idx_keep[-(len(idx_keep)-2):])
            for j in idx_temp:
                idx_delete.append(j)
            idx_covered[idx_same_time] = False
            if np.any(idx_covered):
                i = np.where(idx_covered)[0][0]
            else:
                break
        # update the collection by deleting all those images that have same timestamp
        # and different utm projection
        im_list_flt = [x for k,x in enumerate(im_list) if k not in idx_delete]

    return im_list_flt


def merge_overlapping_images(metadata,inputs):
    """
    Merge simultaneous overlapping images that cover the area of interest.
    When the area of interest is located at the boundary between 2 images, there
    will be overlap between the 2 images and both will be downloaded from Google
    Earth Engine. This function merges the 2 images, so that the area of interest
    is covered by only 1 image.

    KV WRL 2018

    Arguments:
    -----------
    metadata: dict
        contains all the information about the satellite images that were downloaded
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

    Returns:
    -----------
    metadata_updated: dict
        updated metadata

    """

    # only for Sentinel-2 at this stage (not sure if this is needed for Landsat images)
    sat = 'S2'
    filepath = os.path.join(inputs['filepath'], inputs['sitename'])
    filenames = metadata[sat]['filenames']
    total_images = len(filenames)
    # nested function
    def duplicates_dict(lst):
        "return duplicates and indices"
        def duplicates(lst, item):
                return [i for i, x in enumerate(lst) if x == item]
        return dict((x, duplicates(lst, x)) for x in set(lst) if lst.count(x) > 1)    
      
    # first pass on images that have the exact same timestamp
    duplicates = duplicates_dict([_.split('_')[0] for _ in filenames])
    total_removed_step1 = 0
    if len(duplicates) > 0:
        # loop through each pair of duplicates and merge them
        for key in duplicates.keys():
            idx_dup = duplicates[key]
            # get full filenames (3 images and .txtt) for each index and bounding polygons
            fn_im, polygons, im_epsg = [], [], []
            for index in range(len(idx_dup)):
                # image names
                fn_im.append([os.path.join(filepath, 'S2', '10m', filenames[idx_dup[index]]),
                      os.path.join(filepath, 'S2', '20m',  filenames[idx_dup[index]].replace('10m','20m')),
                      os.path.join(filepath, 'S2', '60m',  filenames[idx_dup[index]].replace('10m','60m')),
                      os.path.join(filepath, 'S2', 'meta', filenames[idx_dup[index]].replace('_10m','').replace('.tif','.txt'))])
                # bounding polygons
                polygons.append(Toolbox.get_image_bounds(fn_im[index][0]))
                im_epsg.append(metadata[sat]['epsg'][idx_dup[index]])
            # check if epsg are the same, print a warning message
            if len(np.unique(im_epsg)) > 1:
                print('WARNING: there was an error as two S2 images do not have the same epsg,'+
                      ' please open an issue on Github at https://github.com/kvos/CoastSat/issues'+
                      ' and include your script so I can find out what happened.')
            # find which images contain other images
            contain_bools_list = []
            for i,poly1 in enumerate(polygons):
                contain_bools = []
                for k,poly2 in enumerate(polygons):
                    if k == i: 
                        contain_bools.append(True)
                        # print('%d*: '%k+str(poly1.contains(poly2)))
                    else:
                        # print('%d: '%k+str(poly1.contains(poly2)))
                        contain_bools.append(poly1.contains(poly2))
                contain_bools_list.append(contain_bools)
            # look if one image contains all the others
            contain_all = [np.all(_) for _ in contain_bools_list]
            # if one image contains all the others, keep that one and delete the rest
            if np.any(contain_all):
                idx_keep = np.where(contain_all)[0][0]
                for i in [_ for _ in range(len(idx_dup)) if not _ == idx_keep]:
                    # print('removed %s'%(fn_im[i][-1]))
                    # remove the 3 .tif files + the .txt file
                    for k in range(4):  
                        os.chmod(fn_im[i][k], 0o777)
                        os.remove(fn_im[i][k])
                    total_removed_step1 += 1
        # load metadata again and update filenames
        metadata = get_metadata(inputs) 
        filenames = metadata[sat]['filenames']
    
    # find the pairs of images that are within 5 minutes of each other and merge them
    time_delta = 5*60 # 5 minutes in seconds
    dates = metadata[sat]['dates'].copy()
    pairs = []
    for i,date in enumerate(metadata[sat]['dates']):
        # dummy value so it does not match it again
        dates[i] = pytz.utc.localize(datetime(1,1,1) + timedelta(days=i+1))
        # calculate time difference
        time_diff = np.array([np.abs((date - _).total_seconds()) for _ in dates])
        # find the matching times and add to pairs list
        boolvec = time_diff <= time_delta
        if np.sum(boolvec) == 0:
            continue
        else:
            idx_dup = np.where(boolvec)[0][0]
            pairs.append([i,idx_dup])
    total_merged_step2 = len(pairs)        
    # because they could be triplicates in S2 images, adjust the pairs for consecutive merges
    for i in range(1,len(pairs)):
        if pairs[i-1][1] == pairs[i][0]:
            pairs[i][0] = pairs[i-1][0]
            
    # check also for quadruplicates and remove them 
    pair_first = [_[0] for _ in pairs]
    idx_remove_pair = []
    for idx in np.unique(pair_first):
        # quadruplicate if trying to merge 3 times the same image with a successive image
        if sum(pair_first == idx) == 3: 
            # remove the last image: 3 .tif files + the .txt file
            idx_last = [pairs[_] for _ in np.where(pair_first == idx)[0]][-1][-1]
            fn_im = [os.path.join(filepath, 'S2', '10m', filenames[idx_last]),
                     os.path.join(filepath, 'S2', '20m',  filenames[idx_last].replace('10m','20m')),
                     os.path.join(filepath, 'S2', '60m',  filenames[idx_last].replace('10m','60m')),
                     os.path.join(filepath, 'S2', 'meta', filenames[idx_last].replace('_10m','').replace('.tif','.txt'))]
            for k in range(4):  
                os.chmod(fn_im[k], 0o777)
                os.remove(fn_im[k]) 
            # store the index of the pair to remove it outside the loop
            idx_remove_pair.append(np.where(pair_first == idx)[0][-1])
    # remove quadruplicates from list of pairs
    pairs = [i for j, i in enumerate(pairs) if j not in idx_remove_pair]
    
    # for each pair of image, first check if one image completely contains the other
    # in that case keep the larger image. Otherwise merge the two images.
    for i,pair in enumerate(pairs):
        # get filenames of all the files corresponding to the each image in the pair
        fn_im = []
        for index in range(len(pair)):
            fn_im.append([os.path.join(filepath, 'S2', '10m', filenames[pair[index]]),
                  os.path.join(filepath, 'S2', '20m',  filenames[pair[index]].replace('10m','20m')),
                  os.path.join(filepath, 'S2', '60m',  filenames[pair[index]].replace('10m','60m')),
                  os.path.join(filepath, 'S2', 'meta', filenames[pair[index]].replace('_10m','').replace('.tif','.txt'))])
        # get polygon for first image
        polygon0 = Toolbox.get_image_bounds(fn_im[0][0])
        im_epsg0 = metadata[sat]['epsg'][pair[0]]
        # get polygon for second image
        polygon1 = Toolbox.get_image_bounds(fn_im[1][0])
        im_epsg1 = metadata[sat]['epsg'][pair[1]]  
        # check if epsg are the same
        if not im_epsg0 == im_epsg1:
            print('WARNING: there was an error as two S2 images do not have the same epsg,'+
                  ' please open an issue on Github at https://github.com/kvos/CoastSat/issues'+
                  ' and include your script so we can find out what happened.')
            break
        # check if one image contains the other one
        if polygon0.contains(polygon1):  
            # if polygon0 contains polygon1, remove files for polygon1
            for k in range(4):  # remove the 3 .tif files + the .txt file
                os.chmod(fn_im[1][k], 0o777)
                os.remove(fn_im[1][k])
            # print('removed 1')
            continue
        elif polygon1.contains(polygon0):
            # if polygon1 contains polygon0, remove image0
            for k in range(4):   # remove the 3 .tif files + the .txt file
                os.chmod(fn_im[0][k], 0o777)
                os.remove(fn_im[0][k])
            # print('removed 0')
            # adjust the order in case of triplicates
            if i+1 < len(pairs):
                if pairs[i+1][0] == pair[0]: pairs[i+1][0] = pairs[i][1]
            continue
        # otherwise merge the two images after masking the nodata values
        else:
            for index in range(len(pair)):
                # read image
                im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata = Image_Processing.preprocess_single(fn_im[index], sat, False)
                # in Sentinel2 images close to the edge of the image there are some artefacts,
                # that are squares with constant pixel intensities. They need to be masked in the
                # raster (GEOTIFF). It can be done using the image standard deviation, which
                # indicates values close to 0 for the artefacts.
                if len(im_ms) > 0:
                    # calculate image std for the first 10m band
                    im_std = Toolbox.image_std(im_ms[:,:,0],1)
                    # convert to binary
                    im_binary = np.logical_or(im_std < 1e-6, np.isnan(im_std))
                    # dilate to fill the edges (which have high std)
                    mask10 = morphology.dilation(im_binary, morphology.square(3))
                    # mask the 10m .tif file (add no_data where mask is True)
                    Toolbox.mask_raster(fn_im[index][0], mask10)    
                    # now calculate the mask for the 20m band (SWIR1)
                    # for the older version of the ee api calculate the image std again 
                    if int(ee.__version__[-3:]) <= 201:
                        # calculate std to create another mask for the 20m band (SWIR1)
                        im_std = Toolbox.image_std(im_extra,1)
                        im_binary = np.logical_or(im_std < 1e-6, np.isnan(im_std))
                        mask20 = morphology.dilation(im_binary, morphology.square(3))    
                    # for the newer versions just resample the mask for the 10m bands
                    else:
                        # create mask for the 20m band (SWIR1) by resampling the 10m one
                        mask20 = ndimage.zoom(mask10,zoom=1/2,order=0)
                        mask20 = transform.resize(mask20, im_extra.shape, mode='constant',
                                                  order=0, preserve_range=True)
                        mask20 = mask20.astype(bool)     
                    # mask the 20m .tif file (im_extra)
                    Toolbox.mask_raster(fn_im[index][1], mask20)
                    # create a mask for the 60m QA band by resampling the 20m one
                    mask60 = ndimage.zoom(mask20,zoom=1/3,order=0)
                    mask60 = transform.resize(mask60, im_QA.shape, mode='constant',
                                              order=0, preserve_range=True)
                    mask60 = mask60.astype(bool)
                    # mask the 60m .tif file (im_QA)
                    Toolbox.mask_raster(fn_im[index][2], mask60)   
                    # make a figure for quality control/debugging
                    # im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
                    # fig,ax= plt.subplots(2,3,tight_layout=True)
                    # ax[0,0].imshow(im_RGB)
                    # ax[0,0].set_title('RGB original')
                    # ax[1,0].imshow(mask10)
                    # ax[1,0].set_title('Mask 10m')
                    # ax[0,1].imshow(mask20)
                    # ax[0,1].set_title('Mask 20m')
                    # ax[1,1].imshow(mask60)
                    # ax[1,1].set_title('Mask 60 m')
                    # ax[0,2].imshow(im_QA)
                    # ax[0,2].set_title('Im QA')
                    # ax[1,2].imshow(im_nodata)
                    # ax[1,2].set_title('Im nodata') 
                else:
                    continue
    
            # once all the pairs of .tif files have been masked with no_data, merge the using gdal_merge
            fn_merged = os.path.join(filepath, 'merged.tif')
            for k in range(3):  
                # merge masked bands
                gdal_merge.main(['', '-o', fn_merged, '-n', '0', fn_im[0][k], fn_im[1][k]])
                # remove old files
                os.chmod(fn_im[0][k], 0o777)
                os.remove(fn_im[0][k])
                os.chmod(fn_im[1][k], 0o777)
                os.remove(fn_im[1][k])
                # rename new file
                fn_new = fn_im[0][k].split('.')[0] + '_merged.tif'
                os.chmod(fn_merged, 0o777)
                os.rename(fn_merged, fn_new)
    
            # open both metadata files
            metadict0 = dict([])
            with open(fn_im[0][3], 'r') as f:
                metadict0['filename'] = f.readline().split('\t')[1].replace('\n','')
                metadict0['acc_georef'] = float(f.readline().split('\t')[1].replace('\n',''))
                metadict0['epsg'] = int(f.readline().split('\t')[1].replace('\n',''))
            metadict1 = dict([])
            with open(fn_im[1][3], 'r') as f:
                metadict1['filename'] = f.readline().split('\t')[1].replace('\n','')
                metadict1['acc_georef'] = float(f.readline().split('\t')[1].replace('\n',''))
                metadict1['epsg'] = int(f.readline().split('\t')[1].replace('\n',''))
            # check if both images have the same georef accuracy
            if np.any(np.array([metadict0['acc_georef'],metadict1['acc_georef']]) == -1):
                metadict0['georef'] = -1
            # add new name
            metadict0['filename'] =  metadict0['filename'].split('.')[0] + '_merged.tif'
            # remove the old metadata.txt files
            os.chmod(fn_im[0][3], 0o777)
            os.remove(fn_im[0][3])
            os.chmod(fn_im[1][3], 0o777)
            os.remove(fn_im[1][3])        
            # rewrite the .txt file with a new metadata file
            fn_new = fn_im[0][3].split('.')[0] + '_merged.txt'
            with open(fn_new, 'w') as f:
                for key in metadict0.keys():
                    f.write('%s\t%s\n'%(key,metadict0[key]))  
                    
            # update filenames list (in case there are triplicates)
            filenames[pair[0]] = metadict0['filename']
     
    print('%d out of %d Sentinel-2 images were merged (overlapping or duplicate)'%(total_removed_step1+total_merged_step2,
                                                                                   total_images))

    # update the metadata dict
    metadata_updated = get_metadata(inputs)

    return metadata_updated



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
    """
    Download PlanetScope imagery directly using the Planet API.
    FM 2022

    Parameters
    ----------
    Sat : list
        List of satellite image names per platform.

    Returns
    -------
    idURLs : TYPE
        DESCRIPTION.

    """
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