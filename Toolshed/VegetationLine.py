"""This file contains the main functions for COASTGUARD coastal vegetation edge extraction (VedgeSat).
Freya Muir, University of Glasgow
"""
# load modules
import os
import numpy as np
import matplotlib.pyplot as plt
import pdb
import scipy

# image processing modules
import ee
import skimage.filters as filters
import skimage.measure as measure
import skimage.morphology as morphology
from shapely import geometry
from shapely.geometry import Point, LineString, Polygon
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from geoarray import GeoArray 

# machine learning modules
import sklearn
if sklearn.__version__[:4] == '0.20':
    from sklearn.externals import joblib
else:
    import joblib
from shapely.geometry import LineString

# other modules
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.cm as cm
from matplotlib import colors
from matplotlib import gridspec
import pickle
from datetime import datetime
from pylab import ginput

# CoastSat modules
from Toolshed import Toolbox, Image_Processing

np.seterr(all='ignore') # raise/ignore divisions by 0 and nans

# Main function for batch vegline detection
def extract_veglines(metadata, settings, polygon, dates, savetifs=True):
    """
    Main function to extract vegetation edges from satellite imagery (Landsat 5-9, Sentinel-2 or local images (e.g. Planet)).
    
    FM Aug 2022    
    
    Parameters
    ----------
    metadata : dict
        Dictionary of sat image filenames, georeferencing info, EPSGs and dates of capture.
    settings : dict
        Dictionary of user-defined settings used for the veg edge extraction.
    polygon : list
        List of 5 WGS84 coordinate pairs marking rectangle of interest.
    dates : list
        Start and end dates of interest as yyyy-mm-dd strings.


    Returns
    -------
    output : dict
        Dictionary of extracted veg edges and associated info with each.
    output_latlon : dict
        Dictionary of extracted veg edges and associated info with each (in WGS84).
    output_proj : dict
        Dictionary of extracted veg edges and associated info with each (in chosen CRS).

    """

    sitename = settings['inputs']['sitename']
    # ref_line = np.delete(settings['reference_shoreline'],2,1)
    filepath_data = settings['inputs']['filepath']
    filepath_models = os.path.join(os.getcwd(), 'Classification', 'models')

    # Initialise counter for run success rates
    skipped = {
        'empty_poor': [],
        'missing_mask':[],
        'cloudy': [],
        'no_classes': [],
        'no_contours': []}
    
    # initialise output structure
    output = dict([])
    output_latlon = dict([])
    output_proj = dict([])
    # create a subfolder to store the .jpg images showing the detection
    filepath_jpg = os.path.join(filepath_data, sitename, 'jpg_files', 'detection')
    if not os.path.exists(filepath_jpg):
            os.makedirs(filepath_jpg)
    # close all open figures
    plt.close('all')

    print('Mapping veglines:')

    imgcount = 0
    
    # loop through satellite list
    for satname in metadata.keys():

        imgcount += len(metadata[satname]['filenames'])
        # get images
        #filepath = Toolbox.get_filepath(settings['inputs'],satname)
        filenames = metadata[satname]['filenames']
        datelist = metadata[satname]['dates']
        # Collate filenames of images per platform
        imgs = []
        for i in range(len(filenames)):
            imgs.append(ee.Image(filenames[i]))
        
        # initialise the output variables
        output_date = []       # datetime at which the image was acquired (YYYY-MM-DD)
        output_time = []            # UTC timestamp
        output_vegline = []         # vector of vegline points
        output_vegline_latlon = []
        output_vegline_proj = []
        output_shoreline = []       # vector of waterline points
        output_shoreline_latlon = []
        output_shoreline_proj = []
        output_filename = []        # filename of the images from which the veglines are derived
        output_cloudcover = []      # cloud cover of the images
        output_geoaccuracy = []     # georeferencing accuracy of the images
        output_idxkeep = []         # index that were kept during the analysis (cloudy images are skipped)
        output_t_ndvi = []          # NDVI threshold used to map the vegline
        output_t_ndwi = []          # NDWI threshold used to map the vegline
        
        # get pixel sizes and image collections for each platform
        if satname == 'L5':
            pixel_size = 15
            clf_model = 'MLPClassifier_Veg_L5L8S2.pkl'
            ImgColl = ee.ImageCollection.fromImages(imgs).select(['B1','B2','B3','B4','B5','QA_PIXEL'])
        elif satname == 'L7':
            pixel_size = 15
            clf_model = 'MLPClassifier_Veg_L5L8S2.pkl'
            ImgColl = ee.ImageCollection.fromImages(imgs).select(['B1','B2','B3','B4','B5','B8','QA_PIXEL'])
        elif satname == 'L8':
            pixel_size = 15
            clf_model = 'MLPClassifier_Veg_L5L8S2.pkl'
            ImgColl = ee.ImageCollection.fromImages(imgs).select(['B2','B3','B4','B5', 'B6','B7','B10','B11','QA_PIXEL'])
        elif satname == 'L9':
            pixel_size = 15
            clf_model = 'MLPClassifier_Veg_L5L8S2.pkl' 
            ImgColl = ee.ImageCollection.fromImages(imgs).select(['B2','B3','B4','B5', 'B6','B8','B10','B11','QA_PIXEL'])
        elif satname == 'S2':
            pixel_size = 10
            clf_model = 'MLPClassifier_Veg_L5L8S2.pkl' 
            ImgColl = ee.ImageCollection.fromImages(imgs).filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 98.5))
        else: # Planet or local
            pixel_size = metadata[settings['inputs']['sat_list'][0]]['acc_georef'][0][0] #pull first image's pixel size from transform matrix
            clf_model = 'MLPClassifier_Veg_PSScene.pkl' 
            
        # clf_model = settings['clf_model']
        # load in trained classifier pkl file
        clf = joblib.load(os.path.join(filepath_models, clf_model))
            
        # convert settings['min_beach_area'] and settings['buffer_size'] from metres to pixels
        # TO DO: figure out why these exist
        buffer_size_pixels = np.ceil(settings['buffer_size']/pixel_size)
        min_beach_area_pixels = np.ceil(settings['min_beach_area']/pixel_size**2)


        # loop through the images
        for fn in range(len(filenames)):
            
            print('\r%s:   %0.3f %% ' % (satname,((fn+1)/len(filenames))*100), end='')

            # Image acqusition date
            acqdate = metadata[satname]['dates'][fn]
            # preprocess image (cloud mask + pansharpening/downsampling)
            im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata, acqtime = Image_Processing.preprocess_single(ImgColl, fn, datelist, filenames, satname, settings, polygon, dates, skipped)

            if im_ms is None:
                continue
            
            if cloud_mask is None:
                continue
            
            # get image spatial reference system (epsg code) from refline location
            image_epsg = settings['projection_epsg']
            # compute cloud_cover percentage (with no data pixels)
            cloud_cover_combined = np.divide(sum(sum(cloud_mask.astype(int))),
                                    (cloud_mask.shape[0]*cloud_mask.shape[1]))
            if cloud_cover_combined > 0.95: # if 99% of cloudy pixels in image skip
                continue
            # remove no data pixels from the cloud mask 
            # (for example L7 bands of no data should not be accounted for)
            cloud_mask_adv = np.logical_xor(cloud_mask, im_nodata) 
            # compute updated cloud cover percentage (without no data pixels)
            cloud_cover = np.divide(sum(sum(cloud_mask_adv.astype(int))),
                                    (sum(sum((~im_nodata).astype(int)))))
            # skip image if cloud cover is above user-defined threshold
            if cloud_cover > settings['cloud_thresh']:
                continue

            # calculate a buffer around the reference shoreline
            im_ref_buffer_og = BufferShoreline(settings,settings['reference_shoreline'],georef,cloud_mask)
            if fn == 0: # if the first image in a sat set, use the ref shoreline
                im_ref_buffer = im_ref_buffer_og
            else:
                im_ref_buffer = im_ref_buffer_og
            # otherwise use the most recent shoreline found, so buffer updates through time
            # TO DO: figure out way to update refline ONLY if no gaps in previous line exist (length-based? based on number of coords?)
            # elif output_shoreline[-1].length < im_ref_buffer_og: 
            #     output_shorelineArr = Toolbox.GStoArr(output_shoreline[-1])
            #     im_ref_buffer = BufferShoreline(settings,output_shorelineArr,georef,pixel_size,cloud_mask)
            # # im_ref_buffer = BufferShoreline(settings,georef,pixel_size,cloud_mask)
        
            # Coregistration of satellite images based on AROSICS phase shifts
            # Uses GeoArray(array, geotransform, projection)
            # Read in provided reference image (if it has been provided)
            if settings['reference_coreg_im'] is not None:
                # reference image to coregister to (given as filepath)
                refArr = GeoArray(settings['reference_coreg_im'])
                # target sat image to register (current image in loop)
                trgArr = GeoArray(im_ms[:,:,0:3], georef, 'EPSG:'+str(image_epsg))
                # add reference shoreline buffer (and cloud mask) to region for avoiding tie point creation
                refArr.mask_baddata = im_ref_buffer
                trgArr.mask_baddata = cloud_mask + im_ref_buffer
                
                georef = Image_Processing.Coreg(refArr,trgArr,georef)
            
            if savetifs == True:
                Image_Processing.save_RGB_NDVI(im_ms, cloud_mask, georef, filenames[fn], settings)
             
            # classify image with NN classifier
            im_classif, im_labels = classify_image_NN(im_ms, im_extra, cloud_mask, min_beach_area_pixels, clf)
            # if extracting shorelines alongside (using original CoastSat NN)
            if settings['wetdry'] == True:
                if satname in ['L5','L7','L8','L9']:
                    sh_clf = joblib.load(os.path.join(filepath_models, 'NN_4classes_Landsat_new.pkl'))
                    PS = False
                elif satname == 'S2':
                    sh_clf = joblib.load(os.path.join(filepath_models, 'NN_4classes_S2_new.pkl'))
                    PS = False
                else: # Planet or local image with no SWIR
                    sh_clf = joblib.load(os.path.join(filepath_models, 'NN_4classes_PS_NARRA_new.pkl'))
                    PS = True
                sh_classif, sh_labels = classify_image_NN_shore(im_ms, im_extra, cloud_mask, min_beach_area_pixels, sh_clf, PS)
            
            # if classified image comes back with almost no pixels in either class (<5%), skip
            if (np.count_nonzero(im_labels[:,:,0])/(len(im_labels) * len(im_labels[0]))) < 0.05 or (np.count_nonzero(im_labels[:,:,1])/(len(im_labels) * len(im_labels[0]))) < 0.05:
                skipped['no_classes'].append([filenames[fn], satname, acqdate+' '+acqtime])
                print(' - Skipped: classifier cannot find enough variety of classes')
                continue
            
            # save classified image and transition zone mask after classification takes place
            Image_Processing.save_ClassIm(im_classif, im_labels, cloud_mask, georef, filenames[fn], settings)
            Image_Processing.save_TZone(im_ms, im_labels, cloud_mask, im_ref_buffer, georef, filenames[fn], settings)
                
            # compute NDVI image (NIR-R)
            im_ndvi = Toolbox.nd_index(im_ms[:,:,3], im_ms[:,:,2], cloud_mask)

            # contours_ndvi, t_ndvi = FindShoreContours_Enhc(im_ndvi, im_labels, cloud_mask, im_ref_buffer)
            contours_ndvi, t_ndvi = FindShoreContours_WP(im_ndvi, im_labels, cloud_mask, im_ref_buffer)
            if contours_ndvi is None:
                skipped['no_contours'].append([filenames[fn], satname, acqdate+' '+acqtime])
                print(' - Poor image quality: no contours generated.')
                continue
            
            if settings['wetdry'] == True:
                im_ndwi = Toolbox.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)
                contours_ndwi, t_ndwi = FindShoreContours_Water(im_ndwi, sh_labels, cloud_mask, im_ref_buffer)
                if contours_ndvi is None:
                    skipped['no_contours'].append([filenames[fn], satname, acqdate+' '+acqtime])
                    print(' - Poor image quality: no water contours generated.')
                    continue
                
            # process the contours into a vegline
            vegline, vegline_latlon, vegline_proj = ProcessShoreline(contours_ndvi, cloud_mask, georef, image_epsg, settings)
            if settings['wetdry'] == True:
                shoreline, shoreline_latlon, shoreline_proj = ProcessShoreline(contours_ndwi, cloud_mask, georef, image_epsg, settings)


            # if adjust_detection is True, let the user adjust the detected shoreline
            if settings['adjust_detection']:
                date = metadata[satname]['dates'][fn]
                if settings['wetdry'] == True:
                    skip_image, vegline, vegline_latlon, vegline_proj, t_ndvi = adjust_detection(im_ms, cloud_mask, im_labels, im_ref_buffer, vegline, vegline_latlon, vegline_proj,
                                                                                                 image_epsg, georef, settings, date, satname, contours_ndvi, t_ndvi,
                                                                                                  sh_classif, sh_labels, contours_ndwi, t_ndwi)
                else:
                    skip_image, vegline, vegline_latlon, vegline_proj, t_ndvi = adjust_detection(im_ms, cloud_mask, im_labels, im_ref_buffer, vegline, vegline_latlon, vegline_proj,
                                                                                                 image_epsg, georef,settings, date, satname, contours_ndvi, t_ndvi)
                # if the user decides to skip the image, continue and do not save the mapped vegline
                if skip_image:
                    continue
            
            else:
                if settings['check_detection'] or settings['save_figure']:
                    date = metadata[satname]['dates'][fn]
                    if not settings['check_detection']:
                        plt.ioff() # turning interactive plotting off
                    if settings['wetdry'] == True:
                        skip_image = show_detection(im_ms, cloud_mask, im_labels, im_ref_buffer,
                                                    image_epsg, georef, settings, date, satname, contours_ndvi, t_ndvi,
                                                    sh_classif, sh_labels, contours_ndwi, t_ndwi)
                    else:
                        skip_image = show_detection(im_ms, cloud_mask, im_labels, im_ref_buffer,
                                                    image_epsg, georef, settings, date, satname, contours_ndvi, t_ndvi)
                        
                        
                        # if the user decides to skip the image, continue and do not save the mapped vegline
                    if skip_image:
                        continue
            

            # append to output variables
            output_date.append(acqdate)
            output_time.append(acqtime)
            output_vegline.append(vegline)
            output_vegline_latlon.append(vegline_latlon)
            output_vegline_proj.append(vegline_proj)
            if settings['wetdry'] == True:
                output_shoreline.append(shoreline)
                output_shoreline_latlon.append(shoreline_latlon)
                output_shoreline_proj.append(shoreline_proj)
                output_t_ndwi.append(t_ndwi)
            else: # if not doing waterlines, fill with nans
                output_shoreline.append(np.nan)
                output_shoreline_latlon.append(np.nan)
                output_shoreline_proj.append(np.nan)
                output_t_ndwi.append(np.nan)
            output_filename.append(filenames[fn])
            output_cloudcover.append(cloud_cover)
            output_geoaccuracy.append(metadata[satname]['acc_georef'][fn])
            output_idxkeep.append(fn)
            output_t_ndvi.append(t_ndvi)

        
        # create dictionary of output
        output[satname] = {
                'dates': output_date,
                'times':output_time,
                'veglines': output_vegline,
                'waterlines':output_shoreline,
                'filename': output_filename,
                'cloud_cover': output_cloudcover,
                'idx': output_idxkeep,
                'vthreshold': output_t_ndvi,
                'wthreshold': output_t_ndwi
                }
        print('')
    
        output_latlon[satname] = {
                'dates': output_date,
                'times':output_time,
                'veglines': output_vegline_latlon,
                'waterlines':output_shoreline_latlon,
                'filename': output_filename,
                'cloud_cover': output_cloudcover,
                'idx': output_idxkeep,
                'vthreshold': output_t_ndvi,
                'wthreshold': output_t_ndwi
                }
        
        output_proj[satname] = {
                'dates': output_date,
                'times':output_time,
                'veglines': output_vegline_proj,
                'waterlines':output_shoreline_proj,
                'filename': output_filename,
                'cloud_cover': output_cloudcover,
                'idx': output_idxkeep,
                'vthreshold': output_t_ndvi,
                'wthreshold': output_t_ndwi
                }
        

        dates_sat = []
        for i in range(len(output_date)):
            dates_sat_str = output_date[i] +' '+output_time[i]
            dates_sat.append(datetime.strptime(dates_sat_str, '%Y-%m-%d %H:%M:%S.%f'))
        
        # Water elevations for each image are grabbed from FES2014 to be able to 
        # filter veg lines by tidal stage (e.g. disregard low tide veg edges)
        if os.path.isfile(os.path.join(settings['inputs']['filepath'],'tides',settings['inputs']['sitename']+'_tides.csv')):
            output_waterelev = Toolbox.GetWaterElevs(settings, dates_sat)
        else:
            print('No tide data exists - skipping GetWaterElevs.\n')
            output_waterelev = list(np.empty(len(dates_sat)) * np.nan)
        
        output[satname]['tideelev'] = output_waterelev
        output_latlon[satname]['tideelev'] = output_waterelev
        output_proj[satname]['tideelev'] = output_waterelev
        
    # change the format to have one list sorted by date with all the veglines (easier to use)
    output = Toolbox.merge_output(output)
    output_latlon = Toolbox.merge_output(output_latlon)
    output_proj = Toolbox.merge_output(output_proj)
    
    
    # print statistics of run
    for reason in skipped.keys():
        print(f"Skipped due to {reason}: {len(skipped[reason])} / {imgcount} ({round(len(skipped[reason])/imgcount,4)}%)")
        
    # save output structure as output.pkl
    print('saving output pickle files ...')
    filepath = os.path.join(filepath_data, sitename)
    with open(os.path.join(filepath, sitename + '_output.pkl'), 'wb') as f:
        pickle.dump(output, f)
    
    with open(os.path.join(filepath, sitename + '_settings.pkl'), 'wb') as f:
        pickle.dump(settings, f)

    with open(os.path.join(filepath, sitename + '_output_latlon.pkl'), 'wb') as f:
        pickle.dump(output_latlon, f)
        
    with open(os.path.join(filepath, sitename + '_output_proj.pkl'), 'wb') as f:
        pickle.dump(output_proj, f)
        
    with open(os.path.join(filepath, sitename + '_skip_stats.pkl'), 'wb') as f:
        pickle.dump(skipped, f)
    
    # close figure window if still open
    if plt.get_fignums():
        plt.close()
        
    return output, output_latlon, output_proj


###################################################################################################
# IMAGE CLASSIFICATION FUNCTIONS
###################################################################################################

def calculate_features(im_ms, cloud_mask, im_bool):
    """
    Calculates features on the image that are used for the supervised classification. 
    The features include spectral normalized-difference indices and standard 
    deviation of the image for all the bands and indices.

    KV WRL 2018

    Arguments:
    -----------
    im_ms: np.array
        RGB + downsampled NIR and SWIR
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_bool: np.array
        2D array of boolean indicating where on the image to calculate the features

    Returns:    
    -----------
    features: np.array
        matrix containing each feature (columns) calculated for all
        the pixels (rows) indicated in im_bool
        
    """

    # add all the multispectral bands
    features = np.expand_dims(im_ms[im_bool,0],axis=1)
    for k in range(1,im_ms.shape[2]):
        feature = np.expand_dims(im_ms[im_bool,k],axis=1)
        features = np.append(features, feature, axis=-1)
    # if im_ms.shape[2]>4: # FM: exception for if SWIR band doesn't exist 
    # SWIR-G
    im_SWIRG = Toolbox.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
    features = np.append(features, np.expand_dims(im_SWIRG[im_bool],axis=1), axis=-1)
    # SWIR-NIR
    im_SWIRNIR = Toolbox.nd_index(im_ms[:,:,4], im_ms[:,:,3], cloud_mask)
    features = np.append(features, np.expand_dims(im_SWIRNIR[im_bool],axis=1), axis=-1)
    # NIR-G
    im_NIRG = Toolbox.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)
    features = np.append(features, np.expand_dims(im_NIRG[im_bool],axis=1), axis=-1)
    # NIR-R
    im_NIRR = Toolbox.nd_index(im_ms[:,:,3], im_ms[:,:,2], cloud_mask)
    features = np.append(features, np.expand_dims(im_NIRR[im_bool],axis=1), axis=-1)
    # B-R
    im_BR = Toolbox.nd_index(im_ms[:,:,0], im_ms[:,:,2], cloud_mask)
    features = np.append(features, np.expand_dims(im_BR[im_bool],axis=1), axis=-1)
    
    # calculate standard deviation of individual bands
    for k in range(im_ms.shape[2]):
        im_std =  Toolbox.image_std(im_ms[:,:,k], 1)
        features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    # calculate standard deviation of the spectral indices
    # if im_ms.shape[2]>4: # FM: exception for if SWIR band doesn't exist
    # SWIR-G   
    im_std = Toolbox.image_std(im_SWIRG, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    #SWIR-NIR
    im_std = Toolbox.image_std(im_SWIRNIR, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    # NIR-G
    im_std = Toolbox.image_std(im_NIRG, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    # NIR-R
    im_std = Toolbox.image_std(im_NIRR, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    #B-R
    im_std = Toolbox.image_std(im_BR, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)

    # Total feature sets should be 20 for V+NIR+SWIR (5 bands, 5 indices, stdev on each)

    return features

def calculate_vegfeatures(im_ms, cloud_mask, im_bool):
    """
    Calculates features on the image that are used for the supervised classification. 
    of vegetation. The features include band differences, normalized-difference indices and 
    standard deviation on all image bands and indices.
    Differs from original calculate_features() in that the indices used are for veg (and not water),
    and only NIR is required (not SWIR).

    FM 2022

    Arguments:
    -----------
    im_ms: np.array
        RGB + downsampled NIR and SWIR
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_bool: np.array
        2D array of boolean indicating where on the image to calculate the features

    Returns:    
    -----------
    features: np.array
        matrix containing each feature (columns) calculated for all
        the pixels (rows) indicated in im_bool
        
    """

    # add all the multispectral bands
    features = np.expand_dims(im_ms[im_bool,0],axis=1)
    
    for k in range(1,im_ms.shape[2]):
        feature = np.expand_dims(im_ms[im_bool,k],axis=1)
        features = np.append(features, feature, axis=-1)
    # NDVI (NIR - R)
    im_NIRR = Toolbox.nd_index(im_ms[:,:,3], im_ms[:,:,2], cloud_mask)
    features = np.append(features, np.expand_dims(im_NIRR[im_bool],axis=1), axis=-1)
    # NDWI (NIR-G)
    im_NIRG = Toolbox.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)
    features = np.append(features, np.expand_dims(im_NIRG[im_bool],axis=1), axis=-1)
    # R-G
    im_RG = Toolbox.nd_index(im_ms[:,:,2], im_ms[:,:,1], cloud_mask)
    features = np.append(features, np.expand_dims(im_NIRG[im_bool],axis=1), axis=-1)
    # SAVI
    im_SAVI = Toolbox.savi_index(im_ms[:,:,3], im_ms[:,:,2], cloud_mask)
    features = np.append(features, np.expand_dims(im_SAVI[im_bool],axis=1), axis=-1)
    # RB-NDVI (NIR -+ (R + B))
    im_RBNDVI = Toolbox.rbnd_index(im_ms[:,:,3], im_ms[:,:,2], im_ms[:,:,0], cloud_mask)
    features = np.append(features, np.expand_dims(im_RBNDVI[im_bool],axis=1), axis=-1)
    
    # calculate standard deviation of individual bands
    for k in range(im_ms.shape[2]):
        im_std =  Toolbox.image_std(im_ms[:,:,k], 1)
        features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    
    # calculate standard deviation of the spectral indices
    # NDVI  (NIR - R)
    im_std = Toolbox.image_std(im_NIRR, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    # NDWI (NIR-G)
    im_std = Toolbox.image_std(im_NIRG, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    # R-G
    im_std = Toolbox.image_std(im_RG, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    # SAVI
    im_std = Toolbox.image_std(im_SAVI, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    # RB-NDVI
    im_std = Toolbox.image_std(im_RBNDVI, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)

    # Total feature num should be 20 (5 bands, 5 band indices, stdev on each)
    # or 18 for Planet (4 bands, 5 band indices, stdev on each)
    return features

def calculate_WV_features(im_ms, cloud_mask, im_bool):
    """
    Calculates features on the image that are used for the supervised classification. 
    of vegetation. The features include band differences, normalized-difference indices and 
    standard deviation on all image bands and indices.
    Differs from original calculate_features() in that the indices used are for veg (and not water),
    and only NIR is required (not SWIR).

    FM 2023

    Arguments:
    -----------
    im_ms: np.array
        RGB + downsampled NIR and SWIR
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_bool: np.array
        2D array of boolean indicating where on the image to calculate the features

    Returns:    
    -----------
    features: np.array
        matrix containing each feature (columns) calculated for all
        the pixels (rows) indicated in im_bool
        
    """

    # add all the multispectral bands
    features = np.expand_dims(im_ms[im_bool,0],axis=1)
    
    for k in range(1,im_ms.shape[2]):
        feature = np.expand_dims(im_ms[im_bool,k],axis=1)
        features = np.append(features, feature, axis=-1)
    if im_ms.shape[2]>4: # FM: exception for if SWIR band doesn't exist 
        # SWIR-G
        im_SWIRG = Toolbox.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
        features = np.append(features, np.expand_dims(im_SWIRG[im_bool],axis=1), axis=-1)
        # SWIR-NIR
        im_SWIRNIR = Toolbox.nd_index(im_ms[:,:,4], im_ms[:,:,3], cloud_mask)
        features = np.append(features, np.expand_dims(im_SWIRNIR[im_bool],axis=1), axis=-1)
    # NDVI (NIR - R)
    im_NIRR = Toolbox.nd_index(im_ms[:,:,3], im_ms[:,:,2], cloud_mask)
    features = np.append(features, np.expand_dims(im_NIRR[im_bool],axis=1), axis=-1)
    # NIR-G
    im_NIRG = Toolbox.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)
    features = np.append(features, np.expand_dims(im_NIRG[im_bool],axis=1), axis=-1)
    # R-G
    im_RG = Toolbox.nd_index(im_ms[:,:,2], im_ms[:,:,1], cloud_mask)
    features = np.append(features, np.expand_dims(im_NIRG[im_bool],axis=1), axis=-1)
    # B-R
    im_BR = Toolbox.nd_index(im_ms[:,:,0], im_ms[:,:,2], cloud_mask)
    features = np.append(features, np.expand_dims(im_BR[im_bool],axis=1), axis=-1)
    
    # calculate standard deviation of individual bands
    for k in range(im_ms.shape[2]):
        im_std =  Toolbox.image_std(im_ms[:,:,k], 1)
        features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    if im_ms.shape[2]>4: # FM: exception for if SWIR band doesn't exist
        # SWIR-G   
        im_std = Toolbox.image_std(im_SWIRG, 1)
        features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
        #SWIR-NIR
        im_std = Toolbox.image_std(im_SWIRNIR, 1)
        features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    # calculate standard deviation of the spectral indices
    # NDVI  (NIR - R)
    im_std = Toolbox.image_std(im_NIRR, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    # NIR-G
    im_std = Toolbox.image_std(im_NIRG, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    # R-G
    im_std = Toolbox.image_std(im_RG, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    #B-R
    im_std = Toolbox.image_std(im_BR, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)

    
    # Total feature sets should be 22 for V+NIR+SWIR (5 bands, 6 indices, stdev on each)
    # and 16 for V+NIR (4 bands, 4 indices, stdev on each)
    return features


def calculate_features_PS(im_ms, cloud_mask, im_bool):
    """
    Calculates features on the image that are used for the supervised classification. 
    The features include spectral normalized-difference indices and standard 
    deviation of the image for all the bands and indices.

    KV WRL 2018
    
    Modified for PS data by YD 2020

    Arguments:
    -----------
    im_ms: np.array
        RGB + downsampled NIR and SWIR
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_bool: np.array
        2D array of boolean indicating where on the image to calculate the features

    Returns:    
    -----------
    features: np.array
        matrix containing each feature (columns) calculated for all
        the pixels (rows) indicated in im_bool
        
    """

    # add all the multispectral bands
    features = np.expand_dims(im_ms[im_bool,0],axis=1)
    for k in range(1,im_ms.shape[2]):
        feature = np.expand_dims(im_ms[im_bool,k],axis=1)
        features = np.append(features, feature, axis=-1)
        
    # NIR-G
    im_NIRG = Toolbox.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)
    features = np.append(features, np.expand_dims(im_NIRG[im_bool],axis=1), axis=-1)
    
    # NIR-B
    im_NIRB = Toolbox.nd_index(im_ms[:,:,3], im_ms[:,:,0], cloud_mask)
    features = np.append(features, np.expand_dims(im_NIRB[im_bool],axis=1), axis=-1)
    
    # NIR-R
    im_NIRR = Toolbox.nd_index(im_ms[:,:,3], im_ms[:,:,2], cloud_mask)
    features = np.append(features, np.expand_dims(im_NIRR[im_bool],axis=1), axis=-1)
        
    # B-R
    im_BR = Toolbox.nd_index(im_ms[:,:,0], im_ms[:,:,2], cloud_mask)
    features = np.append(features, np.expand_dims(im_BR[im_bool],axis=1), axis=-1)
    
    # calculate standard deviation of individual bands
    for k in range(im_ms.shape[2]):
        im_std =  Toolbox.image_std(im_ms[:,:,k], 2)
        features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
        
    # calculate standard deviation of the spectral indices
    im_std = Toolbox.image_std(im_NIRG, 2)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = Toolbox.image_std(im_NIRB, 2)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = Toolbox.image_std(im_NIRR, 2)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = Toolbox.image_std(im_BR, 2)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)

    return features


def classify_image_NN(im_ms, im_extra, cloud_mask, min_beach_area, clf):
    """
    Classifies every pixel in the image into classes.

    The classifier is a Neural Network that is already trained.

    FM Aug 2022

    Arguments:
    -----------
    im_ms : np.array
        Pansharpened RGB + downsampled NIR and SWIR
    im_extra :
        only used for Landsat 7 and 8 where im_extra is the panchromatic band
    cloud_mask : np.array
        2D cloud mask with True where cloud pixels are
    min_beach_area : int
        minimum number of pixels that have to be connected to belong to the SAND class
    clf : joblib object
        pre-trained classifier

    Returns:    
    -----------
    im_classif : np.array
        2D image containing pixel labels
    im_labels : np.array
        3D boolean raster containing an image for each class (im_classif == label)

    """

    # calculate features
    vec_features = calculate_vegfeatures(im_ms, cloud_mask, np.ones(cloud_mask.shape).astype(bool))
    vec_features[np.isnan(vec_features)] = 1e-9 # NaN values are create when std is too close to 0

    # remove NaNs and cloudy pixels
    vec_cloud = cloud_mask.reshape(cloud_mask.shape[0]*cloud_mask.shape[1])
    vec_nan = np.any(np.isnan(vec_features), axis=1)
    vec_mask = np.logical_or(vec_cloud, vec_nan)
    vec_features = vec_features[~vec_mask, :]
    
    #labels = clf[0].predict(vec_features_new) # old classifier was subscriptable
    labels = clf.predict(vec_features)
    
    # recompose image
    vec_classif = np.nan*np.ones((cloud_mask.shape[0]*cloud_mask.shape[1]))
    vec_classif[~vec_mask] = labels
    im_classif = vec_classif.reshape((cloud_mask.shape[0], cloud_mask.shape[1]))
    # create a stack of boolean images for each label
    im_veg = im_classif == 1
    im_nonveg = im_classif == 2

    # remove small patches of sand or water that could be around the image (usually noise)
    im_veg = morphology.remove_small_objects(im_veg, min_size=min_beach_area, connectivity=2)
    im_nonveg = morphology.remove_small_objects(im_nonveg, min_size=min_beach_area, connectivity=2)
    
    im_labels = np.stack((im_veg,im_nonveg), axis=-1)

    return im_classif, im_labels

def classify_image_NN_shore(im_ms, im_extra, cloud_mask, min_beach_area, clf, PS):
    """
    Classifies every pixel in the image in one of 4 classes:
        - sand                                          --> label = 1
        - whitewater (breaking waves and swash)         --> label = 2
        - water                                         --> label = 3
        - other (vegetation, buildings, rocks...)       --> label = 0
    The classifier is a Neural Network that is already trained.
    KV WRL 2018
    
    Adapted to include Planet classifier FM Jan 2024
    
    Arguments:
    -----------
    im_ms: np.array
        Pansharpened RGB + downsampled NIR and SWIR
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    min_beach_area: int
        minimum number of pixels that have to be connected to belong to the SAND class
    clf: joblib object
        pre-trained classifier
    Returns:    
    -----------
    im_classif: np.array
        2D image containing labels
    im_labels: np.array of booleans
        3D image containing a boolean image for each class (im_classif == label)
    """

    # calculate features
    if PS is True: # If PlanetScope, calculate features using 16-feature version (no SWIR)
        vec_features = calculate_features_PS(im_ms, cloud_mask, np.ones(cloud_mask.shape).astype(bool))
    else:
        vec_features = calculate_features(im_ms, cloud_mask, np.ones(cloud_mask.shape).astype(bool))
    vec_features[np.isnan(vec_features)] = 1e-9 # NaN values are create when std is too close to 0

    # remove NaNs and cloudy pixels
    vec_cloud = cloud_mask.reshape(cloud_mask.shape[0]*cloud_mask.shape[1])
    vec_nan = np.any(np.isnan(vec_features), axis=1)
    vec_inf = np.any(np.isinf(vec_features), axis=1)    
    vec_mask = np.logical_or(vec_cloud,np.logical_or(vec_nan,vec_inf))
    vec_features = vec_features[~vec_mask, :]

    # classify pixels
    labels = clf.predict(vec_features)

    # recompose image
    vec_classif = np.nan*np.ones((cloud_mask.shape[0]*cloud_mask.shape[1]))
    vec_classif[~vec_mask] = labels
    im_classif = vec_classif.reshape((cloud_mask.shape[0], cloud_mask.shape[1]))

    # create a stack of boolean images for each label
    im_sand = im_classif == 1
    im_swash = im_classif == 2
    im_water = im_classif == 3
    # remove small patches of sand or water that could be around the image (usually noise)
    im_sand = morphology.remove_small_objects(im_sand, min_size=min_beach_area, connectivity=2)
    im_water = morphology.remove_small_objects(im_water, min_size=min_beach_area, connectivity=2)

    im_labels = np.stack((im_sand,im_swash,im_water), axis=-1)

    return im_classif, im_labels

###################################################################################################
# CONTOUR MAPPING FUNCTIONS
###################################################################################################

def FindShoreContours_Trad(im_ndi, cloud_mask, im_ref_buffer):
    """
    Traditional method for shoreline detection using a global threshold.
    Finds the water line by thresholding the Normalized Difference Water Index 
    and applying the Marching Squares Algorithm to contour the iso-value 
    corresponding to the threshold.
    FM Oct 2022, adapted from KV WRL 2018
    Parameters
    -----------
    im_ndi: np.ndarray
        Image (2D) with the normalised difference index
    cloud_mask: np.ndarray
        2D cloud mask with True where cloud pixels are
    im_ref_buffer: np.array
        Binary image marking a buffer around the reference shoreline
    Returns    
    -----------
    contours: list of np.arrays
        contains the coordinates of the contour lines
    t_mwi: float
        Otsu threshold used to map the contours
    """
    nrows = cloud_mask.shape[0]
    ncols = cloud_mask.shape[1]
    # use im_ref_buffer and dilate it by 5 pixels
    se = morphology.disk(5)
    im_ref_buffer_extra = morphology.binary_dilation(im_ref_buffer,se)
    vec_buffer = im_ref_buffer_extra.reshape(nrows*ncols)
    # reshape spectral index image to vector
    vec_ndvi = im_ndi.reshape(nrows*ncols)
    # keep pixels that are in the buffer and not in the cloud mask
    vec_mask = cloud_mask.reshape(nrows*ncols)
    vec = vec_ndvi[np.logical_and(vec_buffer,~vec_mask)]
    # apply otsu's threshold
    vec = vec[~np.isnan(vec)]
    t_otsu = filters.threshold_otsu(vec)
    # use Marching Squares algorithm to detect contours on ndwi image
    im_ndi_buffer = np.copy(im_ndi)
    im_ndi_buffer[~im_ref_buffer] = np.nan
    contours = measure.find_contours(im_ndi_buffer, t_otsu)
    # remove contours that contain NaNs (due to cloud pixels in the contour)
    contours = process_contours(contours)

    return contours, t_otsu

def FindShoreContours_Enhc(im_ndi, im_labels, cloud_mask, im_ref_buffer):
    """
    New robust method for extracting veglines. Incorporates the NN classification
    component to refine the Otsu threshold and make it specific to the inter-class interface.
    FM Oct 2022, adapted from KV WRL 2018
    Arguments:
    -----------
    im_ms: np.array
        RGB + downsampled NIR and SWIR
    im_labels: np.array
        3D image containing a boolean image for each class in the order (sand, swash, water)
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_ref_buffer: np.array
        binary image containing a buffer around the reference shoreline
    Returns:    
    -----------
    contours_mwi: list of np.arrays
        contains the coordinates of the contour lines extracted from the
        Normalized Difference Index of choice
    t_mwi: float
        Otsu sand/water threshold used to map the contours
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
    # create a buffer around the sandy beach
    vec_buffer = im_ref_buffer_extra.reshape(nrows*ncols)
    
    # select water/sand pixels that are within the buffer
    int_veg = vec_ndi[np.logical_and(vec_buffer,vec_veg)]
    int_nonveg = vec_ndi[np.logical_and(vec_buffer,vec_nonveg)]

    # make sure both classes have the same number of pixels before thresholding
    if len(int_veg) > 0 and len(int_nonveg) > 0:
        if np.argmin([int_veg.shape[0],int_nonveg.shape[0]]) == 1:
            int_veg = int_veg[np.random.choice(int_veg.shape[0],int_nonveg.shape[0], replace=False)]
        else:
            int_nonveg = int_nonveg[np.random.choice(int_nonveg.shape[0],int_veg.shape[0], replace=False)]

    # threshold the sand/water intensities
    int_all = np.append(int_veg,int_nonveg, axis=0)
    t_ndi = filters.threshold_otsu(int_all)

    # find contour with Marching-Squares algorithm
    im_ndi_buffer = np.copy(im_ndi)
    im_ndi_buffer[~im_ref_buffer] = np.nan
    contours_ndi = measure.find_contours(im_ndi_buffer, t_ndi)
    # remove contour points that are NaNs (around clouds)
    contours_ndi = process_contours(contours_ndi)

    # return contours and threshold value
    return contours_ndi, t_ndi

def FindShoreContours_Water(im_ndi, im_labels, cloud_mask, im_ref_buffer):
    """
    New robust method for extracting wet-dry boundaries. Incorporates the NN classification
    component to refine the Otsu threshold and make it specific to the inter-class interface.
    FM Oct 2022, adapted from KV WRL 2018
    Arguments:
    -----------
    im_ms: np.array
        RGB + downsampled NIR and SWIR
    im_labels: np.array
        3D image containing a boolean image for each class in the order (sand, swash, water)
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_ref_buffer: np.array
        binary image containing a buffer around the reference shoreline
    Returns:    
    -----------
    contours_mwi: list of np.arrays
        contains the coordinates of the contour lines extracted from the
        MNDWI (Modified Normalized Difference Water Index) image
    t_mwi: float
        Otsu sand/water threshold used to map the contours
    """

    nrows = cloud_mask.shape[0]
    ncols = cloud_mask.shape[1]
    
    # reshape spectral index image to vector
    vec_ndi = im_ndi.reshape(nrows*ncols)

    # reshape labels into vectors (0 is veg, 1 is nonveg)
    vec_water = im_labels[:,:,2].reshape(ncols*nrows)
    vec_nonwater = im_labels[:,:,0].reshape(ncols*nrows)

    # use im_ref_buffer and dilate it by 5 pixels
    # TO DO: alternative to expanding buffer; loop expansion until you get longer distance of contour? or acceptable amount of water vs not water?
    se = morphology.disk(5)
    im_ref_buffer_extra = morphology.binary_dilation(im_ref_buffer, se)
    # create a buffer around the sandy beach
    # vec_buffer = im_ref_buffer_extra.reshape(nrows*ncols)
    # to catch low tide images where ref line prioritises veg, create dummy 'buffer' of all Trues
    vec_buffer = np.full((nrows*ncols), True)

    # select water/sand pixels that are within the buffer
    int_water = vec_ndi[np.logical_and(vec_buffer,vec_water)]
    int_nonwater = vec_ndi[np.logical_and(vec_buffer,vec_nonwater)]
    # Empty/low quality images
    if len(int_water) == 0 or len(int_nonwater) == 0:
        return None, None

    # make sure both classes have the same number of pixels before thresholding
    if len(int_water) > 0 and len(int_nonwater) > 0:
        if np.argmin([int_water.shape[0],int_nonwater.shape[0]]) == 1:
            int_water = int_water[np.random.choice(int_water.shape[0],int_nonwater.shape[0], replace=False)]
        else:
            int_nonwater = int_nonwater[np.random.choice(int_nonwater.shape[0],int_water.shape[0], replace=False)]

    # threshold the sand/water intensities
    int_all = np.append(int_water,int_nonwater, axis=0)
    t_ndi = filters.threshold_otsu(int_all)

    # find contour with Marching-Squares algorithm
    im_ndi_buffer = np.copy(im_ndi)
    # im_ndi_buffer[~im_ref_buffer] = np.nan
    contours_ndi = measure.find_contours(im_ndi_buffer, t_ndi)
    # remove contour points that are NaNs (around clouds)
    contours_ndi = process_contours(contours_ndi)

    # only return MNDWI contours and threshold
    return contours_ndi, t_ndi



def FindShoreContours_WP(im_ndi, im_labels, cloud_mask, im_ref_buffer):
    """
    New robust method for extracting veglines. Incorporates the NN classification
    component to make the threshold specific to the inter-class interface. Uses 
    Weighted Peaks method rather than Otsu. 
    FM Oct 2022, adapted from KV WRL 2018
    
    Arguments:
    -----------
    im_ms: np.array
        RGB + downsampled NIR and SWIR
    im_labels: np.array
        3D image containing a boolean image for each class in the order (sand, swash, water)
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_ref_buffer: np.array
        binary image containing a buffer around the reference shoreline
    Returns:    
    -----------
    contours_ndi: list of np.arrays
        contains the coordinates of the contour lines extracted from the
        Normalized Difference Index image
    t_ndi: float
        threshold used to map the contours
    """
    
    # clip down classified band index values to coastal buffer
    int_veg, int_nonveg = Image_Processing.ClipIndexVec(cloud_mask, im_ndi, im_labels, im_ref_buffer)
    # Empty/low quality images
    if int_veg is None or int_nonveg is None:
        return None, None
    
    t_ndi, _ = Toolbox.FindWPThresh(int_veg, int_nonveg)
    # find contour with Marching-Squares algorithm
    im_ndi_buffer = np.copy(im_ndi)
    im_ndi_buffer[~im_ref_buffer] = np.nan
    contours_ndi = measure.find_contours(im_ndi_buffer, t_ndi)
    # remove contour points that are NaNs (around clouds)
    contours_ndi = process_contours(contours_ndi)

    # only return contours and threshold
    return contours_ndi, t_ndi


def find_wl_contours1_old(im_ndvi, cloud_mask, im_ref_buffer, satname):
    """
    Traditional method for shoreline detection using a global threshold.
    Finds the water line by thresholding the Normalized Difference Water Index 
    and applying the Marching Squares Algorithm to contour the iso-value 
    corresponding to the threshold.

    KV WRL 2018

    Arguments:
    -----------
    im_ndvi: np.ndarray
        Image (2D) with the NDVI (vegetation index)
    cloud_mask: np.ndarray
        2D cloud mask with True where cloud pixels are
    im_ref_buffer: np.array
        Binary image containing a buffer around the reference shoreline

    Returns:    
    -----------
    contours: list of np.arrays
        contains the coordinates of the contour lines
    t_ndvi: float
        Otsu threshold used to map the contours

    """
    # reshape image to vector
    vec_ndvi = im_ndvi.reshape(im_ndvi.shape[0] * im_ndvi.shape[1])
    vec_mask = cloud_mask.reshape(cloud_mask.shape[0] * cloud_mask.shape[1])
    vec = vec_ndvi[~vec_mask]
    # apply otsu's threshold
    vec = vec[~np.isnan(vec)]
    t_otsu = filters.threshold_otsu(vec)
    # if satname=='S2':
    #     t_otsu+=0.09
    # else:
    #     t_otsu=+0.205
    # use Marching Squares algorithm to detect contours on ndvi image
    im_ndvi_buffer = np.copy(im_ndvi)
    im_ndvi_buffer[~im_ref_buffer] = np.nan
    contours = measure.find_contours(im_ndvi_buffer, t_otsu)
    # remove contours that contain NaNs (due to cloud pixels in the contour)
    contours = process_contours(contours)

    return contours, t_otsu

def find_wl_contours2_old(im_ms, im_labels, cloud_mask, buffer_size, im_ref_buffer,satname):
    """
    New robust method for extracting shorelines. Incorporates the classification
    component to refine the treshold and make it specific to the sand/water interface.

    KV WRL 2018

    Arguments:
    -----------
    im_ms: np.array
        RGB + downsampled NIR and SWIR
    im_labels: np.array
        3D image containing a boolean image for each class in the order (sand, swash, water)
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    buffer_size: int
        size of the buffer around the sandy beach over which the pixels are considered in the
        thresholding algorithm.
    im_ref_buffer: np.array
        binary image containing a buffer around the reference shoreline

    Returns:    
    -----------
    contours_ndvi: list of np.arrays
        contains the coordinates of the contour lines extracted from the
        NDVI (Modified Normalized Difference Water Index) image
    t_ndvi: float
        Otsu sand/water threshold used to map the contours

    """

    nrows = cloud_mask.shape[0]
    ncols = cloud_mask.shape[1]

    # calculate Normalized Difference (NIR - R)
    im_ndvi = Toolbox.nd_index(im_ms[:,:,3], im_ms[:,:,2], cloud_mask)
    
    vec_ind = im_ndvi.reshape(nrows*ncols,1)
    
    # reshape labels into vectors
    vec_sand = im_labels[:,:,0].reshape(ncols*nrows)
    vec_veg = im_labels[:,:,1].reshape(ncols*nrows)
    #vec_water = im_labels[:,:,2].reshape(ncols*nrows)
    #vec_urb = im_labels[:,:,3].reshape(ncols*nrows)

    # create a buffer around the sandy beach
    se = morphology.disk(buffer_size)
    im_buffer = morphology.binary_dilation(im_labels[:,:,0], se)
    vec_buffer = im_buffer.reshape(nrows*ncols)

    # select water/sand/swash pixels that are within the buffer
    int_nonveg = vec_ind[np.logical_and(vec_buffer,vec_veg),:]
    int_veg = vec_ind[np.logical_and(vec_buffer,vec_sand),:]

    # make sure both classes have the same number of pixels before thresholding
    if len(int_nonveg) > 0 and len(int_veg) > 0:
        if np.argmin([int_veg.shape[0],int_nonveg.shape[0]]) == 1:
            int_veg = int_veg[np.random.choice(int_veg.shape[0],int_nonveg.shape[0], replace=False),:]
        else:
            int_nonveg = int_nonveg[np.random.choice(int_nonveg.shape[0],int_veg.shape[0], replace=False),:]

    # threshold the sand/water intensities
    int_all = np.append(int_nonveg,int_veg, axis=0)
    t_ndvi = filters.threshold_otsu(int_all[:,0])
    # if satname=='S2':
    #     t_ndvi+=0.09
    # else:
    #     t_ndvi=+0.205
    # find contour with MS algorithm
    im_ndvi_buffer = np.copy(im_ndvi)
    ### This is the problematic bit
    im_ndvi_buffer[~im_ref_buffer] = np.nan
    #contours_wi = measure.find_contours(im_wi_buffer, t_wi)
    contours_ndvi = measure.find_contours(im_ndvi_buffer, t_ndvi)
    # remove contour points that are NaNs (around clouds)
    #contours_wi = process_contours(contours_wi)
    contours_ndvi = process_contours(contours_ndvi)

    # only return NDVI contours and threshold
    return contours_ndvi, t_ndvi


###################################################################################################
# SHORELINE PROCESSING FUNCTIONS
###################################################################################################

def create_shoreline_buffer(im_shape, georef, image_epsg, pixel_size, settings, epsg):
    """
    Creates a buffer around the reference shoreline. The size of the buffer is 
    given by settings['max_dist_ref'].

    KV WRL 2018

    Arguments:
    -----------
    im_shape: np.array
        size of the image (rows,columns)
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
    image_epsg: int
        spatial reference system of the image from which the contours were extracted
    pixel_size: int
        size of the pixel in metres (15 for Landsat, 10 for Sentinel-2)
    settings: dict with the following keys
        'output_epsg': int
            output spatial reference system
        'reference_shoreline': np.array
            coordinates of the reference shoreline
        'max_dist_ref': int
            maximum distance from the reference shoreline in metres

    Returns:    
    -----------
    im_buffer: np.array
        binary image, True where the buffer is, False otherwise

    """
    #initialise the image buffer
    im_buffer = np.ones(im_shape).astype(bool)

    # convert reference shoreline to pixel coordinates
    ref_sl = settings['reference_shoreline'][:,:-1]

    #ref_sl_conv = Toolbox.convert_epsg(ref_sl, epsg, image_epsg)
    ref_sl_pix = Toolbox.convert_world2pix(ref_sl, georef)

    ref_sl_pix_rounded = np.round(ref_sl_pix).astype(int)
    # make sure that the pixel coordinates of the reference shoreline are inside the image
    idx_row = np.logical_and(ref_sl_pix_rounded[:,0] > 0, ref_sl_pix_rounded[:,0] < im_shape[1])
    idx_col = np.logical_and(ref_sl_pix_rounded[:,1] > 0, ref_sl_pix_rounded[:,1] < im_shape[0])
    idx_inside = np.logical_and(idx_row, idx_col)

    ref_sl_pix_rounded = ref_sl_pix_rounded[idx_inside,:]

    # create binary image of the reference shoreline (1 where the shoreline is 0 otherwise)
    im_binary = np.zeros(im_shape)
    for j in range(len(ref_sl_pix_rounded)):
        im_binary[ref_sl_pix_rounded[j,1], ref_sl_pix_rounded[j,0]] = 1
    im_binary = im_binary.astype(bool)

    # dilate the binary image to create a buffer around the reference shoreline
    max_dist_ref_pixels = np.ceil(settings['max_dist_ref']/pixel_size)
    se = morphology.disk(max_dist_ref_pixels)
    im_buffer = morphology.binary_dilation(im_binary, se)
    
    return im_buffer


def BufferShoreline(settings,refline,georef,cloud_mask):
    """
    Buffer reference line and utilise geopandas to generate boolean mask of where shoreline swath is.
    FM 2022

    Parameters
    ----------
    settings : dict
        Process settings.
    georef : list
        Affine transformation matrix of satellite image [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale].
    cloud_mask : array
        Boolean array masking out where clouds have been identified in satellite image.

    Returns
    -------
    im_buffer : boolean array
        Array with same dimensions as sat image, with True where buffered reference shoreline exists.

    """
    if type(refline) == np.ndarray:
        refGS = Toolbox.ArrtoGS(refline, georef)
    else: # if refline is read in as shapefile
        refGS = gpd.GeoSeries(refline['geometry'])
    
    buffDist = settings['max_dist_ref']/georef[1] # convert from metres to pixels using georef cell size
    refLSBuffer = refGS.buffer(buffDist)
    refShapes = ((geom,value) for geom, value in zip(refLSBuffer.geometry, np.ones(len(refLSBuffer))))
    im_buffer_float = features.rasterize(refShapes,out_shape=cloud_mask.shape)
    # convert to bool
    im_buffer = im_buffer_float > 0 
    
    return im_buffer

def process_contours(contours):
    """
    Remove contours that contain NaNs, usually these are contours that are in contact 
    with clouds.
    
    KV WRL 2020
    
    Arguments:
    -----------
    contours: list of np.array
        image contours as detected by the function skimage.measure.find_contours    
    
    Returns:
    -----------
    contours: list of np.array
        processed image contours (only the ones that do not contains NaNs) 
        
    """
    
    # initialise variable
    contours_nonans = []
    # loop through contours and only keep the ones without NaNs
    for k in range(len(contours)):
        if np.any(np.isnan(contours[k])):
            index_nan = np.where(np.isnan(contours[k]))[0]
            contours_temp = np.delete(contours[k], index_nan, axis=0)
            if len(contours_temp) > 1:
                contours_nonans.append(contours_temp)
        else:
            contours_nonans.append(contours[k])
    
    return contours_nonans

def ProcessShoreline(contours, cloud_mask, georef, image_epsg, settings):
    """
    Converts contours from image coordinates to world coordinates and cleans them
    based on distance to clouds and threshold length. Improvements to the
    coordinate array conversion also made by incorporating geopandas; instead of dealing
    with broken shorelines as one long array of coords, multiline features are preserved. 

    FM Aug 2022

    Arguments:
    -----------
    contours: np.array or list of np.array
        image contours as detected by the function find_contours
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
    image_epsg: int
        spatial reference system of the image from which the contours were extracted
    settings: dict with the following keys
        'output_epsg': int
            output spatial reference system
        'min_length_sl': float
            minimum length of shoreline contour to be kept (in meters)

    Returns:
    -----------
    shoreline: np.array
        Array of points with the X and Y coordinates of the shoreline.
    shoreline_latlon :
        Array of points with the X and Y coordinates of the shoreline (in lat-long).
    shoreline_proj :
        Array of points with the X and Y coordinates of the shoreline (in chosen projection system).
    """
    
    # convert pixel coordinates to world coordinates
    contours_world = Toolbox.convert_pix2world(contours, georef)
    
    # remove any coordinates that fall within cloud pixels
    if sum(sum(cloud_mask)) > 0:
        # get the coordinates of the cloud pixels
        idx_cloud = np.where(cloud_mask)
        idx_cloud = np.array([(idx_cloud[0][k], idx_cloud[1][k]) for k in range(len(idx_cloud[0]))])
        # convert to world coordinates and same epsg as the shoreline points
        if idx_cloud.dtype == 'int64': # fix for S2 pix coords being int64 and unwriteable as float64
            idx_cloud = idx_cloud.astype('float64')
        coords_cloud = Toolbox.convert_epsg(Toolbox.convert_pix2world(idx_cloud, georef),
                                               image_epsg, settings['output_epsg'])[:,:-1]
        # only keep the shoreline points that are at least 30m from any cloud pixel
        if type(contours_world) == list: # for multilines; extra nested loop
            idx_keep = [] # initialise indexes of coords to keep
            for j in range(len(contours_world)):  # for every line feature
                idx_keep.append(np.ones(len(contours_world[j][:])).astype(bool)) # length should be number of coord pairs
                for k in range(len(contours_world[j])): # for every coord pair in each line feature
                    if np.any(np.linalg.norm(contours_world[j][k] - coords_cloud, axis=1) < 30):
                        idx_keep[j][k] = False
            contours_world_list = [] # initialise contour list
            for i in range(len(contours_world)): # for each multiline feature
                contour_world = contours_world[i]
                # only keep coords away from clouds in each line feature
                contours_world_list.append(contour_world[idx_keep[i]])  
            contour_world = contours_world_list    
        elif type(contours_world) == np.ndarray:
            # only keep the shoreline points that are at least 30m from any cloud pixel
            idx_keep = np.ones(len(contours_world)).astype(bool)
            for k in range(len(contours_world)): # for every coord pair in the one line feature
                if np.any(np.linalg.norm(contours_world[k,:] - coords_cloud, axis=1) < 30):
                    idx_keep[k] = False
            contours_world = contours_world[idx_keep]
    
    # world coordinates array to geoseries
    contoursGS = gpd.GeoSeries(map(LineString,contours_world),crs=image_epsg)
    # remove any lines that fall below the threshold length defined by user
    shoreline = contoursGS[contoursGS.length > settings['min_length_sl']]
    
    # convert shorelines to different coord systems
    shoreline = shoreline.to_crs(settings['output_epsg'])
    shoreline_latlon = shoreline.to_crs(settings['ref_epsg'])
    shoreline_proj = shoreline.to_crs(settings['projection_epsg'])
        
    return shoreline, shoreline_latlon, shoreline_proj
    
def process_shoreline(contours, cloud_mask, georef, image_epsg, settings):
    """
    Converts the contours from image coordinates to world coordinates. 
    This function also removes the contours that are too small to be a shoreline 
    (based on the parameter settings['min_length_sl'])

    KV WRL 2018

    Arguments:
    -----------
    contours: np.array or list of np.array
        image contours as detected by the function find_contours
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
    image_epsg: int
        spatial reference system of the image from which the contours were extracted
    settings: dict with the following keys
        'output_epsg': int
            output spatial reference system
        'min_length_sl': float
            minimum length of shoreline contour to be kept (in meters)

    Returns:
    -----------
    shoreline: np.array
        array of points with the X and Y coordinates of the shoreline

    """
    # convert pixel coordinates to world coordinates
    contours_world = Toolbox.convert_pix2world(contours, georef)
    # convert world coordinates to desired spatial reference system
    contours_epsg = Toolbox.convert_epsg(contours_world, image_epsg, settings['output_epsg'])
    #contours_epsg = contours_world
    
    contour_latlon = Toolbox.convert_epsg(contours_world, image_epsg, 4326)
    
    contour_proj = Toolbox.convert_epsg(contour_latlon, 4326, settings['projection_epsg'])
    # remove contours that have a perimeter < min_length_sl (provided in settings dict)
    # this enables to remove the very small contours that do not correspond to the shoreline
    contours_long = []
    for l, wl in enumerate(contours_epsg):
        coords = [(wl[k,0], wl[k,1]) for k in range(len(wl))]
        a = LineString(coords) # shapely LineString structure
        if a.length >= settings['min_length_sl']:
            contours_long.append(wl)
    # format points into np.array
    x_points = np.array([])
    y_points = np.array([])
    for k in range(len(contours_long)):
        x_points = np.append(x_points,contours_long[k][:,0])
        y_points = np.append(y_points,contours_long[k][:,1])
    contours_array = np.transpose(np.array([x_points,y_points]))
    shoreline = contours_array
    
    contours_latlon_long = []
    for l, wl in enumerate(contour_latlon):
        coords = [(wl[k,0], wl[k,1]) for k in range(len(wl))]
        a = LineString(coords) # shapely LineString structure
        if a.length >= 0.00000000000000001:
            contours_latlon_long.append(wl)
    # format points into np.array
    x_points = np.array([])
    y_points = np.array([])
    
    for k in range(len(contours_latlon_long)):
        x_points = np.append(x_points,contours_latlon_long[k][:,0])
        y_points = np.append(y_points,contours_latlon_long[k][:,1])
    contours_latlon_array = np.transpose(np.array([x_points,y_points]))
    
    shoreline_latlon = contours_latlon_array
    
    contours_proj_long = []
    for l, wl in enumerate(contour_proj):
        coords = [(wl[k,0], wl[k,1]) for k in range(len(wl))]
        a = LineString(coords) # shapely LineString structure
        if a.length >= 0.01:
            contours_proj_long.append(wl)
    # format points into np.array
    x_points = np.array([])
    y_points = np.array([])
    for k in range(len(contours_proj_long)):
        x_points = np.append(x_points,contours_proj_long[k][:,0])
        y_points = np.append(y_points,contours_proj_long[k][:,1])
    contours_proj_array = np.transpose(np.array([x_points,y_points]))
    
    shoreline_proj = contours_proj_array

    # now remove any shoreline points that are attached to cloud pixels
    if sum(sum(cloud_mask)) > 0:
        # get the coordinates of the cloud pixels
        idx_cloud = np.where(cloud_mask)
        idx_cloud = np.array([(idx_cloud[0][k], idx_cloud[1][k]) for k in range(len(idx_cloud[0]))])
        # convert to world coordinates and same epsg as the shoreline points
        if idx_cloud.dtype == 'int64': # fix for S2 pix coords being int64 and unwriteable as float64
            idx_cloud = idx_cloud.astype('float64')
        coords_cloud = Toolbox.convert_epsg(Toolbox.convert_pix2world(idx_cloud, georef),
                                               image_epsg, settings['output_epsg'])[:,:-1]
        # only keep the shoreline points that are at least 30m from any cloud pixel
        idx_keep = np.ones(len(shoreline)).astype(bool)
        for k in range(len(shoreline)):
            if np.any(np.linalg.norm(shoreline[k,:] - coords_cloud, axis=1) < 30):
                idx_keep[k] = False
        shoreline = shoreline[idx_keep]
    return shoreline, shoreline_latlon, shoreline_proj

###################################################################################################
# PLOTTING FUNCTIONS
###################################################################################################

def SetUpDetectPlot(sitename, settings, im_ms, im_RGB, im_class, im_labels,
                    im_ref_buffer, date, satname,
                    fig, ax1, ax2, ax3, ax4,
                    contours_ndvi, t_ndvi, cloud_mask, georef, image_epsg,
                    sh_classif, sh_labels, contours_ndwi, t_ndwi):
    """
    Set up full plot window for showing/adjusting veg edge detection.
    FM Oct 2023

    Parameters
    ----------
    sitename : str
        Name of site.
    settings : dict
        Dictionary of user-defined settings used for the veg edge extraction.
    im_ms : np.array
        3D array representing multispectral satellite image.
    im_RGB : np.array
        3D array representing red-green-blue satellite image.
    im_class : np.array
        Boolean 2D array representing classified satellite image.
    im_labels: np.array
        3D array containing a boolean 2D image for each class (im_classif == label)
    im_ref_buffer : np.array
        Boolean 2D array matching image dimensions, with ref shoreline buffer zone = 1.
    date : str
        Satellite image capture date.
    satname : str
        Satellite image platform name.
    fig : matplotlib.figure.Figure object
        Figure window.
    ax1, ax2, ax3, ax4 : matplotlib.axes.Axes object
        Subplot axes within fig.
    contours_ndi : list of np.arrays
        Contains the coordinates of the contour lines extracted from the
        Normalized Difference Index image.
    t_ndi : float
        Threshold used to define the contours along.
    cloud_mask : array
        Mask containing cloudy pixels to be removed/ignored.
    georef : list
        Affine transformation matrix of satellite image [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale].
    image_epsg : int
        Coordinate projection code of image.
    sh_classif : np.array
        Boolean 2D array representing classified satellite image (CoastSat classes).
    sh_labels: np.array
        3D array containing a boolean 2D image for each class (CoastSat classes).
    contours_ndwi : list of np.arrays
        Contains the coordinates of the contour lines extracted from the
        Normalized Difference Water Index image.
    t_ndwi : float
        Threshold used to define the waterline contours along.

    Returns
    -------
    fig : matplotlib.figure.Figure object
        Updated figure window.
    ax1, ax2, ax3, ax4 : TYPE
        Updated subplot axes within fig.
    t_line : matplotlib.lines.Line2D
        Line object representing optimum threshold value in prob. dens. func. of NDVI/NDWI.
    im_ndvi_buffer : np.array
        2D array representing NDVI with reference shoreline buffer applied as mask.
    vlplots : list
        List of scatter plot objects of veg lines for each subplot.

    """
    
    cmap = cm.get_cmap('tab20c')
    colorpalette = cmap(np.arange(0,17,1))
    colours = np.zeros((4,4))
    # each row is RGBA for each class
    colours[0,:] = colorpalette[9]  # veg
    colours[1,:] = colorpalette[15]  # non-veg
    colours[2,:] = colorpalette[1] # water
    # colours[3,:] = colorpalette[16] # other
    
    # set RGB levels for each class in image to the colours matching the classes above (using original RGB as template)
    # for each class
    for cl in range(0,im_labels.shape[2]):
        # for each of the three colour values per class (R,G,B)
        for ic, colour in enumerate(colours[cl,:3]):
            # slice each class and each of the three colour arrays to set it to the matching class colours
            im_class[im_labels[:,:,cl],ic] = colour
        
    if settings['wetdry'] == True:
        for ic, colour in enumerate(colours[2,:3]):
            im_class[sh_labels[:,:,2],ic] = colour

    # compute NDVI grayscale image (NIR - R)
    im_ndvi = Toolbox.nd_index(im_ms[:,:,3], im_ms[:,:,2], cloud_mask)
    # buffer NDVI using reference shoreline
    im_ndvi_buffer = np.copy(im_ndvi)
    im_ndvi_buffer[~im_ref_buffer] = np.nan

    # change the color of nans to either black (0.0) or white (1.0) or somewhere in between
    nan_color = 1.0
    im_RGB = np.where(np.isnan(im_RGB), nan_color, im_RGB)
    im_class = np.where(np.isnan(im_class), 1.0, im_class)

    # create image 1 (RGB)
    ax1.imshow(im_RGB)
    im_ref_buffer_3d = np.repeat(im_ref_buffer[:,:,np.newaxis],3,axis=2)
    im_RGB_masked = im_RGB * im_ref_buffer_3d
    ax1.imshow(im_RGB_masked, alpha=0.3) # plot refline mask over top
    
    ax1.axis('off')
    ax1.set_title(sitename, fontweight='bold', fontsize=16)

    # create image 2 (classification)
    ax2.imshow(im_class)
    # if settings['wetdry'] == True:
    #     ax2.imshow(sh_class)
    ax2.axis('off')
    purple_patch = mpatches.Patch(color=colours[0,:], label='Vegetation')
    green_patch = mpatches.Patch(color=colours[1,:], label='Non-Vegetation')
    black_line = mlines.Line2D([],[],color='k',linestyle='-', label='Vegetation Line')
    if settings['wetdry'] == True:
        blue_patch = mpatches.Patch(color=colours[2,:], label='Water')
        blue_line = mlines.Line2D([],[],color='b',linestyle='-', label='Water Line')
        ax2handles = [purple_patch,green_patch,blue_patch,black_line,blue_line]
    else:
        ax2handles = [purple_patch,green_patch,black_line]
    ax2.legend(handles=ax2handles,
               bbox_to_anchor=(1, 1), fontsize=10)
    ax2.set_title(date, fontweight='bold', fontsize=16)

    # create image 3 (NDVI)
    ndviplot = ax3.imshow(im_ndvi, cmap='bwr')
    int_veg = im_ndvi[im_labels[:,:,0]]
    int_nonveg = im_ndvi[im_labels[:,:,1]] 
    labels_other = np.logical_and(~im_labels[:,:,0],~im_labels[:,:,1]) # for only veg/nonveg
    int_other = im_ndvi[labels_other]
    
    # clip down classified band index values to coastal buffer
    int_veg_clip, int_nonveg_clip = Image_Processing.ClipIndexVec(cloud_mask, im_ndvi, im_labels, im_ref_buffer)
    # FM: create transition zone mask
    TZbuffer = Toolbox.TZValues(int_veg_clip, int_nonveg_clip)
    
    im_TZ = Toolbox.TZimage(im_ndvi,TZbuffer)
    
    cmap = colors.ListedColormap(['orange'])
    tzplot = ax3.imshow(im_TZ, cmap=cmap, alpha=0.7)       
    
    ax3.axis('off')
    orange_patch = mpatches.Patch(color='orange', label='Transition Zone', alpha=0.7)
    ax3.legend(handles=[orange_patch],
               bbox_to_anchor=(1, 1), fontsize=10) #bbox_to_anchor=(1.1, 0.5)
    ax3.set_title('NDVI', fontsize=12)
    # plt.colorbar(ndviplot, ax=ax3, location='right', anchor=(0, 0.5), shrink=0.5)
    ax3.set_title(satname, fontweight='bold', fontsize=16)

    # additional options
    #    ax1.set_anchor('W')
    #    ax2.set_anchor('W')
    # cb = plt.colorbar(ndviplot, ax=ax3)
    # cb.ax.tick_params(labelsize=10)
    # cb.set_label('NDVI values')
    #    ax3.set_anchor('W')

    # plot histogram of NDVI values    
    binwidth = 0.01
    ax4.set_facecolor('0.7')
    ax4.yaxis.grid(color='w', linestyle='--', linewidth=0.5)
    ax4ticks = np.arange(-1, 1.1, 0.1)
    ax4.set_xticks(ax4ticks)
    ax4.set(ylabel='PDF',yticklabels=[], xlim=[-1,1])    
    if len(int_nonveg_clip) > 0 and sum(~np.isnan(int_nonveg_clip)) > 0:
        bins = np.arange(-1, 1, binwidth)
        ax4.hist(int_nonveg_clip, bins=bins, density=True, color=colours[1,:], label='Non-Vegetation')
    if len(int_veg_clip) > 0 and sum(~np.isnan(int_veg_clip)) > 0:
        bins = np.arange(-1, 1, binwidth)
        ax4.hist(int_veg_clip, bins=bins, density=True, color=colours[0,:], label='Vegetation', alpha=0.6)
    # if len(int_other) > 0 and sum(~np.isnan(int_other)) > 0:
    #     bins = np.arange(-1, 1, binwidth)
    #     ax4.hist(int_other, bins=bins, density=True, color='C7', label='other', alpha=0.5) 
     
    # process the contours into a vegline
    vegline, vegline_latlon, vegline_proj = ProcessShoreline(contours_ndvi, cloud_mask, georef, image_epsg, settings)
    if settings['wetdry'] == True: # also process shoreline if available
        shoreline, shoreline_latlon, shoreline_proj = ProcessShoreline(contours_ndwi, cloud_mask, georef, image_epsg, settings)
        
    # convert shoreline to pixels
    if len(vegline) > 0:
        # shoreline dataframe back to array
        veglineArr = Toolbox.GStoArr(vegline)
        vl_pix = Toolbox.convert_world2pix(veglineArr, georef)
    else: 
        vl_pix = np.array([[np.nan, np.nan],[np.nan, np.nan]])
        
    if settings['wetdry'] == True:
        if len(shoreline) > 0:
            # shoreline dataframe back to array
            shorelineArr = Toolbox.GStoArr(shoreline)
            sl_pix = Toolbox.convert_world2pix(shorelineArr, georef)
        else: 
            sl_pix = np.array([[np.nan, np.nan],[np.nan, np.nan]])
            
    # plot the shoreline on the images
    vl_plot1 = ax1.scatter(vl_pix[:,0], vl_pix[:,1], c='k', marker='.', s=5)
    vl_plot2 = ax2.scatter(vl_pix[:,0], vl_pix[:,1], c='k', marker='.', s=5)
    vl_plot3 = ax3.scatter(vl_pix[:,0], vl_pix[:,1], c='k', marker='.', s=5)
    t_line = ax4.axvline(x=t_ndvi,ls='--', c='k', lw=1.5, label='Threshold')
    if settings['wetdry'] == True:
        sl_plot1 = ax1.scatter(sl_pix[:,0], sl_pix[:,1], c='#0000A8', marker='.', s=5)
        sl_plot2 = ax2.scatter(sl_pix[:,0], sl_pix[:,1], c='#0000A8', marker='.', s=5)
        sl_plot3 = ax3.scatter(sl_pix[:,0], sl_pix[:,1], c='#0000A8', marker='.', s=5)
    # FM: plot vert lines where edges of overlapping classes reach (transition zone)
    TZmin = ax4.axvspan(TZbuffer[0],TZbuffer[1], color='C1',alpha=0.4,label='Transition Zone')

    vlplots = [vl_plot1, vl_plot2, vl_plot3]
    
    ax4.legend(loc=1)
    plt.draw() # to update the plot

    return fig, ax1, ax2, ax3, ax4, t_line, im_ndvi_buffer, vlplots


def show_detection(im_ms, cloud_mask, im_labels, im_ref_buffer, image_epsg, georef,
                   settings, date, satname, contours_ndvi, t_ndvi,
                   sh_classif=None, sh_labels=None, contours_ndwi=None, t_ndwi=None):
    """
    

    Parameters
    ----------
    im_ms : np.array
        3D array representing multispectral satellite image.
    cloud_mask : array
        Mask containing cloudy pixels to be removed/ignored.
    im_labels: np.array
        3D array containing a boolean 2D image for each class (im_classif == label)
    im_ref_buffer : np.array
        Boolean 2D array matching image dimensions, with ref shoreline buffer zone = 1.
    image_epsg : int
        Coordinate projection code of image.
    georef : list
        Affine transformation matrix of satellite image [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale].
    settings : dict
        Dictionary of user-defined settings used for the veg edge extraction.
    date : str
        Satellite image capture date.
    satname : str
        Satellite image platform name.
    contours_ndvi : list of np.arrays
        Contains the coordinates of the contour lines extracted from the
        Normalized Difference Index image.
    t_ndvi : float
        Threshold used to define the contours along.
    sh_classif : np.array
        Boolean 2D array representing classified satellite image (CoastSat classes). The default is None.
    sh_labels: np.array
        3D array containing a boolean 2D image for each class (CoastSat classes). The default is None.
    contours_ndwi : list of np.arrays
        Contains the coordinates of the contour lines extracted from the
        Normalized Difference Water Index image. The default is None.
    t_ndwi : float
        Threshold used to define the waterline contours along. The default is None.

    Raises
    ------
    StopIteration
        If escape key is pressed, end the shoreline detection checking process.

    Returns
    -------
    skip_image : bool
        Depending on whether left or right direction key is clicked, continue with
        plotting the image or skip it in the edge detection process.

    """
    sitename = settings['inputs']['sitename']
    filepath_data = settings['inputs']['filepath']
    # format date
    if satname != 'S2':
        date_str = datetime.strptime(date,'%Y-%m-%d').strftime('%Y-%m-%d')
    else:
        date_str = datetime.strptime(date,'%Y-%m-%d').strftime('%Y-%m-%d')

    im_RGB = Image_Processing.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
    # compute colours for classified image
    im_class = np.copy(im_RGB)
    # sh_class = np.copy(im_RGB)
    if plt.get_fignums():
            # get open figure if it exists
            fig = plt.gcf()
            ax1 = fig.axes[0]
            ax2 = fig.axes[1]
            ax3 = fig.axes[2]
            ax4 = fig.axes[3]
    else:
        # else create a new figure
        fig = plt.figure()
        fig.set_size_inches([18, 9])
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        # according to the image shape, decide whether it is better to have the images
        # in vertical subplots or horizontal subplots
        if im_RGB.shape[1] > 2.5*im_RGB.shape[0]:
            # vertical subplots (plot in rows)
            gs = gridspec.GridSpec(4, 1)
            gs.update(bottom=0.05, top=0.95, left=0.03, right=0.97)
            ax1 = fig.add_subplot(gs[0,0])
            ax2 = fig.add_subplot(gs[1,0], sharex=ax1, sharey=ax1)
            ax3 = fig.add_subplot(gs[2,0], sharex=ax1, sharey=ax1)
            ax4 = fig.add_subplot(gs[3,0])

        else:
            # horizontal subplots (plot in columns)
            gs = gridspec.GridSpec(2, 3, height_ratios=[4,1])
            gs.update(bottom=0.05, top=0.95, left=0.05, right=0.95)
            ax1 = fig.add_subplot(gs[0,0])
            ax2 = fig.add_subplot(gs[0,1], sharex=ax1, sharey=ax1)
            ax3 = fig.add_subplot(gs[0,2], sharex=ax1, sharey=ax1)
            ax4 = fig.add_subplot(gs[1,:])
        
    fig, ax1, ax2, ax3, ax4, t_line, im_ndvi_buffer, vlplots = SetUpDetectPlot(sitename, settings, im_ms, im_RGB, im_class, im_labels,
                                                      im_ref_buffer, date, satname,
                                                      fig, ax1, ax2, ax3, ax4,
                                                      contours_ndvi, t_ndvi, cloud_mask, georef, image_epsg,
                                                      sh_classif, sh_labels, contours_ndwi, t_ndwi)
    

    # if check_detection is True, let user manually accept/reject the images
    skip_image = False

    if settings['check_detection']:

        # set a key event to accept/reject the detections (see https://stackoverflow.com/a/15033071)
        # this variable needs to be immuatable so we can access it after the keypress event
        key_event = {}
        def press(event):
            # store what key was pressed in the dictionary
            key_event['pressed'] = event.key
        # let the user press a key, right arrow to keep the image, left arrow to skip it
        # to break the loop the user can press 'escape'
        while True:
            btn_keep = plt.text(1.1, 0.9, 'keep ⇨', size=12, ha="right", va="top",
                                transform=ax1.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            btn_skip = plt.text(-0.1, 0.9, '⇦ skip', size=12, ha="left", va="top",
                                transform=ax1.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            btn_esc = plt.text(0.5, 0, '<esc> to quit', size=12, ha="center", va="top",
                                transform=ax1.transAxes,
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

    # if save_figure is True, save a .jpg under /jpg_files/detection
    filepath = os.path.join(filepath_data, sitename, 'jpg_files', 'detection')
    if settings['save_figure'] and not skip_image:
        fig.savefig(os.path.join(filepath, date + '_' + satname + '.jpg'), dpi=150)

    # don't close the figure window, but remove all axes and settings, ready for next plot
    for ax in fig.axes:
        ax.clear()

    return skip_image


def adjust_detection(im_ms, cloud_mask, im_labels, im_ref_buffer, vegline, vegline_latlon, vegline_proj,
                     image_epsg, georef, settings, date, satname, contours_ndvi, t_ndvi,
                     sh_classif=None, sh_labels=None, contours_ndwi=None, t_ndwi=None):
    """
    

    Parameters
    ----------
    im_ms : np.array
        3D array representing multispectral satellite image.
    cloud_mask : array
        Mask containing cloudy pixels to be removed/ignored.
    im_labels: np.array
        3D array containing a boolean 2D image for each class (im_classif == label)
    im_ref_buffer : np.array
        Boolean 2D array matching image dimensions, with ref shoreline buffer zone = 1.
    vegline : TYPE
        DESCRIPTION.
    image_epsg : int
        Coordinate projection code of image.
    georef : list
        Affine transformation matrix of satellite image [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale].
    settings : dict
        Dictionary of user-defined settings used for the veg edge extraction.
    date : str
        Satellite image capture date.
    satname : str
        Satellite image platform name.
    contours_ndvi : list of np.arrays
        Contains the coordinates of the contour lines extracted from the
        Normalized Difference Vegetation Index image.
    t_ndi : float
        Threshold used to define the contours along.
    sh_classif : np.array
        Boolean 2D array representing classified satellite image (CoastSat classes). The default is None.
    sh_labels: np.array
        3D array containing a boolean 2D image for each class (CoastSat classes). The default is None.
    contours_ndwi : list of np.arrays
        Contains the coordinates of the contour lines extracted from the
        Normalized Difference Water Index image. The default is None.
    t_ndwi : float
        Threshold used to define the waterline contours along. The default is None.

    Raises
    ------
    StopIteration
        If escape key is pressed, end the shoreline detection checking process.

    Returns
    -------
    skip_image : bool
        Depending on whether left or right direction key is clicked, continue with
        plotting the image or skip it in the edge detection process.
    vegline: np.array
        Updated array of points with the X and Y coordinates of the shoreline.
    vegline_latlon :
        Updated array of points with the X and Y coordinates of the shoreline (in lat-long).
    vegline_proj :
        Updated array of points with the X and Y coordinates of the shoreline (in chosen projection system).
    t_ndi : float
        Updated threshold used to define the contours along.

    """

    sitename = settings['inputs']['sitename']
    filepath_data = settings['inputs']['filepath']
    # subfolder where the .jpg is stored if the user accepts the shoreline detection
    filepath = os.path.join(filepath_data, sitename, 'jpg_files','detection')
    # format date
    if satname != 'S2':
        date_str = datetime.strptime(date,'%Y-%m-%d').strftime('%Y-%m-%d')
    else:
        date_str = datetime.strptime(date,'%Y-%m-%d').strftime('%Y-%m-%d')

    im_RGB = Image_Processing.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
    # compute colours for classified image
    im_class = np.copy(im_RGB)
    # sh_class = np.copy(im_RGB)
    if plt.get_fignums():
            # get open figure if it exists
            fig = plt.gcf()
            ax1 = fig.axes[0]
            ax2 = fig.axes[1]
            ax3 = fig.axes[2]
            ax4 = fig.axes[3]
    else:
        # else create a new figure
        fig = plt.figure()
        fig.set_size_inches([18, 9])
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        # according to the image shape, decide whether it is better to have the images
        # in vertical subplots or horizontal subplots
        if im_RGB.shape[1] > 2.5*im_RGB.shape[0]:
            # vertical subplots (plot in rows)
            gs = gridspec.GridSpec(4, 1)
            gs.update(bottom=0.05, top=0.95, left=0.03, right=0.97)
            ax1 = fig.add_subplot(gs[0,0])
            ax2 = fig.add_subplot(gs[1,0], sharex=ax1, sharey=ax1)
            ax3 = fig.add_subplot(gs[2,0], sharex=ax1, sharey=ax1)
            ax4 = fig.add_subplot(gs[3,0])

        else:
            # horizontal subplots (plot in columns)
            gs = gridspec.GridSpec(2, 3, height_ratios=[4,1])
            gs.update(bottom=0.05, top=0.95, left=0.05, right=0.95)
            ax1 = fig.add_subplot(gs[0,0])
            ax2 = fig.add_subplot(gs[0,1], sharex=ax1, sharey=ax1)
            ax3 = fig.add_subplot(gs[0,2], sharex=ax1, sharey=ax1)
            ax4 = fig.add_subplot(gs[1,:])
        
    fig, ax1, ax2, ax3, ax4, t_line, im_ndvi_buffer, vlplots = SetUpDetectPlot(sitename, settings, im_ms, im_RGB, im_class, im_labels,
                                                      im_ref_buffer, date, satname,
                                                      fig, ax1, ax2, ax3, ax4,
                                                      contours_ndvi, t_ndvi, cloud_mask, georef, image_epsg,
                                                      sh_classif, sh_labels, contours_ndwi, t_ndwi)
    
    
    # adjust the threshold manually by letting the user change the threshold
    ax4.set_title('Click on the plot below to change the threshold and adjust the line detection. When finished, press <Enter>')
    while True:  
        # let the user click on the threshold plot
        pt = ginput(n=1, show_clicks=True, timeout=-1)
        # if a point was clicked
        if len(pt) > 0: 
            # if user clicked somewhere wrong and value is not between -1 and 1
            if np.abs(pt[0][0]) >= 1: continue
            # update the threshold value
            t_ndvi = pt[0][0]
            # update the plot
            t_line.set_xdata([t_ndvi,t_ndvi])
            # map contours with new threshold
            contours = measure.find_contours(im_ndvi_buffer, t_ndvi)
            # remove contours that contain NaNs (due to cloud pixels in the contour)
            contours = process_contours(contours) 
            # process the contours into a shoreline
            vegline, vegline_latlon, vegline_proj = ProcessShoreline(contours, cloud_mask, georef, image_epsg, settings)
            
            
            # convert line to pixels
            if len(vegline) > 0:
                veglineArr = Toolbox.GStoArr(vegline)
                vl_pix = Toolbox.convert_world2pix(Toolbox.convert_epsg(veglineArr,image_epsg,image_epsg)[:,[0,1]], georef)
            else: 
                vl_pix = np.array([[np.nan, np.nan],[np.nan, np.nan]])
            # update the plotted shorelines
            vlplots[0].set_offsets(vl_pix)
            vlplots[1].set_offsets(vl_pix)
            vlplots[2].set_offsets(vl_pix)
            
            fig.canvas.draw_idle()
        else:
            ax4.set_title('NDVI pixel intensities and threshold')
            break
    
    # let user manually accept/reject the image
    skip_image = False
    # set a key event to accept/reject the detections (see https://stackoverflow.com/a/15033071)
    # this variable needs to be immuatable so we can access it after the keypress event
    key_event = {}
    def press(event):
        # store what key was pressed in the dictionary
        key_event['pressed'] = event.key
    # let the user press a key, right arrow to keep the image, left arrow to skip it
    # to break the loop the user can press 'escape'
    while True:
        btn_keep = plt.text(1.1, 0.9, 'keep ⇨', size=12, ha="right", va="top",
                            transform=ax1.transAxes,
                            bbox=dict(boxstyle="square", ec='k',fc='w'))
        btn_skip = plt.text(-0.1, 0.9, '⇦ skip', size=12, ha="left", va="top",
                            transform=ax1.transAxes,
                            bbox=dict(boxstyle="square", ec='k',fc='w'))
        btn_esc = plt.text(0.5, 0, '<esc> to quit', size=12, ha="center", va="top",
                            transform=ax1.transAxes,
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

    # if save_figure is True, save a .jpg under /jpg_files/detection
    if settings['save_figure'] and not skip_image:
        fig.savefig(os.path.join(filepath, date + '_' + satname + '.jpg'), dpi=150)
        

    # don't close the figure window, but remove all axes and settings, ready for next plot
    for ax in fig.axes:
        ax.clear()

    return skip_image, vegline, vegline_latlon, vegline_proj, t_ndvi


