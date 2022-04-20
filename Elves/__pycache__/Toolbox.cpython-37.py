# uncompyle6 version 3.8.0
# Python bytecode 3.7.0 (3394)
# Decompiled from: Python 3.9.7 (default, Sep 16 2021, 13:09:58) 
# [GCC 7.5.0]
# Embedded file name: /media/14TB_RAID_Array/User_Homes/Freya_Muir/PhD/Year2/ModelsFrameworks/CoastWatch-main/Elves/Toolbox.py
# Compiled at: 2022-01-28 10:14:22
# Size of source mod 2**32: 30562 bytes
"""
This module contains utilities to work with satellite images
    
Author: Kilian Vos, Water Research Laboratory, University of New South Wales
"""
import os, numpy as np
import matplotlib.pyplot as plt
import pdb
from osgeo import gdal, osr
import geopandas as gpd
from shapely import geometry
from shapely.geometry import Point, Polygon, LineString, MultiLineString, MultiPoint
import skimage.transform as transform
from astropy.convolution import convolve
from datetime import datetime
from IPython.display import clear_output
import ee, pickle, math
np.seterr(all='ignore')

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
    aff_mat = np.array([[georef[1], georef[2], georef[0]],
     [
      georef[4], georef[5], georef[3]],
     [
      0, 0, 1]])
    tform = transform.AffineTransform(aff_mat)
    if type(points) is list:
        points_converted = []
        for i, arr in enumerate(points):
            tmp = arr[:, [1, 0]]
            points_converted.append(tform(tmp))

    else:
        if type(points) is np.ndarray:
            tmp = points[:, [1, 0]]
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
                
    Returns:    
    -----------
    points_converted: np.array or list of np.array 
        converted coordinates (pixel row and column)
    
    """
    aff_mat = np.array([[georef[1], georef[2], georef[0]], [georef[4], georef[5], georef[3]], [0, 0, 1]])
    tform = transform.AffineTransform(aff_mat)
    if type(points) is list:
        points_converted = []
        for i, arr in enumerate(points):
            points_converted.append(tform.inverse(points))

    else:
        if type(points) is np.ndarray:
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
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(epsg_in)
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(epsg_out)
    coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    if type(points) is list:
        points_converted = []
        for i, arr in enumerate(points):
            points_converted.append(np.array(coordTransform.TransformPoints(arr)))

    else:
        if type(points) is np.ndarray:
            points_converted = np.array(coordTransform.TransformPoints(points))
        else:
            raise Exception('invalid input type')
    return points_converted


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
    vec_mask = cloud_mask.reshape(im1.shape[0] * im1.shape[1])
    vec_nd = np.ones(len(vec_mask)) * np.nan
    vec1 = im1.reshape(im1.shape[0] * im1.shape[1])
    vec2 = im2.reshape(im2.shape[0] * im2.shape[1])
    temp = np.divide(vec1[(~vec_mask)] - vec2[(~vec_mask)], vec1[(~vec_mask)] + vec2[(~vec_mask)])
    vec_nd[~vec_mask] = temp
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
    image = image.astype(float)
    image_padded = np.pad(image, radius, 'reflect')
    win_rows, win_cols = radius * 2 + 1, radius * 2 + 1
    win_mean = convolve(image_padded, (np.ones((win_rows, win_cols))), boundary='extend', normalize_kernel=True,
      nan_treatment='interpolate',
      preserve_nan=True)
    win_sqr_mean = convolve((image_padded ** 2), (np.ones((win_rows, win_cols))), boundary='extend', normalize_kernel=True,
      nan_treatment='interpolate',
      preserve_nan=True)
    win_var = win_sqr_mean - win_mean ** 2
    win_std = np.sqrt(win_var)
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
    raster = gdal.Open(fn, gdal.GA_Update)
    for i in range(raster.RasterCount):
        out_band = raster.GetRasterBand(i + 1)
        out_data = out_band.ReadAsArray()
        out_band.SetNoDataValue(0)
        no_data_value = out_band.GetNoDataValue()
        out_data[mask] = no_data_value
        out_band.WriteArray(out_data)

    raster = None


def get_filepath(inputs, satname):
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
    if satname == 'L5':
        filepath = os.path.join(filepath_data, sitename, satname, '30m')
    else:
        if satname == 'L7':
            filepath_pan = os.path.join(filepath_data, sitename, 'L7', 'pan')
            filepath_ms = os.path.join(filepath_data, sitename, 'L7', 'ms')
            filepath = [filepath_pan, filepath_ms]
        else:
            if satname == 'L8':
                filepath_pan = os.path.join(filepath_data, sitename, 'L8', 'pan')
                filepath_ms = os.path.join(filepath_data, sitename, 'L8', 'ms')
                filepath = [filepath_pan, filepath_ms]
            else:
                if satname == 'S2':
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
        filename_ms = filename.replace('pan', 'ms')
        fn = [os.path.join(filepath[0], filename),
         os.path.join(filepath[1], filename_ms)]
    if satname == 'S2':
        filename20 = filename.replace('10m', '20m')
        filename60 = filename.replace('10m', '60m')
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
    output_all = dict([])
    satnames = list(output.keys())
    for key in output[satnames[0]].keys():
        output_all[key] = []

    output_all['satname'] = []
    for satname in list(output.keys()):
        for key in output[satnames[0]].keys():
            output_all[key] = output_all[key] + output[satname][key]

        output_all['satname'] = output_all['satname'] + [_ for _ in np.tile(satname, len(output[satname]['dates']))]

    idx_sorted = sorted((range(len(output_all['dates']))), key=(output_all['dates'].__getitem__))
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

    def duplicates_dict(lst):
        """return duplicates and indices"""

        def duplicates(lst, item):
            return [i for i, x in enumerate(lst) if x == item]

        return dict(((x, duplicates(lst, x)) for x in set(lst) if lst.count(x) > 1))

    dates = output['dates']
    dates_str = [datetime.strptime(_, '%Y-%m-%d').strftime('%Y-%m-%d') for _ in dates]
    dupl = duplicates_dict(dates_str)
    if dupl:
        output_no_duplicates = dict([])
        idx_remove = []
        for k, v in dupl.items():
            idx_remove.append(v[0])

        idx_remove = sorted(idx_remove)
        idx_all = np.linspace(0, len(dates_str) - 1, len(dates_str))
        idx_keep = list(np.where(~np.isin(idx_all, idx_remove))[0])
        for key in output.keys():
            output_no_duplicates[key] = [output[key][i] for i in idx_keep]

        print('%d duplicates' % len(idx_remove))
        return output_no_duplicates
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
    if dates[0] < dates_ts[0] or dates[(-1)] > dates_ts[(-1)]:
        raise Exception('Time-series do not cover the range of your input dates')
    temp = []

    def find(item, lst):
        start = 0
        start = lst.index(item, start)
        return start

    for i, date in enumerate(dates):
        print(('\rExtracting closest points: %d%%' % int((i + 1) * 100 / len(dates))), end='')
        temp.append(values_ts[find(min((item for item in dates_ts if item > date)), dates_ts)])

    values = np.array(temp)
    return values


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
    with open(fn) as (kmlFile):
        doc = kmlFile.read()
    str1 = '<coordinates>'
    str2 = '</coordinates>'
    subdoc = doc[doc.find(str1) + len(str1):doc.find(str2)]
    coordlist = subdoc.split('\n')
    polygon = []
    for i in range(1, len(coordlist) - 1):
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
        transects[gdf.loc[(i, 'name')]] = np.array(gdf.loc[(i, 'geometry')].coords)

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
    counter = 0
    for i in range(len(output['shorelines'])):
        if len(output['shorelines'][i]) == 0:
            continue
        else:
            if geomtype == 'lines':
                for j in range(len(output['shorelines'][i])):
                    abbba = []
                    abbba.append(output['shorelines'][i][j][1])
                    abbba.append(output['shorelines'][i][j][0])
                    output['shorelines'][i][j] = abbba

                geom = geometry.LineString(output['shorelines'][i])
            else:
                if geomtype == 'points':
                    coords = output['shorelines'][i]
                    geom = geometry.MultiPoint([(coords[(_, 1)], coords[(_, 0)]) for _ in range(coords.shape[0])])
                else:
                    raise Exception('geomtype %s is not an option, choose between lines or points' % geomtype)
            gdf = gpd.GeoDataFrame(geometry=(gpd.GeoSeries(geom)))
            gdf.index = [i]
            gdf.loc[(i, 'date')] = datetime.strftime(datetime.strptime(output['dates'][0], '%Y-%m-%d'), '%Y-%m-%d %H:%M:%S')
            gdf.loc[(i, 'satname')] = output['satname'][i]
            gdf.loc[(i, 'cloud_cover')] = output['cloud_cover'][i]
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
    for i, key in enumerate(list(transects.keys())):
        geom = geometry.LineString(transects[key])
        gdf = gpd.GeoDataFrame(geometry=(gpd.GeoSeries(geom)))
        gdf.index = [i]
        gdf.loc[(i, 'name')] = key
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

    def GetExtent(gt, cols, rows):
        """Return list of corner coordinates from a geotransform"""
        ext = []
        xarr = [
         0, cols]
        yarr = [0, rows]
        for px in xarr:
            for py in yarr:
                x = gt[0] + px * gt[1] + py * gt[2]
                y = gt[3] + px * gt[4] + py * gt[5]
                ext.append([x, y])

            yarr.reverse()

        return ext

    data = gdal.Open(fn, gdal.GA_ReadOnly)
    gt = data.GetGeoTransform()
    cols = data.RasterXSize
    rows = data.RasterYSize
    ext = GetExtent(gt, cols, rows)
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
        with open(os.path.join(filepath, filename), 'rb') as (f):
            metadata = pickle.load(f)
        return metadata
    metadata = dict([])
    for i in range(len(sat_list)):
        metadata[sat_list[i]] = {'filenames':[],  'acc_georef':[],  'epsg':[],  'dates':[]}

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
                d = datetime.strptime(Sat[i].getInfo().get('features')[j]['properties']['DATATAKE_IDENTIFIER'][5:13], '%Y%m%d')
                metadata[sat_list[i]]['dates'].append(str(d.strftime('%Y-%m-%d')))
            print((sat_list[i]), ': ', (100 * j / len(Sat[i].getInfo().get('features'))), '%', end='')

    with open(os.path.join(filepath, sitename + '_metadata.pkl'), 'wb') as (f):
        pickle.dump(metadata, f)
    return metadata


def image_retrieval(inputs):
    point = ee.Geometry.Point(inputs['polygon'][0][0])
    Sat = []
    if 'L5' in inputs['sat_list']:
        Landsat5 = ee.ImageCollection('LANDSAT/LT05/C01/T1_TOA').filterBounds(point).filterDate(inputs['dates'][0], inputs['dates'][1])
        Sat.append(Landsat5)
    if 'L8' in inputs['sat_list']:
        Landsat8 = ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA').filterBounds(point).filterDate(inputs['dates'][0], inputs['dates'][1]).filter(ee.Filter.lt('CLOUD_COVER', 95))
        Sat.append(Landsat8)
    if 'S2' in inputs['sat_list']:
        Sentinel2 = ee.ImageCollection('COPERNICUS/S2').filterBounds(point).filterDate(inputs['dates'][0], inputs['dates'][1]).filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 80))
        Sat.append(Sentinel2)
    return Sat


def save_shapefiles(output, geomtype, name_prefix, sitename):
    output_geom = gpd.GeoSeries(map(MultiPoint, output['shorelines']))
    outputGDF = gpd.GeoDataFrame(output, crs='EPSG:27700', geometry=output_geom)
    outputGDF = outputGDF.drop('shorelines', axis=1)
    outputGDF.to_file(name_prefix + sitename + str(min(output['dates'])) + '_' + str(min(output['dates'])) + '_veglines.shp')


def Separate_TimeSeries_year(cross_distance, output, key):
    Date_Organised = [
     [
      datetime.strptime(min(output['dates']), '%Y-%m-%d').year]]
    Distance_Organised = [[]]
    Month_Organised = [[]]
    for i in range(len(output['dates'])):
        appended = False
        for j in range(len(Date_Organised)):
            if datetime.strptime(output['dates'][i], '%Y-%M-%d').year == Date_Organised[j][0]:
                Date_Organised[j].append(datetime.strptime(output['dates'][i], '%Y-%m-%d').year)
                Month_Organised[j].append(datetime.strptime(output['dates'][i], '%Y-%m-%d').month)
                Distance_Organised[j].append((cross_distance[key] - np.nanmedian(cross_distance[key]))[i])
                appended = True

        if appended == False:
            Date_Organised.append([datetime.strptime(output['dates'][i], '%Y-%m-%d').year])
            Month_Organised.append([datetime.strptime(output['dates'][i], '%Y-%m-%d').month])
            Distance_Organised.append([(cross_distance[key] - np.nanmedian(cross_distance[key]))[i]])

    DateArr = []
    DistanceAvgArr = []
    for i in range(len(Date_Organised)):
        DateArr.append(Date_Organised[i][0])
        DistanceAvgArr.append(np.nanmean(Distance_Organised[i]))

    return (Date_Organised, Month_Organised, Distance_Organised, DateArr, DistanceAvgArr)


def Separate_TimeSeries_month(cross_distance, output, key):
    Date_Organised = [[]]
    Distance_Organised = [[]]
    Month_Organised = [
     [
      1]]
    for i in range(len(output['dates'])):
        appended = False
        for j in range(len(Month_Organised)):
            if datetime.strptime(output['dates'][i], '%Y-%m-%d').month == Month_Organised[j][0]:
                Date_Organised[j].append(datetime.strptime(output['dates'][i], '%Y-%m-%d').year)
                Month_Organised[j].append(datetime.strptime(output['dates'][i], '%Y-%m-%d').month)
                Distance_Organised[j].append((cross_distance[key] - np.nanmedian(cross_distance[key]))[i])
                appended = True

        if appended == False:
            Date_Organised.append([datetime.strptime(output['dates'][i], '%Y-%m-%d').year])
            Month_Organised.append([datetime.strptime(output['dates'][i], '%Y-%m-%d').month])
            Distance_Organised.append([(cross_distance[key] - np.nanmedian(cross_distance[key]))[i]])

    DateArr = []
    DistanceAvgArr = []
    for i in range(len(Distance_Organised)):
        DateArr.append(Month_Organised[i][0])
        temp_list = Distance_Organised[i]
        newlist = [x for x in temp_list if math.isnan(x) == False]
        DistanceAvgArr.append(np.nanmean(newlist))

    return (Date_Organised, Month_Organised, Distance_Organised, DateArr, DistanceAvgArr)


def daterange(date1, date2):
    for n in range(int(date2.year) - int(date1.year) + 1):
        yield int(date1.year) + n