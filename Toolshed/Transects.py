"""
This module contains functions to analyze the 2D shorelines along shore-normal
transects
    
Martin Hurst, Freya Muir
"""

# load modules
import os
from osgeo import ogr
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import pdb
from datetime import datetime
from pathlib import Path
import pyproj
from pyproj import Proj

# other modules
import skimage.transform as transform
from pylab import ginput

from Toolshed.Coast import *


def ProduceTransectsAll(SmoothingWindowSize, NoSmooths, TransectSpacing, DistanceInland, DistanceOffshore, proj, BasePath):
    
    for subdir, dirs, files in os.walk(BasePath):
        for direc in dirs:
            FileSpec = '/' + str(os.path.join(direc)) + '/' + str(os.path.join(direc)) + '.shp'
            ReprojSpec = '/' + str(os.path.join(direc)) + '/Reproj.shp'
            TransectSpec = '/' + str(os.path.join(direc)) + '/Transect.shp'
            CoastSpec = '/' + str(os.path.join(direc)) + '/Coast.shp'
            Filename2SaveCoast = '/' + str(os.path.join(direc)) + '/' + "My_Baseline.shp"
        
            #Reprojects shape file from EPSG 4326 to 27700 (britain)
        
            shape = gpd.read_file(BasePath+FileSpec)
            #shape = shape.set_crs(4326)
            # change CRS to epsg 27700
            shape = shape.to_crs(crs=proj,epsg=4326)
            # write shp file
            shape.to_file(BasePath+ReprojSpec)
        
            #Creates coast objects
            CellCoast = Coast(BasePath+ReprojSpec, MinLength=5)

            if not CellCoast.BuiltTransects:
            
                # may need to think carefully about how much to smooth
                CellCoast.SmoothCoastLines(WindowSize=SmoothingWindowSize,NoSmooths=NoSmooths)
            
                # make sure each line is correctly orientated with sea on left as you look down the line
                # this is something we'll need to think about replacing
                # CellCoast.CheckOrientation(str(SoftPath),str(MLWSPath))
        
                # write smoothed coast/bathy to file
                CellCoast.WriteCoastShp(BasePath+CoastSpec)
    
                # create some initial dummy transects, check inland/offshore the right way around
                CellCoast.GenerateTransects(TransectSpacing, DistanceInland, DistanceOffshore, CheckTopology=False)
        
                CellCoast.BuiltTransects = True
            
                CellCoast.WriteTransectsShp(BasePath+TransectSpec)
            
                # SAVE ENTIRE COAST OBJECT
                with open(str(BasePath+Filename2SaveCoast), 'wb') as PFile:
                    pickle.dump(CellCoast, PFile)
    return

def ProduceTransects(SmoothingWindowSize, NoSmooths, TransectSpacing, DistanceInland, DistanceOffshore, proj, sitename, BasePath, RefShapePath):
        
    FileSpec = RefShapePath
    ReprojSpec = BasePath + '/Baseline_Reproj.shp'
    TransectSpec = os.path.join(BasePath, sitename+'_Transects.shp')
    CoastSpec = BasePath + '/Coast.shp'
    Filename2SaveCoast = BasePath + '/Coast.pydata'
    
    if (SmoothingWindowSize % 2) == 0:
        SmoothingWindowSize = SmoothingWindowSize + 1
        print('Window size should be odd; changed to %s m' % SmoothingWindowSize)
    
    shape = gpd.read_file(FileSpec)
    # change CRS to epsg 27700
    shape = shape.to_crs(epsg=27700)
    # write shp file
    shape.to_file(ReprojSpec)
        
    #Creates coast objects
    CellCoast = Coast(ReprojSpec, MinLength=10)

    if not CellCoast.BuiltTransects:
            
        CellCoast.SmoothCoastLines(WindowSize=SmoothingWindowSize,NoSmooths=NoSmooths)
            
        CellCoast.WriteCoastShp(CoastSpec)
    
        CellCoast.GenerateTransects(TransectSpacing, DistanceInland, DistanceOffshore, CheckTopology=False)
        
        CellCoast.BuiltTransects = True
            
        CellCoast.WriteTransectsShp(TransectSpec)
            
        with open(str(Filename2SaveCoast), 'wb') as PFile:
            pickle.dump(CellCoast, PFile)
            
    return
    
def GetIntersections(BasePath, TransectGDF, VeglineGDF):
    
    # Filename2SaveCoast = BasePath + '/Coast.pydata'
    # with open(str(Filename2SaveCoast), 'rb') as PFile:
    #     CellCoast = pickle.load(PFile)
        
    # HistoricalShorelinesShp = PathToVegShp
    # CellCoast.ExtractSatShorePositions(HistoricalShorelinesShp)
    
    
    # IntersectPoints = TransectGDF.unary_union.intersection(VeglineGDF.unary_union)
    # IntersectPoints = gpd.GeoDataFrame(geometry=gpd.GeoSeries(IntersectPoints)).explode()
    # IntersectGDF = gpd.sjoin(TransectGDF ,VeglineGDF)
        
    # TransectDict = TransectGDF.to_dict()
    # TransectDict['distances'] = {}
    # for T in range(len(TransectDict['TransectID'])):
    #     TransectDict['distances'][T] = 
    
    columns_data = []
    geoms = []
    for _, _, ID, TrGeom in TransectGDF.itertuples():
        for _,dates,filename,cloud,ids,otsu,satn,VGeom in VeglineGDF.itertuples():
            Intersects = TrGeom.intersection(VGeom)
            columns_data.append((ID,dates,filename,cloud,ids,otsu,satn))
            geoms.append(Intersects)
    AllIntersects = gpd.GeoDataFrame(columns_data,geometry=geoms,columns=['TransectID','dates','filename','cloud_cove','idx','Otsu_thres','satname'])
    AllIntersectsS = AllIntersects[~AllIntersects.is_empty].reset_index().drop('index',axis=1)
    AllIntersectsS['interpnt'] = AllIntersectsS['geometry']
    interpnt
    
    return AllIntersects
    

def compute_intersection(output, transects, settings, linetype):
    """
    Computes the intersection between the 2D shorelines and the shore-normal.
    transects. It returns time-series of cross-shore distance along each transect.
    
    KV WRL 2018       

    Arguments:
    -----------
    output: dict
        contains the extracted shorelines and corresponding metadata
    transects: dict
        contains the X and Y coordinates of each transect
    settings: dict with the following keys
        'along_dist': int
            alongshore distance considered caluclate the intersection
              
    Returns:    
    -----------
    cross_dist: dict
        time-series of cross-shore distance along each of the transects. 
        Not tidally corrected.        
    """  
    
    """
    if (linetype+'transect_time_series.csv') in os.listdir(settings['inputs']['filepath']):
        print('Cross-distance calculations already exist and were loaded')
        with open(os.path.join(settings['inputs']['filepath'], linetype+'transect_time_series.csv'), 'rb') as f:
            cross_dist = pickle.load(f)
        return cross_dist
    """
    
    # loop through shorelines and compute the median intersection    
    intersections = np.zeros((len(output['shorelines']),len(transects)))
    for i in range(len(output['shorelines'])):

        sl = output['shorelines'][i]
        
        print(" \r\tShoreline %4d / %4d" % (i+1, len(output['shorelines'])), end="")
        
        for j,key in enumerate(list(transects.keys())): 
            
            # compute rotation matrix
            X0 = transects[key][0,0]
            Y0 = transects[key][0,1]
            temp = np.array(transects[key][-1,:]) - np.array(transects[key][0,:])
            phi = np.arctan2(temp[1], temp[0])
            Mrot = np.array([[np.cos(phi), np.sin(phi)],[-np.sin(phi), np.cos(phi)]])
    
            # calculate point to line distance between shoreline points and the transect
            p1 = np.array([X0,Y0])
            p2 = transects[key][-1,:]
            d_line = np.abs(np.cross(p2-p1,sl-p1)/np.linalg.norm(p2-p1))
            # calculate the distance between shoreline points and the origin of the transect
            d_origin = np.array([np.linalg.norm(sl[k,:] - p1) for k in range(len(sl))])
            # find the shoreline points that are close to the transects and to the origin
            # the distance to the origin is hard-coded here to 1 km 
            idx_dist = np.logical_and(d_line <= settings['along_dist'], d_origin <= 1000)
            # find the shoreline points that are in the direction of the transect (within 90 degrees)
            temp_sl = sl - np.array(transects[key][0,:])
            phi_sl = np.array([np.arctan2(temp_sl[k,1], temp_sl[k,0]) for k in range(len(temp_sl))])
            diff_angle = (phi - phi_sl)
            idx_angle = np.abs(diff_angle) < np.pi/2
            # combine the transects that are close in distance and close in orientation
            idx_close = np.where(np.logical_and(idx_dist,idx_angle))[0]     
            
            # in case there are no shoreline points close to the transect 
            if len(idx_close) == 0:
                intersections[i,j] = np.nan
            else:
                # change of base to shore-normal coordinate system
                xy_close = np.array([sl[idx_close,0],sl[idx_close,1]]) - np.tile(np.array([[X0],
                                   [Y0]]), (1,len(sl[idx_close])))
                xy_rot = np.matmul(Mrot, xy_close)
                # compute the median of the intersections along the transect
                intersections[i,j] = np.nanmedian(xy_rot[0,:])
    
    # fill the a dictionnary
    cross_dist = dict([])
    cross_dist['dates'] = output['dates']
    for j,key in enumerate(list(transects.keys())): 
        cross_dist[key] = intersections[:,j]  
    
    
    # save a .csv file for Excel users
    out_dict = dict([])
    out_dict['dates'] = output['dates']
    for key in transects.keys():
        out_dict['Transect '+ key] = cross_dist[key]
    df = pd.DataFrame(out_dict)
    fn = os.path.join(settings['inputs']['filepath'],settings['inputs']['sitename'],linetype+
                      'transect_time_series.csv')
    df.to_csv(fn, sep=',')
    print('Time-series of the shoreline change along the transects saved as:\n%s'%fn)
    
    return cross_dist

def stuffIntoLibrary(geo, image_epsg, projection_epsg, filepath, sitename):
    
    print('Reading transects into library for further analysis...')
    transect = Path("Data/" + sitename + "/transect_proj.pkl")
    
    proj1 = Proj(init="epsg:"+str(projection_epsg)) 
    proj2 = Proj(init="epsg:"+str(4326))
    proj3 = Proj(init="epsg:"+str(image_epsg))
    
    if transect.is_file():
        with open(os.path.join(filepath, sitename + '_transect_proj' + '.pkl'), 'rb') as f:
            transects_proj = pickle.load(f)
        with open(os.path.join(filepath, sitename + '_transect_latlon' + '.pkl'), 'rb') as f:
            transects_latlon = pickle.load(f)
            
        return transects_latlon, transects_proj
    
    transects_latlon = dict([])
    transects_proj = dict([])

    for i in range (len(geo['geometry'])):
        
        lib = 'Transect_'+str(i+1)
    
        x,y = geo['geometry'][i].coords.xy
        
        # convert to lat lon
        xy0 = pyproj.transform(proj1,proj2,y[0],x[0])
        xy1 = pyproj.transform(proj1,proj2,y[1],x[1])
        coord0_latlon = [xy0[1],xy0[0]]
        coord1_latlon = [xy1[1],xy1[0]]

        transects_latlon[lib] = np.array([coord0_latlon, coord1_latlon])
        #x,y = pyproj.transform(proj2,proj3,transects_latlon[lib][0][1],transects_latlon[lib][0][0])
        #x1,y1 = pyproj.transform(proj2,proj3,transects_latlon[lib][1][1],transects_latlon[lib][1][0])
        transects_proj[lib] = np.array([[x[1],y[1]],[x[0],y[0]]])

        print(" \r\tCurrent Progress:",np.round(i/len(geo['geometry'])*100,2),"%",end='')
    
    with open(os.path.join(filepath, sitename + '_transect_proj.pkl'), 'wb') as f:
            pickle.dump(transects_proj, f)
            
    with open(os.path.join(filepath, sitename + '_transect_latlon.pkl'), 'wb') as f:
            pickle.dump(transects_latlon, f)
            
    return transects_latlon, transects_proj

def transect_compiler(Rows, transect_proj, transect_range, output):
    
    cross_distance_condensed = dict([])
    standard_err_condensed = dict([])
    transect_condensed = dict([])
    Dates = dict([])
    new_Transect = 1

    cross_arr = []
    trans_arr = []

    for i in range(len(transect_range)):
        cross_arr = []
        trans_arr = []
        for j in range(transect_range[i][0],transect_range[i][1]):
            try:
                arr = []
                for k in range(len(Rows)-1):
                    try:
                        arr.append(float(Rows[k][j]))
                    except:
                        arr.append(np.nan)
                cross_arr.append(arr)
                trans_arr.append(transect_proj[list(transect_proj.keys())[j]])
            except:
                continue
        std = np.nanstd(cross_arr,0)
        for j in range(len(std)):
            std[j] = std[j]/(abs(transect_range[i][0]-transect_range[i][1]))**0.5

        NaN_mask = np.isfinite(np.nanmean(cross_arr,0))
        cross_distance_condensed['Transect_'+str(transect_range[i][0])+'-'+str(transect_range[i][1])] = np.nanmean(cross_arr,0).astype(np.double)[NaN_mask]
        standard_err_condensed['Transect_'+str(transect_range[i][0])+'-'+str(transect_range[i][1])] = std.astype(np.double)[NaN_mask]
        Dates['Transect_'+str(transect_range[i][0])+'-'+str(transect_range[i][1])] = np.array(output['dates'])[NaN_mask]
        transect_condensed['Transect_'+str(transect_range[i][0])+'-'+str(transect_range[i][1])] = np.mean(trans_arr,0).astype(np.double)#[NaN_mask]
        
    return cross_distance_condensed, standard_err_condensed, transect_condensed, Dates