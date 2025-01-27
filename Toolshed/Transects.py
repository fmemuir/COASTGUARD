"""
This module contains functions to analyze the 2D shorelines along shore-normal
transects
    
Martin Hurst, Freya Muir - University of Glasgow
"""

# load modules
import os
import glob
import pickle

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import pyproj
from pyproj import Proj

# other modules
from sklearn.linear_model import LinearRegression
from scipy import stats
from pylab import ginput
import rasterio as rio
from rasterio.features import shapes
from shapely.geometry import Point, Polygon, LineString, MultiLineString, MultiPoint


from Toolshed import Toolbox, Waves, Slope
from Toolshed.Coast import *


def ProduceTransectsAll(SmoothingWindowSize, NoSmooths, TransectSpacing, DistanceInland, DistanceOffshore, proj, BasePath):
    """
    UNUSED
    Produce transects using CoastalMappingTools
    FM Sept 2022


    """
    for subdir, dirs, files in os.walk(BasePath):
        for direc in dirs:
            FileSpec = '/' + str(os.path.join(direc)) + '/' + str(os.path.join(direc)) + '.shp'
            ReprojShp = '/' + str(os.path.join(direc)) + '/Reproj.shp'
            TransectSpec = '/' + str(os.path.join(direc)) + '/Transect.shp'
            CoastSpec = '/' + str(os.path.join(direc)) + '/Coast.shp'
            Filename2SaveCoast = '/' + str(os.path.join(direc)) + '/' + "My_Baseline.shp"
        
            #Reprojects shape file from EPSG 4326 to 27700 (britain)
        
            shape = gpd.read_file(BasePath+FileSpec)
            #shape = shape.set_crs(4326)
            # change CRS to epsg 27700
            shape = shape.to_crs(crs=proj,epsg=4326)
            # write shp file
            shape.to_file(BasePath+ReprojShp)
        
            #Creates coast objects
            CellCoast = Coast(BasePath+ReprojShp, MinLength=5)

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

def ProduceTransects(settings, SmoothingWindowSize, NoSmooths, TransectSpacing, DistanceInland, DistanceOffshore, VegBasePath, referenceLinePath):
    """
    Produce shore-normal transects using CoastalMappingTools
    FM Oct 2022

    Parameters
    ----------
    settings : dict
        Dictionary of user-defined settings used for the veg edge/waterline extraction.
    SmoothingWindowSize : int
        Smoothing window size in metres (should be odd to account for indexing).
    NoSmooths : int
        Number of times to repeat the smoothing process.
    TransectSpacing : int
        Alongshore space between transects in metres.
    DistanceInland : int
        Distance in metres to extend transects inland.
    DistanceOffshore : TYPE
        Distance in metres to extend transects out to sea.
    VegBasePath : str
        Filepath to where veglines are stored (to also save transects to).
    referenceLinePath : str
        Filepath to reference shoreline shapefile.

    Returns
    -------
    TransectGDF : GeoDataFrame
        GoeDataFrame of cross-shore transects generated from smoothed reference shoreline.

    """
    
    sitename = settings['inputs']['sitename']
    ReprojShp = VegBasePath + '/Baseline_Reproj.shp'
    TransectPath = os.path.join(VegBasePath, sitename+'_Transects.shp')
    CoastSpec = VegBasePath + '/Coast.shp'
    Filename2SaveCoast = VegBasePath + '/Coast.pydata'
    
    if (SmoothingWindowSize % 2) == 0:
        SmoothingWindowSize = SmoothingWindowSize + 1
        print('Window size should be odd; changed to %s m' % SmoothingWindowSize)
    
    refGDF = gpd.read_file(os.path.join('Data','referenceLines',referenceLinePath))
    refGDF = gpd.GeoDataFrame(geometry=refGDF['geometry'])
        
    # change CRS to desired projected EPSG
    # projection_epsg = settings['projection_epsg']
    refGDF = refGDF.to_crs(epsg=settings['output_epsg'])
    
    # Check line orientation to ensure sea is on the right (line should be in UTM of choice)
    refGDF = Toolbox.CheckRefOrientation(refGDF)
    
    # write shp file
    refGDF.to_file(ReprojShp)
        
    #Creates coast objects
    CellCoast = Coast(ReprojShp, MinLength=10)

    if not CellCoast.BuiltTransects:
            
        CellCoast.SmoothCoastLines(WindowSize=SmoothingWindowSize,NoSmooths=NoSmooths)
            
        CellCoast.WriteCoastShp(CoastSpec)
    
        CellCoast.GenerateTransects(TransectSpacing, DistanceInland, DistanceOffshore, CheckTopology=False)
        
        CellCoast.BuiltTransects = True
            
        CellCoast.WriteSimpleTransectsShp(TransectPath)
            
        with open(str(Filename2SaveCoast), 'wb') as PFile:
            pickle.dump(CellCoast, PFile)
            
    TransectGDF = gpd.read_file(TransectPath)
    
    # Add reference line intersect points to raw transect GDF
    TransectGDF.set_crs(epsg=settings['output_epsg'], inplace=True)
    
    # intersect each transect with original baseline to get ref line points
    columnsdata = []
    geoms = []
    for _,LineID,ID,TrGeom in TransectGDF.itertuples():
        for _,refGeom in refGDF.itertuples():
            intersect = TrGeom.intersection(refGeom)
            columnsdata.append((LineID, ID))
            geoms.append(intersect)
    allintersection = gpd.GeoDataFrame(columnsdata, geometry=geoms, columns=['LineID','TransectID'])
    
    # take only first point if any multipoint intersections
    for inter in range(len(allintersection)):
        if allintersection['geometry'][inter].geom_type == 'MultiPoint':
            allintersection['geometry'][inter] = list(allintersection['geometry'][inter].geoms)[0]
    
    TransectGDF['reflinepnt'] = allintersection['geometry']
    
    # Re-export transects to pkl to retain reflinepnt field
    with open(TransectPath[:-3]+'pkl', 'wb') as Tfile:
        pickle.dump(TransectGDF,Tfile)
    
    return TransectGDF
    
def GetIntersections(BasePath, TransectGDF, ShorelineGDF):
    """
    New intersection between transects and shorelines, based on geopandas GDFs/shapefiles 
    rather than shorelines represented as points.
    
    FM Sept 2022

    Parameters
    ----------
    BasePath : str
        Path to shapefiles of transects.
    TransectGDF : GeoDataFrame
        GDF of shore-normal transects created.
    ShorelineGDF : GeoDataFrame
        GDF of lines extracted from sat images.

    Returns
    -------
    TransectInterGDF : GeoDataFrame
        Transects with newly added intersection info.

    """
     
    print("performing intersections between transects...")
    
    # checking for mismatched coordinate systems
    if TransectGDF.crs != ShorelineGDF.crs:
        print("Your coordinate systems are mismatched; changing transect CRS to match shorelines CRS...")
        TransectGDF.to_crs(ShorelineGDF.crs, inplace=True)
    # initialise where each intersection between lines and transects will be saved
    ColumnData = []
    Geoms = []
    # for each row/feature in transect
    for _, _, ID, TrGeom, refpnt in TransectGDF.itertuples():
        # for each row/feature shoreline
        for _,dates,times,filename,cloud,ids,vthresh,wthresh,tideelev,satn,SGeom in ShorelineGDF.itertuples():
            # calculate intersections between each transect and shoreline
            Intersects = TrGeom.intersection(SGeom)
            ColumnData.append((ID,refpnt,dates,times,filename,cloud,ids,vthresh,wthresh,tideelev,satn))
            Geoms.append(Intersects)
            
    # create GDF from appended lists of intersections        
    AllIntersects = gpd.GeoDataFrame(ColumnData,geometry=Geoms,columns=['TransectID','reflinepnt','dates','times','filename','cloud_cove','idx','vthreshold','wthreshold','tideelev','satname'])
    # remove any rows with no intersections
    AllIntersects = AllIntersects[~AllIntersects.is_empty].reset_index().drop('index',axis=1)
    # duplicate geom column to save point intersections
    AllIntersects['interpnt'] = AllIntersects['geometry']
    # take only first point on any transects which intersected a single shoreline more than once
    for inter in range(len(AllIntersects)):
        if AllIntersects['interpnt'][inter].geom_type == 'MultiPoint':
            AllIntersects['interpnt'][inter] = list(AllIntersects['interpnt'][inter].geoms)[0] # list() accesses individual points in MultiPoint
    # AllIntersects = AllIntersects.drop('geometry',axis=1)
    AllIntersects = AllIntersects.rename_geometry('pntgeometry')

    # attribute join on transect ID to get transect geometry back
    AllIntersects = AllIntersects.merge(TransectGDF[['TransectID','geometry']], on='TransectID')
    AllIntersects = AllIntersects.drop('pntgeometry',axis=1)

    
    print("formatting into GeoDataFrame...")
    # initialise distances of intersections 
    distances = []
    # for each intersection
    for i in range(len(AllIntersects)):
        # calculate distance of intersection along transect
        distances.append(np.sqrt( 
            (AllIntersects['interpnt'][i].x - AllIntersects['geometry'][i].coords[0][0])**2 + 
            (AllIntersects['interpnt'][i].y - AllIntersects['geometry'][i].coords[0][1])**2 ))
    AllIntersects['distances'] = distances
    
    TransectDict = TransectGDF.to_dict('list')
    for Key in AllIntersects.drop(['TransectID','geometry'],axis=1).keys():
        TransectDict[Key] = {}
    TransectDict['interpnt'] = AllIntersects['interpnt'].copy()
    TransectDict['distances'] = AllIntersects['distances'].copy()
    

    #initialise lists used for storing each transect's intersection values
    reflinepnt, dates, times, filename, cloud_cove, idx, vthreshold, wthreshold, satname, tideelev, distances, interpnt = ([] for i in range(12)) # per-transect lists of values

    Key = [reflinepnt,dates,times,filename,cloud_cove,idx,vthreshold, wthreshold,satname,tideelev,  distances, interpnt]
    KeyName = ['reflinepnt','dates','times','filename','cloud_cove','idx','vthreshold', 'wthreshold','tideelev','satname', 'distances', 'interpnt']
    
    # for each column name
    for i in range(len(Key)):
        # for each transect
        for Tr in range(len(TransectGDF['TransectID'])):
            # refresh per-transect list
            TrKey = []
            # for each matching intersection on a single transect
            for j in range(len(AllIntersects.loc[AllIntersects['TransectID']==Tr])):
                # append each intersection value to a list for each transect
                # iloc used so index doesn't restart at 0 each loop
                TrKey.append(AllIntersects[KeyName[i]].loc[AllIntersects['TransectID']==Tr].iloc[j]) 
            Key[i].append(TrKey)
    
        TransectDict[KeyName[i]] = Key[i]
    
    print("TransectDict with intersections created.")
    
    TransectInterGDF = gpd.GeoDataFrame(TransectDict, crs=ShorelineGDF.crs)

    return TransectInterGDF
    
 

def GetBeachWidth(BasePath, TransectGDF, TransectInterGDF, WaterlineGDF, settings, output, AvBeachSlope=None):
    """
    Intersect waterlines with transects, based on geopandas GDFs/shapefiles.
    Waterlines are tidally corrected using either a DEM of slopes, CoastSat.slope,
    or a single slope value for all transects.
    
    FM Sept 2022
    Updated Oct 2024

    Parameters
    ----------
    BasePath : str
        Path to shapefiles of transects.
    TransectGDF : GeoDataFrame
        GDF of shore-normal transects created.
    TransectGDF : GeoDataFrame
        GDF of shore-normal transects created, with veg edge intersection data stored.
    WaterlineGDF : TYPE
        GeoDataFrame of waterlines extracted from satellite images.
    settings : dict
        Dictionary of user-defined settings used for the veg edge/waterline extraction.
    output : dict
        Dictionary of extracted veg edges (and waterlines) and associated info with each edge.
    AvBeachSlope : float, optional
        Average tan(Beta) value across the intertidal zone. The default is None.

    Returns
    -------
    TransectInterGDF : GeoDataFrame
        GeoDataFrame of cross-shore transects with width between 

    """
     
    print("performing intersections between transects and waterlines...")
    
    TransectInterGDFWater = TransectInterGDF.copy()
    # checking for mismatched coordinate systems
    if TransectGDF.crs != WaterlineGDF.crs:
        print("Your coordinate systems are mismatched; changing transect CRS to match shorelines CRS...")
        TransectGDF.to_crs(WaterlineGDF.crs, inplace=True)
    # initialise where each intersection between lines and transects will be saved
    ColumnData = []
    Geoms = []
    # for each row/feature in transect
    for _, _, ID, TrGeom, refpnt in TransectGDF.itertuples():
        # Extend transect line out to sea and inland
        TrGeom = Toolbox.ExtendLine(TrGeom, 300)
        # for each row/feature shoreline
        for _,dates,_,_,_,_,_,_,_,_,SGeom in WaterlineGDF.itertuples():
            # calculate intersections between each transect and shoreline
            Intersects = TrGeom.intersection(SGeom)
            ColumnData.append((ID,dates))
            Geoms.append(Intersects)
            
    # create GDF from appended lists of intersections        
    AllIntersects = gpd.GeoDataFrame(ColumnData,geometry=Geoms,columns=['TransectID', 'wldates'])
    # remove any rows with no intersections
    AllIntersects = AllIntersects[~AllIntersects.is_empty].reset_index().drop('index',axis=1)
    # duplicate geom column to save point intersections
    AllIntersects['wlinterpnt'] = AllIntersects['geometry']
    # take only first point on any transects which intersected a single shoreline more than once
    for inter in range(len(AllIntersects)):
        if AllIntersects['wlinterpnt'][inter].geom_type == 'MultiPoint':
            AllIntersects['wlinterpnt'][inter] = list(AllIntersects['wlinterpnt'][inter].geoms)[0] # list() accesses individual points in MultiPoint
    # AllIntersects = AllIntersects.drop('geometry',axis=1)
    AllIntersects = AllIntersects.rename_geometry('pntgeometry')

    # attribute join on transect ID to get transect geometry back
    AllIntersects = AllIntersects.merge(TransectGDF[['TransectID','geometry']], on='TransectID')
    AllIntersects = AllIntersects.drop('pntgeometry',axis=1)


    print("formatting into GeoDataFrame...")
    # initialise distances of intersections 
    distances = []
    # for each intersection
    for i in range(len(AllIntersects)):
        # calculate distance of intersection along transect
        distances.append(np.sqrt( 
            (AllIntersects['wlinterpnt'][i].x - AllIntersects['geometry'][i].coords[0][0])**2 + 
            (AllIntersects['wlinterpnt'][i].y - AllIntersects['geometry'][i].coords[0][1])**2 ))
    AllIntersects['wldists'] = distances

    #initialise lists used for storing each transect's intersection values
    dates, distances, interpnt = ([] for i in range(3)) # per-transect lists of values

    Key = [dates, distances, interpnt]
    KeyName = ['wldates','wldists','wlinterpnt']
       
    # for each column name
    for i in range(len(Key)):
        # for each transect
        for Tr in range(len(TransectGDF['TransectID'])):
            # refresh per-transect list
            TrKey = []
            # for each matching intersection on a single transect
            for j in range(len(AllIntersects.loc[AllIntersects['TransectID']==Tr])):
                # append each intersection value to a list for each transect
                # iloc used so index doesn't restart at 0 each loop
                TrKey.append(AllIntersects[KeyName[i]].loc[AllIntersects['TransectID']==Tr].iloc[j]) 
            Key[i].append(TrKey)
    
        TransectInterGDFWater[KeyName[i]] = Key[i]
        
    
    # Create beach width attribute
    # must initialise with list of same length as waterline dates
    TransectInterGDFWater['beachwidth'] = TransectInterGDFWater['wldates'].copy()
    print('calculating tidally corrected cross-shore distances...')
    # Tidal correction to get corrected distances along transects
    TransectInterGDFWater = TidalCorrection(settings, output, TransectInterGDFWater, AvBeachSlope)
    # Field representing beach zone dependent on tidal height range split into 3 (upper, middle or lower)
    TideSteps = Toolbox.BeachTideLoc(settings)
    
    # for each transect    
    for Tr in range(len(TransectGDF['TransectID'])):
        
        ShoreLevels = []
        # for each water elevation obs in each transect
        for welev in TransectInterGDFWater['tideelev'].iloc[Tr]:
            if welev >= TideSteps[0] and welev <= TideSteps[1]:
                ShoreLevels.append('lower')
            elif welev >= TideSteps[1] and welev <= TideSteps[2]:
                ShoreLevels.append('middle')
            elif welev >= TideSteps[2] and welev <= TideSteps[3]:
                ShoreLevels.append('upper')

        TransectInterGDFWater['tidezone'].iloc[Tr] = ShoreLevels
        
        
        print('calculating distances between veg and water lines...')
        # dates into transect-specific list
        WLDateList = [datetime.strptime(date, '%Y-%m-%d') for date in TransectInterGDFWater['wldates'].iloc[Tr]]
        VLDateList = [datetime.strptime(date, '%Y-%m-%d') for date in TransectInterGDFWater['dates'].iloc[Tr]]
        # find index of closest waterline date to each vegline date
        VLSLDists = []
        for D, WLDate in enumerate(WLDateList):
            # index of matching nearest date
            if VLDateList != []:
                DateLoc = Toolbox.NearDate(WLDate,VLDateList)
                if DateLoc == False:
                    VLSLDists.append(np.nan)
                    continue
                else:
                    DateIndex = VLDateList.index(DateLoc)
            else:
                continue
            # use date index to identify matching distance along transect
            # and calculate distance between two intersections (veg - water means +ve is veg measured seaward towards water)
            VLSLDists.append(TransectInterGDFWater['wlcorrdist'].iloc[Tr][D] - TransectInterGDFWater['distances'].iloc[Tr][DateIndex])
        TransectInterGDFWater['beachwidth'].iloc[Tr] = VLSLDists
        
        
    print("TransectDict with beach width and waterline intersections created.")
        
    return TransectInterGDFWater
    

def GetWaterIntersections(BasePath, TransectGDF, TransectInterGDF, WaterlineGDF, settings, output):
    """
    IN DEVELOPMENT: This is an attempt to make GetBeachWidth() more efficient.
    
    Intersect waterlines with transects, based on geopandas GDFs/shapefiles.
    Waterlines are tidally corrected using either a DEM of slopes, CoastSat.slope,
    or a single slope value for all transects.
    
    
    FM Sept 2022
    Updated Oct 2024

    Parameters
    ----------
    BasePath : str
        Path to shapefiles of transects.
    TransectGDF : GeoDataFrame
        GDF of shore-normal transects created.
    TransectInterGDF : GeoDataFrame
        GDF of shore-normal transects, with veg edge intersection data stored.
    
    WaterlineGDF : TYPE
        GeoDataFrame of waterlines extracted from satellite images.
    settings : dict
        Dictionary of user-defined settings used for the veg edge/waterline extraction.
    output : dict
        Dictionary of extracted veg edges (and waterlines) and associated info with each edge.


    Returns
    -------
    TransectInterGDFWater : GeoDataFrame
        GeoDataFrame of cross-shore transects with width between 

    """
     
    print("performing intersections between transects and waterlines...")
    
    # checking for mismatched coordinate systems
    if TransectGDF.crs != WaterlineGDF.crs:
        print("Your coordinate systems are mismatched; changing transect CRS to match shorelines CRS...")
        TransectGDF.to_crs(WaterlineGDF.crs, inplace=True)
    # initialise where each intersection between lines and transects will be saved
    ColumnData = []
    Geoms = []
    
    # START of faster code
    # Spatial indexing with `sindex`
    WaterlineGDF_sindex = WaterlineGDF.sindex
    # Perform intersection operations
    for row in TransectGDF.itertuples():
        _, _, ID, TrGeom, refpnt = row
        # Extend transect line out to sea and inland
        TrGeom = Toolbox.ExtendLine(TrGeom, 300)
        # Use spatial indexing to find possible intersecting shorelines
        MatchesID = list(WaterlineGDF_sindex.intersection(TrGeom.bounds))
        Matches = WaterlineGDF.iloc[MatchesID]
        # for each row/feature shoreline
        for WL_row in Matches.itertuples():
            _, dates, times, _, _, _, _, _, _, _, SGeom = WL_row
            Intersects = TrGeom.intersection(SGeom)
            # ignore any rows with no intersections
            if not Intersects.is_empty:
                ColumnData.append((ID, dates, times))
                Geoms.append(Intersects)
    
    # Create GeoDataFrame of intersections and filter empty intersections
    AllIntersects = gpd.GeoDataFrame(ColumnData, geometry=Geoms, columns=['TransectID', 'wldates', 'wltimes'])
    
    # take only first point on any transects which intersected a single shoreline more than once
    AllIntersects['wlinterpnt'] = AllIntersects['geometry'].apply(
        lambda geom: geom.geoms[0] if geom.geom_type == 'MultiPoint' else geom)
    # END of faster code
    
    # AllIntersects = AllIntersects.drop('geometry',axis=1)
    AllIntersects = AllIntersects.rename_geometry('pntgeometry')

    # attribute join on transect ID to get transect geometry back
    AllIntersects = AllIntersects.merge(TransectGDF[['TransectID','geometry']], on='TransectID')
    AllIntersects = AllIntersects.drop('pntgeometry',axis=1)
    
    # START of faster code
    # calculate distances of intersections 
    AllIntersects['wldists'] = AllIntersects.apply(
        lambda row: row['wlinterpnt'].distance(Point(row['geometry'].coords[0])) 
        if row['wlinterpnt'] 
        else np.nan, axis=1)

    print("formatting into GeoDataFrame...")
    TransectInterGDFWater = TransectInterGDF.copy()
    # Initialise dictionary with empty lists for each TransectID in TransectInterGDFWater
    WLData = {name: [[] for _ in range(len(TransectInterGDFWater))] for name in ['wldates', 'wltimes', 'wldists', 'wlinterpnt']}

    # Create a mapping of TransectID to index for quick assignment
    Tr_Ind = {Tr: idx for idx, Tr in enumerate(TransectInterGDFWater['TransectID'])}

    # Populate WLData only for TransectIDs with intersections
    for TrID, TrGroup in AllIntersects.groupby('TransectID'):
        # Sort the wldists lists to maintain temporal order
        TrGroup_sort = TrGroup.sort_values(by='wldates')
        idx = Tr_Ind[TrID]  # Get the index in TransectInterGDF for this TransectID
        for datakey in ['wldates','wltimes','wldists','wlinterpnt']:
            WLData[datakey][idx] = TrGroup_sort[datakey].tolist()

    # Assign now-filled lists to the corresponding TransectInterGDFWater columns
    for key, data in WLData.items():
        TransectInterGDFWater[key] = data
    # END of faster code

    return TransectInterGDFWater


def CalcBeachWidth(settings, TransectGDF, TransectInterGDFWater):
    # Create beach width and attributes
    # must initialise with list of same length as waterline dates
    TransectInterGDFWater['beachwidth'] = TransectInterGDFWater['wldates'].copy()
    TideSteps = Toolbox.BeachTideLoc(settings)
    TransectInterGDFWater['tidezone'] = TransectInterGDFWater['tideelev'].copy()
    # for each transect    
    for Tr in range(len(TransectGDF['TransectID'])):
        
        # Field representing beach zone dependent on tidal height range split into 3 (upper, middle or lower)
        ShoreLevels = []
        # for each water elevation obs in each transect
        for welev in TransectInterGDFWater['tideelev'].iloc[Tr]:
            if welev >= TideSteps[0] and welev <= TideSteps[1]:
                ShoreLevels.append('lower')
            elif welev >= TideSteps[1] and welev <= TideSteps[2]:
                ShoreLevels.append('middle')
            elif welev >= TideSteps[2] and welev <= TideSteps[3]:
                ShoreLevels.append('upper')

        TransectInterGDFWater['tidezone'].iloc[Tr] = ShoreLevels
        
        # dates into transect-specific list
        WLDateList = [datetime.strptime(date, '%Y-%m-%d') for date in TransectInterGDFWater['wldates'].iloc[Tr]]
        VLDateList = [datetime.strptime(date, '%Y-%m-%d') for date in TransectInterGDFWater['dates'].iloc[Tr]]
        # find index of closest waterline date to each vegline date
        VLSLDists = []
        for D, WLDate in enumerate(WLDateList):
            # index of matching nearest date
            if VLDateList != []:
                DateLoc = Toolbox.NearDate(WLDate,VLDateList)
                if DateLoc == False:
                    VLSLDists.append(np.nan)
                    continue
                else:
                    DateIndex = VLDateList.index(DateLoc)
            else:
                continue
            # use date index to identify matching distance along transect
            # and calculate distance between two intersections (veg - water means +ve is veg measured seaward towards water)
            VLSLDists.append(TransectInterGDFWater['wlcorrdist'].iloc[Tr][D] - TransectInterGDFWater['distances'].iloc[Tr][DateIndex])


        TransectInterGDFWater['beachwidth'].iloc[Tr] = VLSLDists
        
        
    print("\nTransectDict with beach width and waterline intersections created.")
        
    return TransectInterGDFWater


def TidalCorrection(settings, output, TransectInterGDFWater, AvBeachSlope=None):
    """
    Correct cross-shore waterline distances to remove the effects of tides. Uses
    the equation "x_tide = x + ( z_tide / tan(Beta) )", where x is cross-shore
    distance along transect of waterline intersection, z_tide is the tidal stage
    at a chosen elevation above sea level, and Beta is the rise/run of the beach
    between mean sea level and mean high water spring. 
    
    FM Nov 2022
    Updated Oct 2024

    Parameters
    ----------
    settings : dict
        Dictionary of user-defined settings used for the veg edge/waterline extraction.
    output : dict
        Dictionary of extracted veg edges (and waterlines) and associated info with each edge.
    IntersectDF : GeoDataFrame
        AllIntersects GeoDataFrame with information from transect-veg edge intersections extracted.
    AvBeachSlope : float, optional
        Average tan(Beta) value across the intertidal zone. The default value is None.

    Returns
    -------
    CorrIntDistances : list
        Corrected cross-shore waterline distances per transect.
    TidalStages : list
        Tidal elevations per transect.

    """
    # get the tide level corresponding to the time of sat image acquisition
    dates_sat = []
    for i in range(len(output['dates'])):
        dates_sat_str = output['dates'][i] +' '+output['times'][i]
        dates_sat.append(datetime.strptime(dates_sat_str, '%Y-%m-%d %H:%M:%S.%f'))
    
    # more efficient to get all possible dates and matching tides, then use as
    # lookup for each transect (rather than repeating GetWaterElevs() per-transect needlessly)
    tide_sat = Toolbox.GetWaterElevs(settings,dates_sat)
    tides_sat = np.array(tide_sat)
    tide_dict = dict(zip(dates_sat, tides_sat))
    
    # tidal correction along each transect
    # elevation at which you would like the shoreline time-series to be
    RefElev = 0
    
    BeachSlopes = [] # single value per transect
    TidalStages = [] # timeseries value per transect
    CorrectedDists = [] # timeseries value per transect
    
    for Tr in range(len(TransectInterGDFWater)):
        print(f"\r{Tr} / {len(TransectInterGDFWater)}", end='\r')        
        
        dates_dt_tr = [datetime.strptime(date_str, '%Y-%m-%d').date() for date_str in TransectInterGDFWater['wldates'].iloc[Tr]]
        dates_sat_tr = [] # attach times to per-transect dates
        for date in dates_dt_tr:
            for dt in dates_sat:
                if dt.date() == date:
                    dates_sat_tr.append(datetime.combine(date, dt.time()))
        tides_sat_tr = [tide_dict[date] for date in dates_sat_tr]
        
        cross_distances = TransectInterGDFWater['wldists'].iloc[Tr]
        
        # TO DO: figure out way of running this per transect
        DEMpath = os.path.join(settings['inputs']['filepath'],'tides',settings['inputs']['sitename']+'_DEM.tif')
        if os.path.exists(DEMpath):
            MSL = 1.0
            MHWS = 0.1
            BeachSlope = GetBeachSlopesDEM(MSL, MHWS, DEMpath)
        
        elif AvBeachSlope is None: # no average slope provided, calculate slope
            # if only a few observations exist, just use global-constant beach slope of tan(Beta) = 0.1
            if len(dates_sat_tr) < 10:
                BeachSlope = 0.1
            else:
                BeachSlope = Slope.CoastSatSlope(dates_sat_tr, tides_sat_tr, cross_distances)
        
        else: # just use user-provided beach-average slope
            BeachSlope = AvBeachSlope
        
        # After calculating tidal stages (and beach slopes if needed), perform 
        # tidal correction on each waterline position in each transect
        CorrectedDistsTr = [] # per timestep
        for ts, cross_distance in enumerate(cross_distances):
            TidalElev = tides_sat_tr[ts] - RefElev
            Correction = TidalElev / BeachSlope
            CorrectedDistsTr.append(cross_distance + Correction)
        # append list of single corrections per-transect back onto larger list for GDF
        CorrectedDists.append(CorrectedDistsTr)    
        
        # generate per-transect tidal elevation and slope list
        BeachSlopes.append(BeachSlope)
        TidalStages.append(tides_sat_tr)
        
    
    # Once each transect has been corrected, add finished lists to geodataframe
    TransectInterGDFWater['wlcorrdist'] = CorrectedDists
    TransectInterGDFWater['tideelev'] = TidalStages
    TransectInterGDFWater['beachslope'] = BeachSlopes
    
    return TransectInterGDFWater


def WLCorrections(settings, output, TransectInterGDFWater, TransectInterGDFWave=None, AvBeachSlope=None):
    """
    IN DEVELOPMENT: This is an attempt to make TidalCorrection() more efficient.
    
    Correct cross-shore waterline distances to remove the effects of tides. Uses
    the equation "x_corr = x + ( z / tan(beta) )", where x is cross-shore
    distance along transect of waterline intersection, z is the tidal stage
    at a chosen elevation above sea level (+ runup if available), and beta is 
    the rise/run of the beach between mean sea level and mean high water spring. 
    
    FM Nov 2022
    Updated Oct 2024

    Parameters
    ----------
    settings : dict
        Dictionary of user-defined settings used for the veg edge/waterline extraction.
    output : dict
        Dictionary of extracted veg edges (and waterlines) and associated info with each edge.
    TransectInterGDFWater : GeoDataFrame
        GeoDataFrame with information from transect-veg edge intersections extracted.
    AvBeachSlope : float, optional
        Average tan(Beta) value across the intertidal zone. The default value is None.

    Returns
    -------
    TransectInterGDFWater : GeoDataFrame
        GeoDataFrame with information from transect-veg edge intersections extracted, updated with
        corrected WL distances, tidal elevation and beach slopes.

    """
    
    print('Correcting waterline positions...')
    # Parse satellite acquisition datetime and create lookup dictionary
    dates_sat = [
        datetime.strptime(f"{date} {time}", '%Y-%m-%d %H:%M:%S.%f') 
        for date, time in zip(output['dates'], output['times'])
    ]
    tide_dict = dict(zip(dates_sat, Toolbox.GetWaterElevs(settings, dates_sat)))
    
    # If wave data is provided, add runups to tidal correction
    if TransectInterGDFWave is not None:
        print('using runup as well as tides...')
        # need to set any nans in runup to 0 to avoid setting all operations to nan
        runup_dict = dict(zip(TransectInterGDFWave['WaveDates'].iloc[0], [0 if np.isnan(x) else x for x in TransectInterGDFWave['Runups'].iloc[0]])) # first entry should have all runups
        TWL_dict = dict(zip(dates_sat, [tide_dict[date] + runup_dict[date] for date in dates_sat]))
    else: # if no runups available, just use tides
        TWL_dict = tide_dict.copy()
    
    # Reference elevation
    RefElev = 0
    
    # Initialize lists for each transect's final data
    BeachSlopes, TidalStages, CorrectedDists = [], [], []
    
    # Path to DEM file for slope calculation (if available)
    DEMpath = os.path.join(settings['inputs']['filepath'], 'tides', f"{settings['inputs']['sitename']}_DEM.tif")
    
    # Precomputed beach slope if available
    if os.path.exists(DEMpath):
        print('beach slope being derived from provided DEM...')
        MSL, MHWS = 1.0, 0.1
        BeachSlopeDEM = GetBeachSlopesDEM(MSL, MHWS, DEMpath)
    else:
        BeachSlopeDEM = None  # No DEM slope
    
    # Process each transect
    for Tr, transect in TransectInterGDFWater.iterrows():
        print(f"\r corrected {Tr} / {len(TransectInterGDFWater)} transects", end='')       
        # Gather and match dates for each transect's observations
        dates_dt_tr = [datetime.strptime(date_str, '%Y-%m-%d').date() for date_str in transect['wldates']]
        dates_sat_tr = [datetime.combine(date, next(dt.time() for dt in dates_sat if dt.date() == date)) for date in dates_dt_tr]
        tides_sat_tr = [tide_dict[date] for date in dates_sat_tr]
        TWL_tr = [TWL_dict[date] for date in dates_sat_tr]
        
        # Retrieve transect cross distances
        cross_distances = transect['wldists']
    
        # Determine beach slope for this transect
        if BeachSlopeDEM is not None:
            BeachSlope = BeachSlopeDEM
        elif AvBeachSlope is not None:
            BeachSlope = AvBeachSlope
        else:
            # Calculate beach slope dynamically if needed
            if len(dates_sat_tr) < 10:
                BeachSlope = 0.1
            else:
                # tides needs to be used bc if any nan, then fine_tide_peak fails
                BeachSlope = Slope.CoastSatSlope(dates_sat_tr, tides_sat_tr, cross_distances)
    
        # Correct each cross-shore distance for tidal elevation
        CorrectedDistsTr = [
            cross_distance + ((z - RefElev) / BeachSlope)
            for cross_distance, z in zip(cross_distances, TWL_tr)
        ]
           
        # Append results for this transect
        CorrectedDists.append(CorrectedDistsTr)
        BeachSlopes.append(BeachSlope)
        TidalStages.append(tides_sat_tr)
        
    # Add results back to TransectInterGDFWater
    TransectInterGDFWater['wlcorrdist'] = CorrectedDists
    TransectInterGDFWater['tideelev'] = TidalStages
    TransectInterGDFWater['beachslope'] = BeachSlopes

    return TransectInterGDFWater


def CalcIribarrens(TransectInterGDFWave, TransectInterGDFWater):
    
    Iribarrens = []
    for Tr in range(len(TransectInterGDFWave)):
        WaveHs = TransectInterGDFWave['WaveHs'].iloc[Tr]
        WaveTp = TransectInterGDFWave['WaveTp'].iloc[Tr]
        beta = TransectInterGDFWater['beachslope'].iloc[Tr]
        IribarrenTr = []
        for i in range(len(WaveHs)):
            # Deepwater wave length
            L0 = (9.81 * WaveTp[i]**2) / (2 * np.pi)
            # Iribarren number (dynamic beach steepness)
            zeta = beta / (WaveHs[i] * L0) 
            
            # Append per-wave value to list
            IribarrenTr.append(zeta)
            
        Iribarrens.append(IribarrenTr)
            
    # Add per-transect Iribarrens lists to full list
    TransectInterGDFWave['Iribarren'] = Iribarrens
    
    return TransectInterGDFWave


def GetBeachSlopesDEM(MSL, MHWS, DEMpath):
    """
    IN DEVELOPMENT
    Extract a list of cross-shore slopes from a DEM using provided water levels.
    FM Nov 2022
    
    Parameters
    ----------
    MSL : float
        Elevation at which mean sea level sits.
    MHWS : float
        Elevation at which mean high water spring sits.
    DEMpath : str
        Filepath to digital terrain model.

    Returns
    -------
    None.

    """
    
    

def SaveIntersections(TransectInterGDF, LinesGDF, BasePath, sitename):
    
    """
    Calculate rates of change of veg edges along transects. 
    Save transects with intersection info (and cross-shore change rates) as shapefile. 
    FM Sept 2022

    Parameters
    ----------
    TransectInterGDF : GeoDataFrame
        Cross-shore transects with newly added veg edge intersection info.
    LinesGDF : GeoDataFrame
        Satellite-derived veg edge lines GeoDataFrame.
    BasePath : str
        Path to shapefiles of transects.
    sitename : str
        Name of site.

    Returns
    -------
    TransectInterGDF : GeoDataFrame
        GeoDataFrame of transects with intersection (and rate of change info).
    """
    
    print('saving new transect shapefile ...')
         

    olddate, youngdate, oldyoungT, oldyoungRt, recentT, recentRt = ([] for i in range(6))
    for Tr in range(len(TransectInterGDF)):
        FullDateTime = []
        RecentDateTime = []
        DateRange = []
        Slopes = []
        if len(TransectInterGDF['dates'].iloc[Tr]) > 0: #
            DateRange.append(TransectInterGDF['dates'].iloc[Tr][0]) # oldest date
            if len(TransectInterGDF['dates'].iloc[Tr]) > 1:
                DateRange.append(TransectInterGDF['dates'].iloc[Tr][-2]) # second youngest date
                DateRange.append(TransectInterGDF['dates'].iloc[Tr][-1]) # youngest date
            else: # for transects with only two dates, take first and last for both 'full' and 'recent' rates
                DateRange.append(TransectInterGDF['dates'].iloc[Tr][-1])
                DateRange.append(TransectInterGDF['dates'].iloc[Tr][-1])
            
            # for each Tr, find difference between converted oldest and youngest dates and transform to decimal years
            FullDateTime = round((float((datetime.strptime(DateRange[2],'%Y-%m-%d')-datetime.strptime(DateRange[0],'%Y-%m-%d')).days)/365.2425),4)
            RecentDateTime = round((float((datetime.strptime(DateRange[2],'%Y-%m-%d')-datetime.strptime(DateRange[1],'%Y-%m-%d')).days)/365.2425),4)
            # convert dates to ordinals for linreg
            OrdDates = [datetime.strptime(i,'%Y-%m-%d').toordinal() for i in TransectInterGDF['dates'].iloc[Tr]]
            
            for idate in [0,-2]:
                X = np.array(OrdDates[idate:]).reshape((-1,1))
                y = np.array(TransectInterGDF['distances'][Tr][idate:])
                model = LinearRegression(fit_intercept=True).fit(X,y)
                
                
                Slope = round(model.coef_[0]*365.2425, 2) # ordinal dates means slope is in m/day, converts to m/yr
                Slopes.append(Slope)

            olddate.append(DateRange[0]) # oldest date in timeseries
            youngdate.append(DateRange[-1]) # youngest date in timeseries
            oldyoungT.append(FullDateTime) # time difference in years between oldest and youngest date
            oldyoungRt.append(Slopes[0]) # rate of change from oldest to youngest veg edge in m/yr
            recentT.append(RecentDateTime) # time difference in years between second youngest and youngest date
            recentRt.append(Slopes[1]) # rate of change from second youngest to youngest veg edge in m/yr

        else: # if empty (< 2 intersections), just write empty values to Tr (to keep same no. of entries vs no. of Tr)
            olddate.append(np.nan) # oldest date in timeseries
            youngdate.append(np.nan) # youngest date in timeseries
            oldyoungT.append(np.nan) # time difference in years between oldest and youngest date
            oldyoungRt.append(np.nan) # rate of change from oldest to youngest veg edge in m/yr
            recentT.append(np.nan) # time difference in years between second youngest and youngest date
            recentRt.append(np.nan) # rate of change from second youngest to youngest veg edge in m/yr
    
    TransectInterGDF['olddate'] = olddate # oldest date in timeseries
    TransectInterGDF['youngdate'] = youngdate # youngest date in timeseries
    TransectInterGDF['oldyoungT'] = oldyoungT # time difference in years between oldest and youngest date
    TransectInterGDF['oldyoungRt'] = oldyoungRt # rate of change from oldest to youngest veg edge in m/yr
    TransectInterGDF['recentT'] = recentT # time difference in years between second youngest and youngest date
    TransectInterGDF['recentRt'] = recentRt # rate of change from second youngest to youngest veg edge in m/yr
    
    TransectInterShp = TransectInterGDF.copy()

    # reformat fields with lists to strings
    KeyName = list(TransectInterShp.select_dtypes(include='object').columns)
    for Key in KeyName:
        # round any floating points numbers before export
        realInd = next(i for i, j in enumerate(TransectInterShp[Key]) if j)
            
        if type(TransectInterShp[Key][realInd]) == list: # for lists of intersected values per transect
            if type(TransectInterShp[Key][realInd][0]) == np.float64:  
                for Tr in range(len(TransectInterShp[Key])):
                    TransectInterShp[Key][Tr] = [round(i,2) for i in TransectInterShp[Key][Tr]]
        else: # for singular values per transect
            if type(TransectInterShp[Key][realInd]) == np.float64: 
                for Tr in range(len(TransectInterShp[Key])):
                    TransectInterShp[Key][Tr] = [round(i,2) for i in TransectInterShp[Key][Tr]]
                    
        TransectInterShp[Key] = TransectInterShp[Key].astype(str)
    
    TransectInterShp.to_file(os.path.join(BasePath,sitename+'_Transects_Intersected.shp'))
    
    print("Shapefile with sat intersections saved.")
    
    return TransectInterGDF

    
def SaveWaterIntersections(TransectInterGDFWater, LinesGDF, BasePath, sitename):
    """
    Save transects with waterline and beach width intersection info as shapefile.
    FM Sept 2022

    Parameters
    ----------
    TransectInterGDF : GeoDataFrame
        Cross-shore transects with newly added waterline intersection info.
    LinesGDF : GeoDataFrame
        Satellite-derived waterline lines GeoDataFrame.
    BasePath : str
        Path to shapefiles of transects.
    sitename : str
        Name of site.

    Returns
    -------
    TransectInterGDFWater : GeoDataFrame
        GDF of transects with waterline intersection and rates of change info.
    """
    
    
    print('saving new transect shapefile ...')
    
    olddate, youngdate, oldyoungT, oldyoungRt, oldyoungME, recentT, recentRt, recentME = ([] for i in range(8))
    for Tr in range(len(TransectInterGDFWater)):
        FullDateTime = []
        RecentDateTime = []
        DateRange = []
        Slopes = []
        MoEs = []
        if len(TransectInterGDFWater['wldates'].iloc[Tr]) > 0:
            DateRange.append(TransectInterGDFWater['wldates'].iloc[Tr][0]) # oldest date
            if len(TransectInterGDFWater['wldates'].iloc[Tr]) > 1:
                DateRange.append(TransectInterGDFWater['wldates'].iloc[Tr][-2]) # second youngest date
                DateRange.append(TransectInterGDFWater['wldates'].iloc[Tr][-1]) # youngest date
            else: # for transects with only two dates, take first and last for both 'full' and 'recent' rates
                DateRange.append(TransectInterGDFWater['wldates'].iloc[Tr][-1])
                DateRange.append(TransectInterGDFWater['wldates'].iloc[Tr][-1])
            
            # for each Tr, find difference between converted oldest and youngest dates and transform to decimal years
            FullDateTime = round((float((datetime.strptime(DateRange[2],'%Y-%m-%d')-datetime.strptime(DateRange[0],'%Y-%m-%d')).days)/365.2425),4)
            RecentDateTime = round((float((datetime.strptime(DateRange[2],'%Y-%m-%d')-datetime.strptime(DateRange[1],'%Y-%m-%d')).days)/365.2425),4)
            # convert dates to ordinals for linreg
            OrdDates = [datetime.strptime(i,'%Y-%m-%d').toordinal() for i in TransectInterGDFWater['wldates'].iloc[Tr]]
            
            for idate in [0,-2]:
                X = np.array(OrdDates[idate:]).reshape((-1,1))
                y = np.array(TransectInterGDFWater['wlcorrdist'][Tr][idate:])
                model = LinearRegression(fit_intercept=True).fit(X,y)
                Slope_mday = model.coef_[0]
                Intercept_mday = model.intercept_
                
                n = len(X)
                meanX = np.mean(X)
                sumsqX = np.sum((X - meanX) ** 2)
                resids = y - (Slope_mday * X + Intercept_mday)
                s = np.sqrt(np.sum(resids ** 2) / (n - 2))
                SE_slope = s / np.sqrt(sumsqX)
                
                CL = 0.95
                t_value = stats.t.ppf((1 + CL) / 2, df=(n-2))
                MoE_mday = t_value * SE_slope
                
                # Convert to metres per year
                MoE = round(MoE_mday*365.2425, 2)
                Slope = round(Slope_mday*365.2425, 2)
                Slopes.append(Slope)
                MoEs.append(MoE)
        
            olddate.append(DateRange[0]) # oldest date in timeseries
            youngdate.append(DateRange[-1]) # youngest date in timeseries
            oldyoungT.append(FullDateTime) # time difference in years between oldest and youngest date
            oldyoungRt.append(Slopes[0]) # rate of change from oldest to youngest veg edge in m/yr
            oldyoungME.append(MoEs[0]) # margin of error (plus or minus) on old to young rate in m/yr
            recentT.append(RecentDateTime) # time difference in years between second youngest and youngest date
            recentRt.append(Slopes[1]) # rate of change from second youngest to youngest veg edge in m/yr
            recentME.append(MoEs[1]) # margin or error (plus or minus) on second youngest to youngest rate in m/yr

        else: # if empty (< 2 intersections), just write empty values to Tr (to keep same no. of entries vs no. of Tr)
            olddate.append(np.nan) # oldest date in timeseries
            youngdate.append(np.nan) # youngest date in timeseries
            oldyoungT.append(np.nan) # time difference in years between oldest and youngest date
            oldyoungRt.append(np.nan) # rate of change from oldest to youngest veg edge in m/yr
            oldyoungME.append(np.nan) # margin of error (plus or minus) on old to young rate in m/yr
            recentT.append(np.nan) # time difference in years between second youngest and youngest date
            recentRt.append(np.nan) # rate of change from second youngest to youngest veg edge in m/yr
            recentME.append(np.nan) # margin or error (plus or minus) on second youngest to youngest rate in m/yr
    
    TransectInterGDFWater['olddateW'] = olddate # oldest date in timeseries
    TransectInterGDFWater['youngdateW'] = youngdate # youngest date in timeseries
    TransectInterGDFWater['oldyoungTW'] = oldyoungT # time difference in years between oldest and youngest date
    TransectInterGDFWater['oldyungRtW'] = oldyoungRt # rate of change from oldest to youngest veg edge in m/yr
    TransectInterGDFWater['oldyungMEW'] = oldyoungME # margin of error (plus or minus) on old to young rate in m/yr
    TransectInterGDFWater['recentTW'] = recentT # time difference in years between second youngest and youngest date
    TransectInterGDFWater['recentRtW'] = recentRt # rate of change from second youngest to youngest veg edge in m/yr
    TransectInterGDFWater['recentMEW'] = recentME # margin or error (plus or minus) on second youngest to youngest rate in m/yr
    
    TransectInterShp = TransectInterGDFWater.copy()

    # reformat fields with lists to strings
    # get only the columns that are made of lists
    KeyName = list(TransectInterShp.select_dtypes(include='object').columns)
    for Key in KeyName:
        # round any floating points numbers before export
        realInd = next(i for i, j in enumerate(TransectInterShp[Key]) if j)
            
        if type(TransectInterShp[Key][realInd]) == list: # for lists of intersected values
            if type(TransectInterShp[Key][realInd][0]) == np.float64:  
                for Tr in range(len(TransectInterShp[Key])):
                    TransectInterShp[Key][Tr] = [round(i,2) for i in TransectInterShp[Key][Tr]]
        else: # for singular values
            if type(TransectInterShp[Key][realInd]) == np.float64: 
                for Tr in range(len(TransectInterShp[Key])):
                    TransectInterShp[Key][Tr] = [round(i,2) for i in TransectInterShp[Key][Tr]]
    
        TransectInterShp[Key] = TransectInterShp[Key].astype(str)
        
    TransectInterShp.to_file(os.path.join(BasePath,sitename+'_Transects_Intersected_Water.shp'))

    
    print("Shapefile with sat intersections saved.")
    
    return TransectInterGDFWater


def CalculateChanges(TransectInterGDF):
    """
    Calculate distances of each veg edge intersect along transect, normalised to transect midpoint.
    FM Sept 2022

    Parameters
    ----------
    TransectInterGDF : GeoDataFrame
        GDF of transects with veg edge intersection info.

    Returns
    -------
    TransectInterGDF : GeoDataFrame
        GDF of transects with veg edge intersection info (plus new normalised dists).

    """
    # must initialise with list of same length as veg dists
    TransectInterGDF['normdists'] = TransectInterGDF['distances'].copy()
    # for each transect
    for Tr in range(len(TransectInterGDF['TransectID'])):
        Dists = []
        # for each intersection on each transect
        for i, Dist in enumerate(TransectInterGDF['distances'][Tr]):
            # intersection distance along transect minus midpoint distance gives +ve for seaward and -ve for landward
            Dists.append(Dist - TransectInterGDF.geometry[Tr].length/2)
        TransectInterGDF['normdists'][Tr] = Dists
    
    print("TransectDict updated with distances between sat lines.")
            
    return TransectInterGDF


def TZIntersect(settings, TransectInterGDF, VeglinesGDF, BasePath):
    """
    Intersections between coastal indicator lines and veg Transition Zone rasters.
    FM June 2023

    Parameters
    ----------
    settings : dict
        Dictionary of user-defined settings used for the veg edge extraction.
    TransectInterGDF : GeoDataFrame
        GeoDataFrame of transects with veg edge intersection info assigned.
    VeglinesGDF : GeoDataFrame
        GoeDataFrame representing shapefile of vegetation edge lines.
    BasePath : str
        Filepath to where veg edge and transect shapefiles sit.

    Returns
    -------
    TransectInterGDF : GeoDataFrame
        Updated GeoDataFrame with new info attached to each transect.

    """
    
    print('Intersecting transects with transition zones... ')
    # Initialise empty field that matches dimensions of each intersection
    WidthFields = []
    for Tr in range(len(TransectInterGDF)):
        WidthFields.append([np.nan]*len(TransectInterGDF['filename'].iloc[Tr]))
        
    fpath = os.path.join(settings['inputs']['filepath'], settings['inputs']['sitename'])
    # read in Transition Zone tifs
    fnames = [os.path.basename(x) for x in glob.glob(os.path.join(fpath,'jpg_files', '*_TZ.tif'))]

    for fnum, fname in enumerate(fnames): # for each TZ raster (and therefore image date)
        with rio.Env():
            with rio.open(os.path.join(fpath, 'jpg_files', fname)) as src:
                img = src.read(1).astype("float32") # first band
                results = (
                {'properties': {'raster_val': v}, 'geometry': s}
                for i, (s, v) 
                in enumerate(
                    shapes(img, mask=None, transform=src.transform)))
        # TZ to polygon
        geoms = list(results)
        TZpoly = gpd.GeoDataFrame.from_features(geoms, src.crs)
        TZpoly = TZpoly[TZpoly['raster_val'] == 1] # get rid of nan polygons
        
        f = fname[:-7] # get rid of '_TZ' and extension
        # Calculate area of polygons
        TZpoly['area'] = TZpoly.area        
        # Get matching veg line and buffer by ref line buffer amount
        VeglinesGDF['imagename'] = [os.path.basename(x) for x in VeglinesGDF['filename']]
        Vegline = VeglinesGDF[VeglinesGDF['imagename'].isin([f])]
        VeglineBuff = Vegline.buffer(settings['max_dist_ref'])
        # convert to matching CRS for clipping (also ensures same CRS for Tr intersect)
        TZpoly = TZpoly.to_crs(VeglineBuff.crs) 
        # Clip TZ polys to matching image's vegline buffer
        TZpolyClip = gpd.clip(TZpoly,VeglineBuff)
        TZpolyClip = TZpolyClip.explode()
                    
        # Intersection between polygon edges and Tr
        for Tr in range(len(TransectInterGDF)):
            print('\r %0.3f %% images processed' % ( ((fnum)/len(fnames))*100 ), end='')
            # list of filenames on each transect from intersections with VEs
            TrFiles = [os.path.basename(x) for x in TransectInterGDF['filename'].iloc[Tr]]
            # get matching image index in list of transect's VE filenames 
            try:
                ImInd = TrFiles.index(f)
            except: # if filename doesn't exist in list of files on transect, skip
                TZwidth = np.nan
                continue # step out of current Tr loop and start on next Tr
            
            TransectGeom = TransectInterGDF['geometry'].iloc[Tr]
            # Distances of each intersection pair from VE
            TZpolyClip['pntdist'] = [np.nan]*len(TZpolyClip)
            # Intersect Tr with TZ polygon
            TZpolyClip['TrIntersect'] = TZpolyClip.exterior.intersection(TransectGeom)
            # Remove empty geoms from TZ dataframe
            TZpolyClipInter = TZpolyClip[TZpolyClip['TrIntersect'].is_empty == False][TZpolyClip['TrIntersect'].isna() == False]
            
            # if Transect ends inside TZ polygon, extend length until multipoint is achieved
            while (TZpolyClipInter['TrIntersect'].geom_type == 'Point').sum() > 0:
                TransectGeom = Toolbox.ExtendLine(TransectGeom, 10)
                TZpolyClip['TrIntersect'] = TZpolyClip.exterior.intersection(TransectGeom)
                # Remove empty geoms from TZ dataframe
                TZpolyClipInter = TZpolyClip[TZpolyClip['TrIntersect'].is_empty == False][TZpolyClip['TrIntersect'].isna() == False]
            
            # fill in distance between first TZ intersection point and VE-Tr intersection
            for i in range(len(TZpolyClipInter)):
                Point1 = list(TZpolyClipInter['TrIntersect'].iloc[i].geoms)[0]  # Extract the first point from the MultiPoint
                TZpolyClipInter['pntdist'].iloc[i] = Point1.distance(TransectInterGDF['interpnt'][Tr][ImInd])
                # below line causes error for iterating over multipoints
                # TZpolyClipInter['pntdist'].iloc[i] = list(TZpolyClipInter['TrIntersect'].iloc[i])[0].distance(TransectInterGDF['interpnt'][Tr][ImInd])
            # TZpolyClip['pntdist'] = TZpolyClip.centroid.distance(TransectInterGDF['interpnt'][Tr][ImInd])
            
            # if Transect doesn't intersect with any TZ polygons
            if len(TZpolyClipInter) == 0:
                TZwidth = np.nan
            else:                
                TZpolyClose = TZpolyClipInter['TrIntersect'][TZpolyClipInter['pntdist'] == TZpolyClipInter['pntdist'].min()]
                # TZ width (Distance between intersect points)
                TZwidth = TZpolyClose.explode(index_parts=True).iloc[0].distance(TZpolyClose.explode(index_parts=True).iloc[1])
            # Info stored back onto the matching Tr ID
            WidthFields[Tr][ImInd] = TZwidth
    
    print('Adding TZ widths to transect shapefile... ')
    TransectInterGDF['TZwidth'] = WidthFields
    
    # initialise and fill field with median TZ widths across each Tr's timeseries
    TransectInterGDF['TZwidthMn'] = np.zeros(len(TransectInterGDF))
    for i in range(len(TransectInterGDF)):
        TransectInterGDF['TZwidthMn'].iloc[i] = np.nanmean(TransectInterGDF['TZwidth'].iloc[i])
    
    TransectInterShp = TransectInterGDF.copy()
    
    # reformat fields with lists to strings
    KeyName = list(TransectInterShp.select_dtypes(include='object').columns)
    for Key in KeyName:
        # round any floating points numbers before export
        realInd = next(i for i, j in enumerate(TransectInterShp[Key]) if j)
            
        if type(TransectInterShp[Key][realInd]) == list: # for lists of intersected values
            if type(TransectInterShp[Key][realInd][0]) == np.float64:  
                for Tr in range(len(TransectInterShp[Key])):
                    TransectInterShp[Key][Tr] = [round(i,2) for i in TransectInterShp[Key][Tr]]
        else: # for singular values
            if type(TransectInterShp[Key][realInd]) == np.float64: 
                for Tr in range(len(TransectInterShp[Key])):
                    TransectInterShp[Key][Tr] = [round(i,2) for i in TransectInterShp[Key][Tr]]
        
        TransectInterShp[Key] = TransectInterShp[Key].astype(str)
                    
    # Save as shapefile of intersected transects
    TransectInterShp.to_file(os.path.join(BasePath,settings['inputs']['sitename']+'_Transects_Intersected_TZ.shp'))
        
    return TransectInterGDF    


def SlopeIntersect(settings, TransectInterGDF, VeglinesGDF, BasePath, DTMfile=None):
    """
    Intersections between coastal indicator lines and topographic slope raster.
    FM June 2023

    Parameters
    ----------
    settings : dict
        Dictionary of user-defined settings used for the veg edge extraction.
    TransectInterGDF : GeoDataFrame
        GeoDataFrame of transects with veg edge intersection info assigned.
    VeglinesGDF : GeoDataFrame
        GoeDataFrame representing shapefile of vegetation edge lines.
     BasePath : str
         Filepath to where veg edge and transect shapefiles sit.
    DTMfile : str, optional
        Filepath to slope raster of choice. The default is None.

    Returns
    -------
    TransectInterGDF : GeoDataFrame
        Updated GeoDataFrame with new info attached to each transect.

    """
    
    if DTMfile is None:
        print('No DTM file provided.')
        return TransectInterGDF
    
    else:
        print('Intersecting transects with slope ... ')
        
        src = rio.open(DTMfile)
        
        # DTM should be in same CRS as Transects; reproject using gdal if not  
        if src.crs != TransectInterGDF.crs:
            dstDTMfile = os.path.join(os.path.splitext(DTMfile)[0] + '_reproj.tif')
            command = 'gdalwarp ' + DTMfile + ' ' + dstDTMfile + ' -t_srs ' + str(TransectInterGDF.crs)
            os.system(command)
            # read back in reprojected .tif
            src = rio.open(dstDTMfile)

        MaxSlope = []
        MeanSlope = []
        
        for Tr in range(len(TransectInterGDF)):
            print('\r %0.3f %% transects processed' % ( (Tr/len(TransectInterGDF))*100 ), end='')

            # Only want 20 m either side of veg intersect
            InterPnts = TransectInterGDF['interpnt'].iloc[Tr]
            # If there are no line intersections on that transect
            if InterPnts == []:
                MaxSlope.append(np.nan)
                MeanSlope.append(np.nan)
            else:
                # Take average vegline intersection point of each transect to swath points along
                InterPnt = Point(np.mean([Pnt.x for Pnt in InterPnts]), np.mean([Pnt.y for Pnt in InterPnts]))
                
                # Extend Tr in either direction along transect from intersection point
                intx, Trx, inty, Try = InterPnt.coords.xy[0][0], TransectInterGDF.iloc[Tr].geometry.coords.xy[0][0], InterPnt.coords.xy[1][0],TransectInterGDF.iloc[Tr].geometry.coords.xy[1][0]
                # Distance decided by cross-shore width of TZ plus extra 5m buffer
                if np.isnan(TransectInterGDF['TZwidthMn'].iloc[Tr]) == True:
                    dist = 5 # buffer transects with no TZ by 5m
                else:
                    dist = round(TransectInterGDF['TZwidthMn'].iloc[Tr]) + 5
                # calculate vector
                v = (Trx-intx, Try-inty)
                v_ = np.sqrt((Trx-intx)**2 + (Try-inty)**2)
                # calculate normalised vector
                vnorm = v / v_
                # use norm vector to extend 
                x_1, y_1 = (intx, inty) - (dist*vnorm)
                x_2, y_2 = (intx, inty) + (dist*vnorm)
                
                # New linestring from extended points
                NewTr = gpd.GeoDataFrame(index=[0], crs=TransectInterGDF.crs, geometry=[LineString([(x_1,y_1),(x_2,y_2)])])
                NewTrGeom = NewTr.geometry
                # Generate regularly spaced points along each transect
                distance_delta = 1
                distances = np.arange(0, float(NewTrGeom.length), distance_delta)
                points = [NewTrGeom.interpolate(distance) for distance in distances]
                points = [(float(point.x), float(point.y)) for point in points]
                
                # Extract slope values at each point along Tr
                MaxSlopeTr = np.max([val[0] for val in src.sample(points)])
                MeanSlopeTr = np.mean([val[0] for val in src.sample(points)])
                if MaxSlopeTr == -9999: # nodata value
                    MaxSlopeTr = np.nan
                    MeanSlopeTr = np.nan
                MaxSlope.append(MaxSlopeTr)
                MeanSlope.append(MeanSlopeTr)
        
        TransectInterGDF['SlopeMax'] = MaxSlope
        TransectInterGDF['SlopeMean'] = MeanSlope
        
        
        TransectInterShp = TransectInterGDF.copy()
        
        # reformat fields with lists to strings
        KeyName = list(TransectInterShp.select_dtypes(include='object').columns)
        for Key in KeyName:
            # round any floating points numbers before export
            realInd = next(i for i, j in enumerate(TransectInterShp[Key]) if j)
                
            if type(TransectInterShp[Key][realInd]) == list: # for lists of intersected values
                if type(TransectInterShp[Key][realInd][0]) == np.float64:  
                    for Tr in range(len(TransectInterShp[Key])):
                        TransectInterShp[Key][Tr] = [round(i,2) for i in TransectInterShp[Key][Tr]]
            else: # for singular values
                if type(TransectInterShp[Key][realInd]) == np.float64: 
                    for Tr in range(len(TransectInterShp[Key])):
                        TransectInterShp[Key][Tr] = [round(i,2) for i in TransectInterShp[Key][Tr]]
            
            TransectInterShp[Key] = TransectInterShp[Key].astype(str)
                        
        # Save as shapefile of intersected transects
        TransectInterShp.to_file(os.path.join(BasePath,settings['inputs']['sitename']+'_Transects_Intersected_Slope.shp'))
            
        return TransectInterGDF    
            


def WavesIntersect(settings, TransectInterGDF, BasePath, output, lonmin, lonmax, latmin, latmax):
    """
    Intersections between coastal indicator lines and wave hindcast data from 
    Copernicus Marine Service.
    FM June 2023

    Parameters
    ----------
    settings : dict
        Dictionary of user-defined settings used for the veg edge extraction.
    TransectInterGDF : GeoDataFrame
        GeoDataFrame of transects with veg edge intersection info assigned.
    BasePath : str
        Filepath to where veg edge and transect shapefiles sit.
    output : dict
        Dictionary of extracted veg edges and associated info with each edge.
    lonmin, lonmax, latmin, latmax : float
        Longitudes and latitudes of area of interest bounding box

    Returns
    -------
    TransectInterGDFWave : GeoDataFrame
        Updated GeoDataFrame with new wave info attached to each transect.

    """
    
    # Convert bbox coords back to WGS84
    BBox = gpd.GeoDataFrame(crs=4326,geometry=[Polygon([[lonmin, latmin],
                    [lonmax,latmin],
                    [lonmax,latmax],
                    [lonmin, latmax]])])
    BBox.to_crs(epsg=4326, inplace=True)
    # lonmin, lonmax, latmin, latmax = 
    
    # Avoid overwriting anything
    TransectInterGDFWave = TransectInterGDF.copy()
    
    # Download wave hindcast for given time frame and location
    WavePath, WaveOutFile = Waves.GetHindcastWaveData(settings, output, lonmin, lonmax, latmin, latmax)
    WaveFilePath = os.path.join(WavePath, WaveOutFile)
    
    # Sample waves from CMEMS hindcast
    # WaveDates, WaveHs, WaveDir, WaveTp, NormWaveHs, NormWaveDir, NormWaveTp, StDevWaveHs, StDevWaveDir, StDevWaveTp, WaveDiffusivity, WaveStability, ShoreAngles = Waves.SampleWaves(settings, output, TransectInterGDF, WaveFilePath)
    WaveDates, WaveHs, WaveDir, WaveTp, NormWaveHs, NormWaveDir, NormWaveTp, StDevWaveHs, StDevWaveDir, StDevWaveTp, WaveDiffusivity, WaveStability, ShoreAngles = Waves.SampleWavesSimple(settings, output, TransectInterGDF, WaveFilePath)
    WaveAlphas = []
    for Tr in range(len(TransectInterGDFWave)):
        WaveAlphas.append(Waves.CalcAlpha(WaveDir, ShoreAngles, Tr))
    
    TransectInterGDFWave['WaveDates'] = WaveDates
    TransectInterGDFWave['WaveHs'] = WaveHs
    TransectInterGDFWave['WaveDir'] = WaveDir
    TransectInterGDFWave['WaveAlpha'] = WaveAlphas
    TransectInterGDFWave['WaveTp'] = WaveTp
    TransectInterGDFWave['WaveDiffus'] = WaveDiffusivity
    TransectInterGDFWave['WaveStabil'] = WaveStability
    TransectInterGDFWave['ShoreAngle'] = ShoreAngles
    
    # Calculate wave runup from extracted wave conditions
    Runups = Waves.CalcRunup(WaveHs)
    TransectInterGDFWave['Runups'] = Runups
    
    # If any directions (and WaveAlphas) are still masked, set to nan
    for Key in TransectInterGDFWave.select_dtypes(include='object').columns:
        for Tr in range(len(TransectInterGDFWave[Key])):
            data = TransectInterGDFWave[Key].iloc[Tr]
            if isinstance(data,list):
                TransectInterGDFWave.at[Tr, Key] = [np.nan if str(x) == 'masked' or str(x) == '--' else x for x in data]
    
    TransectInterShp = TransectInterGDFWave.copy()
    
    # reformat fields with lists to strings
    KeyName = list(TransectInterShp.select_dtypes(include='object').columns)
    for Key in KeyName:
        # round any floating points numbers before export
        realInd = next(i for i, j in enumerate(TransectInterShp[Key]) if j)
            
        if type(TransectInterShp[Key][realInd]) == list: # for lists of intersected values
            if type(TransectInterShp[Key][realInd][0]) == np.float64:  
                for Tr in range(len(TransectInterShp[Key])):
                    TransectInterShp[Key][Tr] = [round(i,3) for i in TransectInterShp[Key][Tr]]
        else: # for singular values
            if type(TransectInterShp[Key][realInd]) == np.float64: 
                for Tr in range(len(TransectInterShp[Key])):
                    TransectInterShp[Key][Tr] = [round(i,3) for i in TransectInterShp[Key][Tr]]
        
        TransectInterShp[Key] = TransectInterShp[Key].astype(str)
                    
    # Save as shapefile of intersected transects
    TransectInterShp.to_file(os.path.join(BasePath,settings['inputs']['sitename']+'_Transects_Intersected_Waves.shp'))
        
    
    return TransectInterGDFWave


def GetFutureData(sitename, DateMin, DateMax, CoastalDF):
    
    with open(os.path.join(os.getcwd(),'Data', sitename, sitename + '_settings.pkl'), 'rb') as f:
        settings = pickle.load(f)
    

    # Download wave forecasts from Copernicus Marine
    WavePath, WaveOutFile = Waves.GetForecastWaveData(settings, DateMin, DateMax)
    WaveFilePath = os.path.join(WavePath, WaveOutFile)
    
    # Sample future waves using coastal DF transects
    WaveDates, WaveHs, WaveDir, WaveTp, WaveDiffusivity, WaveStability, ShoreAngles = Waves.SampleWavesFuture(CoastalDF, WaveFilePath)
    


def ValidateIntersects(ValidationShp, DatesCol, TransectGDF, TransectDict):
    """
    UNUSED/LEGACY
    Intersects transects with validation lines from shapefile, matches date of
    each validation line to nearest sat line, and calculates distance along 
    transect between them.
    FM Oct 2022

    Parameters
    ----------
    ValidationShp : str
        File path to validation line shapefile.
    DatesCol : str
        Name of attribute field where dates are stored.
    TransectGDF : GeoDataFrame
        Transect GDF with no attributes, just geometries.
    TransectDict : dict
        Transect dictionary with attributes.

    Returns
    -------
    ValidDict : dict
        Dictionary holding validation line info.
        
    """
    
    print('performing transect intersects on validation lines...')
    ValidGDF = gpd.read_file(ValidationShp)
    if DatesCol in ValidGDF.keys():
        ValidGDF = ValidGDF[[DatesCol,'geometry']]
    else:
        print('No date column found - check your spelling')
        return
    # initialise where each intersection between lines and transects will be saved
    ColumnData = []
    Geoms = []
     # for each row/feature in transect
    for _, _, ID, TrGeom, refpnt in TransectGDF.itertuples():
         # for each row/feature shoreline
        for _,dates,SGeom in ValidGDF.itertuples():
             # calculate intersections between each transect and shoreline
            Intersects = TrGeom.intersection(SGeom)
            ColumnData.append((ID,dates))
            Geoms.append(Intersects)
            
    # create GDF from appended lists of intersections        
    AllIntersects = gpd.GeoDataFrame(ColumnData,geometry=Geoms,columns=['TransectID','Vdates'])
    # remove any rows with no intersections
    AllIntersects = AllIntersects[~AllIntersects.is_empty].reset_index().drop('index',axis=1)
    # duplicate geom column to save point intersections
    AllIntersects['Vinterpnt'] = AllIntersects['geometry']
    # take only first point on any transects which intersected a single shoreline more than once
    for inter in range(len(AllIntersects)):
        if AllIntersects['Vinterpnt'][inter].geom_type == 'MultiPoint':
            AllIntersects['Vinterpnt'][inter] = list(AllIntersects['Vinterpnt'][inter].geoms)[0] # list() accesses individual points in MultiPoint
    # AllIntersects = AllIntersects.drop('geometry',axis=1)
    AllIntersects = AllIntersects.rename_geometry('pntgeometry')

    # attribute join on transect ID to get transect geometry back
    AllIntersects = AllIntersects.merge(TransectGDF[['TransectID','geometry']], on='TransectID')
    AllIntersects = AllIntersects.drop('pntgeometry',axis=1)

    print("formatting into GeoDataFrame...")
    # initialise distances of intersections 
    distances = []
    # for each intersection
    for i in range(len(AllIntersects)):
        # calculate distance of intersection along transect
        distances.append(Toolbox.CalcDistance(AllIntersects['Vinterpnt'][i], AllIntersects['geometry'][i]))
    AllIntersects['Vdists'] = distances
    
    #initialise lists used for storing each transect's intersection values
    Vdates,Vdists, Vinterpnt = ([] for i in range(3)) # per-transect lists of values

    Key = [Vdates, Vdists, Vinterpnt]
    KeyName = ['Vdates', 'Vdists', 'Vinterpnt']
    ValidDict = TransectDict.copy()
    
    # for each column name
    for i in range(len(Key)):
        # for each transect
        for Tr in range(len(TransectGDF['TransectID'])):
            # refresh per-transect list
            TrKey = []
            # for each matching intersection on a single transect
            for j in range(len(AllIntersects.loc[AllIntersects['TransectID']==Tr])):
                # append each intersection value to a list for each transect
                # iloc used so index doesn't restart at 0 each loop
                TrKey.append(AllIntersects[KeyName[i]].loc[AllIntersects['TransectID']==Tr].iloc[j]) 
            Key[i].append(TrKey)
    
        ValidDict[KeyName[i]] = Key[i]
    
    print('calculating distances between validation and sat lines...')
    ValidDict['valsatdist'] = ValidDict['TransectID'].copy()
    # for each transect
    for Tr in range(len(TransectGDF['TransectID'])):
        # dates into transect-specific list
        VDateList = [datetime.strptime(date, '%Y-%m-%d') for date in ValidDict['Vdates'][Tr]]
        DateList = [datetime.strptime(date, '%Y-%m-%d') for date in ValidDict['dates'][Tr]]
        # find index of closest sat date to each validation date
        ValSatDists = []
        for D, VDate in enumerate(VDateList):
            # index of matching nearest date
            if DateList != []:
                DateIndex = DateList.index(Toolbox.NearDate(VDate,DateList))
            else:
                continue
            # use date index to identify matching distance along transect
            # and calculate distance between two intersections (sat - validation means +ve is seaward/-ve is landward)
            ValSatDists.append(ValidDict['distances'][Tr][DateIndex] - ValidDict['Vdists'][Tr][D])
            
        ValidDict['valsatdist'][Tr] = ValSatDists
        
    print("TransectDict with intersections created.")
    
    return ValidDict


def ValidateSatIntersects(sitename, ValidationShp, DatesCol, TransectGDF, TransectInterGDF):
    """
    Intersects transects with validation lines from shapefile, matches date of
    each sat line to nearest valid. line, and calculates distance along 
    transect between them.
    
    FM Oct 2022

    Parameters
    ----------
    ValidationShp : str
        File path to validation line shapefile.
    DatesCol : str
        Name of attribute field where dates are stored.
    TransectGDF : GeoDataFrame
        Transect GDF with no attributes, just geometries.
    TransectDict : dict
        Transect dictionary with attributes.

    Returns
    -------
    ValidDict : dict
        DESCRIPTION.
        
    """
    
    validpath = os.path.join(os.getcwd(), 'Data', sitename, 'validation')
    if os.path.isdir(validpath) is False:
        os.mkdir(validpath)
    
    print('performing transect intersects on validation lines...')
    ValidGDF = gpd.read_file(ValidationShp)
    if DatesCol in ValidGDF.keys():
        ValidGDF = ValidGDF[[DatesCol,'geometry']]
    else:
        print('No date column found - check your spelling')
        return
    # initialise where each intersection between lines and transects will be saved
    ColumnData = []
    Geoms = []
     # for each row/feature in transect
    for _, _, ID, TrGeom, refpnt in TransectGDF.itertuples():
         # for each row/feature shoreline
        for _,dates,SGeom in ValidGDF.itertuples():
             # calculate intersections between each transect and shoreline
            Intersects = TrGeom.intersection(SGeom)
            ColumnData.append((ID,dates))
            Geoms.append(Intersects)
            
    # create GDF from appended lists of intersections        
    AllIntersects = gpd.GeoDataFrame(ColumnData,geometry=Geoms,columns=['TransectID','Vdates'])
    # remove any rows with no intersections
    AllIntersects = AllIntersects[~AllIntersects.is_empty].reset_index().drop('index',axis=1)
    # duplicate geom column to save point intersections
    AllIntersects['Vinterpnt'] = AllIntersects['geometry']
    # take only first point on any transects which intersected a single shoreline more than once
    for inter in range(len(AllIntersects)):
        if AllIntersects['Vinterpnt'][inter].geom_type == 'MultiPoint':
            AllIntersects['Vinterpnt'][inter] = list(AllIntersects['Vinterpnt'][inter].geoms)[0] # list() accesses individual points in MultiPoint
    # AllIntersects = AllIntersects.drop('geometry',axis=1)
    AllIntersects = AllIntersects.rename_geometry('pntgeometry')

    # attribute join on transect ID to get transect geometry back
    AllIntersects = AllIntersects.merge(TransectGDF[['TransectID','geometry']], on='TransectID')
    AllIntersects = AllIntersects.drop('pntgeometry',axis=1)
    
    print("formatting into GeoDataFrame...")
    # initialise distances of intersections 
    distances = []
    # for each intersection
    for i in range(len(AllIntersects)):
        # calculate distance of intersection along transect
        distances.append(Toolbox.CalcDistance(AllIntersects['Vinterpnt'][i], AllIntersects['geometry'][i]))
    AllIntersects['Vdists'] = distances
    
    #initialise lists used for storing each transect's intersection values
    Vdates,Vdists, Vinterpnt = ([] for i in range(3)) # per-transect lists of values

    Key = [Vdates, Vdists, Vinterpnt]
    KeyName = ['Vdates', 'Vdists', 'Vinterpnt']
    ValidInterGDF = TransectInterGDF.copy()
    
    # for each column name
    for i in range(len(Key)):
        # for each transect
        for Tr in range(len(TransectGDF['TransectID'])):
            # refresh per-transect list
            TrKey = []
            # for each matching intersection on a single transect
            for j in range(len(AllIntersects.loc[AllIntersects['TransectID']==Tr])):
                # append each intersection value to a list for each transect
                # iloc used so index doesn't restart at 0 each loop
                TrKey.append(AllIntersects[KeyName[i]].loc[AllIntersects['TransectID']==Tr].iloc[j]) 
            Key[i].append(TrKey)
    
        ValidInterGDF[KeyName[i]] = Key[i]
    
    print('calculating distances between validation and sat lines...')
    # must initialise with list of same length as veg dates
    ValidInterGDF['valsatdist'] = ValidInterGDF['dates'].copy()
    ValidInterGDF['valsatdate'] = ValidInterGDF['dates'].copy()
    # for each transect
    for Tr in range(len(TransectGDF['TransectID'])):
        # dates into transect-specific list
        VDateList = [datetime.strptime(date, '%Y-%m-%d') for date in ValidInterGDF['Vdates'].iloc[Tr]]
        DateList = [datetime.strptime(date, '%Y-%m-%d') for date in ValidInterGDF['dates'].iloc[Tr]]
        # find index of closest validation date to each sat date
        # previously was empty list with nans or distances appended
        # now should start with list of nans with n=n(dates)
        ValSatDists = list(np.empty(len(ValidInterGDF['dates'].iloc[Tr]))*np.nan)
        ValSatDates = list(np.empty(len(ValidInterGDF['dates'].iloc[Tr]))*np.nan)
        for D, Date in enumerate(DateList):
            # index of matching nearest date
            if VDateList != []:
                NearestDate = Toolbox.NearDate(Date,VDateList)
                if NearestDate == False: # if no matching validation date exists, add nan to the list
                    continue
                else:
                    # use date index to identify matching distance along transect
                    # and calculate distance between two intersections (sat - validation means +ve is seaward/-ve is landward)
                    DateStr = datetime.strftime(Date,'%Y-%m-%d')
                    SDateIndex = ValidInterGDF['dates'].iloc[Tr].index(DateStr)
                    VDateIndex = VDateList.index(NearestDate)
                    ValSatDists[SDateIndex] = ValidInterGDF['distances'].iloc[Tr][D] - ValidInterGDF['Vdists'].iloc[Tr][VDateIndex]
                    ValSatDates[SDateIndex] = ValidInterGDF['Vdates'].iloc[Tr][VDateIndex]
            else:
                continue
       

        ValidInterGDF['valsatdist'].iloc[Tr] = ValSatDists
        ValidInterGDF['valsatdate'].iloc[Tr] = ValSatDates
        
    print("ValidInterGDF with intersections created.")
    
    return ValidInterGDF


# def compute_intersection(output, transects, settings, linetype):
#     """
#     UNUSED/LEGACY
#     Computes the intersection between the 2D shorelines and the shore-normal.
#     transects. It returns time-series of cross-shore distance along each transect.
    
#     KV WRL 2018       

#     Arguments:
#     -----------
#     output: dict
#         contains the extracted shorelines and corresponding metadata
#     transects: dict
#         contains the X and Y coordinates of each transect
#     settings: dict with the following keys
#         'along_dist': int
#             alongshore distance considered caluclate the intersection
              
#     Returns:    
#     -----------
#     cross_dist: dict
#         time-series of cross-shore distance along each of the transects. 
#         Not tidally corrected.        
#     """  
    
#     """
#     if (linetype+'transect_time_series.csv') in os.listdir(settings['inputs']['filepath']):
#         print('Cross-distance calculations already exist and were loaded')
#         with open(os.path.join(settings['inputs']['filepath'], linetype+'transect_time_series.csv'), 'rb') as f:
#             cross_dist = pickle.load(f)
#         return cross_dist
#     """
    
#     # loop through shorelines and compute the median intersection    
#     intersections = np.zeros((len(output['shorelines']),len(transects)))
#     for i in range(len(output['shorelines'])):

#         sl = output['shorelines'][i]
        
#         print(" \r\tShoreline %4d / %4d" % (i+1, len(output['shorelines'])), end="")
        
#         for j,key in enumerate(list(transects.keys())): 
            
#             # compute rotation matrix
#             X0 = transects[key][0,0]
#             Y0 = transects[key][0,1]
#             temp = np.array(transects[key][-1,:]) - np.array(transects[key][0,:])
#             phi = np.arctan2(temp[1], temp[0])
#             Mrot = np.array([[np.cos(phi), np.sin(phi)],[-np.sin(phi), np.cos(phi)]])
    
#             # calculate point to line distance between shoreline points and the transect
#             p1 = np.array([X0,Y0])
#             p2 = transects[key][-1,:]
#             d_line = np.abs(np.cross(p2-p1,sl-p1)/np.linalg.norm(p2-p1))
#             # calculate the distance between shoreline points and the origin of the transect
#             d_origin = np.array([np.linalg.norm(sl[k,:] - p1) for k in range(len(sl))])
#             # find the shoreline points that are close to the transects and to the origin
#             # the distance to the origin is hard-coded here to 1 km 
#             idx_dist = np.logical_and(d_line <= settings['along_dist'], d_origin <= 1000)
#             # find the shoreline points that are in the direction of the transect (within 90 degrees)
#             temp_sl = sl - np.array(transects[key][0,:])
#             phi_sl = np.array([np.arctan2(temp_sl[k,1], temp_sl[k,0]) for k in range(len(temp_sl))])
#             diff_angle = (phi - phi_sl)
#             idx_angle = np.abs(diff_angle) < np.pi/2
#             # combine the transects that are close in distance and close in orientation
#             idx_close = np.where(np.logical_and(idx_dist,idx_angle))[0]     
            
#             # in case there are no shoreline points close to the transect 
#             if len(idx_close) == 0:
#                 intersections[i,j] = np.nan
#             else:
#                 # change of base to shore-normal coordinate system
#                 xy_close = np.array([sl[idx_close,0],sl[idx_close,1]]) - np.tile(np.array([[X0],
#                                    [Y0]]), (1,len(sl[idx_close])))
#                 xy_rot = np.matmul(Mrot, xy_close)
#                 # compute the median of the intersections along the transect
#                 intersections[i,j] = np.nanmedian(xy_rot[0,:])
    
#     # fill the a dictionnary
#     cross_dist = dict([])
#     cross_dist['dates'] = output['dates']
#     for j,key in enumerate(list(transects.keys())): 
#         cross_dist[key] = intersections[:,j]  
    
    
#     # save a .csv file for Excel users
#     out_dict = dict([])
#     out_dict['dates'] = output['dates']
#     for key in transects.keys():
#         out_dict['Transect '+ key] = cross_dist[key]
#     df = pd.DataFrame(out_dict)
#     fn = os.path.join(settings['inputs']['filepath'],settings['inputs']['sitename'],linetype+
#                       'transect_time_series.csv')
#     df.to_csv(fn, sep=',')
#     print('Time-series of the shoreline change along the transects saved as:\n%s'%fn)
    
#     return cross_dist

# def stuffIntoLibrary(geo, image_epsg, projection_epsg, filepath, sitename):
#     """
#     UNUSED/LEGACY

#     Parameters
#     ----------
#     geo : TYPE
#         DESCRIPTION.
#     image_epsg : int
#         EPSG code of satellite image.
#     projection_epsg : int
#         EPSG code of desired projection.
#     filepath : TYPE
#         DESCRIPTION.
#     sitename : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     transects_latlon : TYPE
#         DESCRIPTION.
#     transects_proj : TYPE
#         DESCRIPTION.

#     """
#     print('Reading transects into library for further analysis...')
#     transect = Path("Data/" + sitename + "/transect_proj.pkl")
    
#     proj1 = Proj(init="epsg:"+str(projection_epsg)) 
#     proj2 = Proj(init="epsg:"+str(4326))
#     proj3 = Proj(init="epsg:"+str(image_epsg))
    
#     if transect.is_file():
#         with open(os.path.join(filepath, sitename + '_transect_proj' + '.pkl'), 'rb') as f:
#             transects_proj = pickle.load(f)
#         with open(os.path.join(filepath, sitename + '_transect_latlon' + '.pkl'), 'rb') as f:
#             transects_latlon = pickle.load(f)
            
#         return transects_latlon, transects_proj
    
#     transects_latlon = dict([])
#     transects_proj = dict([])

#     for i in range (len(geo['geometry'])):
        
#         lib = 'Transect_'+str(i+1)
    
#         x,y = geo['geometry'][i].coords.xy
        
#         # convert to lat lon
#         xy0 = pyproj.transform(proj1,proj2,y[0],x[0])
#         xy1 = pyproj.transform(proj1,proj2,y[1],x[1])
#         coord0_latlon = [xy0[1],xy0[0]]
#         coord1_latlon = [xy1[1],xy1[0]]

#         transects_latlon[lib] = np.array([coord0_latlon, coord1_latlon])
#         #x,y = pyproj.transform(proj2,proj3,transects_latlon[lib][0][1],transects_latlon[lib][0][0])
#         #x1,y1 = pyproj.transform(proj2,proj3,transects_latlon[lib][1][1],transects_latlon[lib][1][0])
#         transects_proj[lib] = np.array([[x[1],y[1]],[x[0],y[0]]])

#         print(" \r\tCurrent Progress:",np.round(i/len(geo['geometry'])*100,2),"%",end='')
    
#     with open(os.path.join(filepath, sitename + '_transect_proj.pkl'), 'wb') as f:
#             pickle.dump(transects_proj, f)
            
#     with open(os.path.join(filepath, sitename + '_transect_latlon.pkl'), 'wb') as f:
#             pickle.dump(transects_latlon, f)
            
#     return transects_latlon, transects_proj

# def transect_compiler(Rows, transect_proj, transect_range, output):
    
#     cross_distance_condensed = dict([])
#     standard_err_condensed = dict([])
#     transect_condensed = dict([])
#     Dates = dict([])
#     new_Transect = 1

#     cross_arr = []
#     trans_arr = []

#     for i in range(len(transect_range)):
#         cross_arr = []
#         trans_arr = []
#         for j in range(transect_range[i][0],transect_range[i][1]):
#             try:
#                 arr = []
#                 for k in range(len(Rows)-1):
#                     try:
#                         arr.append(float(Rows[k][j]))
#                     except:
#                         arr.append(np.nan)
#                 cross_arr.append(arr)
#                 trans_arr.append(transect_proj[list(transect_proj.keys())[j]])
#             except:
#                 continue
#         std = np.nanstd(cross_arr,0)
#         for j in range(len(std)):
#             std[j] = std[j]/(abs(transect_range[i][0]-transect_range[i][1]))**0.5

#         NaN_mask = np.isfinite(np.nanmean(cross_arr,0))
#         cross_distance_condensed['Transect_'+str(transect_range[i][0])+'-'+str(transect_range[i][1])] = np.nanmean(cross_arr,0).astype(np.double)[NaN_mask]
#         standard_err_condensed['Transect_'+str(transect_range[i][0])+'-'+str(transect_range[i][1])] = std.astype(np.double)[NaN_mask]
#         Dates['Transect_'+str(transect_range[i][0])+'-'+str(transect_range[i][1])] = np.array(output['dates'])[NaN_mask]
#         transect_condensed['Transect_'+str(transect_range[i][0])+'-'+str(transect_range[i][1])] = np.mean(trans_arr,0).astype(np.double)#[NaN_mask]
        
#     return cross_distance_condensed, standard_err_condensed, transect_condensed, Dates

