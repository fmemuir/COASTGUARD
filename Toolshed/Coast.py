"""
Coast object for analysing coastal morphology and predicting future coastal change

Martin D. Hurst
University of Glasgow
June 2019

"""

# import modules
import os, sys, time, pickle, bisect, pdb
from pathlib import Path
import numpy as np
from scipy.interpolate import splprep, splev
import numpy.ma as ma
from sklearn.cluster import KMeans

import shapefile
import itertools
import rasterio
import geopandas as gp
from shapely.geometry import Point, Polygon, LineString, MultiLineString, MultiPoint
from shapely.ops import nearest_points, linemerge

#from Toolshed import Line
from Toolshed.Line import *
from IPython.display import clear_output

# might do some multiprocessing?
from multiprocessing import Pool

class Coast:
    """
    Description of object goes here

    """

    def __init__(self, CoastShp="", MinLength=0.):
        
        """
        MDH, June 2019
        """

        print("Coast: Initialising Coast object")

        self.Cell = None
        self.SubCell = None
        self.CMU = None
        self.Method = None
        self.CoastShp = CoastShp
        self.NoCoastLines = 0
        self.CoastLines = []
        self.Contours = []
        self.MLWSLines = []
        self.CliffTopLines = []
        self.CliffToeLines = []
        self.BarrierFrontTopLines = []
        self.BarrierFrontToeLines = []
        self.BarrierBackTopLines = []
        self.BarrierBackToeLines = []
        self.CrestLines = []
        self.ExtFrontLines_Low = []
        self.ExtFrontLines_Med = []
        self.ExtFrontLines_High = []
        self.ExtBackLines_Low = []
        self.ExtBackLines_Med = []
        self.ExtBackLines_High = []
        self.FutureShoreLinesYears = []
        self.FutureShoreLines = []
        self.FutureVegEdgeLines = []
        self.FutureMinUncertainty = []
        self.FutureMaxUncertainty = []
        self.FutureMinError = []
        self.FutureMaxError = []
        self.WriteFutureLines = []
        self.WriteRecentLines = []
        self.Projection = ""
        self.OverallOrientation = 0.
        self.TransectsSpacing = 10.
        self.NodeSpacing = 10.
        self.TransectsLength2Sea = 200.
        self.TransectsLength2Land = 1000.
        self.ExtremeWaterLevels = []
        self.MHWS = None
        self.UniqueDEMList = []
            
        # some tracking bools
        self.BuiltTransects = False
        self.GotHistoricShorelines = False
        self.SampledDEMs = False
        self.PredictedFutureShorelines = False
        self.MorphologyAnalysed = False

        if CoastShp:
            self.ReadCoastShp(CoastShp, MinLength)
            
        else:
            print("Coast: Generating empty coast object")

    def __str__(self):
        String = "Coast Object:\n\tFile: %s\n\tNumber of Coastlines:%d\n\t" % (str(self.CoastShp), self.NoCoastLines)
        return String

    # a function to save to a pickle file
    def Save(self, PickleFile):

        """
        """
        print("Coast.Save: Saving Coast Object")
        with open(PickleFile, 'wb') as PFile:
            pickle.dump(self, PFile)

    # read coast from a shapefile
    def ReadCoastShp(self, CoastShp, MinLength=0.):
        
        """
        """

        # Open coast polyline file for reading
        SF = shapefile.Reader(CoastShp)
        Shapes = SF.shapes()
        
        # I HAVE DELETED THE RECORDING OF SHAPES AND RECORDS INTO THE OBJECT DUE TO COMPATIBILITY ISSUES
        # WITH PICKLING THAT I CANT UNDERSTAND!!!!

        # Get number of coast segments to work on
        self.NoCoastLines = len(Shapes)
        print("Coast.ReadCoastShp: Read Coastline, no of coast segments is", self.NoCoastLines)
    
        # Generate coast nodes for each segment
        for i in range(0,self.NoCoastLines):
            
            print(" \r\tCoastline %4d / %4d" % (i+1, self.NoCoastLines), end="")

            # get X and Y coordinates of segment
            try:
                X, Y = np.array(Shapes[i].points).T
            except:
                continue
                
            # Set up a line object for each
            ThisLine = Line(str(i), X, Y)
            
            # append to list of coast lines
            if ThisLine.TotalLength > MinLength:
                self.CoastLines.append(ThisLine)

        # get new number of coastal segments based on the list built
        self.NoCoastLines = len(self.CoastLines)

        print("")    

        # get projection strings
        f = open(CoastShp.rstrip("shp")+"cpg")
        self.Projection = f.read()
        f.close()

 
    def WriteCoastShp(self, CoastShp):
        
        """
        Writes the contents of a list of coast line objects to polyline shape file

        MDH, June 2019

        """
        # print action to screen
        print("Coast.WriteCoastShp: Writing coast line to a shapefile")

        self.WriteLinesShp("CoastLines", CoastShp)

    def WriteCliffShp(self, CliffShp):
        
        """
        Writes the contents of a list of cliff line objects to polyline shape file

        MDH, June 2019

        """
        
        # print action to screen
        print("Coast.WriteCliffShp: Writing cliffs to shapefiles")

        if len(self.CliffTopLines) == 0:
            self.GetCliffLines()

        CliffTopShp = CliffShp.split(".")[0]+"_Top.shp"
        CliffToeShp = CliffShp.split(".")[0]+"_Toe.shp"
        self.WriteLinesShp("CliffTopLines", CliffTopShp)
        self.WriteLinesShp("CliffToeLines", CliffToeShp)
        self.WritePatchesShp("CliffTopLines", "CliffToeLines", CliffShp)

    def WriteBarrierShp(self, BarrierShp):

        """
        Writes the contents of a list of barrier line objects to polyline shape file

        MDH, June 2019

        """

        # print action to screen
        print("Coast.WriteBarrierShp: Writing barrier line objects to polyline a shapefile")

        if len(self.BarrierFrontTopLines) == 0:
            self.GetBarrierLines()

        # set up individual file names
        BarrierFrontTopShp = BarrierShp.split(".")[0]+"_Front_Top.shp"
        BarrierFrontToeShp = BarrierShp.split(".")[0]+"_Front_Toe.shp"
        BarrierBackTopShp = BarrierShp.split(".")[0]+"_Back_Top.shp"
        BarrierBackToeShp = BarrierShp.split(".")[0]+"_Back_Toe.shp"
        BarrierTopPatchesShp = BarrierShp.split(".")[0]+"_Top.shp"
        BarrierToePatchesShp = BarrierShp.split(".")[0]+"_Toe.shp"
                
        # launch polyline shapefile writer
        self.WriteLinesShp("BarrierFrontTopLines", BarrierFrontTopShp)
        self.WriteLinesShp("BarrierFrontToeLines", BarrierFrontToeShp)
        self.WriteLinesShp("BarrierBackTopLines", BarrierBackTopShp)
        self.WriteLinesShp("BarrierBackToeLines", BarrierBackToeShp)

        # launch polygon patches shapefile writer
        self.WritePatchesShp("BarrierFrontTopLines", "BarrierBackTopLines", BarrierTopPatchesShp)
        self.WritePatchesShp("BarrierFrontToeLines", "BarrierBackToeLines", BarrierToePatchesShp)

    def WriteExtremeLevelsShp(self, ExtremeShp):

        """
        Writes the contents of a list of barrier line objects to polyline shape file

        MDH, June 2019

        """

        if len(self.ExtFrontLines_Low) == 0:
            self.GetExtremeLines()

        # print action to screen
        print("Coast.WriteExtremeLevelsShp: Writing extreme water line objects to polyline and polygon shapefile")

        # loop through extreme water levels
        for i, Level in enumerate(["Low", "Med","High"]):

            # set up individual file names
            ExtFrontShp = ExtremeShp.split(".")[0]+"_"+Level+"_Front.shp"
            ExtBackShp = ExtremeShp.split(".")[0]+"_"+Level+"_Back.shp"
            ExtPatchesShp = ExtremeShp.split(".")[0]+"_"+Level+".shp"
                
            # launch polyline shapefile writer
            self.WriteLinesShp("ExtFrontLines_"+Level, ExtFrontShp)
            self.WriteLinesShp("ExtBackLines_"+Level, ExtBackShp)
            
            # launch polygon patches shapefile writer
            self.WritePatchesShp("ExtFrontLines_"+Level, "ExtBackLines_"+Level, ExtPatchesShp)
    
    def WriteErodedAreaShp(self, ErosionShp, StartYear=2020, Year=2100,Smooth=True):
        
        """
        Writes future shorelines to polygon patches

        MDH, Jan 2020

        """
        
        # print action to screen
        #print("Coast.WriteErodedAreaShp: Writing predicted erosion area to polygon file")
        
        # retrieve future shorelines
        self.GetFutureShoreLines()

        # get lists of lines for year of prediction and most recent shoreline position
        Indices = [i for i, Line in enumerate(self.FutureShoreLines) if Line.Year == Year]
        self.WriteFutureLines = [self.FutureShoreLines[i] for i in Indices]
        Indices = [i for i, Line in enumerate(self.FutureShoreLines) if Line.Year == StartYear]
        self.WriteRecentLines = [self.FutureShoreLines[i] for i in Indices]
        
        # set up files to write
        ErosionFrontShp = ErosionShp.split(".")[0]+"_temp.shp"
        ErosionBackShp = ErosionShp.split(".")[0]+"_temp2.shp"

        # write lines then patches
        self.WriteLinesShp("WriteFutureLines", ErosionBackShp, Smooth=True)
        self.WriteLinesShp("WriteRecentLines", ErosionFrontShp, Smooth=True)
        self.WritePatchesShp("WriteFutureLines", "WriteRecentLines", ErosionShp, Smooth=True)

    def WriteErosionProximityShp(self, ProximityShp, BufferDistance=10., Year=2100, Smooth=True):

        """
        Writes Erosion Proximity polygon patches for a given decade

        MDH, Feb, 2021
        
        """

        # retrieve future shorelines
        self.GetFutureShoreLines()
        Lines = self.GetFutureShoreLinesProximity(BufferDistance)

        # get lists of lines for year of prediction and most recent shoreline position
        Indices = [i for i, Line in enumerate(self.FutureShoreLines) if Line.Year == Year]
        self.WriteFutureLines = [self.FutureShoreLines[i] for i in Indices]
        Indices = [i for i, Line in enumerate(Lines) if Line.Year == Year]
        self.WriteBufferLines = [Lines[i] for i in Indices]
        
        # set up files to write
        ErosionFutureShp = ProximityShp.split(".")[0]+"_temp.shp"
        ErosionBufferShp = ProximityShp.split(".")[0]+"_temp2.shp"

        # write lines then patches
        self.WriteLinesShp("WriteFutureLines", ErosionFutureShp, Smooth=True)
        self.WriteLinesShp("WriteBufferLines", ErosionBufferShp, Smooth=True)
        self.WritePatchesShp("WriteFutureLines", "WriteBufferLines", ProximityShp, Smooth=True)
    
    
    def WriteFutureShorelinesShp(self, FutureShoreLinesShp, Smooth=True):

        """
        Writes the contents of a list of future shoreline objects to polyline shape file

        MDH, June 2019

        Added functionality to write spline of future line prediction to get smoothed
        shape that is faithful to predictions

        MDH, Jan 2020

        """

        # extract future shoreline positions from transects
        self.GetFutureShoreLines()

        # print action to screen
        print("Coast.WriteFutureShorelinesShp: Writing future MHWS line objects to polyline shapefiles")

        # open new shapefile        
        WL = shapefile.Writer(FutureShoreLinesShp,shapeType=shapefile.POLYLINE)
       
        # Create Fields
        self.Fields = [('DeletionFlag','C',1,0),['Cell','C', 2, 0], ['SubCell','C', 2, 0], ['Line_ID', 'C', 20, 0],['Year','N', 4, 0],['Method','C', 5, 0]]
        WL.fields = self.Fields[1:] 

        for Line in self.FutureShoreLines:
            
            if Smooth:
                Line.SmoothLine(WindowSize=11)

            # Find Loops
            Line.MakeSimple()
                
            # get line node positions
            X, Y = Line.get_XY()

            if Smooth and len(X) > 5:

                XSmooth = X[1:-1]
                YSmooth = Y[1:-1]
                # calculate distance
                Dist = np.zeros(XSmooth.shape)
                Dist[1:] = np.sqrt((XSmooth[1:] - XSmooth[:-1])**2 + (YSmooth[1:] - YSmooth[:-1])**2)
                Dist = np.cumsum(Dist)
                
                # build a spline representation of the line
                Spline, u = splprep([XSmooth, YSmooth], u=Dist, s=0)

                # resample it at smaller distance intervals
                Interp_Dist = np.arange(0, Dist[-1], 1.)
                XSmooth, YSmooth = splev(Interp_Dist, Spline)

                XSmooth = np.insert(XSmooth,0,X[0])
                YSmooth = np.insert(YSmooth,0,Y[0])
                X = np.append(XSmooth,X[-1])
                Y = np.append(YSmooth,Y[-1])
                
#            # check for loops here and remove?
#            TempLine = LineString(zip(X,Y))
#            
#            if not TempLine.is_simple:
#                
#                print("Spline line is not simple")
#                
#                X, Y = TempLine.coords.xy
#                X = np.array(X)
#                Y = np.array(Y)
#
#                #"Union" method will split self-intersection linestring.
#                Result = TempLine.union(Point(X[0],Y[0]))
#                KeepBool = np.zeros(len(X),dtype=bool)
#                Index = 0
#                NewIndex = 0
#
#                for L in Result:
#
#                    x,y = L.coords.xy
#                    Index = NewIndex
#                    NewIndex = Index+len(x)
#
#                    if not Point(L.coords[0]).distance(Point(L.coords[-1])) == 0:
#                        KeepBool[Index:NewIndex-1] = 1
#                        
#                # get line node positions
#                KeepBool[-1] = True
#                X = X[KeepBool]
#                Y = Y[KeepBool]
#                TempLine = LineString(zip(X,Y))


            # convert to list for writing to shapefile
            WriteLine = [np.column_stack([X,Y]).tolist()]
            
            # generate record
            if self.Method == None:
                import pdb
                pdb.set_trace()
             
            Record = [str(Line.Cell), str(Line.SubCell),str(Line.ID),str(Line.Year), str(self.Method)]

            # write line and record
            WL.line(WriteLine)
            WL.record(*Record) ####### ISSUE WITH RECORDS NEEDS FIXING ########
        
        # close the shapefiles and clean up
        WL.close()
            
        # create the projection file    
        f = open(FutureShoreLinesShp.rstrip("shp")+"prj","w")
        f.write(self.Projection)
        f.close()

    def WriteFutureUncertaintyShp(self, UncertaintyShp, Year=2100):

        """
        Writes future shoreline uncertainty estimates to a polygon
        for a particular year

        MDH, March 2020

        """
        
        self.FutureMinUncertainty = []
        self.FutureMaxUncertainty = []
        
        # predict and extract future shoreline positions from transects
        self.PredictFutureShorelinesUncertainty(Year)
        self.GetFutureShorelineUncertainty(Year)

        # print action to screen
        print("Coast.WriteFutureUncertaintyShp: Writing uncertainty area to polygon file")
        
        # set up files to write
        FutureMinShp = UncertaintyShp.split(".")[0]+"_Min.shp"
        FutureMaxShp = UncertaintyShp.split(".")[0]+"_Max.shp"

        # spleen for smooth line?
        
        # write lines then patches
        self.WriteLinesShp("FutureMinUncertainty", FutureMinShp)
        self.WriteLinesShp("FutureMaxUncertainty", FutureMaxShp)
        self.WritePatchesShp("FutureMinUncertainty", "FutureMaxUncertainty", UncertaintyShp)

    def WriteFutureErrorShp(self, ErrorShp, Year=2100):

        """
        Writes future shoreline error estimates to a polygon
        for a particular year

        MDH, October 2020

        """

        # predict and extract future shoreline positions from transects
        self.PredictFutureShorelinesError(Year)
        self.GetFutureShorelineError(Year)

        # print action to screen
        print("Coast.WriteFutureErrorShp: Writing uncertainty area to polygon file %d", Year)

        # set up files to write
        FutureMinShp = ErrorShp.split(".")[0]+"_Min.shp"
        FutureMaxShp = ErrorShp.split(".")[0]+"_Max.shp"

        # spleen for smooth line?
        
        # write lines then patches
        self.WriteLinesShp("FutureMinError", FutureMinShp)
        self.WriteLinesShp("FutureMaxError", FutureMaxShp)
        self.WritePatchesShp("FutureMinError", "FutureMaxError", ErrorShp)

    def WriteFutureVegEdgeShp(self, FutureVegEdgeShp, Smooth=False):

        """
        Writes the contents of a list of future veg edge objects to polyline shape file

        MDH, Feb 2020

        Added functionality to write spline of future line prediction to get smoothed
        shape that is faithful to predictions

        MDH, Jan 2020

        """

        if len(self.FutureVegEdgeLines) == 0:
            self.GetFutureVegEdgeLines()

        # print action to screen
        print("Coast.WriteFutureVegEdgeShp: Writing future veg edge line objects to polyline shapefiles")

        # open new shapefile        
        WL = shapefile.Writer(FutureVegEdgeShp,shapeType=shapefile.POLYLINE)
       
        # Create Fields
        self.Fields = [('DeletionFlag','C',1,0),['Line_ID', 'C', 20, 0],['Year','N', 4, 0]]
        WL.fields = self.Fields[1:] 

        for Line in self.FutureVegEdgeLines:
            
            if Smooth:
                Line.SplineLine()

            # get line node positions
            X, Y = Line.get_XY()

            
            # convert to list for writing to shapefile
            WriteLine = [np.column_stack([X, Y]).tolist()]
            
            # generate record
            Record = [str(Line.ID),str(Line.Year)]

            # write line and record
            WL.line(WriteLine)
            WL.record(*Record) 
        
        # close the shapefiles and clean up
        WL.close()
            
        # create the projection file    
        f = open(FutureVegEdgeShp.rstrip("shp")+"prj","w")
        f.write(self.Projection)
        f.close()

    def WriteFutureShorelineSegmentsShp(self, FutureShoreLinesShp):

        """
        Writes the contents of a list of future shoreline objects to polyline shape file
        organised into individual segments with attributes

        PLAN FOR DOING THIS ON A SPLINE
        loop through future shoreline lines and get the spline
        loop through transects and intersect with historic shoreline position and then get nearerst nodes from spline??

        MDH, January 2020

        """

        # print action to screen
        print("Coast.WriteFutureShorelineSegmentsShp: Writing future MHWS line objects to polyline shapefiles")
        
        
        self.GetFutureShoreLines()
        
        # open new shapefile        
        WL = shapefile.Writer(FutureShoreLinesShp,shapeType=shapefile.POLYLINE)
       
        # Create Fields
        self.Fields = [('DeletionFlag','C', 1, 0), ['Line_ID', 'C', 3, 0], ['Transect_ID','C', 5, 0],
                        ['Cell','C', 2, 0], ['SubCell','C', 2, 0], ['CMU','C', 3, 0],
                        ['Year','N', 4, 0], ['Distance','N', 6, 2], ['Rate','N', 4, 4]]
        WL.fields = self.Fields[1:] 

        # Loop through prediction years
        for i, Line in enumerate(self.FutureShoreLines):
            
            # keep track of no of coastal segments for IDs
            FutureCount = 0
            
            # get line node positions
            X, Y = Line.get_XY()
            
            # get nodes for spline
            Interp_X = X[1:-1]
            Interp_Y = Y[1:-1]

            # calculate distance
            Dist = np.zeros(Interp_X.shape)
            Dist[1:] = np.sqrt((Interp_X[1:] - Interp_X[:-1])**2 + (Interp_Y[1:] - Interp_Y[:-1])**2)
            Dist = np.cumsum(Dist)
            
            # build a spline representation of the line
            K = 3 # by default

            if len(Interp_X) < 2:
                continue

            elif len(Interp_X) < 4:
                K = len(Interp_X)-1

            Spline, u = splprep([Interp_X, Interp_Y], u=Dist, s=0, k=K)

            # resample it at smaller distance intervals
            Interp_Dist = np.arange(0, Dist[-1], 1.)
            Interp_X, Interp_Y = splev(Interp_Dist, Spline)

            # add start and end nodes back on
            Interp_X = np.insert(Interp_X, 0, (X[0]+X[1])/2.)
            Interp_Y = np.insert(Interp_Y, 0, (Y[0]+Y[1])/2.)
            Interp_X = np.append(Interp_X, (X[-1]+X[-2])/2.)
            Interp_Y = np.append(Interp_Y, (Y[-1]+Y[-2])/2.)
            
            # convert to a linestring
            SplineLine = LineString((tuple(zip(Interp_X,Interp_Y))))
            SplinePoints = MultiPoint((tuple(zip(Interp_X,Interp_Y))))
            
            # loop through transects and get contiguous future prediction lines
            for CoastLine in self.CoastLines:
                
                # set up empty list of intersection indices with spline
                TransectsList = []
                IntersectionIndices = []

                # get a list of nearest indices on interpolated lines
                for j, Transect in enumerate(CoastLine.Transects):
                    
                    # intersect extended transect with spline to find index
                    X1 = Transect.EndNode.X + 1000 * np.sin( np.radians( Transect.Orientation ) )
                    Y1 = Transect.EndNode.Y + 1000 * np.cos( np.radians( Transect.Orientation ) )
                    
                    TransectLine = LineString(((Transect.StartNode.X,Transect.StartNode.Y),(X1,Y1)))
                    Intersection = TransectLine.intersection(SplineLine)

                    # catch no intersections and flag for deletion?
                    if Intersection.geom_type == "GeometryCollection":
                        continue

                    # check there arent multiple intersections, if there are just get the nearest
                    elif Intersection.geom_type == "MultiPoint":
                        Intersection = Intersection[0]

                    Distances = [SplinePoint.distance(Intersection) for SplinePoint in SplinePoints]
                    TransectsList.append(j)
                    IntersectionIndices.append(Distances.index(min(Distances)))
                
                # loop across transects again
                for j in range(0, len(TransectsList)):
                    
                    if j == 0:
                        StartIndex = IntersectionIndices[j]
                    else:
                        StartIndex = EndIndex
                    
                    if j == len(TransectsList)-1:
                        EndIndex = IntersectionIndices[j]
                    else:
                        EndIndex = int((IntersectionIndices[j+1]+IntersectionIndices[j])/2)
                    
                    if StartIndex == EndIndex:
                        continue

                    # initiate dummy lists for nodes
                    X = Interp_X[StartIndex:EndIndex]
                    Y = Interp_Y[StartIndex:EndIndex]
                    
                    # get shoreline position in the future
                    Transect = CoastLine.Transects[TransectsList[j]]
                    FutureNode = Transect.get_FuturePosition(Line.Year)

                    # get line node positions
                    WriteLine = [np.column_stack([X,Y]).tolist()]
            
                    # calculate additional attributes
                    RecentNode = Transect.get_RecentPosition()
                    
                    if not FutureNode:
                        continue

                    if not FutureNode:
                        continue

                    Distance = np.sqrt((FutureNode.X-RecentNode.X)**2. + (FutureNode.Y-RecentNode.Y)**2.)
                    Rate = Transect.get_FutureShorelineRate(Line.Year)

                    # generate record (strs?)
                    Record = [str(CoastLine.ID), str(Transect.ID), str(Transect.Cell), str(Transect.SubCell),
                    str(Transect.CMU), str(Line.Year), str(Distance), str(Rate)]

                    # write line and record
                    WL.line(WriteLine)
                    WL.record(*Record) ####### ISSUE WITH RECORDS NEEDS FIXING ########

        # close the shapefiles and clean up
        WL.close()
            
        # create the projection file 
        f = open(FutureShoreLinesShp.rstrip("shp")+"prj","w")
        f.write(self.Projection)
        f.close()

    
    def WriteLinesShp(self, DictionaryKey, CoastShp, Smooth=False):
        
        """
        Writes the contents of a list of line objects to polyline shape file
        List of line objects is part of the Coast object and identified by 
        the dictionary key

        Need to add optional conditional statement?

        MDH, June 2019

        """

        # print action to screen
        #print("Coast.WriteLinesShp: Writing a list of lines to a polyline shapefile")

        # open new shapefile        
        WL = shapefile.Writer(CoastShp,shapeType=shapefile.POLYLINE)
       
        # Create Fields
        self.Fields = [('DeletionFlag','C',1,0),['Line_ID', 'C', 3, 0],['Method', 'C', 5, 0]]
        WL.fields = self.Fields[1:] 

        for Line in self.__dict__[DictionaryKey]:
            
            if Smooth:
                Line.SmoothLine(WindowSize=11)

            # Find Loops
            Line.MakeSimple()
                
            # get line node positions
            X, Y = Line.get_XY()

            if Smooth and len(X) > 5:

                XSmooth = X[1:-1]
                YSmooth = Y[1:-1]
                # calculate distance
                Dist = np.zeros(XSmooth.shape)
                Dist[1:] = np.sqrt((XSmooth[1:] - XSmooth[:-1])**2 + (YSmooth[1:] - YSmooth[:-1])**2)
                Dist = np.cumsum(Dist)
                
                # build a spline representation of the line
                Spline, u = splprep([XSmooth, YSmooth], u=Dist, s=0)

                # resample it at smaller distance intervals
                Interp_Dist = np.arange(0, Dist[-1], 1.)
                XSmooth, YSmooth = splev(Interp_Dist, Spline)

                XSmooth = np.insert(XSmooth,0,X[0])
                YSmooth = np.insert(YSmooth,0,Y[0])
                X = np.append(XSmooth,X[-1])
                Y = np.append(YSmooth,Y[-1])

            # get line node positions
            WriteLine = [np.column_stack([X,Y]).tolist()]
            
            # generate record
            Record = [str(Line.ID),str(self.Method)]

            # write line and record
            WL.line(WriteLine)
            WL.record(*Record) ####### ISSUE WITH RECORDS NEEDS FIXING ########
        
        # close the shapefiles and clean up
        WL.close()
            
        # create the projection file    
        f = open(CoastShp.rstrip("shp")+"prj","w")
        f.write(self.Projection)
        f.close()
    
    def WritePatchesShp(self, DictionaryKey1, DictionaryKey2, PatchShp, Smooth=False):

        """

        Writes polygon patches between two lines to a polygon shapefile

        Dictionary Key refers

        MDH, June 2019

        """

        # print action to screen
        #print("Coast.WritePatchesShp: Writing patch between two lines to a polygon shapefile")

        if len(self.__dict__[DictionaryKey1]) == 0:
            print("Coast.WritePatchesShp (Error): Trying to write from empty list of lines", DictionaryKey1, DictionaryKey2)
            
        # open new shapefile        
        WS = shapefile.Writer(PatchShp,shapeType=shapefile.POLYGON)
       
        # Create Fields
        self.Fields = [('DeletionFlag','C',1,0),['Poly_ID', 'C', 3, 0],['Method', 'C', 5, 0]]
        WS.fields = self.Fields[1:] 

        for Line1, Line2 in zip(self.__dict__[DictionaryKey1],self.__dict__[DictionaryKey2]):
            
            # get line node positions
            X1, Y1 = Line1.get_XY()

            if Smooth and len(X1) > 5:

                XSmooth = X1[1:-1]
                YSmooth = Y1[1:-1]
                
                # calculate distance
                Dist = np.zeros(XSmooth.shape)
                Dist[1:] = np.sqrt((XSmooth[1:] - XSmooth[:-1])**2 + (YSmooth[1:] - YSmooth[:-1])**2)
                Dist = np.cumsum(Dist)
                
                # build a spline representation of the line
                Spline, u = splprep([XSmooth, YSmooth], u=Dist, s=0)

                # resample it at smaller distance intervals
                Interp_Dist = np.arange(0, Dist[-1], 1.)
                XSmooth, YSmooth = splev(Interp_Dist, Spline)

                XSmooth = np.insert(XSmooth,0,X1[0])
                YSmooth = np.insert(YSmooth,0,Y1[0])
                X1 = np.append(XSmooth,X1[-1])
                Y1 = np.append(YSmooth,Y1[-1])

            # get line node positions
            X2, Y2 = Line2.get_XY()

            if Smooth and len(X2) > 5:

                XSmooth = X2[1:-1]
                YSmooth = Y2[1:-1]
                # calculate distance
                Dist = np.zeros(XSmooth.shape)
                Dist[1:] = np.sqrt((XSmooth[1:] - XSmooth[:-1])**2 + (YSmooth[1:] - YSmooth[:-1])**2)
                Dist = np.cumsum(Dist)
                
                # build a spline representation of the line
                Spline, u = splprep([XSmooth, YSmooth], u=Dist, s=0)

                # resample it at smaller distance intervals
                Interp_Dist = np.arange(0, Dist[-1], 1.)
                XSmooth, YSmooth = splev(Interp_Dist, Spline)

                XSmooth = np.insert(XSmooth,0,X2[0])
                YSmooth = np.insert(YSmooth,0,Y2[0])
                X2 = np.append(XSmooth,X2[-1])
                Y2 = np.append(YSmooth,Y2[-1])

            # combine, reversing the order of the second line to make a patch
            X = np.concatenate((X1,X2[::-1]))
            Y = np.concatenate((Y1,Y2[::-1]))
            WritePoly = [np.column_stack([X,Y]).tolist()]
            
            # generate record
            Record = [str(Line1.ID), str(self.Method)]

            # write line and record
            WS.poly(WritePoly)
            WS.record(*Record) 
        
        # close the shapefiles and clean up
        WS.close()
            
        # create the projection file    
        f = open(PatchShp.rstrip("shp")+"prj","w")
        f.write(self.Projection)
        f.close()

    def WritePointsShp(self, PointsShp):
        """
        Function to write transect points to a point shape file

        MDH, June 2019
        
        """

        # print action to screen
        print("Coast.WritePointsShp: Writing points to a shapefile")

        WP = shapefile.Writer(PointsShp, shapeType=shapefile.POINT)
        
        # Create Fields
        Fields = [('DeletionFlag','C',1,0),['Line_ID', 'C', 3, 0],['Transect_ID', 'C', 5, 0]] #['Segment_ID','C', 3, 0], might add 
        WP.fields = Fields[1:]

        for Line in self.CoastLines:
            for Transect in Line.Transects:
                
                # Create the record
                Record = [str(Line.ID), str(Transect.ID)]

                # add the line and record
                WP.point(Transect.CoastNode.X, Transect.CoastNode.Y)
                WP.record(*Record)

        # close the shapefiles and clean up
        WP.close()
            
        # create the projection file    
        f = open(PointsShp.rstrip("shp")+"prj","w")
        f.write(self.Projection)
        f.close()

    def WriteTransectsShp(self, TransectsShp):

        """
        Writes the transects of a Coast object to polyline shape file

        builds a large attribute table with all transect properties

        MDH, June 2019

        """

        # print action to screen
        print("Coast.WriteTransectsShp: Writing coastal transects and attributes to a shapefile")

        # open new shapefile        
        WL = shapefile.Writer(TransectsShp,shapeType=shapefile.POLYLINE)
        
        # Check length of extreme water levels
        if len(self.ExtremeWaterLevels) != 3:
            self.ExtremeWaterLevels = [[],[],[]]

        # Create Fields
        Fields = [('DeletionFlag','C',1,0), ['LineID', 'C', 3, 0], ['TransectID', 'C', 5, 0], 
        ['Cliff_H','N', 5, 2],['Cliff_S','N', 5, 2],
        ['Rocky','N', 2, 1], 
        ['Bar_FH','N', 5, 2], ['Bar_FS','N', 5, 2],
        ['Bar_BH','N', 5, 2], ['Bar_BS','N', 5, 2],
        ['Bar_ToeW','N', 6, 2], ['Bar_TopW','N', 6, 2],
        ['Bar_Volume','N', 7, 2], ['Crest_Elev','N', 5, 2], 
        ['ST_W_low','N', 6, 2], ['ST_V_low','N', 7, 2],
        ['ST_W_med','N', 6, 2], ['ST_V_med','N', 7, 2],
        ['ST_W_high','N', 6, 2], ['ST_V_high','N', 7, 2],
        ['LT_W_low','N', 6, 2], ['LT_V_low','N', 7, 2],
        ['LT_W_med','N', 6, 2], ['LT_V_med','N', 7, 2],
        ['LT_W_high','N', 6, 2], ['LT_V_high','N', 7, 2]]
        
        WL.fields = Fields[1:]

        
        for Line in self.CoastLines:
            for Transect in Line.Transects:

                # get transect node positions
                X, Y = Transect.get_XY()
                
                WriteTransect = [np.column_stack([X,Y]).tolist()]

                # Create the record this could become a function in transect object...
                Record = [int(Line.ID), int(Transect.ID), Transect.CliffHeight, Transect.CliffSlope, 
                            Transect.Rocky,
                            Transect.FrontHeight, Transect.FrontSlope, 
                            Transect.BackHeight, Transect.BackSlope,
                            Transect.ToeWidth, Transect.TopWidth,
                            Transect.BarrierVolume, Transect.CrestElevation,
                            Transect.ExtremeWidths[0], Transect.ExtremeVolumes[0],
                            Transect.ExtremeWidths[1], Transect.ExtremeVolumes[1],
                            Transect.ExtremeWidths[2], Transect.ExtremeVolumes[2],
                            Transect.ExtremeTotalWidths[0], Transect.ExtremeTotalVolumes[0],
                            Transect.ExtremeTotalWidths[1], Transect.ExtremeTotalVolumes[1],
                            Transect.ExtremeTotalWidths[2], Transect.ExtremeTotalVolumes[2]]

                # write transect and record
                WL.line(WriteTransect)
                try:
                    WL.record(*Record) 
                except:
                    print(Transect.ID)
                    print(Record)
                    #print(Transect.ExtremeWidths)
                    sys.exit()
                
        
        # close the shapefiles and clean up
        WL.close()
            
        # create the projection file    
        f = open(TransectsShp.rstrip("shp")+"prj","w")
        f.write(self.Projection)
        f.close()
    
    def WriteSimpleTransectsShp(self, TransectsShp):

        """
        Writes the transects of a Coast object to polyline shape file

        simplified attribute table for use with only shoreline intersects.

        FM Sept 2022

        """

        # print action to screen
        print("Coast.WriteSimpleTransectsShp: Writing coastal transects and attributes to a shapefile")

        # open new shapefile        
        WL = shapefile.Writer(TransectsShp,shapeType=shapefile.POLYLINE)
        
        # Check length of extreme water levels
        if len(self.ExtremeWaterLevels) != 3:
            self.ExtremeWaterLevels = [[],[],[]]

        # Create Fields
        Fields = [('DeletionFlag','C',1,0), ['LineID', 'N', 3, 0], ['TransectID', 'N', 5, 0]]

        
        WL.fields = Fields[1:]

        
        for Line in self.CoastLines:
            for Transect in Line.Transects:

                # get transect node positions
                X, Y = Transect.get_XY()
                
                WriteTransect = [np.column_stack([X,Y]).tolist()]

                # Create the record this could become a function in transect object...
                Record = [int(Line.ID), int(Transect.ID)]

                # write transect and record
                WL.line(WriteTransect)
                try:
                    WL.record(*Record) 
                except:
                    print(Transect.ID)
                    print(Record)
                    #print(Transect.ExtremeWidths)
                    sys.exit()
                
        
        # close the shapefiles and clean up
        WL.close()
            
        # create the projection file    
        f = open(TransectsShp.rstrip("shp")+"prj","w")
        f.write(self.Projection)
        f.close()
    
    def WriteFutureTransectsShp(self, TransectsShp):

        """
        Writes the transects of a Coast object to polyline shape file

        builds a large attribute table with all future shoreline info

        MDH, Sept 2020

        """

        # print action to screen
        print("Coast.WriteFutureTransectsShp: Writing coastal transects and attributes to a shapefile")

        # open new shapefile        
        WL = shapefile.Writer(TransectsShp,shapeType=shapefile.POLYLINE)
        
        # Check length of extreme water levels
        if len(self.ExtremeWaterLevels) != 3:
            self.ExtremeWaterLevels = [[],[],[]]

        # Create Fields
        Fields = [('DeletionFlag','C',1,0), 
        ['Cell', 'C', 3, 0], ['SubCell', 'C', 3, 0], ['CMU','C', 20, 0],
        ['LineID', 'N', 3, 0], ['TransectID', 'N', 5, 0], ['Hist_Rate','N', 4, 4],
        ['CalibYr','N', 4, 0], ['BaseLYr','N', 4, 0], ['BaseLSrc','C', 50, 0], 
        ['Extrap2050','N', 6, 4], ['Extrap2100','N', 6, 4], ['FirstEYr','N',4, 4],
        ['Dist_2030', 'N', 6, 4], ['Rate_2030', 'N', 6, 4], 
        ['Dist_2040', 'N', 6, 4], ['Rate_2040', 'N', 6, 4], 
        ['Dist_2050', 'N', 6, 4], ['Rate_2050', 'N', 6, 4], 
        ['Dist_2060', 'N', 6, 4], ['Rate_2060', 'N', 6, 4], 
        ['Dist_2070', 'N', 6, 4], ['Rate_2070', 'N', 6, 4], 
        ['Dist_2080', 'N', 6, 4], ['Rate_2080', 'N', 6, 4], 
        ['Dist_2090', 'N', 6, 4], ['Rate_2090', 'N', 6, 4], 
        ['Dist_2100', 'N', 6, 4], ['Rate_2100', 'N', 6, 4], 
        ['RCP85_2100', 'N', 4, 3],
        ['DC1_SvEn_B','N', 4, 0], ['DC1_SvEn_C','N', 4, 0], 
        ['DC1_DistV','N', 6, 4], ['DC1_RateBC','N', 6, 4],
        ['OS_2020_Yr','N',4,0], ['Method','C', 5, 0]
        ]
        
        WL.fields = Fields[1:]

        
        for Line in self.CoastLines:
            for Transect in Line.Transects:

                if Transect.Future:
                    # get transect node positions
                    X, Y = Transect.get_XY()
                    
                    WriteTransect = [np.column_stack([X,Y]).tolist()]
                    
                    if not Transect.DC1:
                        Transect.DC1 = ["","","",""]
                    else:
                        Transect.DC1[3] = Transect.DC1[2]/(Transect.DC1[1]-Transect.DC1[0])
                    
                    # Create the record this could become a function in transect object...
                    Record = [str(self.Cell), str(self.SubCell), str(self.CMU), str(Line.ID), str(Transect.ID),
                                Transect.ChangeRate, 
                                Transect.CalibrationYear, Transect.HistoricShorelinesYears[-1], Transect.HistoricShorelinesSources[-1], 
                                Transect.get_ExtrapDistance(2050), Transect.get_ExtrapDistance(2100), Transect.get_FirstFutureErosionYear(),
                                Transect.get_FuturePositionChange(2020, 2030), Transect.get_FutureRate(2020, 2030),
                                Transect.get_FuturePositionChange(2030, 2040), Transect.get_FutureRate(2030, 2040),
                                Transect.get_FuturePositionChange(2040, 2050), Transect.get_FutureRate(2040, 2050),
                                Transect.get_FuturePositionChange(2050, 2060), Transect.get_FutureRate(2050, 2060),
                                Transect.get_FuturePositionChange(2060, 2070), Transect.get_FutureRate(2060, 2070),
                                Transect.get_FuturePositionChange(2070, 2080), Transect.get_FutureRate(2070, 2080),
                                Transect.get_FuturePositionChange(2080, 2090), Transect.get_FutureRate(2080, 2090),
                                Transect.get_FuturePositionChange(2090, 2100), Transect.get_FutureRate(2090, 2100),
                                Transect.FutureSeaLevels[-1],
                                
                                Transect.DC1[0], Transect.DC1[1], Transect.DC1[2], Transect.DC1[3],
                                Transect.OSYear, self.Method]
                    
                                
    
                    # write transect and record
                    WL.line(WriteTransect)
                    WL.record(*Record) 
                                    
        # close the shapefiles and clean up
        WL.close()
            
        # create the projection file    
        f = open(TransectsShp.rstrip("shp")+"prj","w")
        f.write(self.Projection)
        f.close()

    def WriteCrestLinesShp(self, CrestLineShp):

        """
        Writes the crest line of barriers to shape file

        MDH, July 2019

        """

        print("Coast.WriteCrestLinesShp: Writing barrier crest locations to polyline shapefile")
        
        if len(self.CrestLines) == 0:
            self.GetBarrierLines()

        # launch polyline shapefile writer
        self.WriteLinesShp("CrestLines", CrestLineShp)

    def WriteCrestPointsShp(self, CrestPointsShp):

        """
        Writes the crest lines points of barriers to shape file

        builds a large attribute table with all transect properties

        MDH, July 2019

        """

        print("Coast.WriteCrestPointsShp: Writing barrier crest locations to point shapefile")

        # open new shapefile        
        WP = shapefile.Writer(CrestPointsShp,shapeType=shapefile.POINTZ)
        
        # Check length of extreme water levels
        if len(self.ExtremeWaterLevels) != 3:
            print("Coast.WriteTransectsShp (Error): No extreme water levels info to write to attributes")
            self.ExtremeWaterLevels = [[],[],[]]

        # Create Fields
        Fields = [('DeletionFlag','C',1,0), ['LineID', 'C', 3, 0], ['TransectID', 'C', 5, 0], 
        ['Cliff_H','N', 5, 2],['Cliff_S','N', 5, 2],
        ['Rocky','N', 2, 1], 
        ['Bar_FH','N', 5, 2], ['Bar_FS','N', 5, 2],
        ['Bar_BH','N', 5, 2], ['Bar_BS','N', 5, 2],
        ['Bar_ToeW','N', 6, 2], ['Bar_TopW','N', 6, 2],
        ['Bar_Volume','N', 7, 2], ['Crest_Elev','N', 5, 2], 
        ['ST_W_low','N', 6, 2], ['ST_V_low','N', 7, 2],
        ['ST_W_med','N', 6, 2], ['ST_V_med','N', 7, 2],
        ['ST_W_high','N', 6, 2], ['ST_V_high','N', 7, 2],
        ['LT_W_low','N', 6, 2], ['LT_V_low','N', 7, 2],
        ['LT_W_med','N', 6, 2], ['LT_V_med','N', 7, 2],
        ['LT_W_high','N', 6, 2], ['LT_V_high','N', 7, 2]]

        WP.fields = Fields[1:]

        for Line in self.CoastLines:
            for Transect in Line.Transects:

                # get crest position
                try:
                    X, Y, Z = Transect.get_CrestPosition()
                except:
                    continue
                
                # Create the record
                Record = [str(Line.ID), str(Transect.ID), Transect.CliffHeight, Transect.CliffSlope, 
                            Transect.Rocky,
                            Transect.FrontHeight, Transect.FrontSlope, 
                            Transect.BackHeight, Transect.BackSlope,
                            Transect.ToeWidth, Transect.TopWidth,
                            Transect.BarrierVolume, Transect.CrestElevation,
                            Transect.ExtremeWidths[0], Transect.ExtremeVolumes[0],
                            Transect.ExtremeWidths[1], Transect.ExtremeVolumes[1],
                            Transect.ExtremeWidths[2], Transect.ExtremeVolumes[2],
                            Transect.ExtremeTotalWidths[0], Transect.ExtremeTotalVolumes[0],
                            Transect.ExtremeTotalWidths[1], Transect.ExtremeTotalVolumes[1],
                            Transect.ExtremeTotalWidths[2], Transect.ExtremeTotalVolumes[2]]


                # write transect and record
                WP.pointz(X, Y, Z)
                WP.record(*Record) 
        
        # close the shapefiles and clean up
        WP.close()
            
        # create the projection file    
        f = open(CrestPointsShp.rstrip("shp")+"prj","w")
        f.write(self.Projection)
        f.close()
  
    def WriteFrontPointsShp(self, FrontPointsShp):

        """
        Writes the front lines points of barriers to shape file

        builds a large attribute table with all transect properties

        MDH, July 2019

        """

        print("Coast.WriteFrontPointsShp: Writing barrier front locations to point shapefile")

        # open new shapefile        
        WP = shapefile.Writer(FrontPointsShp,shapeType=shapefile.POINTZ)
        
        # Check length of extreme water levels
        if len(self.ExtremeWaterLevels) != 3:
            print("Coast.WriteTransectsShp (Error): No extreme water levels info to write to attributes")
            self.ExtremeWaterLevels = [[],[],[]]

        # Create Fields
        Fields = [('DeletionFlag','C',1,0), ['LineID', 'C', 3, 0], ['TransectID', 'C', 5, 0], 
        ['Cliff_H','N', 5, 2],['Cliff_S','N', 5, 2],
        ['Rocky','N', 2, 1], 
        ['Bar_FH','N', 5, 2], ['Bar_FS','N', 5, 2],
        ['Bar_BH','N', 5, 2], ['Bar_BS','N', 5, 2],
        ['Bar_ToeW','N', 6, 2], ['Bar_TopW','N', 6, 2],
        ['Bar_Volume','N', 7, 2], ['Crest_Elev','N', 5, 2], 
        ['ST_W_low','N', 6, 2], ['ST_V_low','N', 7, 2],
        ['ST_W_med','N', 6, 2], ['ST_V_med','N', 7, 2],
        ['ST_W_high','N', 6, 2], ['ST_V_high','N', 7, 2],
        ['LT_W_low','N', 6, 2], ['LT_V_low','N', 7, 2],
        ['LT_W_med','N', 6, 2], ['LT_V_med','N', 7, 2],
        ['LT_W_high','N', 6, 2], ['LT_V_high','N', 7, 2]]

        WP.fields = Fields[1:]

        for Line in self.CoastLines:
            for Transect in Line.Transects:


                # get crest position
                try:
                    X, Y, Z = Transect.get_FrontPosition()
                except:
                    continue
                
                # Create the record
                Record = [str(Line.ID), str(Transect.ID), Transect.CliffHeight, Transect.CliffSlope, 
                            Transect.Rocky,
                            Transect.FrontHeight, Transect.FrontSlope, 
                            Transect.BackHeight, Transect.BackSlope,
                            Transect.ToeWidth, Transect.TopWidth,
                            Transect.BarrierVolume, Transect.CrestElevation,
                            Transect.ExtremeWidths[0], Transect.ExtremeVolumes[0],
                            Transect.ExtremeWidths[1], Transect.ExtremeVolumes[1],
                            Transect.ExtremeWidths[2], Transect.ExtremeVolumes[2],
                            Transect.ExtremeTotalWidths[0], Transect.ExtremeTotalVolumes[0],
                            Transect.ExtremeTotalWidths[1], Transect.ExtremeTotalVolumes[1],
                            Transect.ExtremeTotalWidths[2], Transect.ExtremeTotalVolumes[2]]

                # write transect and record
                WP.pointz(X, Y, Z)
                WP.record(*Record) 
        
        # close the shapefiles and clean up
        WP.close()
            
        # create the projection file    
        f = open(FrontPointsShp.rstrip("shp")+"prj","w")
        f.write(self.Projection)
        f.close()

    def WriteTransectsCSV(self,Folder=os.getcwd()):

        """

        Writes all transects to csv files in the folder specified or
        by default in the current working directory

        args: Folder in which to put files

        MDH, July 2019

        """
        
        print("Coast.WriteTransectsCSV: Writing all topographic transects to csv files")
        
        # Track progress
        NoTransects = np.sum([Line.NoTransects for Line in self.CoastLines])
        CurrentTransect = 0

        for Line in self.CoastLines:
            for Transect in Line.Transects:
                
                # print progress to screen
                print(" \r\tTransect %3d / %3d" % (CurrentTransect, NoTransects), end="")

                # write transect    
                Transect.Write(Folder)

                # update counter
                CurrentTransect += 1

        print("")

    def WriteBarriersTextFile(self, Filename, delimiter=","):
        
        """
        MDH, July 2020
        """
        
        # define filename and open for writing
        f = open(Filename,'w')
        
        # write headers
        f.write("LineID" + delimiter + "TransectID" + delimiter + "FrontToeElev" + delimiter + "BackToeElev" + delimiter + "CrestElev" + delimiter + "ToeWidth" + delimiter + "Volume" + "\n")
        
        for Line in self.CoastLines:
            for Transect in Line.Transects:
                if Transect.Barrier:
                    Width, Volume = Transect.ExtractBarrierWidthVolume()
                    f.write(str(Line.ID) + delimiter)
                    f.write(str(Transect.ID) + delimiter)
                    f.write(str(Transect.Elevation[Transect.FrontToeInd]) + delimiter)
                    f.write(str(Transect.Elevation[Transect.BackToeInd]) + delimiter)
                    f.write(str(Transect.Elevation[Transect.CrestInd]) + delimiter)
                    f.write(str(Width) + delimiter)
                    f.write(str(Volume) + "\n")
                
        f.close()
        
    def MergeReverseCoastLines(self):

        """
        Identifies individual coast Lines that are touching at one end 
        and combines them into a single Line

        Reversal of line directions might cause bugs, works so far

        MDH, June 2019
        """

        print("Coast.MergeCoastLines: Merging coastlines")

        # set up Flag for lines being flipped
        FlagReverse = 1

        Pass = 0

        while FlagReverse:

            # print progress to screen
            print(" \r\tPass %3d" % (Pass))
            Pass += 1

            # Update Flag
            FlagReverse = 0

            # Empty lists to populate with new shapes and records
            NewCoastLines = []
            
            # create list of joins
            JoinsList = np.zeros(self.NoCoastLines,dtype=int)-9999
            JoinedByList = np.zeros(self.NoCoastLines,dtype=int)-9999
            ReversedList = []
            
            # get start and end nodes from line sections
            StartNodes = [CoastLine.Nodes[0] for CoastLine in self.CoastLines]
            EndNodes = [CoastLine.Nodes[-1] for CoastLine in self.CoastLines]
            
            # compare start nodes and end nodes to populate join list
            # this could probably be done better!
            for i, StartNode in enumerate(StartNodes):
                for j, EndNode in enumerate(EndNodes):
                    if i == j:
                        continue
                    elif StartNode == EndNode:
                        JoinsList[j] = i
                        JoinedByList[i] = j

                # check for line direction reversals
                for k, TestNode in enumerate(StartNodes):
                    if i == k:
                        continue
                    elif StartNode == TestNode:
                        if not i in ReversedList:
                            ReversedList.append(k)
                            FlagReverse = 1
            
            # get list of line sections to start at
            StartList = np.where(JoinedByList < 0)[0]
            
            for i, StartLine in enumerate(StartList):
                
                # print progress to screen
                print(" \r\tLine %4d / %4d" % (i, len(StartList)), end="")
                
                # get vector of line section
                X1, Y1 = self.CoastLines[StartLine].get_XY()
                
                # get first line section to join
                JoinLine = JoinsList[StartLine]

                while JoinLine > -1:
                    
                    # get next line
                    X2, Y2 = self.CoastLines[JoinLine].get_XY()

                    # join the lines
                    X1 = np.concatenate((X1,X2[1:]))
                    Y1 = np.concatenate((Y1,Y2[1:]))

                    # get next n
                    JoinLine = JoinsList[JoinLine]

                # reverse any vectors needing reversing
                if StartLine in ReversedList:
                    X1 = X1[::-1]
                    Y1 = Y1[::-1]

                # write new line, and update shape and records lists
                NewCoastLines.append(Line(self.CoastLines[StartLine].ID, X1, Y1))

            # update object properties with merged geometries
            self.CoastLines = NewCoastLines
            
            # update number of shapes
            self.NoCoastLines = len(self.CoastLines)
        
        print("\r\t Done.")

    def MergeCoastLines(self, SnapDistance=0.1):

        """
        Identifies individual coast Lines that are touching at one end 
        and combines them into a single Line using shapely

        Distance to snap end points in m

        MDH, Feb, 2020
        """

        print("Coast.MergeCoastLines: Merging coastlines...")

        # get start and end nodes from line sections
        StartNodes = [CoastLine.Nodes[0] for CoastLine in self.CoastLines]
        EndNodes = [CoastLine.Nodes[-1] for CoastLine in self.CoastLines]

        # first check if any start nodes are the same within tolerance
        Distances = np.ones(len(StartNodes))*-9999.
        for i, StartNode in enumerate(StartNodes):
            for ii, StartNode2 in enumerate(StartNodes):
                if i == ii:
                    continue
                Distance = StartNode.get_Distance(StartNode2)
                if Distance < SnapDistance:
                    print("Snapping")
                    self.CoastLines[ii].Nodes[0] = StartNode

        # now check if any end nodes are the same within tolerance
        for i, StartNode in enumerate(StartNodes):
            for j, EndNode in enumerate(EndNodes):
                if i == j:
                    continue
                Distance = StartNode.get_Distance(EndNode)
                if Distance < SnapDistance:
                    print("Snapping")
                    self.CoastLines[j].Nodes[-1] = StartNode

        # create a list of linestrings to merge
        #LineString((tuple(zip(Interp_X,Interp_Y))))
        LinesList = []
        for TempLine in self.CoastLines:
            X,Y = TempLine.get_XY()
            LinesList.append(LineString((tuple(zip(X,Y)))))
        
        # LinesList = [LineString(tuple(zip(Line.get_XY()))) for Line in self.CoastLines]
        MultiLine = MultiLineString(LinesList)
        SimplifiedMultiLine = MultiLine.simplify(0.2)

        # check geom_type before attempting merge
        if SimplifiedMultiLine.geom_type == "MultiLineString": 
            MergedLine = linemerge(SimplifiedMultiLine)
        else:
            MergedLine = SimplifiedMultiLine

        #reset object
        self.CoastLines = []

        # add line or multiple lines depending on result of merge
        if MergedLine.geom_type == "LineString":
            
            # get x and y and add to CoastLine object as Line
            X, Y = MergedLine.xy
            self.CoastLines.append(Line("0", X, Y))
            
        elif MergedLine.geom_type == "MultiLineString":
            
            # loop through lines in MultiLineString
            for i, TempLine in enumerate(MergedLine):
                
                # get x and y and add to CoastLine object as Line
                X, Y = TempLine.xy
                self.CoastLines.append(Line(str(i), X, Y))

        else:
            print("Geometry not recognised!")
            sys.exit()
        
        # update no of coastlines
        self.NoCoastLines = len(self.CoastLines)

    def SmoothCoastLines(self, WindowSize=1001, NoSmooths=2, Resample=True, NodeSpacing=5., PolyOrder=4):
        
        """
        Smooths the CoastLines contained in Coast object
        Wrapper to the function in the Line object
        Calls scipy.signal.savgol_filter

        Savitzky and Golay (1964) smoothing filter
    
        Savitzky, A. and Golay, M. J.: Smoothing and differentiation of data
        by simplified least squares procedures, Anal. Chem., 36, 1627-
        1639, 1964.

        https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.signal.savgol_filter.html

        MDH, June 2019

        Parameters
        ----------
        WindowLength : int
            The length of the filter window (i.e. the number of coefficients). 
            WindowLength must be a positive odd integer.
        PolyOrder : int
            The order of the polynomial used to fit the samples. 
            PolyOrder must be less than window_length.
        NoSmooths : int
            Default is 1
        Resampling : bool
            Whether or not to resample the line to regular spaced nodes
            Default is True
        NodeSpacing : float
            Node spacing for resampleing
            Default is 10 m
        
        """

        print("Coast: Smoothing CoastLines")
        
        
        
        for i in range(0, NoSmooths):
            for Line in self.CoastLines:
                
                if Resample:
                    Line.ResampleNodes(NodeSpacing)
                    
                # smooth the line
                Line.SmoothLine(WindowSize, PolyOrder)

                

    def SplineCoastLines(self):
        
        """
        Splines and resamples the CoastLines contained in Coast object
        Wrapper to the function in the Line object
        
        MDH, March 2020

        """

        print("Coast: Generating Spline of CoastLines")

        for Line in self.CoastLines:
            
            # smooth the line
            Line.SplineLine()


    def ReverseCoastLines(self):
        """
        Function to reverse lines are ordered along the coast
        and line segments progress along the coast. The "along coast" direction
        is always that which results in the water being on the left as you look
        down the coastal vector.

        This might be buggy as anything and need lots more work. Should be run
        after MergeCoast and SmoothCoast but before Transects are built, though 
        if Transects have been built they will get rebuilt

        ***add argument to include shoreline shape then for each node
            find nearest on shoreline shape and calc orientation
            then use mean orientation to assess MDH, Feb 2020

        MDH, Feb 2020
        """

        for Line in self.CoastLines:
            Line.ReverseLine()

        # could add something here to do look up based on distance from starts to ends

    def ReconfigureCoastLines(self, Direction2OpenWater):
        """
        Function to arrange coastline so that lines are ordered along the coast
        and line segments progress along the coast. The "along coast" direction
        is always that which results in the water being on the left as you look
        down the coastal vector.

        This might be buggy as anything and need lots more work. Should be run
        after MergeCoast and SmoothCoast but before Transects are built, though 
        if Transects have been built they will get rebuilt

        ***add argument to include shoreline shape then for each node
            find nearest on shoreline shape and calc orientation
            then use mean orientation to assess MDH, Feb 2020

        MDH, June 2019

        Parameters
        ----------
        Direction2OpenWater: str
            Text-based description of the general direction to open water
            Cardinal direction
            e.g. "E", "east", "East"
        """

        # get start nodes and end nodes of each line
        StartNodes = []
        EndNodes = []

        for Line in self.CoastLines:

            # check line is oriented in the correct order
            StartNode = Line.Nodes[0]
            EndNode = Line.Nodes[-1]

            if str(Direction2OpenWater).lower()[0] == "e":
                
                # reverse the line and update start and end nodes if required
                if StartNode.Y < EndNode.Y:
                    Line.ReverseLine()
                    StartNode = Line.Nodes[0]
                    EndNode = Line.Nodes[-1]
                
            elif Direction2OpenWater.lower()[0] == "s":
                ErrorString = ("Coast.ReconfigureCoastLine (ERROR): "
                    "This direction top open water [s] has not been implemented yet")
                sys.exit(ErrorString)

            elif Direction2OpenWater.lower()[0] == "w":
                if StartNode.Y > EndNode.Y:
                    Line.ReverseLine()
                    StartNode = Line.Nodes[0]
                    EndNode = Line.Nodes[-1]
                
            elif Direction2OpenWater.lower()[0] == "n":
                ErrorString = ("Coast.ReconfigureCoastLine (ERROR): "
                    "This direction top open water [n] has not been implemented yet")
                sys.exit(ErrorString)

            else:
                ErrorString = ("Coast.ReconfigureCoastLine (ERROR): "
                    "The string representing direction to open water not recognised; "
                    "\n\tshould be [e]ast, [s]outh, [w]est or [n]orth")
                sys.exit(ErrorString)

            StartNodes.append(Line.Nodes[0])
            EndNodes.append(Line.Nodes[-1])
        
        # check the lines are organised in the correct order
        if Direction2OpenWater.lower()[0] == "e":
            
            # sort the lines based on Y or their start node
            # needs to be an array to apply negative sign in order to get descending order
            DescendingIndices = np.argsort(-np.array([Node.Y for Node in StartNodes]))
            
            # here comes some bullshit to convert list to numpy array 
            # in order to sort and then turn back into a list :(
            self.CoastLines = list(np.array(self.CoastLines)[DescendingIndices])
            for i, Line in enumerate(self.CoastLines):
                Line.ID = str(i)

        if len(self.CoastLines[0].Transects) != 0:
            self.GenerateTransects(self.TransectsSpacing, self.TransectsLength2Sea, self.TransectsLength2Land)

        # calculate overall orientation
        StartNode = self.CoastLines[0].Nodes[0]
        EndNode = self.CoastLines[-1].Nodes[-1]

        #calculate the spatial change
        dx = EndNode.X - StartNode.X
        dy = EndNode.Y - StartNode.Y

        #Calculate the orientation of the line from ThisNode to NextNode
        if dx > 0 and dy > 0:
            self.OverallOrientation = np.degrees( np.arctan( dx / dy ) )
        elif dx > 0 and dy < 0:
            self.OverallOrientation = 180.0 + np.degrees( np.arctan( dx / dy ) )
        elif dx < 0 and dy < 0:
            self.OverallOrientation = 180.0 + np.degrees( np.arctan( dx / dy ) )
        elif dx < 0 and dy > 0:
            self.OverallOrientation = 360 + np.degrees( np.arctan( dx / dy ) )

    def CheckOrientation(self, ShorelineShp, OffshoreShp):

        """
        Wrapper to function in the line object to check and correct coast orientation
        relative to a shoreline and a deeper contour e.g. a bathy line or MLWS

        MDH, May 2020

        """

        print("Coast.CheckOrientation: Checking CoastLine Orientation Geometry")
        
        # generate transects along each line
        for Line in self.CoastLines:
            
            # generate transects along each line
            Line.CheckLineOrientation(ShorelineShp, OffshoreShp)

    # function to do something    
    def GenerateTransects(self, TransectSpacing, TransectLength2Sea=5000, TransectLength2Land=5000, CheckTopology=True):
        """
        Wrapper to the function in the Line object

        Generates transects perpendicular to the coastline

        MDH, June 2019

        Parameters
        ----------
        TransectSpacing : float
            The distance between consecutive transects along the CoastLines
            in map units, spatial units depend on units of the CoastLine read in,
            Should be [m]
        TransectLength2Sea : float
            The length of the transect in the direction of sea in map units, 
            spatial units depend on units of the CoastLine read in, Should be [m]
        TransectLength2Land : float
            The length of the transect in the direction of land in map units, 
            spatial units depend on units of the CoastLine read in, Should be [m]
        CheckTopology : bool
            Whether to check for overlapping transects and correct. Default is true.
                    
        """
        print("Coast.GenerateTransectNormals: Generating CoastLine transects perpendicular to the coast")

        self.TransectsSpacing = TransectSpacing
        self.TransectsLength2Sea = TransectLength2Sea
        self.TransectsLength2Land = TransectLength2Land

        # generate transects along each line
        for Line in self.CoastLines:

            # generate transects along each line
            Line.GenerateTransects(TransectSpacing, TransectLength2Sea, TransectLength2Land, CheckTopology)

    def GetShorefaceSlopes(self,BathyShp):
        
        """
        
        Wrapper to the function in the Line object
        
        MDH, August 2020
    
        """
        print("Coast.GetShorefaceSlope: Finding distance between shoreline and -10m bathy contour to calculate slope")
        
        for Line in self.CoastLines:
            Line.GetShorefaceSlope(BathyShp)

    def GetShorefaceSlopesMLWS(self):

        """

        Wrapper to function in the Transect object

        MDH, Dec 20202

        """
        
        print("Coast.GetShorefaceSlopeMLWS: Finding distance between shoreline and -10m bathy contour to calculate slope")
        
        for Line in self.CoastLines:
            for Transect in Line.Transects:
                Transect.CalculateIntertidalSlope()            
            
    def GenerateTransectsBetweenContoursShp(self, ContourShp1, ContourShp2, Distance2Sea=8000., Distance2Land=8000., TransectSpacing=20., CheckTopology=True):
        """
        Wrapper to the function in the Line object

        Generates transects perpendicular to the coastline and extends them to 
        the nearest point on another shapefile line

        MDH, September 2019

        Parameters
        ----------
        ContourShp : str
            The name of a shapefile containing a contour/contours 
            to base transects on.
        TransectSpacing : float
            The distance between consecutive transects along the CoastLines
            in map units, spatial units depend on units of the CoastLine read in,
            Should be [m]
        """
        print("Coast.GenerateTransectsBetweenContoursShp: Generating CoastLine transects perpendicular to the coast")

        self.TransectsSpacing = TransectSpacing
        
        for Line in self.CoastLines:

            # generate transects along each line
            Line.GenerateTransectsBetweenContours(ContourShp1,ContourShp2,TransectSpacing,Distance2Sea,Distance2Land,CheckTopology)

    def GenerateMidpointLinesBetweenContoursShp(self, ContourShp1, ContourShp2, Distance2Sea=8000., Distance2Land=8000., TransectSpacing=20., CheckTopology=True):
        """
        Wrapper to the function in the Line object

        Generates a midpoint line between two contours for use as a base line
        required to help adjust for differences between bathy and coastal orientations

        MDH, July 2020

        Parameters
        ----------
        ContourShp : str
            The name of a shapefile containing a contour/contours 
            to base transects on.
        TransectSpacing : float
            The distance between consecutive transects along the CoastLines
            in map units, spatial units depend on units of the CoastLine read in,
            Should be [m]
        """
        print("Coast.GenerateTransectsBetweenContoursShp: Generating CoastLine transects perpendicular to the coast")

        self.TransectsSpacing = TransectSpacing
        
        for Line in self.CoastLines:

            # generate transects along each line
            Line.GenerateMidpointLineBetweenContours(ContourShp1,ContourShp2,TransectSpacing,Distance2Sea,Distance2Land,CheckTopology)

    def GenerateTransectsFromContours(self,ContourShp,TransectSpacing=10.):

        """

        Loops along Coast line object and creates transects between coast and nearest
        point on adjacent contour lines

        MDH September 2019

        Parameters
        ----------
        ContourShp : str
            The name of a shapefile containing a contour/contours 
            to base transects on.
        TransectSpacing : float
            The distance between consecutive transects along the CoastLines
            in map units, spatial units depend on units of the CoastLine read in,
            Should be [m]
        """

        self.TransectsSpacing = TransectSpacing

        for Line in self.CoastLines:
            Line.GenerateTransectsFromContour(ContourShp, TransectSpacing)

    def IntersectTransectsWithIntertidal(self, IntertidalPolyShp):

        """
        Wrapper function to loop through transects and intersect with 
        a polygon defining the intertidal zone

        MDH, June 2020

        """
        print("Coast.IntersectTransectsWithIntertidal: Truncating transects to polygons")
        for Line in self.CoastLines:
            print("Line", Line.ID)
            Line.IntersectTransectsWithIntertidal(IntertidalPolyShp)

    def CheckTransectTopology(self):

        """
        Wrapper function to check for overlapping transects and collect
        Run this after transects have been updated for historical shoreline positions.
        Will then need to rerun historical shoreline position analysis

        MDH, Feb 2020

        """

        print("\nCoast.CheckTransectTopology: Checking for overlapping transects")
        for Line in self.CoastLines:
            Line.FindOverlappingTransects()
            #CheckTransectTopology()

    def RemoveNoHistoricalTransects(self):
        """
        Deletes transects with no historical shoreline positions at any time?

        """

    def GenerateNodes(self, NodeSpacing):

        """
        Wrapper to the function in the Line object

        Generates nodes the coastline

        MDH, August 2019

        Parameters
        ----------
        NodeSpacing : float
            The distance between consecutive nodes along the CoastLines
            in map units, spatial units depend on units of the CoastLine read in,
            Should be [m]
        
        """
        print("Coast.GenerateNodes: Generating CoastLine nodes")

        self.NodeSpacing = TransectSpacing
        
        for Line in self.CoastLines:

            # generate transects along each line
            Line.GenerateNodes(NodeSpacing)

    def SampleDC1Data(self,DC1Shp):
        
        """
        Function to extract info from DC1 analysis
        
        MDH, November 2020
        
        """
        
        print("Coast.SampleDC1Data: Sampling data from DC1 to add to transects")
        # read shapefile using geopandas
        GDF = gp.read_file(DC1Shp)
        Lines = GDF['geometry']
        
        if len(Lines) == 0:
            print("No Lines")
            return
        
        # catch situation where only one line
        MultiLines = []

        if len(Lines) == 1:
            MultiLines = Lines[0]

        # deal with invalid geometries on the fly? This is messy!
        else:
            for Line in Lines:
                if not Line:
                    continue
                elif Line.geom_type == "LineString":
                    MultiLines.append(Line)
                elif Line.geom_type == "MultiLineString":
                    for SubLine in Line:
                        if SubLine.geom_type == "LineString":
                            MultiLines.append(SubLine)

            MultiLines = MultiLineString(MultiLines)    
            
        if not MultiLines:
            print("No Lines")
            return
        
        for Line in self.CoastLines:
            for Transect in Line.Transects:
                
                # extend transect line inland to look for intersection
                #Calculate start and end nodes and generate Transect
                TransectLine = LineString(((Transect.StartNode.X,Transect.StartNode.Y),(Transect.EndNode.X,Transect.EndNode.Y)))
            
                # intersect with historical shoreline
                try:
                    Intersections = TransectLine.intersection(MultiLines)
                except:
                    import pdb
                    pdb.set_trace()
                    
                # catch no intersections and flag for deletion?
                if Intersections.geom_type == "GeometryCollection":
                    Transect.DeleteFlag = True
                    continue

                # check there arent multiple intersections
                # get first intersection if so
                if Intersections.geom_type is "MultiPoint":
                    StartPoint = Point(Transect.StartNode.X, Transect.StartNode.Y)
                    Distances = [IntersectPoint.distance(StartPoint) for IntersectPoint in Intersections]
                    Index = Distances.index(min(Distances))
                    Intersection = Intersections[Index]
                    
                else:
                    # check if this is a new endnode by intersecting with line from startnode to endnode
                    Intersection = Intersections
                                    
                # use minimum of line.distance to find line
                # need date attribute if rates are to be calculated
                Distances = Lines.distance(Intersection)
                NearestLine = GDF.iloc[Distances.idxmin()]
                
                Transect.DC1 = []
                Transect.DC1.append(int(NearestLine.Surv_End_B))
                Transect.DC1.append(int(NearestLine.Surv_End_C))
                Transect.DC1.append(float(NearestLine.DIST_V))
                Transect.DC1.append(float(NearestLine.Rate_B_C))
                
        
        # sort out delete flags???
                
    def Check_OS_Years(self):
        
        """
        Function to get and populate OS years from smarter 2020 dataset
        """
        
        for Line in self.CoastLines:
            for Transect in Line.Transects:
                Transect.Check_OS_Year()
        
        
    def ExtractHistoricalShorelinePositions(self,HistoricalShorelinesShp,Reset=False, AllowMultiples=False):

        """
        Function to find nearest historic shoreline position on each transect
        and add nodes to transect dictionary by date

        MDH, August 2019

        Parameters
        ----------
        HistoricalShorelineShp : string
            Filename for polyline shapfile containing historical shoreline positions
        Reset : bool
            Resets all historical shoreline positions
        """
        print("Coast.ExtractHistoricalShorelinePositions: Finding historical shoreline positions from ", end="")
        print(Path(HistoricalShorelinesShp).name)

        # set a distance to look inland to check for intersections
        LookDistance = 0.

        # read shapefile using geopandas
        GDF = gp.read_file(HistoricalShorelinesShp)
        Lines = GDF['geometry']
        
        if len(Lines) == 0:
            print("No Lines")
            import pdb
            pdb.set_trace()
            return
        
        # catch situation where only one line
        MultiLines = []

        if len(Lines) == 1:
            MultiLines = Lines[0]

        # deal with invalid geometries on the fly? This is messy!
        else:
            for Line in Lines:
                if not Line:
                    continue
                elif Line.geom_type == "LineString":
                    MultiLines.append(Line)
                elif Line.geom_type == "MultiLineString":
                    for SubLine in Line:
                        if SubLine.geom_type == "LineString":
                            MultiLines.append(SubLine)

            MultiLines = MultiLineString(MultiLines)    
            #MultiLines = MultiLineString([Line for Line in Lines if Line.geom_type == "LineString"])
            
        if not MultiLines:
            print("No Lines")
            return
        
        for Line in self.CoastLines:
            
            for Transect in Line.Transects:
                
                if Reset:
                    Transect.ResetHistoricShorelines()
                    
                # extend transect line inland to look for intersection
                #Calculate start and end nodes and generate Transect
                X1 = Transect.EndNode.X + LookDistance * np.sin( np.radians( Transect.Orientation ) )
                Y1 = Transect.EndNode.Y + LookDistance * np.cos( np.radians( Transect.Orientation ) )
                TransectLine = LineString(((Transect.StartNode.X,Transect.StartNode.Y),(X1,Y1)))
            
                # intersect with historical shoreline
                Intersections = TransectLine.intersection(MultiLines)
                
                # catch no intersections and flag for deletion?
                if Intersections.geom_type == "GeometryCollection":
                    Transect.DeleteFlag = True
                    continue

                # check there arent multiple intersections
                """
                # store multiple intersections if so
                if Intersections.geom_type is "MultiPoint":
                    StartPoint = Point(Transect.StartNode.X, Transect.StartNode.Y)
                    Distances = [IntersectPoint.distance(StartPoint) for IntersectPoint in Intersections]
                    Index = Distances.index(min(Distances))
                    Indices = np.argsort(np.array(Distances))
                    Distances = np.array(Distances)[Indices]
                    IntersectionsList = [Intersections[i] for i in Indices]
                    
                else:
                    # check if this is a new endnode by intersecting with line from startnode to endnode
                    Distance = Transect.LineString.distance(Intersections)
                    Intersection = Intersections
                    IntersectionsList = [Intersection,]
                """

                # store multiple intersections if so
                if Intersections.geom_type is "MultiPoint":
                    CoastPoint = Point(Transect.CoastNode.X, Transect.CoastNode.Y)
                    Distances = [IntersectPoint.distance(CoastPoint) for IntersectPoint in Intersections]
                    Index = Distances.index(min(Distances))
                    Indices = np.argsort(np.array(Distances))
                    Distances = np.array(Distances)[Indices]
                    IntersectionsList = [Intersections[i] for i in Indices]
                    
                else:
                    # check if this is a new endnode by intersecting with line from startnode to endnode
                    Distance = Transect.LineString.distance(Intersections)
                    Intersection = Intersections
                    IntersectionsList = [Intersection,]
                
                IntersectionYears = []
                
                # loop through intersections and add to struct
                for Intersection in IntersectionsList:
                    #print(Intersection.wkt, end=", ")
                    # use minimum of line.distance to find line
                    # need date attribute if rates are to be calculated
                    Distances = Lines.distance(Intersection)
                    # print(Distances.idxmin())
                    NearestLine = GDF.iloc[Distances.idxmin()]
                
                    # check it hasnt already been read
                    if "FULLSHP_YR" in NearestLine:
                        IntersectionYears.append(int(NearestLine.FULLSHP_YR))
                    elif "Surv_EndYr" in NearestLine:
                        IntersectionYears.append(int(NearestLine.Surv_EndYr))
                    elif "Surv_End_A" in NearestLine:
                        IntersectionYears.append(int(NearestLine.Surv_End_A))
                    elif "Surv_End_B" in NearestLine:
                        IntersectionYears.append(int(NearestLine.Surv_End_B))
                    elif "Surv_End_C" in NearestLine:
                        IntersectionYears.append(int(NearestLine.Surv_End_C))
                    elif "Surv_End_D" in NearestLine:
                        IntersectionYears.append(int(NearestLine.Surv_End_D))
                    elif "versiondat" in NearestLine:
                        IntersectionYears.append(int(NearestLine.versiondat[0:4]))
                    elif "dates" in NearestLine:
                        IntersectionYears.append(int(NearestLine.dates))
                    else:
                        sys.exit("Couldnt find survey year for MHWS historic shoreline position")
                
                # delete intersections for years that already exist?
                if len(IntersectionYears) == 1:
                    if IntersectionYears[0] in Transect.HistoricShorelinesYears:
                        continue
                        
                elif len(IntersectionYears) > 1:
                    Indices = [i for i, Year in enumerate(IntersectionYears) if Year not in Transect.HistoricShorelinesYears]
                    IntersectionsList = [IntersectionsList[i] for i in Indices]
                    IntersectionYears = [IntersectionYears[i] for i in Indices]
                
                if len(IntersectionYears) == 0:
                    continue
                
                if not AllowMultiples:
                    
                    CoastPoint = Point(Transect.CoastNode.X, Transect.CoastNode.Y)
                    TempDistances = [IntersectionPoint.distance(CoastPoint) for IntersectionPoint in IntersectionsList]
                    IntersectionIndex = TempDistances.index(min(TempDistances))
                    Intersection = IntersectionsList[IntersectionIndex]
                    Year = IntersectionYears[IntersectionIndex]
                    
                    if Year not in Transect.HistoricShorelinesYears:
                        
                        # add year to transect
                        Index = bisect.bisect(Transect.HistoricShorelinesYears, Year)
                        Transect.HistoricShorelinesYears.insert(Index, Year)
                        
                        # add shoreline position
                        Position = Node(Intersection.x,Intersection.y)
                        Positions = [Position,]
                        Transect.HistoricShorelinesPositions.insert(Index, Positions)
                        
                        # add distance
                        Distances = [Transect.StartNode.get_Distance(Position),]
                        Transect.HistoricShorelinesDistances.insert(Index, Distances)
                        
                        # add source info
                        Transect.HistoricShorelinesSources.insert(Index, Path(HistoricalShorelinesShp).name)
                        
                        # retrieve positional error
                        if Year < 1970:
                            Error = 5.
                        elif Year < 2000:
                            Error = 2.
                        else:
                            Error = 1.
                            
                        # add error
                        Transect.HistoricShorelinesErrors.insert(Index, Error)
                        
                    else:
                        
                        # find and either add or replace depending on proximity
                        Index = Transect.HistoricShorelinesYears.index(Year)
                        Position = Node(Intersection.x,Intersection.y)
                        
                        MinDistance = 1000.
                        
                        for OldPosition in Transect.HistoricShorelinesPositions[Index]:
                            Distance = OldPosition.get_Distance(Position)
                            if Distance < MinDistance:
                                MinDistance = Distance
                        
                        if MinDistance > 1.:
                        
                            # add to transect
                            Transect.HistoricShorelinesPositions[Index].append(Position)
                            Transect.HistoricShorelinesDistances[Index].append(Distance)

                else:
                
                    # loop through unique years
                    UniqueYears = list(set(IntersectionYears))
                    for Year in UniqueYears:
    
                        # retrieve positional error
                        if Year < 1970:
                            Error = 5.
                        elif Year < 2000:
                            Error = 2.
                        else:
                            Error = 1.
    
                        
                        # isolate intersections for this year
                        Indices = [i for i, ThisYear in enumerate(IntersectionYears) if ThisYear == Year]
                        TempIntersectionsList = [IntersectionsList[i] for i in Indices]
                        CoastPoint = Point(Transect.CoastNode.X, Transect.CoastNode.Y)
                        TempDistances = [IntersectionPoint.distance(CoastPoint) for IntersectionPoint in TempIntersectionsList]
                        IntersectionIndex = TempDistances.index(min(TempDistances))
                        Intersection = TempIntersectionsList[IntersectionIndex]
    
                        if Year not in Transect.HistoricShorelinesYears:
                            
                            # add year to transect
                            Index = bisect.bisect(Transect.HistoricShorelinesYears, Year)
                            Transect.HistoricShorelinesYears.insert(Index, Year)
                            
                            # add shoreline position
                            Position = Node(Intersection.x,Intersection.y)
                            Positions = [Position,]
                            Transect.HistoricShorelinesPositions.insert(Index, Positions)
                            
                            # add distance
                            Distances = [Transect.StartNode.get_Distance(Position),]
                            Transect.HistoricShorelinesDistances.insert(Index, Distances)
                            
                            # add source info
                            Transect.HistoricShorelinesSources.insert(Index, Path(HistoricalShorelinesShp).name)
                            
                            # add error
                            Transect.HistoricShorelinesErrors.insert(Index, Error)
                            
                        else:
                            
                            # find and either add or replace depending on proximity
                            Index = Transect.HistoricShorelinesYears.index(Year)
                            Position = Node(Intersection.x,Intersection.y)
                            
                            MinDistance = 1000.
                            
                            for OldPosition in Transect.HistoricShorelinesPositions[Index]:
                                Distance = OldPosition.get_Distance(Position)
                                if Distance < MinDistance:
                                    MinDistance = Distance
                            
                            if MinDistance > 1.:
                            
                                # add to transect
                                Transect.HistoricShorelinesPositions[Index].append(Position)
                                Transect.HistoricShorelinesDistances[Index].append(Distance)

                """
                for i, Intersection in enumerate(IntersectionsList):
                    
                    # retrieve year
                    Year = IntersectionYears[i]
                    
                    if Year not in Transect.HistoricShorelinesYears:
                       
                        # add year to transect
                        Index = bisect.bisect(Transect.HistoricShorelinesYears, Year)
                        Transect.HistoricShorelinesYears.insert(Index, Year)
                        
                        # add shoreline position
                        Position = Node(Intersection.x,Intersection.y)
                        Positions = [Position,]
                        Transect.HistoricShorelinesPositions.insert(Index, Positions)
                        
                        # add distance
                        Distances = [Transect.StartNode.get_Distance(Position),]
                        Transect.HistoricShorelinesDistances.insert(Index, Distances)
                        
                        # add source info
                        Transect.HistoricShorelinesSources.insert(Index, Path(HistoricalShorelinesShp).name)
                        
                        # add error
                        Transect.HistoricShorelinesErrors.insert(Index, Error)
                        
                    else:
                        
                        # find and either add or replace depending on proximity
                        Index = Transect.HistoricShorelinesYears.index(Year)
                        Position = Node(Intersection.x,Intersection.y)
                        
                        MinDistance = 1000.
                        
                        for OldPosition in Transect.HistoricShorelinesPositions[Index]:
                            Distance = OldPosition.get_Distance(Position)
                            if Distance < MinDistance:
                                MinDistance = Distance
                        
                        if MinDistance > 1.:
                        
                            # add to transect
                            Transect.HistoricShorelinesPositions[Index].append(Position)
                            Transect.HistoricShorelinesDistances[Index].append(Distance)
                """

    def ExtractSatShorePositions(self,HistoricalShorelinesShp,Reset=False, AllowMultiples=False):
    
        """
        Function to find nearest historic shoreline position on each transect
        and add nodes to transect dictionary by date
    
        MDH, August 2019
    
        Parameters
        ----------
        HistoricalShorelineShp : string
            Filename for polyline shapfile containing historical shoreline positions
        Reset : bool
            Resets all historical shoreline positions
        """
        print("Coast.ExtractSatShorePositions: Finding historical shoreline positions from ", end="")
        print(Path(HistoricalShorelinesShp).name)
    
        # set a distance to look inland to check for intersections
        LookDistance = 0.
    
        # read shapefile using geopandas
        GDF = gp.read_file(HistoricalShorelinesShp)
        Lines = GDF['geometry']
        
        if len(Lines) == 0:
            print("No Lines")
            import pdb
            pdb.set_trace()
            return
        
        # catch situation where only one line
        MultiLines = []
    
        if len(Lines) == 1:
            MultiLines = Lines[0]
    
        # deal with invalid geometries on the fly? This is messy!
        else:
            for Line in Lines:
                if not Line:
                    continue
                elif Line.geom_type == "LineString":
                    MultiLines.append(Line)
                elif Line.geom_type == "MultiLineString":
                    for SubLine in Line:
                        if SubLine.geom_type == "LineString":
                            MultiLines.append(SubLine)
    
            MultiLines = MultiLineString(MultiLines)    
            #MultiLines = MultiLineString([Line for Line in Lines if Line.geom_type == "LineString"])
            
        if not MultiLines:
            print("No Lines")
            return
        
        for Line in self.CoastLines:
            
            for Transect in Line.Transects:
                
                if Reset:
                    Transect.ResetHistoricShorelines()
                    
                # extend transect line inland to look for intersection
                #Calculate start and end nodes and generate Transect
                X1 = Transect.EndNode.X + LookDistance * np.sin( np.radians( Transect.Orientation ) )
                Y1 = Transect.EndNode.Y + LookDistance * np.cos( np.radians( Transect.Orientation ) )
                TransectLine = LineString(((Transect.StartNode.X,Transect.StartNode.Y),(X1,Y1)))
            
                # intersect with historical shoreline
                Intersections = TransectLine.intersection(MultiLines)
                
                # catch no intersections and flag for deletion?
                if Intersections.geom_type == "GeometryCollection":
                    Transect.DeleteFlag = True
                    continue
    
    
                # store multiple intersections if so
                if Intersections.geom_type is "MultiPoint":
                    CoastPoint = Point(Transect.CoastNode.X, Transect.CoastNode.Y)
                    Distances = [IntersectPoint.distance(CoastPoint) for IntersectPoint in Intersections]
                    Index = Distances.index(min(Distances))
                    Indices = np.argsort(np.array(Distances))
                    Distances = np.array(Distances)[Indices]
                    IntersectionsList = [Intersections[i] for i in Indices]
                    
                else:
                    # check if this is a new endnode by intersecting with line from startnode to endnode
                    Distance = Transect.LineString.distance(Intersections)
                    Intersection = Intersections
                    IntersectionsList = [Intersection,]
                
                IntersectionDates = []
                
                # loop through intersections and add to struct
                for Intersection in IntersectionsList:
                    #print(Intersection.wkt, end=", ")
                    # use minimum of line.distance to find line
                    # need date attribute if rates are to be calculated
                    Distances = Lines.distance(Intersection)
                    # print(Distances.idxmin())
                    import pdb
                    try:
                        NearestLine = GDF.iloc[Distances.idxmin()]
                    except:
                        continue
                
                    # check it hasnt already been read
                    if "dates" in NearestLine:
                        IntersectionDates.append(NearestLine.dates)
                    else:
                        sys.exit("Couldnt find survey year for MHWS historic shoreline position")
                
                if not AllowMultiples:
                    
                    CoastPoint = Point(Transect.CoastNode.X, Transect.CoastNode.Y)
                    TempDistances = [IntersectionPoint.distance(CoastPoint) for IntersectionPoint in IntersectionsList]
                    IntersectionIndex = TempDistances.index(min(TempDistances))
                    Intersection = IntersectionsList[IntersectionIndex]
                    Year = IntersectionDates[IntersectionIndex]
                    
                    if Year not in Transect.HistoricShorelinesYears:
                        
                        # add year to transect
                        Index = bisect.bisect(Transect.HistoricShorelinesYears, Year)
                        Transect.HistoricShorelinesYears.insert(Index, Year)
                        
                        # add shoreline position
                        Position = Node(Intersection.x,Intersection.y)
                        Positions = [Position,]
                        Transect.HistoricShorelinesPositions.insert(Index, Positions)
                        
                        # add distance
                        Distances = [Transect.StartNode.get_Distance(Position),]
                        Transect.HistoricShorelinesDistances.insert(Index, Distances)
                        
                        # add source info
                        Transect.HistoricShorelinesSources.insert(Index, Path(HistoricalShorelinesShp).name)
                    
                        
                    else:
                        
                        # find and either add or replace depending on proximity
                        Index = Transect.HistoricShorelinesYears.index(Year)
                        Position = Node(Intersection.x,Intersection.y)
                        
                        MinDistance = 1000.
                        
                        for OldPosition in Transect.HistoricShorelinesPositions[Index]:
                            Distance = OldPosition.get_Distance(Position)
                            if Distance < MinDistance:
                                MinDistance = Distance
                        
                        if MinDistance > 1.:
                        
                            # add to transect
                            Transect.HistoricShorelinesPositions[Index].append(Position)
                            Transect.HistoricShorelinesDistances[Index].append(Distance)
    
                else:
                
                    # loop through unique years
                    UniqueYears = list(set(IntersectionDates))
                    for Year in UniqueYears:
                            
                        # isolate intersections for this year
                        Indices = [i for i, ThisYear in enumerate(IntersectionDates) if ThisYear == Year]
                        TempIntersectionsList = [IntersectionsList[i] for i in Indices]
                        CoastPoint = Point(Transect.CoastNode.X, Transect.CoastNode.Y)
                        TempDistances = [IntersectionPoint.distance(CoastPoint) for IntersectionPoint in TempIntersectionsList]
                        IntersectionIndex = TempDistances.index(min(TempDistances))
                        Intersection = TempIntersectionsList[IntersectionIndex]
    
                        if Year not in Transect.HistoricShorelinesYears:
                            
                            # add year to transect
                            Index = bisect.bisect(Transect.HistoricShorelinesYears, Year)
                            Transect.HistoricShorelinesYears.insert(Index, Year)
                            
                            # add shoreline position
                            Position = Node(Intersection.x,Intersection.y)
                            Positions = [Position,]
                            Transect.HistoricShorelinesPositions.insert(Index, Positions)
                            
                            # add distance
                            Distances = [Transect.StartNode.get_Distance(Position),]
                            Transect.HistoricShorelinesDistances.insert(Index, Distances)
                            
                            # add source info
                            Transect.HistoricShorelinesSources.insert(Index, Path(HistoricalShorelinesShp).name)
                            
                            # add error
                            Transect.HistoricShorelinesErrors.insert(Index, Error)
                            
                        else:
                            
                            # find and either add or replace depending on proximity
                            Index = Transect.HistoricShorelinesYears.index(Year)
                            Position = Node(Intersection.x,Intersection.y)
                            
                            MinDistance = 1000.
                            
                            for OldPosition in Transect.HistoricShorelinesPositions[Index]:
                                Distance = OldPosition.get_Distance(Position)
                                if Distance < MinDistance:
                                    MinDistance = Distance
                            
                            if MinDistance > 1.:
                            
                                # add to transect
                                Transect.HistoricShorelinesPositions[Index].append(Position)
                                Transect.HistoricShorelinesDistances[Index].append(Distance)
    
    

    def ExtractMLWS(self,MLWSShp):

        """
        Function to find nearest location of MLWS
        from shapefile for each transect

        MDH, December 2020

        Parameters
        ----------
        MLWSShp : string
            Filename for polyline shapfile containing MLWS
        
        """
        print("Coast.ExtractMLWS: Finding nearest MLWS position")
        
        # read shapefile using geopandas
        GDF = gp.read_file(MLWSShp)
        
        # get lines geometry
        Lines = GDF['geometry']
        
        # catch situation where only one line
        MultiLines = []

        if len(Lines) == 1:
            MultiLines = Lines[0]

        # deal with invalid geometries on the fly? This is messy!
        else:
            for ThisLine in Lines:
                if not ThisLine:
                    continue
                elif ThisLine.geom_type == "LineString":
                    MultiLines.append(ThisLine)
                elif ThisLine.geom_type == "MultiLineString":
                    for SubLine in ThisLine:
                        if SubLine.geom_type == "LineString":
                            MultiLines.append(SubLine)
        
        MultiLines = MultiLineString(MultiLines)
                    
        for ThisLine in self.CoastLines:
            for Transect in ThisLine.Transects:
                
                # shapely goes here
                BasePoint = Point(Transect.CoastNode.X, Transect.CoastNode.Y)
                NearestPoint = nearest_points(MultiLines, BasePoint)[0]
                Transect.MLWS = Node(NearestPoint.x,NearestPoint.y)

    def ExtractContours(self,ContourShp):

        """
        Function to find nearest location of -10 m depth contour
        from contour shapefile for each transect

        MDH, August 2019

        Parameters
        ----------
        ContourShp : string
            Filename for polyline shapfile containing depth contours
        
        """
        print("Coast.ExtractContours: Finding nearest depth contours")
        
        # read shapefile using geopandas
        GDF = gp.read_file(ContourShp)
        
        for Contour in GDF.level.unique():
        
            # isolate closure depth contour
            GDFtemp = GDF[GDF.level == Contour]
        
            # get lines geometry
            Lines = GDFtemp['geometry']
            MultiLines = MultiLineString([Line for Line in Lines])

            for i, ContourLine in enumerate(MultiLines):
                x, y = ContourLine.xy
                TempLine = Line(str(i),x,y,Contour)
                self.Contours.append(TempLine)
            
            for ThisLine in self.CoastLines:
                for Transect in ThisLine.Transects:
                    
                    # shapely goes here
                    BasePoint = Point(Transect.CoastNode.X, Transect.CoastNode.Y)
                    NearestPoint = nearest_points(MultiLines, BasePoint)[0]
                    Transect.Contours.append(Node(str(Contour), NearestPoint.x,NearestPoint.y, Contour))

    


    def SampleHistoricalRSLR(self, PastRSLRRaster):

        """ 
        
        Samples a raster of most recent rates of relative sea level change (rise/fall)
        at each transect location on coast. 

        Gets the nearest point for now, rather than any interpolation

        Parameters
        ----------
        PastRSLRRaster : string
            Filename for raster to be sampled
        
        MDH, September 2019

        """

        print("Coast.SampleHistoricalRSLR: Sampling historical Relative Sea Level raster dataset")

        # open the raster dataset to work on
        with rasterio.open(PastRSLRRaster) as RSLRDataset:
        
            # loop through transects and sample
            for Line in self.CoastLines:
                for i, Transect in enumerate(Line.Transects[:]):
                    for val in RSLRDataset.sample([(Transect.CoastNode.X,Transect.CoastNode.Y)]):
                        Transect.HistoricalRSLR = val[0]

    def SampleMHWSElevation(self,MHWSRaster):

        """
        Samples a raster of MHWS elevation at each transect location on the coast

        Parameters
        ----------

        MHWSRaster : string
            Filename for raster to be sampled
        
        MDH, January 2020

        """

        print("Coast.SampleMHWSElevation: Sampling MHWS elevation raster dataset")

        # open the raster dataset to work on
        with rasterio.open(MHWSRaster) as MHWSDataset:
        
            # loop through transects and sample
            for Line in self.CoastLines:
                for i, Transect in enumerate(Line.Transects[:]):
                    for val in MHWSDataset.sample([(Transect.CoastNode.X,Transect.CoastNode.Y)]):
                        Transect.MHWS = val[0]


    def SampleFutureRSL(self, FutureRSLFolder, RCP=8, Percentile=95, Years=[2020,2030,2040,2050,2060,2070,2080,2090,2100]):

        """ 
        
        Samples a raster of future rates of relative sea level change (rise/fall)
        at each transect location on coast

        Parameters
        ----------
        FutureRSLFolder : string
            Folder containing future sea level elevation rasters for Scotland
        RCP : int
            RCP scenario to use
        Percentile : int
            Percentile scenario to use
        Years : list
            List of integers corresponding to the years to be analysed
        
        MDH, September 2019

        """

        print("Coast.SampleFutureRSL: Sampling future Relative Sea Level raster dataset")

        if self.FutureShoreLinesYears:
            print("\tFuture sea levels already sampled")
            return

        self.FutureShoreLinesYears = Years

        for Year in Years:
            FutureRSLRaster = FutureRSLFolder + "/RCP" + str(RCP) + "_" + str(Percentile) + "th_" + str(Year) + "_filled.tif"

            # open the raster dataset to work on
            with rasterio.open(FutureRSLRaster) as RSLDataset:
            
                # loop through transects and sample
                for Line in self.CoastLines:
                    for i, Transect in enumerate(Line.Transects[:]):
                        for val in RSLDataset.sample([(Transect.CoastNode.X,Transect.CoastNode.Y)]):
                            Transect.FutureSeaLevels.append(val[0])
                            Transect.FutureSeaLevelYears.append(Year)

    def SampleRockHeadPosition(self, UPSMRaster, MaxRockHeadErosionDistance=25.):

        """
        Function to check values of UPSM and identify if a limit on shoreline erosion position 
        is required based on a threshold value of 0.4

        MDH, January 2020

        """

        print("Coast.SampleRockHeadPosition: Sampling rock head dataset to set maximum extent of erosion")

        # open the raster dataset to work on
        with rasterio.open(UPSMRaster) as RockHeadDataset:
        
            # loop through transects and sample
            for Line in self.CoastLines:
                for i, Transect in enumerate(Line.Transects[:]):
                    
                    # generate a list of tuples to sample UPSM
                    X1 = Transect.StartNode.X
                    Y1 = Transect.StartNode.Y
                    X2 = Transect.EndNode.X
                    Y2 = Transect.EndNode.Y
                    X = np.linspace(X1,X2,50.)
                    Y = np.linspace(Y1,Y2,50.)
                    NodeList = tuple(zip(X, Y))

                    # build a list of X,Y values to check along transect to find position of rock head if present
                    #for val in RSLRDataset.sample([(Transect.CoastNode.X,Transect.CoastNode.Y)]):
                    RockHeadVector = np.array([val[0] for val in RockHeadDataset.sample(NodeList)])
                    RockHeadVector[RockHeadVector < 0] = np.nan
                    
                    # if everything is soft, carry on
                    # ignore errors caused by NaNs
                    with np.errstate(invalid='ignore'):
                        RockBool = RockHeadVector < 0.4
        
                    if not RockBool.any():
                        continue
                    
                    # else find the position of the first appearance of 0.4
                    JInd = np.argmax(RockBool)
                    
                    if JInd == len(RockHeadVector)-1:
                        continue

                    # repeat to find to the nearest meter
                    dX = (X[JInd-1] - X[JInd+1])
                    dY = (Y[JInd-1] - Y[JInd+1])
                    NVals = np.int(np.sqrt(dX**2. + dY**2.))
                    
                    X = np.linspace(X[JInd-1], X[JInd+1], NVals)
                    Y = np.linspace(Y[JInd-1], Y[JInd+1], NVals)
                    NodeList = tuple(zip(X, Y))

                    # build a list of X,Y values to check along transect to find position of rock head if present
                    RockHeadVector = np.array([val[0] for val in RockHeadDataset.sample(NodeList)])
                    RockHeadVector[RockHeadVector < 0] = np.nan

                    # else find the position of the first appearance of 0.4
                    # ignore errors caused by NaNs
                    with np.errstate(invalid='ignore'):
                        RockBool = RockHeadVector < 0.4

                    if not RockBool.any():
                        continue
                    
                    JInd = np.argmax(RockBool)
                    
                    # flag position as attribute of transect
                    Transect.RockHeadPosition = Node(X[JInd],Y[JInd])
                    Transect.RockHeadDistance = Transect.StartNode.get_Distance(Transect.RockHeadPosition)

                    # check rockhead position relative to starting shoreline position and adjust to allow 
                    # some erosion to take place or not to take place
                    if Transect.HistoricShorelinesDistances and (Transect.HistoricShorelinesDistances[-1][0] > Transect.RockHeadDistance):
                        Transect.RockHeadDistance = Transect.HistoricShorelinesDistances[-1][0] + MaxRockHeadErosionDistance
                        Transect.RockHeadPosition = Transect.get_Position(Transect.RockHeadDistance)
                    else:
                        Transect.RockHeadDistance += MaxRockHeadErosionDistance
                        Transect.RockHeadPosition = Transect.get_Position(Transect.RockHeadDistance)
                        
    def SampleDefencesPosition(self, DefencesShp, MaxDefencesErosionDistance=0.):

        """
        Function to find defences and identify if a limit on shoreline erosion position 
        
        MDH, January 2021

        """

        print("Coast.SampleDefencesPosition: Sampling position of coastal defences")


        # set a distance to look inland to check for intersections
        LookDistance = 0.

        # read shapefile using geopandas
        GDF = gp.read_file(DefencesShp)
        Lines = GDF['geometry']
        
        if len(Lines) == 0:
            print("No Lines")
            import pdb
            pdb.set_trace()
            return
        
        # catch situation where only one line
        MultiLines = []

        if len(Lines) == 1:
            MultiLines = Lines[0]

        # deal with invalid geometries on the fly? This is messy!
        else:
            for Line in Lines:
                if not Line:
                    continue
                elif Line.geom_type == "LineString":
                    MultiLines.append(Line)
                elif Line.geom_type == "MultiLineString":
                    for SubLine in Line:
                        if SubLine.geom_type == "LineString":
                            MultiLines.append(SubLine)

            MultiLines = MultiLineString(MultiLines)    
            #MultiLines = MultiLineString([Line for Line in Lines if Line.geom_type == "LineString"])
            
        if not MultiLines:
            print("No Lines")
            return
        
        for Line in self.CoastLines:
            for Transect in Line.Transects:
                
                # extend transect line inland to look for intersection
                #Calculate start and end nodes and generate Transect
                X1 = Transect.EndNode.X + LookDistance * np.sin( np.radians( Transect.Orientation ) )
                Y1 = Transect.EndNode.Y + LookDistance * np.cos( np.radians( Transect.Orientation ) )
                TransectLine = LineString(((Transect.StartNode.X,Transect.StartNode.Y),(X1,Y1)))
            
                # intersect with historical shoreline
                try:
                    Intersections = TransectLine.intersection(MultiLines)
                except:
                    import pdb
                    pdb.set_trace()
                    
                # catch no intersections and flag for deletion?
                if Intersections.geom_type == "GeometryCollection":
                    continue

                # check there arent multiple intersections
                StartPoint = Point(Transect.StartNode.X, Transect.StartNode.Y)
                # store multiple intersections if so
                if Intersections.geom_type is "MultiPoint":
                    Distances = [IntersectPoint.distance(StartPoint) for IntersectPoint in Intersections]
                    Index = Distances.index(min(Distances))
                    Distance = Distances[Index]
                    Intersection = Intersections[Index]
                    
                else:
                    # check if this is a new endnode by intersecting with line from startnode to endnode
                    Intersection = Intersections
                    Distance = StartPoint.distance(Intersection)
                
                # assign to transect
                Transect.Defences = True
                Transect.DefencesDistance = Distance+MaxDefencesErosionDistance
                Transect.DefencesPosition = Transect.get_Position(Transect.DefencesDistance)
                
    def PredictFutureShorelines(self):

        """

        Wrapper to call Transects function to predict future shoreline positions

        MDH, September 2019

        """
        print("Coast.PredictFutureShorelines: predicting future shoreline positions")
        # loop through transects and sample
        for Line in self.CoastLines:
            for Transect in Line.Transects:
                Transect.PredictFutureShorelines()

    def PredictFutureShorelinesUncertainty(self, Year=2100):

        """

        Wrapper to call Transects function to predict future shoreline positions uncertainty

        MDH, September 2019
        
        """
        print("Coast.PredictFutureShorelinesUncertainty: predicting future shoreline positions uncertainty %d", Year)
        # loop through transects and sample
        for Line in self.CoastLines:
            for Transect in Line.Transects:
                if Transect.Future:
                    Transect.PredictFutureShorelineUncertainty(Year)

    def PredictFutureShorelinesError(self, Year=2100):

        """

        Wrapper to call Transects function to predict future shoreline positional error

        MDH, September 2020

        """
        print("Coast.PredictFutureShorelines: predicting future shoreline positions error %d", Year)
        # loop through transects and sample
        for Line in self.CoastLines:
            for Transect in Line.Transects:
                if Transect.Future:
                    Transect.PredictFutureShorelineError(Year)

    def PredictFutureVegEdge(self,VegEdgeShp, Year=None):

        """

        Wrapper function to call Transects function to predict future shoreline position
        based on position of vegetation edge provided in a shapefile.


        MDH, Feb 2020

        """

        print("Coast.PredictFutureVegEdge: Finding position of future veg edge ", end="")
        
        # set a distance to look inland to check for intersections
        LookDistance = 100.

        # read shapefile using geopandas
        GDF = gp.read_file(VegEdgeShp)
        Lines = GDF['geometry']
        
        # catch situation where only one line
        if len(Lines) == 0:
            sys.exit("Error: No Veg Edge Lines!")
        elif len(Lines) == 1:
            MultiLines = Lines[0]
        else:
            MultiLines = MultiLineString([Line for Line in Lines if Line])
            MultiLines = MultiLineString([Line for Line in MultiLines if Line.geom_type == "LineString"])
            

        for Line in self.CoastLines:
            for Transect in Line.Transects:
                
                # extend transect line inland to look for intersection
                #Calculate start and end nodes and generate Transect
                X1 = Transect.EndNode.X + LookDistance * np.sin( np.radians( Transect.Orientation ) )
                Y1 = Transect.EndNode.Y + LookDistance * np.cos( np.radians( Transect.Orientation ) )
                TransectLine = LineString(((Transect.StartNode.X,Transect.StartNode.Y),(X1,Y1)))
            
                # intersect with historical shoreline
                Intersection = TransectLine.intersection(MultiLines)

                # catch no intersections and flag for deletion?
                if Intersection.geom_type == "GeometryCollection":
                    Transect.VegEdge = False
                    continue

                # check there arent multiple intersections, if there are just get the nearest
                if Intersection.geom_type is "MultiPoint":
                    StartPoint = Point(Transect.StartNode.X, Transect.StartNode.Y)
                    Distances = [IntersectPoint.distance(StartPoint) for IntersectPoint in Intersection]
                    Index = Distances.index(min(Distances))
                    Intersection = Intersection[Index]

                # check if this is a new endnode by intersecting with line from startnode to endnode
                Distance = Transect.LineString.distance(Intersection)
                
                if Distance > 0.001:
                    
                    # set this as the new end node
                    NewEndNode = Node(Intersection.x,Intersection.y)
                    Transect.Redraw(Transect.StartNode, NewEndNode)

                # use minimum of line.distance to find line
                # need date attribute if rates are to be calculated
                Distances = Lines.distance(Intersection)
                NearestLine = GDF.iloc[Distances.idxmin()]
                
                # check it hasnt already been read
                if not Year:
                    if "Surv_End_A" in NearestLine:
                        Year = int(NearestLine.Surv_End_A)
                    elif "Surv_End_B" in NearestLine:
                        Year = int(NearestLine.Surv_End_B)
                    elif "Surv_End_C" in NearestLine:
                        Year = int(NearestLine.Surv_End_C)
                    elif "Surv_End_D" in NearestLine:
                        Year = int(NearestLine.Surv_End_D)
                    else:
                        sys.exit("Couldnt find survey year for MHWS historic shoreline position")

                # add point to transect
                Transect.VegEdgePosition = Node(Intersection.x,Intersection.y)
                Transect.VegEdgeYear = Year
                Transect.VegEdge = True
                
                # analyse future veg edge
                Transect.PredictFutureVegEdge()

                
    def ExtendTransects2Hinterland(self, Distance):

        """
        Extends transects by a fixed distance into the hinterland in order to 
        measure hinterland topography. N.B. does not extend start/end point but 
        creates a new node in the hinterland.

        MDH, March 2020

        """

        print("Coast.ExtendTransects2Hinterland: Puts a new node landward of existing transect")

        for Line in self.CoastLines:
            for Transect in Line.Transects:
                Transect.ExtendTransect(Distance, 0)
                
    def ExtendTransects2Line(self, LineShp):

        """
        Extends transects to a line shp file

        MDH, August 2020

        """

        print("Coast.ExtendTransects2Hinterland: Puts a new node landward of existing transect")

        # read in the lines object file
        
        for Line in self.CoastLines:
            Line.ExtendTransectsToLineShp(LineShp)
            
    def TruncateTransects(self):
        
        """
        function to cut the length of transects the the extrermes of historical
        or future shoreline positions, including uncertainties
        
        MDH, November 2020
        
        """
        for Line in self.CoastLines:
            for Transect in Line.Transects:
                Transect.Truncate()
                            
    def FindDEM(self, DEMIndexFileShp):

        """
        Identifies which DEMs transects intersect with, where interesction is with
        more than one transect DEMs will get merged.

        Need to think this through more carefully... dont want to end up having to repeatedly open the same DEM
        Get a list of transects that intersect each DEM along the coast object?
        Get a list of unique DEMs that are intersected.
        Intersect Coast lines and transects with DEMIndexFileShp
        Open each DEM and extract topography for all transects that fall within
        What to do about transects crossing from one DEM to another?

        MDH, March 2020

        """

        print("Coast.FindDEM: Identifying DEM for each transect to sample from")

        # read the DEM index file
        PolyGDF = gp.read_file(DEMIndexFileShp)
        
        # list of unique DEMs
        self.UniqueDEMList = []
        
        #print(len(self.CoastLines))
        for Line in self.CoastLines:
            

            # get multilinestring of transects
            Lines = [LineString([(Transect.EndNode.X,Transect.EndNode.Y),(Transect.StartNode.X,Transect.StartNode.Y)]) for Transect in Line.Transects]
            
            if not Lines:
                continue
            
            LineGDF = gp.GeoDataFrame(geometry=Lines,crs=PolyGDF.crs)
            
            JoinGDF = gp.sjoin(LineGDF, PolyGDF, op='intersects')
            
            # set DEMs to list
            self.UniqueDEMList.extend(list(JoinGDF.location.unique()))
        
        # replace extension with *.tif
        #for i, DEMPath in enumerate(self.UniqueDEMList):
        #    self.UniqueDEMList[i] = DEMPath.rstrip("asc")+"tif"

    def ExtractTransectTopography(self, DEMFileList=None):

        """
        Function to sample elevations for transect lines from list of DEM files
        
        MDH, March 2020

        """      
        print("Coast.ExtractTransectTopography: Sampling DEM(s) along transects")

        # set up dem file list
        if DEMFileList:
            # check if list and make list if not
            if not isinstance(DEMFileList, list):
                DEMFileList = [DEMFileList,]
            self.UniqueDEMList = DEMFileList

        # loop through DEMs
        for DEM in self.UniqueDEMList:
            
            print("\t" + DEM.split("/")[-1])

            DTM_Dataset = rasterio.open(DEM)
            DTMArray = DTM_Dataset.read(1)
            NCols = DTM_Dataset.width
            NRows = DTM_Dataset.height
            NDV = DTM_Dataset.nodata
            Resolutions = DTM_Dataset.res
            
            # check if we're missing no data
            if not DTM_Dataset.nodata:
                raise SystemExit("DTM missing no data value")

            # check for square pixels
            if not DTM_Dataset.res[0] == DTM_Dataset.res[1]:
                raise SystemExit("DTM has non-square cells")
        
            # get resolution
            DTM_Resolution = DTM_Dataset.res[0]

            # get extent of DTM and set up polygon of extent
            XMin = DTM_Dataset.bounds[0]
            XMax = DTM_Dataset.bounds[2]
            YMin = DTM_Dataset.bounds[1]
            YMax = DTM_Dataset.bounds[3]
            DTM_Extent = Polygon([[XMin, YMin], [XMin, YMax], [XMax, YMax], [XMax, YMin]])

            # Get vectors of X and Y coordinates, NB reversal of Y in line with 
            # DTM indexing from top left
            XVector = XMin+np.arange(0,NCols)*DTM_Resolution+0.5*DTM_Resolution
            YVector = YMin+DTM_Resolution*np.arange(0,NRows)[::-1]+0.5*DTM_Resolution

            for Line in self.CoastLines:
                for Transect in Line.Transects:
                    
                    # check we have nodes to sample
                    if not Transect.DistanceNodes:
                        Transect.DistanceSpacing = DTM_Dataset.res[0]
                        Transect.GenerateSampleNodes()

                    # check for intersection
                    if not Transect.LineString.intersects(DTM_Extent):
                        continue
                    
                    # get list of points that intersect DTM only
                    Points = [Point(ThisNode.X,ThisNode.Y) for ThisNode in Transect.DistanceNodes]
                    Points = [ThisPoint if ThisPoint.within(DTM_Extent) else Point((0,0)) for ThisPoint in Points]
                    Coords = [(Point.x, Point.y) for Point in Points]
                    Elevations = [Sample[0] for Sample in DTM_Dataset.sample(Coords)]
                    Transect.Elevation = Elevations

                    # problem here gettign back to transects
                    for i, ThisNode in enumerate(Transect.DistanceNodes):
                        
                        if not ThisNode.Z and Elevations[i] > 0:
                            Transect.DistanceNodes[i].Z = Elevations[i]

                    # Set up the mask from NDVs
                    Mask = Elevations == NDV
                    Transect.Distance = ma.masked_where(Mask,Transect.Distance)
                    Transect.Elevation = ma.masked_where(Mask,Elevations)

                    Transect.HaveTopography = True

    def ExtractTransectTopographySwath(self, DTMFile, SwathDistance=-9999):
        """
        Profile to populate transects with topographic data
        Uses swath profile routine to collect elevations within a certain distance
        of each transect line then takes IDW values for the transect topography

        ADD FUNCTIONALITY TO CATCH WHEN DEM EDGE HAS BEEN EXCEEDED? NO TRANSECTS IN THIS CASE

        MDH, June 2019
        
        Parameters
        ----------
        DTMFile : str
            Name of DTM File, must be a *.tif

        SwathDistance : float
            Distance away from transect line to sample elevations in DEM
            Default is 2 times the resolution of the DTM

        """
        
        print("Coast.EstractTransectTopography: Sampling DTMs for each transect")
        
        # load the DTM and get its properties
        print("\tLoading DTM... ", end="")
        DTM_Dataset = rasterio.open(DTMFile)
        DTMArray = DTM_Dataset.read(1)
        NCols = DTM_Dataset.width
        NRows = DTM_Dataset.height
        NDV = DTM_Dataset.nodata
        Resolutions = DTM_Dataset.res
        print("Done")

        # check for square pixels
        if not DTM_Dataset.res[0] == DTM_Dataset.res[1]:
            raise SystemExit("DTM has non-square cells")
        
        # get resolution
        DTM_Resolution = DTM_Dataset.res[0]

        # check swath distance
        if SwathDistance < 0:
            SwathDistance = DTM_Resolution*2.

        # get extent of DTM and set up polygon of extent
        XMin = DTM_Dataset.bounds[0]
        XMax = DTM_Dataset.bounds[2]
        YMin = DTM_Dataset.bounds[1]
        YMax = DTM_Dataset.bounds[3]
        
        DTM_Extent = Polygon([[XMin, YMin], [XMin,YMax], [XMax, YMax], [XMax, YMin]])
            
            
        # Get vectors of X and Y coordinates, NB reversal of Y in line with 
        # DTM indexing from top left
        XVector = XMin+np.arange(0,NCols)*DTM_Resolution+0.5*DTM_Resolution
        YVector = YMin+DTM_Resolution*np.arange(0,NRows)[::-1]+0.5*DTM_Resolution

        # Track progress
        NoTransects = np.sum([Line.NoTransects for Line in self.CoastLines])
        CurrentTransect = 0
                        
        for Line in self.CoastLines:
            for Transect in Line.Transects:
                
                # print progress to screen
                print(" \r\tTransect %3d / %3d" % (CurrentTransect, NoTransects), end="")

                #Get line points
                X1, Y1 = Transect.StartNode.get_XY()
                X2, Y2 = Transect.EndNode.get_XY()
                TransectLine = LineString([(X1, Y1), (X2, Y2)])

                # check for intersection
                if not TransectLine.intersects(DTM_Extent):
                    continue

                #find indices for bounding box
                #need to be careful with reverse indexing
                iStart = np.argmin(np.abs(YVector-np.max([Y1,Y2])))-1
                iEnd = np.argmin(np.abs(YVector-np.min([Y1,Y2])))+1
                jStart = np.argmin(np.abs(XVector-np.min([X1,X2])))-1
                jEnd = np.argmin(np.abs(XVector-np.max([X1,X2])))+1

                #Get Vector X and Y
                dX12 = X2-X1
                dY12 = Y2-Y1

                #Declare list holders for profile data
                X = []
                Y = []
                Z = []
                DistAlong = []
                DistTo = []
                
                for i in range(iStart,iEnd):

                    #get Y position
                    YNode = YMax-DTM_Resolution*i-0.5*DTM_Resolution

                    for j in range(jStart,jEnd):
                        
                        #get X position
                        XNode = XMin + j*DTM_Resolution + 0.5*DTM_Resolution;

                        #Get 2nd Vector Properties in Array
                        dX13 = XNode-X1
                        dY13 = YNode-Y1

                        #Find Dot Product
                        DotProduct = dX12*dX13 + dY12*dY13;

                        #calculate fraction of distance along line
                        t = DotProduct/(dX12*dX12 + dY12*dY12)
                        if ((t < 0.) or (t > 1.)):
                            continue
                    
                        #Find point along line
                        XLine = X1 + t*dX12
                        YLine = Y1 + t*dY12
                        DistanceAlongLine = t*np.sqrt(dX12*dX12 + dY12*dY12)

                        #find distance to point
                        DistanceToLine = np.sqrt((XLine-XNode)*(XLine-XNode) + (YLine-YNode)*(YLine-YNode))

                        if ((DistanceToLine < SwathDistance) and (DTMArray[i][j] != NDV)):
                            X.append(XNode)
                            Y.append(YNode)
                            DistAlong.append(DistanceAlongLine)
                            DistTo.append(DistanceToLine)
                            Z.append(DTMArray[i][j])
                                
                #Sort by distance along line, need to convert to numpy arrays as we go to sort
                Sortedi = np.argsort(DistAlong)
                X = np.asarray(X)[Sortedi]
                Y = np.asarray(Y)[Sortedi]
                DistAlong = np.asarray(DistAlong)[Sortedi]
                DistTo = np.asarray(DistTo)[Sortedi]
                Z = np.asarray(Z)[Sortedi]
                
                #if (WriteSwathDataFlag):
                    # Write results to text file using pandas (easier) for each profile
                    #DF = pd.DataFrame({"X": X, "Y": Y, "Z": Z, "DistAlong": DistAlong, "DistTo": DistTo})
                    #DF.to_pickle(SwathProfsFolder+"Swath_"+str(Transect.ID)+".pkl")
                
                #Create a line for interpolating to
                # determination of distance spacing should be externalised
                LineLength = np.sqrt((X2-X1)**2 + (Y2-Y1)**2)
                NoPoints = (int)(LineLength/(DTM_Resolution*2.))
                Transect.DistanceSpacing = DTM_Resolution*2.
                XLine = np.linspace(X1,X2,NoPoints)
                YLine = np.linspace(Y1,Y2,NoPoints)
                DistAlongTransect = np.zeros(len(XLine))
                ZIDW = np.zeros(len(XLine))
                ZMin = np.zeros(len(XLine))
                ZMax = np.zeros(len(XLine))
                ZStd = np.zeros(len(XLine))
                                
                #Loop along line
                for i in range(0,NoPoints):
                    
                    #Calculate distance along the line
                    DistAlongTransect[i] = i*DTM_Resolution*2.
                    
                    # Sample a reduced array here i.e. a neighbourhood to reduce computation time
                    Neighbourhood = np.abs(DistAlongTransect[i]-DistAlong) < DTM_Resolution*2.
                    ZLocal = Z[Neighbourhood]
                    
                    if len(ZLocal) == 0:
                        
                        # Set to NDV
                        ZIDW[i] = NDV
                        ZMin[i] = NDV
                        ZMax[i] = NDV
                        ZStd[i] = NDV
                        
                        continue
                    
                    # Do IDW
                    # Create a distance vector
                    Dist = np.sqrt(DistAlong[Neighbourhood]**2. + DistTo[Neighbourhood]**2.)
                    
                    # Weights are inverse
                    Weights = 1./Dist**2.
                    
                    # Interpolate Z
                    ZIDW[i]  = np.sum(Z[Neighbourhood]*Weights)/np.sum(Weights)
                    
                    # Other Z Values
                    ZMin[i] = np.min(ZLocal)
                    ZMax[i] = np.max(ZLocal)
                    ZStd[i] = np.std(ZLocal)
                    
                # Set up the mask from NDVs
                Mask = ZIDW == NDV
                DistAlongTransect = ma.masked_where(Mask,DistAlongTransect)
                ZIDW = ma.masked_where(Mask,ZIDW)
                ZMin = ma.masked_where(Mask,ZMin)
                ZMax = ma.masked_where(Mask,ZMax)
                ZStd = ma.masked_where(Mask,ZStd)
                
                Transect.Distance = DistAlongTransect
                Transect.DistanceSpacing = DistAlongTransect[1]-DistAlongTransect[0]
                Transect.Elevation = ZIDW
                Transect.ElevationMin = ZMin
                Transect.ElevationMax = ZMax
                Transect.ElevStd = ZStd

                # update transect no
                CurrentTransect += 1
        
        print("")

    def AnalyseTransectMorphology(self):

        """

        Barrier focus for now

        MDH, June 2019

        """

        print("Coast.AnalyseTransectMorphology: Finding cliff and barrier positions and calculating metrics")

        # Track progress
        NoTransects = np.sum([Line.NoTransects for Line in self.CoastLines])-1
        CurrentTransect = 0

        for Line in self.CoastLines:
            for Transect in Line.Transects:

                # print progress to screen
                print(" \r\tTransect %3d / %3d" % (CurrentTransect, NoTransects), end="")
                
                # # Call analyses
                #if Transect.ID == "13":
                Transect.FindCliff()
                Transect.FindBarrier()
                
                # update transect progress no
                CurrentTransect += 1
        
        print("")

    def AnalyseBarrierWidths(self, WaterElevs):
        
        """
        
        Extracts barrier width at given elevations e.g. high water

        MDH, June 2019

        """

        print("Coast.AnalyseBarrierWidth: Finding barrier positions at a given elevations and calculating metrics")

        # update extreme water levels
        self.ExtremeWaterLevels = WaterElevs

        # Track progress
        NoTransects = np.sum([Line.NoTransects for Line in self.CoastLines])-1
        CurrentTransect = 0

        # loop through transects and get contiguous barrier lines
        for CoastLine in self.CoastLines:
            for Transect in CoastLine.Transects:
                
                # print progress to screen
                print(" \r\tTransect %3d / %3d" % (CurrentTransect, NoTransects), end="")
                    
                # extract barrier width
                #if Transect.ID == "138":
                #    Transect.ExtractBarrierWidths(WaterElevs)
                Transect.ExtractBarrierWidths(WaterElevs)

                # update transect progress no
                CurrentTransect += 1

        print("")
    
    def MapBarrierFeatureExtents(self, WaterElevs, DTM):
        """
        Function to contour the DEM to map extent of elevations above extreme water levels
        but within the zone of analysis from the first to the last topographic intersection

        MDH, October 2019
        
        Parameters
        ----------
        WaterElevsL : list(float)
            List of extreme water surface elevations
        
        DTMFile : str
            Name of DTM File, must be a *.tif

        """
        
        print("Coast.MapBarrierFeatureExtents: Extracting features from DTM")

        # get the max extent of flood protection features

        
        # load the DTM and get its properties
        print("\tLoading DTM... ", end="")
        DTM_Dataset = rasterio.open(DTMFile)
        DTMArray = DTM_Dataset.read(1)
        NCols = DTM_Dataset.width
        NRows = DTM_Dataset.height
        NDV = DTM_Dataset.nodata
        Resolutions = DTM_Dataset.res
        print("Done")

        # check for square pixels
        if not DTM_Dataset.res[0] == DTM_Dataset.res[1]:
            raise SystemExit("DTM has non-square cells")
        
        # get resolution
        DTM_Resolution = DTM_Dataset.res[0]

        # check swath distance
        if SwathDistance < 0:
            SwathDistance = DTM_Resolution*2.

        # get extent of DTM
        XMin = DTM_Dataset.bounds[0]
        XMax = DTM_Dataset.bounds[2]
        YMin = DTM_Dataset.bounds[1]
        YMax = DTM_Dataset.bounds[3]

        # Get vectors of X and Y coordinates, NB reversal of Y in line with 
        # DTM indexing from top left
        XVector = XMin+np.arange(0,NCols)*DTM_Resolution+0.5*DTM_Resolution
        YVector = YMin+DTM_Resolution*np.arange(0,NRows)[::-1]+0.5*DTM_Resolution

        # Track progress
        NoTransects = np.sum([Line.NoTransects for Line in self.CoastLines])
        CurrentTransect = 0

    def FindRockyCoast(self, TidalElevation=2.):

        """
        
        Calculates roughness up to a fixed tidal elevation as the standard deviation of 
        slope and the average standard deviation of local elevations

        uses a kmeans clustering algorithm to split in two based on these in order to split
        rocky from sandy

        MDH, July 2019

        """

        # loop through transects and get contiguous barrier lines
        for CoastLine in self.CoastLines:
            for Transect in CoastLine.Transects:
                Transect.AnalyseRoughness(TidalElevation)
        
        #NoTransects = np.sum([Line.NoTransects for Line in self.CoastLines])-1

        # Get roughness values as arrays
        SlopeRoughness = np.array([Transect.SlopeRoughness for Line in self.CoastLines for Transect in Line.Transects])
        ValueLocs = (np.isnan(SlopeRoughness) == False)
        Locations = np.argwhere(ValueLocs)
        SlopeRoughness = SlopeRoughness[ValueLocs]
        ElevationRoughness = np.array([Transect.ElevationRoughness for Line in self.CoastLines for Transect in Line.Transects])
        ElevationRoughness = ElevationRoughness[ValueLocs]
        Data = np.column_stack((SlopeRoughness,ElevationRoughness))
        
        # perform k-means clustering assuming two clusters
        # set up a KMeans object
        ThisKMeans = KMeans(n_clusters=2)
        ThisKMeans.fit(Data)
        GroupList = ThisKMeans.fit_predict(Data)
        
        # check which way round and correct
        if np.mean(ElevationRoughness[GroupList == 0]) > np.mean(ElevationRoughness[GroupList == 1]):
            GroupList = abs(GroupList-1)
        
        # loop through transects and get contiguous barrier lines
        Counter = 0
        for CoastLine in self.CoastLines:
            for i, Transect in enumerate(CoastLine.Transects):
                Transect.Rocky = GroupList[Counter]
                Counter += 1

    def GetFutureShoreLinesProximity(self, BufferDistance):

        Lines = []

        # Loop through prediction years
        for Year in self.FutureShoreLinesYears:

            # keep track of no of coastal segments for IDs
            FutureCount = 0
            
            # loop through transects and get contiguous cliff lines
            for CoastLine in self.CoastLines:
                
                # find transects with future predictions
                FutureBool = [Transect.Future for Transect in CoastLine.Transects]
                FutureBool.insert(0, False)
                FutureBool = np.array(FutureBool).astype(int)

                # check for lines with no predictions
                if not any(FutureBool):
                    continue
                
                # get a list of the start and end points of contiguous lines
                StartEndFlags = np.diff(FutureBool)
                
                # if first element is true this is a start point
                if FutureBool[0]:
                    StartEndFlags[0] = 1
                
                # if last line finishes on a start flag then remove
                if StartEndFlags[-1] == 1:
                    StartEndFlags[-1] = 0
                
                # if no start flags
                if len(StartEndFlags.nonzero()[0]) == 0:
                    continue
                
                # if last line finishes on last node then flag as end flag
                if StartEndFlags[StartEndFlags.nonzero()[0][-1]] == 1:
                    StartEndFlags[-1] = -1

                StartList = np.argwhere(StartEndFlags == 1).flatten()
                EndList = np.argwhere(StartEndFlags == -1).flatten()
                
                if len(StartList) < 1:
                    continue
                
                if not len(StartList) == len(EndList):
                    print("Start and End lists not the same length")
                    print(len(StartList),len(EndList))
                    import pdb
                    pdb.set_trace()
                    
                for i in range(0,len(StartList)):
                    
                    # catch single node cliff lines and ignore
                    if (EndList[i]-StartList[i]<2):
                        continue

                    # create empty lists for storing future nodes
                    ProximityList = []
                    
                    # add latest MHWS from previous node to start
                    # might need some logic here for first transect
                    if StartList[i] == 0:
                        FirstNode = CoastLine.Transects[StartList[i]].get_RecentPosition()
                        ii = 1
                    else:
                        FirstNode = CoastLine.Transects[StartList[i]-1].get_RecentPosition()
                        ii = 0
                        if not FirstNode:
                            FirstNode = CoastLine.Transects[StartList[i]].get_RecentPosition()
                            ii= 1
                    
                    if not FirstNode:
                        import pdb
                        pdb.set_trace()
                        
                    ProximityList.append(FirstNode)
                    
                    # loop through transects and get future positions
                    for Transect in CoastLine.Transects[StartList[i]+ii:EndList[i]]:
                        
                        if Transect.get_FutureDistance(Year) > Transect.get_RecentDistance():
                            Distance = Transect.get_FutureDistance(Year) + BufferDistance
                            TempNode = Transect.get_Position(Distance)
                            ProximityList.append(TempNode)
                        
                        else:
                            ProximityList.append(Transect.get_RecentPosition())
                        
                                                
                    # add latest MHWS from next node to end
                    # might need some logic here to finish
                    if not CoastLine.Transects[EndList[i]].get_RecentPosition():
                        LastNode = CoastLine.Transects[EndList[i]-1].get_RecentPosition()
                    else:
                        LastNode = CoastLine.Transects[EndList[i]].get_RecentPosition()
                    
                    ProximityList.append(LastNode)
                    
                    # create new line object for top
                    try:
                        X = [ProximityNode.X for ProximityNode in ProximityList]
                        Y = [ProximityNode.Y for ProximityNode in ProximityList]
                    except:
                        import pdb
                        pdb.set_trace()
                        
                    TempLine = Line("Proximity_"+str(FutureCount), X, Y, Year=Year)
                    Lines.append(TempLine)
                    
                    # update counter
                    FutureCount += 1

        return Lines

    def GetFutureShoreLines(self):

        """

        Extracts contiguous lines of future predicted MHWS

        """
        self.FutureShoreLines = []

        # Loop through prediction years
        for Year in self.FutureShoreLinesYears:

            # keep track of no of coastal segments for IDs
            FutureCount = 0
            
            # loop through transects and get contiguous cliff lines
            for CoastLine in self.CoastLines:
                
                # find transects with future predictions
                FutureBool = [Transect.Future for Transect in CoastLine.Transects]
                FutureBool.insert(0, False)
                FutureBool = np.array(FutureBool).astype(int)

                # check for lines with no predictions
                if not any(FutureBool):
                    continue
                
                # get a list of the start and end points of contiguous lines
                StartEndFlags = np.diff(FutureBool)
                
                # if first element is true this is a start point
                if FutureBool[0]:
                    StartEndFlags[0] = 1
                
                # if last line finishes on a start flag then remove
                if StartEndFlags[-1] == 1:
                    StartEndFlags[-1] = 0
                
                # if no start flags
                if len(StartEndFlags.nonzero()[0]) == 0:
                    continue
                
                # if last line finishes on last node then flag as end flag
                if StartEndFlags[StartEndFlags.nonzero()[0][-1]] == 1:
                    StartEndFlags[-1] = -1

                StartList = np.argwhere(StartEndFlags == 1).flatten()
                EndList = np.argwhere(StartEndFlags == -1).flatten()
                
                if len(StartList) < 1:
                    continue
                
                if not len(StartList) == len(EndList):
                    print("Start and End lists not the same length")
                    print(len(StartList),len(EndList))
                    import pdb
                    pdb.set_trace()
                    
                for i in range(0,len(StartList)):
                    
                    # catch single node cliff lines and ignore
                    if (EndList[i]-StartList[i]<2):
                        continue

                    # create empty lists for storing future nodes
                    FutureList = []
                    
                    # add latest MHWS from previous node to start
                    # might need some logic here for first transect
                    if StartList[i] == 0:
                        FirstNode = CoastLine.Transects[StartList[i]].get_RecentPosition()
                        ii = 1
                    else:
                        FirstNode = CoastLine.Transects[StartList[i]-1].get_RecentPosition()
                        ii = 0
                        if not FirstNode:
                            FirstNode = CoastLine.Transects[StartList[i]].get_RecentPosition()
                            ii= 1
                    
                    if not FirstNode:
                        import pdb
                        pdb.set_trace()
                        
                    FutureList.append(FirstNode)
                    
                    # loop through transects and get future positions
                    for Transect in CoastLine.Transects[StartList[i]+ii:EndList[i]]:
                        
                        if Transect.get_FutureDistance(Year) > Transect.get_RecentDistance():
                            TempNode = Transect.get_FuturePosition(Year)
                            FutureList.append(TempNode)
                        
                        else:
                            FutureList.append(Transect.get_RecentPosition())
                        
                                                
                    # add latest MHWS from next node to end
                    # might need some logic here to finish
                    if not CoastLine.Transects[EndList[i]].get_RecentPosition():
                        LastNode = CoastLine.Transects[EndList[i]-1].get_RecentPosition()
                    else:
                        LastNode = CoastLine.Transects[EndList[i]].get_RecentPosition()
                    
                    FutureList.append(LastNode)
                    
                    # create new line object for top
                    try:
                        X = [FutureNode.X for FutureNode in FutureList]
                        Y = [FutureNode.Y for FutureNode in FutureList]
                    except:
                        import pdb
                        pdb.set_trace()
                        
                    TempLine = Line("FutureCoast_"+str(FutureCount), X, Y, Year=Year)
                    self.FutureShoreLines.append(TempLine)
                    
                    # update counter
                    FutureCount += 1
    
    def GetFutureShorelineUncertainty(self, Year=2100):

        """
        
        Extracts contiguous lines of uncertainty on Bruun Rule predictions to 2100
        
        MDH, March 2020

        """

        
        # keep track of no of coastal segments for IDs
        FutureCount = 0
        self.FutureMinUncertainty = []
        self.FutureMaxUncertainty = []

        # loop through transects and get contiguous locations where there are predictions
        for CoastLine in self.CoastLines:
                
            # find transects with future predictions
            FutureBool = [Transect.Future for Transect in CoastLine.Transects]
            FutureBool.insert(0, False)
            FutureBool = np.array(FutureBool).astype(int)

            # check for lines with no predictions
            if not any(FutureBool):
                continue
                
            # get a list of the start and end points of contiguous cliff lines
            StartEndFlags = np.diff(FutureBool)
            
            # if first element is true this is a start point
            if FutureBool[0]:
                StartEndFlags[0] = 1
            
            # if last line finishes on a start flag then remove
            if StartEndFlags[-1] == 1:
                StartEndFlags[-1] = 0
            
            # if no start flags
            if len(StartEndFlags.nonzero()[0]) == 0:
                continue
            
            # if last line finishes on last node then flag as end flag
            if StartEndFlags[StartEndFlags.nonzero()[0][-1]] == 1:
                StartEndFlags[-1] = -1
            
            StartList = np.argwhere(StartEndFlags == 1).flatten()
            EndList = np.argwhere(StartEndFlags == -1).flatten()
            if not len(StartList) == len(EndList):
                print("Start and End lists not the same length")
                print(len(StartList),len(EndList))
                import pdb
                pdb.set_trace()
                
            for i in range(0,len(StartList)):
                
                # catch single node cliff lines and ignore
                if (EndList[i]-StartList[i]<2):
                    continue

                # create empty lists for storing future nodes for min and max predictions
                FutureMinList = []
                FutureMaxList = []
                
                # add latest MHWS from previous node to start
                # might need some logic here for first transect
                if StartList[i] == 0:
                    FirstNode = CoastLine.Transects[StartList[i]].get_RecentPosition()
                    ii = 1
                else:
                    FirstNode = CoastLine.Transects[StartList[i]-1].get_RecentPosition()
                    ii = 0
                    if not FirstNode:
                        FirstNode = CoastLine.Transects[StartList[i]].get_RecentPosition()
                        ii= 1
                            
                FutureMinList.append(FirstNode)
                FutureMaxList.append(FirstNode)

                # loop through transects and get min and max future positions
                for Transect in CoastLine.Transects[StartList[i]+ii:EndList[i]]:
                    Transect.PredictFutureShorelineUncertainty(Year)
                    FutureMinNode = Transect.FutureShorelinesMinNode
                    try:
                        FutureMaxNode = Transect.FutureShorelinesMaxNode
                    except:
                        import pdb
                        pdb.set_trace()
                    FutureMinList.append(FutureMinNode)
                    FutureMaxList.append(FutureMaxNode)
                    
                # add latest MHWS from next node to end
                # might need some logic here to finish
                # add latest MHWS from next node to end
                # might need some logic here to finish
                if not CoastLine.Transects[EndList[i]].get_RecentPosition():
                    LastNode = CoastLine.Transects[EndList[i]-1].get_RecentPosition()
                else:
                    LastNode = CoastLine.Transects[EndList[i]].get_RecentPosition()
                    
                FutureMinList.append(LastNode)
                FutureMaxList.append(LastNode)
                
                try:
                    # create new line object for min and max
                    X = [FutureMinNode.X for FutureMinNode in FutureMinList]
                    Y = [FutureMinNode.Y for FutureMinNode in FutureMinList]
                
                except:
                    import pdb
                    pdb.set_trace()
                    
                TempLine = Line("FutureMin_"+str(FutureCount), X, Y)
                self.FutureMinUncertainty.append(TempLine)

                # create new line object for min and max
                X = [FutureMaxNode.X for FutureMaxNode in FutureMaxList]
                Y = [FutureMaxNode.Y for FutureMaxNode in FutureMaxList]
                
                TempLine = Line("FutureMax_"+str(FutureCount), X, Y)
                self.FutureMaxUncertainty.append(TempLine)
                
                # update counter
                FutureCount += 1

    def GetFutureShorelineError(self, Year=2100):

        """
        
        Extracts contiguous lines of error on Bruun Rule predictions for a given year
        Error is propoagagtion of historical shoreline positional errors only
        
        MDH, October 2020

        """

        # keep track of no of coastal segments for IDs
        FutureCount = 0

        # loop through transects and get contiguous locations where there are predictions
        for CoastLine in self.CoastLines:
                
            # find transects with future predictions
            FutureBool = [Transect.Future for Transect in CoastLine.Transects]
            FutureBool.insert(0, False)
            FutureBool = np.array(FutureBool).astype(int)

            # check for lines with no predictions
            if not any(FutureBool):
                continue
                
            # get a list of the start and end points of contiguous cliff lines
            StartEndFlags = np.diff(FutureBool)

            # if first element is true this is a start point
            if FutureBool[0]:
                StartEndFlags[0] = 1
            
            # if last line finishes on a start flag then remove
            if StartEndFlags[-1] == 1:
                StartEndFlags[-1] = 0
            
            # if no start flags
            if len(StartEndFlags.nonzero()[0]) == 0:
                continue
            
            # if last line finishes on last node then flag as end flag
            if StartEndFlags[StartEndFlags.nonzero()[0][-1]] == 1:
                StartEndFlags[-1] = -1

            StartList = np.argwhere(StartEndFlags == 1).flatten()
            EndList = np.argwhere(StartEndFlags == -1).flatten()
            if not len(StartList) == len(EndList):
                print("Start and End lists not the same length")
                print(len(StartList),len(EndList))
                import pdb
                pdb.set_trace()
                    
            for i in range(0,len(StartList)):
                
                # catch single node cliff lines and ignore
                if (EndList[i]-StartList[i]<2):
                    continue

                # create empty lists for storing future nodes for min and max predictions
                FutureMinList = []
                FutureMaxList = []
                
                # add latest MHWS from previous node to start
                # might need some logic here for first transect
                # might need some logic here for first transect
                if StartList[i] == 0:
                    FirstNode = CoastLine.Transects[StartList[i]].get_RecentPosition()
                    ii = 1
                else:
                    FirstNode = CoastLine.Transects[StartList[i]-1].get_RecentPosition()
                    ii = 0
                    if not FirstNode:
                        FirstNode = CoastLine.Transects[StartList[i]].get_RecentPosition()
                        ii= 1
                
                FutureMinList.append(FirstNode)
                FutureMaxList.append(FirstNode)

                # loop through transects and get min and max future positions
                for Transect in CoastLine.Transects[StartList[i]:EndList[i]]:
                    Transect.PredictFutureShorelineError(Year)
                    FutureMinNode = Transect.FutureShorelinesMinNode
                    FutureMaxNode = Transect.FutureShorelinesMaxNode
                    FutureMinList.append(FutureMinNode)
                    FutureMaxList.append(FutureMaxNode)
                    
                # add latest MHWS from next node to end
                # might need some logic here to finish
                if EndList[i] == CoastLine.NoTransects-1:
                    LastNode = CoastLine.Transects[EndList[i]-1].get_RecentPosition()
                else:
                    LastNode = CoastLine.Transects[EndList[i]].get_RecentPosition()
                    if not LastNode:
                        LastNode = CoastLine.Transects[EndList[i]-1].get_RecentPosition()
                
                FutureMinList.append(LastNode)
                FutureMaxList.append(LastNode)
                
                # create new line object for min and max
                X = [FutureMinNode.X for FutureMinNode in FutureMinList]
                Y = [FutureMinNode.Y for FutureMinNode in FutureMinList]
                
                TempLine = Line("FutureMin_"+str(FutureCount), X, Y)
                self.FutureMinError.append(TempLine)

                # create new line object for min and max
                X = [FutureMaxNode.X for FutureMaxNode in FutureMaxList]
                Y = [FutureMaxNode.Y for FutureMaxNode in FutureMaxList]
                
                TempLine = Line("FutureMax_"+str(FutureCount), X, Y)
                self.FutureMaxError.append(TempLine)
                
                # update counter
                FutureCount += 1

    def GetFutureVegEdgeLines(self):

        """

        Extracts contiguous lines of future predicted vegetation edge

        MDH, Feb 2020

        """

        # Loop through prediction years
        for Year in self.FutureShoreLinesYears[1:]:

            # keep track of no of coastal segments for IDs
            FutureCount = 0

            # loop through transects and get contiguous cliff lines
            for CoastLine in self.CoastLines:
                
                # find transects with future predictions
                VegEdgeBool = [Transect.VegEdge for Transect in CoastLine.Transects]
                VegEdgeBool.insert(0, False)
                VegEdgeBool = np.array(VegEdgeBool).astype(int)
                
                # check for lines with no predictions
                if not any(VegEdgeBool):
                    continue
                
                # get a list of the start and end points of contiguous cliff lines
                StartEndFlags = np.diff(VegEdgeBool)
                
                # if first element is true this is a start point
                if VegEdgeBool[0]:
                    StartEndFlags[0] = 1
                
                # if last line finishes on a start flag then remove
                if StartEndFlags[-1] == 1:
                    StartEndFlags[-1] = 0
                
                # if no start flags
                if len(StartEndFlags.nonzero()[0]) == 0:
                    continue
                
                # if last line finishes on last node then flag as end flag
                if StartEndFlags[StartEndFlags.nonzero()[0][-1]] == 1:
                    StartEndFlags[-1] = -1

                StartList = np.argwhere(StartEndFlags == 1).flatten()
                EndList = np.argwhere(StartEndFlags == -1).flatten()
                if not len(StartList) == len(EndList):
                    print("Start and End lists not the same length")
                    print(len(StartList),len(EndList))
                    import pdb
                    pdb.set_trace()
                    
                for i in range(0,len(StartList)):
                    
                    # catch single node cliff lines and ignore
                    if (EndList[i]-StartList[i]<2):
                        continue

                    # create empty lists for storing clifftop and clifftoe nodes
                    FutureList = []
                    
                    # loop through transects and get future positions
                    
                    for Transect in CoastLine.Transects[StartList[i]:EndList[i]]:
                        FutureNode = Transect.get_FutureVegEdge(Year)
                        FutureList.append(FutureNode)
                        
                    # create new line object for top
                    X = [FutureNode.X for FutureNode in FutureList]
                    Y = [FutureNode.Y for FutureNode in FutureList]
                    
                    TempLine = Line("FutureVegEdge_"+str(FutureCount), X, Y, Year=Year)
                    self.FutureVegEdgeLines.append(TempLine)
                    
                    # update counter
                    FutureCount += 1

    def GetBarrierWidth(self):

        """
        
        Gets barrier at a given elevation e.g. high water

        MDH, June 2019

        """
        
        # keep track of no of barrier locations for IDs
        BarrierCount = 0

        # loop through transects and get contiguous barrier lines
        for CoastLine in self.CoastLines:
            
            # find transects with cliffs
            BarrierBool = [Transect.Intersection for Transect in CoastLine.Transects]
            BarrierBool.insert(0, False)
            BarrierBool = np.array(BarrierBool).astype(int)
            
            # get a list of the start and end points of contiguous barrier lines
            StartEndFlags = np.diff(BarrierBool)
            
            # if first element is true this is a start point
            if BarrierBool[0]:
                StartEndFlags[0] = 1
            
            # if last line finishes on a start flag then remove
            if StartEndFlags[-1] == 1:
                StartEndFlags[-1] = 0
            
            # if no start flags
            if len(StartEndFlags.nonzero()[0]) == 0:
                continue
            
            # if last line finishes on last node then flag as end flag
            if StartEndFlags[StartEndFlags.nonzero()[0][-1]] == 1:
                StartEndFlags[-1] = -1
                
            StartList = np.argwhere(StartEndFlags == 1).flatten()
            EndList = np.argwhere(StartEndFlags == -1).flatten()
            
            if not len(StartList) == len(EndList):
                print("Start and End lists not the same length")
                print(len(StartList),len(EndList))
                import pdb
                pdb.set_trace()
                
            for i in range(0,len(StartList)):
                
                # catch single node cliff lines and ignore
                if (EndList[i]-StartList[i]<2):
                    continue

                # create empty lists for storing clifftop and clifftoe nodes
                FrontList = []
                BackList = []

                # loop through transects and get top and toe positions
                
                for Transect in CoastLine.Transects[StartList[i]:EndList[i]]:
                    FrontNode, BackNode = Transect.get_CliffPosition()
                    FrontList.append(FrontNode)
                    BackList.append(BackNode)
                
                # create new line object for front
                X = [FrontNode.X for FrontNode in FrontList]
                Y = [FrontNode.Y for FrontNode in FrontList]
                
                TempLine = Line("Front_"+str(BarrierCount), X, Y)
                self.ExtremeFrontLines.append(TempLine)
                
                # create new line object for toe
                X = [BackNode.X for BackNode in BackList]
                Y = [BackNode.Y for BackNode in BackList]
                
                TempLine = Line("Back_"+str(BarrierCount), X, Y)
                self.ExtremeBackLines.append(TempLine)

                # update counter
                BarrierCount += 1

    def GetCliffLines(self):
        
        """

        Generate line objects from cliff top and cliff toe positions on transects

        MDH, June 2019

        """

        # keep track of no of cliffs for IDs
        CliffCount = 0

        # loop through transects and get contiguous cliff lines
        for CoastLine in self.CoastLines:
            
            # find transects with cliffs
            CliffBool = [Transect.Cliff for Transect in CoastLine.Transects]
            CliffBool.insert(0, False)
            CliffBool = np.array(CliffBool).astype(int)

            # check for transects with no cliffs
            if not any(CliffBool):
                print("No Cliffs on Line")
                continue
            
            # get a list of the start and end points of contiguous cliff lines
            StartEndFlags = np.diff(CliffBool)

            # if first element is true this is a start point
            if CliffBool[0]:
                StartEndFlags[0] = 1
            
            # if last line finishes on a start flag then remove
            if StartEndFlags[-1] == 1:
                StartEndFlags[-1] = 0
            
            # if no start flags
            if len(StartEndFlags.nonzero()[0]) == 0:
                continue
            
            # if last line finishes on a cliff flag the last element as the end of the cliff
            if StartEndFlags[StartEndFlags.nonzero()[0][-1]] == 1:
                StartEndFlags[-1] = -1

            StartList = np.argwhere(StartEndFlags == 1).flatten()
            EndList = np.argwhere(StartEndFlags == -1).flatten()
            if not len(StartList) == len(EndList):
                print("Start and End lists not the same length")
                print(len(StartList),len(EndList))
                
            for i in range(0,len(StartList)):
                
                # catch single node cliff lines and ignore
                if (EndList[i]-StartList[i]<2):
                    continue

                # create empty lists for storing clifftop and clifftoe nodes
                CliffTopList = []
                CliffToeList = []

                # loop through transects and get top and toe positions
                
                for Transect in CoastLine.Transects[StartList[i]:EndList[i]]:
                    TempTop, TempToe = Transect.get_CliffPosition()
                    CliffTopList.append(TempTop)
                    CliffToeList.append(TempToe)
                
                # create new line object for top
                X = [TempTop.X for TempTop in CliffTopList]
                Y = [TempTop.Y for TempTop in CliffTopList]
                
                TempLine = Line("Cliff_"+str(CliffCount), X, Y)
                self.CliffTopLines.append(TempLine)
                
                # create new line object for toe
                X = [TempToe.X for TempToe in CliffToeList]
                Y = [TempToe.Y for TempToe in CliffToeList]
                
                TempLine = Line("Cliff_"+str(CliffCount), X, Y)
                self.CliffToeLines.append(TempLine)

                # update counter
                CliffCount += 1

    def GetBarrierLines(self):
        
        """

        Generate line objects from cliff top and cliff toe positions on transects,
        Also get crest line

        MDH, June 2019

        """

        # keep track of no of cliffs for IDs
        BarrierCount = 0

        # loop through transects and get contiguous barrier lines
        for CoastLine in self.CoastLines:
            
            # find transects with barriers
            BarrierBool = [Transect.Barrier for Transect in CoastLine.Transects]
            BarrierBool.insert(0, False)
            BarrierBool = np.array(BarrierBool).astype(int)
            
            # get a list of the start and end points of contiguous cliff lines
            StartEndFlags = np.diff(BarrierBool)
            
            # if first element is true this is a start point
            if BarrierBool[0]:
                StartEndFlags[0] = 1
            
            # if last line finishes on a start flag then remove
            if StartEndFlags[-1] == 1:
                StartEndFlags[-1] = 0
            
            # if no start flags
            if len(StartEndFlags.nonzero()[0]) == 0:
                continue
            
            # if last line finishes on last node then flag as end flag
            if StartEndFlags[StartEndFlags.nonzero()[0][-1]] == 1:
                StartEndFlags[-1] = -1
                
            StartList = np.argwhere(StartEndFlags == 1).flatten()
            EndList = np.argwhere(StartEndFlags == -1).flatten()

            if not len(StartList) == len(EndList):
                print("Start and End lists not the same length")
                import pdb
                pdb.set_trace()
                
            for i in range(0,len(StartList)):
                
                # catch single node cliff lines and ignore
                if (EndList[i]-StartList[i]<2):
                    continue

                # create empty lists for storing barrier front and back top and toe nodes
                """
                THIS WHOLE THING COULD PROBABLY BE SIMPLIFIED MASSIVELY BY USING __DICT__
                """
                BarrierFrontTopList = []
                BarrierFrontToeList = []
                BarrierBackTopList = []
                BarrierBackToeList = []
                CrestList = []

                # loop through transects and get top and toe positions
                
                for Transect in CoastLine.Transects[StartList[i]:EndList[i]]:
                    TempFrontTop, TempFrontToe, TempBackTop, TempBackToe, TempCrest = Transect.get_BarrierPosition()
                    BarrierFrontTopList.append(TempFrontTop)
                    BarrierFrontToeList.append(TempFrontToe)
                    BarrierBackTopList.append(TempBackTop)
                    BarrierBackToeList.append(TempBackToe)
                    CrestList.append(TempCrest)
                
                # create new line object for front top
                X = [TempTop.X for TempTop in BarrierFrontTopList]
                Y = [TempTop.Y for TempTop in BarrierFrontTopList]
                
                TempLine = Line("Barrier_"+str(BarrierCount), X, Y)
                self.BarrierFrontTopLines.append(TempLine)
                
                # create new line object for front toe
                X = [TempToe.X for TempToe in BarrierFrontToeList]
                Y = [TempToe.Y for TempToe in BarrierFrontToeList]
                
                TempLine = Line("Barrier_"+str(BarrierCount), X, Y)
                self.BarrierFrontToeLines.append(TempLine)

                # create new line object for back top
                X = [TempTop.X for TempTop in BarrierBackTopList]
                Y = [TempTop.Y for TempTop in BarrierBackTopList]
                
                TempLine = Line("Barrier_"+str(BarrierCount), X, Y)
                self.BarrierBackTopLines.append(TempLine)
                
                # create new line object for back toe
                X = [TempToe.X for TempToe in BarrierBackToeList]
                Y = [TempToe.Y for TempToe in BarrierBackToeList]
                
                TempLine = Line("Barrier_"+str(BarrierCount), X, Y)
                self.BarrierBackToeLines.append(TempLine)

                # create new line object for crest
                X = [TempCrest.X for TempCrest in CrestList]
                Y = [TempCrest.Y for TempCrest in CrestList]
                
                TempLine = Line("Crest_"+str(BarrierCount), X, Y)
                self.CrestLines.append(TempLine)

                # update counter
                BarrierCount += 1

    def GetExtremeLines(self):
        
        """

        Generate line objects from extreme water positions on transects for front feature,
        
        MDH, July 2019

        """

        # loop through extreme water levels
        for i, Level in enumerate(["Low", "Med","High"]):
            
            # keep track of no of cliffs for IDs
            Count = 0
            
            # loop through transects and get contiguous extreme lines
            for CoastLine in self.CoastLines:
            
                # find transects with cliffs
                Widths = np.array([Transect.ExtremeWidths[i] for Transect in CoastLine.Transects])
                ExtremeBool = np.array([False if Width is None else True for Width in Widths])
                ExtremeBool[Widths==0] = False
                ExtremeBool = np.insert(ExtremeBool, 0, False)
                ExtremeBool = np.array(ExtremeBool).astype(int)
                
                # get a list of the start and end points of contiguous cliff lines
                StartEndFlags = np.diff(ExtremeBool)
                
                # if first element is true this is a start point
                if ExtremeBool[0]:
                    StartEndFlags[0] = 1
                
                # if last line finishes on a start flag then remove
                if StartEndFlags[-1] == 1:
                    StartEndFlags[-1] = 0
                
                # if no start flags
                if len(StartEndFlags.nonzero()[0]) == 0:
                    continue
                
                # if last line finishes on last node then flag as end flag
                if StartEndFlags[StartEndFlags.nonzero()[0][-1]] == 1:
                    StartEndFlags[-1] = -1
                    
                StartList = np.argwhere(StartEndFlags == 1).flatten()
                EndList = np.argwhere(StartEndFlags == -1).flatten()

                if not len(StartList) == len(EndList):
                    print("Start and End lists not the same length")
                    import pdb
                    pdb.set_trace()


                for j in range(0,len(StartList)):
                    
                    # catch single node cliff lines and ignore
                    if (EndList[j]-StartList[j]<2):
                        continue

                    # create empty lists for storing barrier front and back top and toe nodes
                    """
                    THIS WHOLE THING COULD PROBABLY BE SIMPLIFIED MASSIVELY BY USING __DICT__
                    """
                    ExtremeFrontList = []
                    ExtremeBackList = []
                    
                    # loop through transects and get front and back positions
                    for Transect in CoastLine.Transects[StartList[j]:EndList[j]]:
                        try:
                            TempFront, TempBack  = Transect.get_ExtremePosition(i)
                            ExtremeFrontList.append(TempFront)
                            ExtremeBackList.append(TempBack)
                        except:
                            continue
                    
                    if len(ExtremeFrontList) < 2:
                        continue
                        
                    # create new line object for front 
                    X = [TempFront.X for TempFront in ExtremeFrontList]
                    Y = [TempFront.Y for TempFront in ExtremeFrontList]
                    
                    TempLine = Line("Ext_"+Level+str(Count), X, Y)
                    self.__dict__["ExtFrontLines_"+Level].append(TempLine)
                    
                    # create new line object for back
                    X = [TempBack.X for TempBack in ExtremeBackList]
                    Y = [TempBack.Y for TempBack in ExtremeBackList]
                    
                    TempLine = Line("Ext_"+Level+str(Count), X, Y)
                    self.__dict__["ExtBackLines_"+Level].append(TempLine)

                    # update counter
                    Count += 1

    def GetExtremeExtent(self):

        """
        Generates shapefiles of the lowest elevation extreme water
        extent that is providing some sort of protective function

        MDH, October 2019

        """

        # loop through extreme water levels
        i = 0
        Level = "Low"
        
        # keep track of no of lines for IDs
        Count = 0
            
        # loop through transects and get contiguous extreme lines
        for CoastLine in self.CoastLines:
        
            # find transects with coastal protection
            ExtremeBool = ([any(isinstance(Transect.Intersections,float)) for Transect in CoastLine.Transect])
            ExtremeBool = np.insert(ExtremeBool, 0, False)
            ExtremeBool = np.array(ExtremeBool).astype(int)
            
            # get a list of the start and end points of contiguous sections with protection
            StartEndFlags = np.diff(ExtremeBool)
            
            # if first element is true this is a start point
            if ExtremeBool[0]:
                StartEndFlags[0] = 1
            
            # if last line finishes on a start flag then remove
            if StartEndFlags[-1] == 1:
                StartEndFlags[-1] = 0
            
            # if no start flags
            if len(StartEndFlags.nonzero()[0]) == 0:
                continue
            
            # if last line finishes on last node then flag as end flag
            if StartEndFlags[StartEndFlags.nonzero()[0][-1]] == 1:
                StartEndFlags[-1] = -1
            
            # start flag is gradient = 1, end flag where gradient = -1
            StartList = np.argwhere(StartEndFlags == 1).flatten()
            EndList = np.argwhere(StartEndFlags == -1).flatten()

            if not len(StartList) == len(EndList):
                print("Start and End lists not the same length")
                import pdb
                pdb.set_trace()


            for j in range(0,len(StartList)):
                
                # catch single node cliff lines and ignore
                if (EndList[j]-StartList[j]<2):
                    continue

                # create empty lists for storing barrier front and back top and toe nodes
                """
                THIS WHOLE THING COULD PROBABLY BE SIMPLIFIED MASSIVELY BY USING __DICT__
                """
                ExtremeFrontList = []
                ExtremeBackList = []
                
                # loop through transects and get front and back positions
                for Transect in CoastLine.Transects[StartList[j]:EndList[j]]:
                    try:
                        TempFront, TempBack  = Transect.get_ExtremePosition(i)
                        ExtremeFrontList.append(TempFront)
                        ExtremeBackList.append(TempBack)
                    except:
                        continue
                
                if len(ExtremeFrontList) < 2:
                    continue
                    
                # create new line object for front 
                X = [TempFront.X for TempFront in ExtremeFrontList]
                Y = [TempFront.Y for TempFront in ExtremeFrontList]
                
                TempLine = Line("Ext_"+Level+str(Count), X, Y)
                self.__dict__["ExtFrontLines_"+Level].append(TempLine)
                
                # create new line object for back
                X = [TempBack.X for TempBack in ExtremeBackList]
                Y = [TempBack.Y for TempBack in ExtremeBackList]
                
                TempLine = Line("Ext_"+Level+str(Count), X, Y)
                self.__dict__["ExtBackLines_"+Level].append(TempLine)

                # update counter
                Count += 1

    def SetMHWS(self,MHWS):

        """
        Sets MHWS on all lines and transects
        Could be replaced with spatially dynamic data later

        MDH, July 2019

        """
        # set MHWS
        self.MHWS = MHWS

        # loop through lines and plot profiles #
        for Line in self.CoastLines:
            for Transect in Line.Transects:
                Transect.MHWS = MHWS

    def SetShorefaceDepth(self,Dsf):

        """
        Sets shoreface depth on all lines and transects
        Could be replaced with spatially dynamic data later

        MDH, November 2019

        """
        # set Shoreface Depth
        self.Dsf = Dsf

        # loop through lines and 
        for Line in self.CoastLines:
            for Transect in Line.Transects:
                Transect.ClosureDepth = Dsf

    def PlotTransects(self, PlotFolder, ReverseFlag=False):
        
        """

        Description goes here

        MDH, June 2019

        """

        #import figure plotting stuff here not globally!
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt

        print("Coast.PlotTransects: Plotting each transect topographic profile")

        # Track progress
        NoTransects = np.sum([Line.NoTransects for Line in self.CoastLines])-1
        CurrentTransect = 0

        # loop through lines and plot profiles #
        for Line in self.CoastLines:
            for Transect in Line.Transects:
                
                # print progress to screen
                print(" \r\tTransect %3d / %3d" % (CurrentTransect, NoTransects), end="")

                # call plotting function
                Transect.Plot(PlotFolder, ReverseFlag)
                    
                CurrentTransect += 1

        print("")

    def PlotBarrierProperties(self, PlotFolder):
        """
        """

        #import figure plotting stuff here not globally!
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt

        # set up a figure
        # in time might want to automatically adjust figure for coast orientation
        fig, ax = plt.figure(figsize=(8,4))
        
        for Line in self.Coastlines:
            
            # get property to plot
            W  = [Transect.ToeWidth for Transect in Line.Transects]
            ax.plot(W,range(0,len(W)),'k-',lw=2)
        
        ax.set_xlabel("Barrier Width at Toe (m)")
        ax.set_ylabel("Transect ID")
        fig.savefig(PlotFolder + "BarrierWidth.png")

        fig.clear()
        plt.close(fig)
    
    def PlotPositions():

        #import figure plotting stuff here not globally!
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt

        print("Coast.PlotPositions: Plotting transect positions")

        # Track progress
        NoTransects = np.sum([Line.NoTransects for Line in self.CoastLines])-1
        CurrentTransect = 0

        # loop through lines and plot profiles #
        for Line in self.CoastLines:
            for Transect in Line.Transects:
                
                # print progress to screen
                print(" \r\tTransect %3d / %3d" % (CurrentTransect, NoTransects), end="")

                # call plotting function
                #if Transect.ID == "0":
                Transect.PlotPositions(PlotFolder)
                    
                CurrentTransect += 1

        print("")

    def get_MeanHistoricErosion(self):

        """ 
        
        Function to calculate the mean historic erosion on transects
        
        MDH, March 2021
        
        """

        HistoricRates = []

        for Line in self.CoastLines:
            for Transect in Line.Transects:

                if not Transect.Future:
                    continue

                HistoricRate = Transect.ChangeRate
                
                if not HistoricRate:
                    continue
                elif not HistoricRate < 0:
                    continue
                else:
                    HistoricRates.append(HistoricRate)
        
        NErodingTransects = len(HistoricRates)
        MeanErosionRate = np.nan_to_num(np.mean(HistoricRates))

        return NErodingTransects, MeanErosionRate

    def get_MeanDC1Erosion(self):

        """ 
        
        Function to calculate the mean historic erosion on transects from DC1 data
        
        MDH, March 2021
        
        """

        HistoricRates = []

        for Line in self.CoastLines:
            for Transect in Line.Transects:

                if not Transect.Future:
                    continue
                    
                if not Transect.DC1:
                    continue
                
                HistoricRate = Transect.DC1[2]/(Transect.DC1[1]-Transect.DC1[0])
                
                if not HistoricRate:
                    continue
                elif not HistoricRate < 0:
                    continue
                else:
                    HistoricRates.append(HistoricRate)
        
        NErodingTransects = len(HistoricRates)
        MeanErosionRate = np.nan_to_num(np.mean(HistoricRates))
        
        return NErodingTransects, MeanErosionRate

    def get_MeanTotalErosion(self, Decade=2100):

        """ 
        
        Function to calculate the mean total erosion on transects for future predictions
        
        MDH, March 2021
        
        """

        ErosionDistances = []

        for Line in self.CoastLines:
            for Transect in Line.Transects:

                if not Transect.Future:
                    continue

                ErosionDistance = Transect.get_TotalErosion(2020,Decade)
                
                if not ErosionDistance:
                    continue
                elif not ErosionDistance < 0:
                    continue
                else:
                    ErosionDistances.append(ErosionDistance)
        
        NErodingTransects = len(ErosionDistances)
        MeanTotalErosion = np.nan_to_num(np.mean(ErosionDistances))

        return NErodingTransects, MeanTotalErosion

    def get_NumberOfTransects(self, Future=True):

        """

        Returns the total number of transects on all lines in the object

        """

        NoTransects = 0

        for Line in self.CoastLines:
            
            FutureTransects = [Transect.Future for Transect in Line.Transects]
            NoTransects += FutureTransects.count(True)
            
        return NoTransects

    def get_RecentShorelinesYearsList(self):

        """

        Function to generate a list of the most recent shorelines

        MDH, March 2021

        """
        List = []
        for Line in self.CoastLines:
            for Transect in Line.Transects:
                List.append(Transect.get_RecentYear())
        
        return List

    def get_ErosionDistancesList(self, Decade=2100):

        """
        Function to generate a list of the most recent shoreline distances

        MDH, March 2021

        """

        ErosionDistances = []

        for Line in self.CoastLines:
            for Transect in Line.Transects:

                if not Transect.Future:
                    continue

                ErosionDistance = Transect.get_TotalErosion(2020,Decade)
                
                if not ErosionDistance:
                    continue
                elif not ErosionDistance < 0:
                    continue
                else:
                    ErosionDistances.append(ErosionDistance)
        
        return ErosionDistances