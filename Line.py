"""
Description of file goes here

Martin D. Hurst
University of Glasgow
June 2019

"""

# import modules
import sys
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev
#from Toolshed import Transect, Node
from Node import *
from Transect import *

import geopandas as gp
from shapely.geometry import Point, LineString, MultiLineString, Polygon, MultiPolygon
from shapely.ops import nearest_points, linemerge

import pdb

class Line:
    
    """
    """

    def __init__(self, ID, X, Y, Contour=None, Year=None, Cell=None, SubCell=None, CMU=None, Flag=None):
        """
        """
        self.ID = ID
        self.Cell = Cell
        self.SubCell = SubCell
        self.CMU = CMU
        self.Year = Year
        self.NoNodes = 0
        self.Nodes = []
        self.RawNodes = []
        self.Projection = ""
        self.Orientation = []
        self.Curvature = []
        self.SegmentLength = []
        self.TotalLength = 0
        self.Transects = []
        self.NoTransects = 0
        self.Points = []
        self.NoPoints = 0
        self.Contour = Contour
        self.GenerateNodes(X, Y)
        self.Flag = Flag

    def __str__(self):
        """
        """
        String = "Line Object:\nID: %s\nNoNodes: %d\nLength: %.2f" % (str(self.ID), self.NoNodes, self.TotalLength)
        return String

    def GenerateNodes(self, X, Y):
        """
        Function to convert X and Y data into Nodes
        """
        # check X and Y are same length
        if len(X) != len(Y):
            sys.exit("Line.GenerateNodes(ERROR): X and Y vectors are not same length.\n\t \
length of X: %d\n\tlength of Y:%d\n\n" % (len(X),len(Y)))

        # reset node list
        self.Nodes = []

        # set the number of nodes on the line
        self.NoNodes = len(X)

        # loop through and create node list
        for x, y in zip(X, Y):
            self.Nodes.append(Node(x,y))
        
        self.CalculateGeometry()
        
        if not self.RawNodes:
            self.RawNodes = self.Nodes

    def ResampleNodes(self, ResampleInterval=10.):
        
        # Get X and Y vectors from Nodes
        X, Y = self.get_XY()

        # set up lists to store new X and Y and put first values
        XNew = [X[0],]
        YNew =[Y[0],]

        # Parameters for tracing along length
        CumulativeLength = 0.
        NextPosition = ResampleInterval

        # Track spacing and generate profile at desired distances
        for i in range(0, self.NoNodes):

            #Update the cumulative length of the line
            CumulativeLength += self.SegmentLength[i]

            # get orientation
            TempOrientation = self.Orientation[i]
            
            # Test to see if we're going to create a new node
            while CumulativeLength > NextPosition:

                #calculate point for section
                DistanceToStepBack = CumulativeLength - NextPosition
                dX = DistanceToStepBack * np.sin( np.radians( TempOrientation ) )
                dY = DistanceToStepBack * np.cos( np.radians( TempOrientation ) )
                
                # find the point for the transect along the line
                XNew.append(self.Nodes[i+1].X - dX)
                YNew.append(self.Nodes[i+1].Y - dY)

                # update to find next point
                NextPosition += ResampleInterval
        
        # add last node
        XNew.append(X[-1])
        YNew.append(Y[-1])

        # Write new X and Y vectors to Nodes and recalc geometry
        self.GenerateNodes(XNew,YNew)
        self.CalculateGeometry()

    def CalculateGeometry(self):
        
        """
        Calculate the orientation, curvature and length along the line
        Orientation is the direction towards the next node in the vector
        Curvature is the difference in orientation between two segments
        SegmentLength is the distance to the next node in the vector

        MDH, June 2019

        """
        # reset arrays
        self.Orientation = np.ones(self.NoNodes)*-9999
        self.SegmentLength = np.ones(self.NoNodes)*-9999
        self.TotalLength = 0

        # loop through the nodes
        for i in range(0,self.NoNodes-1):
            
            # Get the two nodes
            ThisNode = self.Nodes[i]
            NextNode = self.Nodes[i+1]

            #calculate the spatial change
            dx = NextNode.X - ThisNode.X
            dy = NextNode.Y - ThisNode.Y

            #Calculate the orientation of the line from ThisNode to NextNode
            if dx > 0 and dy > 0:
                self.Orientation[i] = np.degrees( np.arctan( dx / dy ) )
            elif dx > 0 and dy < 0:
                self.Orientation[i] = 180.0 + np.degrees( np.arctan( dx / dy ) )
            elif dx < 0 and dy < 0:
                self.Orientation[i] = 180.0 + np.degrees( np.arctan( dx / dy ) )
            elif dx < 0 and dy > 0:
                self.Orientation[i] = 360 + np.degrees( np.arctan( dx / dy ) )
            
            #Calculate the length of the segment
            self.SegmentLength[i] = np.sqrt(dx**2. + dy**2.)

            #Update the cumulative length of the line
            self.TotalLength += self.SegmentLength[i]

        # Properties of last node
        self.Orientation[-1] = self.Orientation[-2]
        self.SegmentLength[-1] = 0
        
    def SmoothLine(self, WindowSize=1001, PolyOrder=2):
        
        """
        Savitzky and Golay (1964) smoothing filter
            
        Savitzky, A. and Golay, M. J.: Smoothing and differentiation of data
            by simplified least squares procedures, Anal. Chem., 36, 1627-
            1639, 1964.
        """

        # Get X and Y vectors from Nodes
        X, Y = self.get_XY()
        
        # smooth X and Y individually with Savitzky Golay filter
        # window size and polyorder must be integers you idiot!
        XSmooth = savgol_filter(X,WindowSize,PolyOrder, mode="mirror")
        YSmooth = savgol_filter(Y,WindowSize,PolyOrder, mode="mirror")

        # add functions to insert first and last node again
        XSmooth = np.insert(XSmooth,0,X[0])
        YSmooth = np.insert(YSmooth,0,Y[0])
        XSmooth = np.append(XSmooth,X[-1])
        YSmooth = np.append(YSmooth,Y[-1])

        # copy nodes to raw
        self.RawNodes = self.Nodes
        
        # Write new X and Y vectors to Nodes
        self.GenerateNodes(XSmooth,YSmooth)
        self.CalculateGeometry()
    
    def MakeSimple(self):

        """

        Function to find and remove complexities (loops) in a line"

        MDH, Jan, 20201

        """

        # Get X and Y vectors from Nodes and write LineString object
        X, Y = self.get_XY()
        LS = LineString(zip(X,Y))

        while not LS.is_simple:
            
            #"Union" method will split self-intersection linestring.
            Result = LS.union(Point(X[0],Y[0]))
            
            # use result to get list of self intersections somehow
            
            # isolate non-looping line segments
            try:
                Lines2Merge = [L for L in Result if not Point(L.coords[0]).distance(Point(L.coords[-1])) < 1]
                LS = linemerge(Lines2Merge)
            
            except:
                LS = Result
                
            # merge lines that do not loop
            while LS.type == "MultilineString":
                Lines2Merge = [L for L in LS if L.is_simple]
                LS = linemerge(Lines2Merge)
                
            
        # Write new X and Y vectors to Nodes
        #X, Y = LS.coords.xy

        try:
            X, Y = LS.coords.xy
            self.GenerateNodes(X,Y)
            self.CalculateGeometry()
        except:
            return
                
    def SplineLine(self):

        """

        Function to populate a spline of the nodes and resample to regular distances
        This could be called in several places within Coast object...

        MDH, March 2020
        
        """

        # Get X and Y vectors from Nodes
        X, Y = self.get_XY()

        XSmooth = np.array(X[1:-1])
        YSmooth = np.array(Y[1:-1])
        
        # calculate distance
        Dist = np.zeros(XSmooth.shape)
        Dist[1:] = np.sqrt((XSmooth[1:] - XSmooth[:-1])**2 + (YSmooth[1:] - YSmooth[:-1])**2)
        Dist = np.cumsum(Dist)
        
        # build a spline representation of the line
        K = 3 # by default

        if len(XSmooth) < 2:
            return

        elif len(XSmooth) < 4:
            K = len(XSmooth)-1

        Spline, u = splprep([XSmooth, YSmooth], u=Dist, s=0, k=K)

        # resample it at smaller distance intervals
        Interp_Dist = np.arange(0, Dist[-1], 1.)
        XSmooth, YSmooth = splev(Interp_Dist, Spline)

        XSmooth = np.insert(XSmooth,0,X[0])
        YSmooth = np.insert(YSmooth,0,Y[0])
        X = np.append(XSmooth,X[-1])
        Y = np.append(YSmooth,Y[-1])

        # copy nodes to raw
        self.RawNodes = self.Nodes

        # Write new X and Y vectors to Nodes
        self.GenerateNodes(XSmooth,YSmooth)
        self.CalculateGeometry()

    def GenerateBuffer(self, Dist1, Dist2):
        
        """
        Description goes here

        MDH, June 2019

        """

        # empty lists for new nodes
        BufferNodesLeft = []
        BufferNodesRight = []

        # Orientation increments by 1 degree when rounding required
        OrientationInc = 1.

        # Node Counter to give each node a unique ID
        NodeCounter = 0

        # loop through nodes 
        for i, ThisNode in enumerate(self.NoNodes):
            
            # this section of code could definately be more efficient
            # or better written but will do for now

            # check if line is convex/concave left
            if not self.Orientation[i] < self.Orientation[i-1]:
                        
                # find point perpendicular to orientation on left side
                TempOrientation = self.Orientation[i]
                XL = ThisNode.X + Dist1 * np.sin( np.radians (TempOrientation-90.) )
                YL = ThisNode.Y + Dist1 * np.cos( np.radians (TempOrientation-90.) )
                BufferNodesLeft.append(Node(NodeCounter, XL, YL))
                NodeCounter += 1

                # increment orientation to complete radius
                while TempOrientation < self.Orientation[i+1]:
                    TempOrientation += OrientationInc
                    XL = ThisNode.X + Dist1 * np.sin( np.radians (TempOrientation-90.) )
                    YL = ThisNode.Y + Dist1 * np.cos( np.radians (TempOrientation-90.) )
                    BufferNodesLeft.append(Node(NodeCounter, XL, YL))
                    NodeCounter += 1

                # find point on right perpendicular to mean orientation
                TempOrientation = np.mean(self.Orientation[i-1:i+1])
                XR = ThisNode.X + Dist2 * np.sin( np.radians (TempOrientation+90.) )
                YR = ThisNode.X + Dist2 * np.sin( np.radians (TempOrientation+90.) )
                BufferNodesRight.append(Node(NodeCounter, XR, YR))
                NodeCounter += 1

            else:

                # find point perpendicular to orientation on right side
                TempOrientation = self.Orientation[i]
                XR = ThisNode.X + Dist2 * np.sin( np.radians (TempOrientation+90.) )
                YR = ThisNode.Y + Dist2 * np.cos( np.radians (TempOrientation+90.) )
                BufferNodesRight.append(Node(NodeCounter, XR, YR))
                NodeCounter += 1

                # increment orientation to complete radius
                while TempOrientation < self.Orientation[i+1]:
                    TempOrientation += OrientationInc
                    XR = ThisNode.X + Dist2 * np.sin( np.radians (TempOrientation+90.) )
                    YR = ThisNode.Y + Dist2 * np.cos( np.radians (TempOrientation+90.) )
                    BufferNodesRight.append(Node(NodeCounter, XR, YR))
                    NodeCounter += 1

                # find point on right perpendicular to mean orientation
                TempOrientation = np.mean(self.Orientation[i-1:i+1])
                XL = ThisNode.X + Dist1 * np.sin( np.radians (TempOrientation-90.) )
                YL = ThisNode.X + Dist1 * np.sin( np.radians (TempOrientation-90.) )
                BufferNodesLeft.append(Node(NodeCounter, XL, YL))
                NodeCounter += 1

        return Line(XL,YL,"LeftBuffer"), Line(XL,YL,"RightBuffer")


    def GenerateTransects(self, Spacing=10., TransectLength2Sea=5000., TransectLength2Land=5000., CheckTopology=True):
        """
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
            Check for overlapping transects and correct. Default is True
        
        """

        # print("Line: Generating Transects perpendicular to the coast")
        
        # if rewriting Transects, empty the Transects list
        if len(self.Transects) != 0:
            self.Transects = []
            self.Points = []

        # Give each transect unique ID
        TransectCount = 0
        
        # Parameters for tracing along length
        CumulativeLength = 0.0
        NextPosition = Spacing

        # Track spacing and generate profile at desired distances
        for i in range(0, self.NoNodes):

            #Update the cumulative length of the line
            CumulativeLength += self.SegmentLength[i]

            # get orientation
            TempOrientation = self.Orientation[i]
            
            # Test to see if we're going to create a cross section
            while CumulativeLength > NextPosition:

                #calculate point for section
                DistanceToStepBack = CumulativeLength - NextPosition
                dX = DistanceToStepBack * np.sin( np.radians( TempOrientation ) )
                dY = DistanceToStepBack * np.cos( np.radians( TempOrientation ) )
                
                # find the point for the transect along the line
                PointX = self.Nodes[i+1].X - dX
                PointY = self.Nodes[i+1].Y - dY

                #Create cross section line
                #Get line orientation
                if TempOrientation < 0:
                    TransectOrientation = TempOrientation + 90.
                else:
                    TransectOrientation = TempOrientation - 90.

                #Calculate start and end nodes and generate Transect
                X1 = PointX + TransectLength2Sea * np.sin( np.radians( TransectOrientation ) )
                Y1 = PointY + TransectLength2Sea * np.cos( np.radians( TransectOrientation ) )
                X2 = PointX - TransectLength2Land * np.sin( np.radians( TransectOrientation ) )
                Y2 = PointY - TransectLength2Land * np.cos( np.radians( TransectOrientation ) )
                self.Transects.append( Transect( Node(PointX, PointY), Node(X1, Y1), Node(X2, Y2), str(self.ID), str(TransectCount) ) )

                # update to find next transect
                TransectCount += 1
                NextPosition += Spacing
        
        # record number of transects
        self.NoTransects = TransectCount   

        # check for overlaps?
        if CheckTopology:
            self.CheckTransectTopology()     

    def GenerateTransectsFromContour(self, ContourShp, Spacing):

        """

        Generates regularly spaced transects along the coastline by 
        finding the nearest point in another line dataset and drawing
        connecting lines
    
        MDH, August 2019

        Parameters
        ----------
        Spacing : float
            The distance between consecutive points along the CoastLines
            in map units, spatial units depend on units of the CoastLine read in,
            Should be [m]

        """

        # if rewriting Transects, empty the Transects list
        if len(self.Points) != 0:
            self.Transects = []
            self.Points = []
        
        # generate points along the line
        self.GeneratePoints(Spacing)

        # load the contour shapefile
        GDF = gp.read_file(ContourShp)
        Lines = GDF['geometry']
        
        # make a multlinestring if there are multiple lines
        LineList = []
        for LineObj in Lines:
            if (LineObj.geom_type == "MultiLineString"):
                for ThisLine in LineObj:
                    LineList.append(ThisLine)
            else:
                LineList.append(LineObj)

        Lines = MultiLineString(LineList)
        
        for ThisPoint in self.Points:
                
                # find nearest point in contour lines
                BasePoint = Point(ThisPoint.X, ThisPoint.Y)
                NearestPoint = nearest_points(Lines, BasePoint)[0]
                
                # build transect using these two points
                self.Transects.append(Transect(str(self.ID), str(ThisPoint.ID), Node(NearestPoint.x, NearestPoint.y), Node(BasePoint.x, BasePoint.y), Node(NearestPoint.x, NearestPoint.y)))

    def CheckLineOrientation(self, ShorelineShp, OffshoreShp):

        """

        Checks a line is in the right orientation by comparison to a coastline and a bathymetry line
    
        MDH, May 2020

        Parameters
        ----------

        ContourShp1 : string
            Name of a shapefile with the first line/contour to look for when
            drawing transects. This should be the line nearest to the coast
        ContourShp2: string
            Name of a shapefile wit hthe second line/contour to look for when
            drawing transects. This should be the offshore line.
        """

        #print("\tChecking Geometry, Line", self.ID)
        #if self.ID == "3":
        #    import pdb
        #    pdb.set_trace()
            
        # load the contour shapefile
        GDF = gp.read_file(ShorelineShp)
        Lines = GDF['geometry']
        
        # make a multlinestring if there are multiple lines
        LineList = []
        for LineObj in Lines:
            if not LineObj:
                continue
            elif (LineObj.geom_type == "MultiLineString"):
                for ThisLine in LineObj:
                    LineList.append(ThisLine)
            elif (LineObj.geom_type == "LineString"):
                LineList.append(LineObj)
            else:
                sys.exit("problem reading lines")

        # catch situation where only one line
        if len(LineList) == 1:
            ShoreLines = LineList[0]
        else:
            ShoreLines = MultiLineString(LineList)

        # load the second contour shapefile
        GDF = gp.read_file(OffshoreShp)
        Lines = GDF['geometry']
        
        # make a multlinestring if there are multiple lines
        LineList = []
        for LineObj in Lines:
            if not LineObj:
                continue
            elif (LineObj.geom_type == "MultiLineString"):
                for ThisLine in LineObj:
                    LineList.append(ThisLine)
            elif (LineObj.geom_type == "LineString"):
                LineList.append(LineObj)
            else:
                sys.exit("problem reading lines")

        # catch situation where only one line
        if len(LineList) == 1:
            BathyLines = LineList[0]
        else:
            BathyLines = MultiLineString(LineList)

        #  define some temporary initial transect lines
        if not self.NoTransects:
            self.GenerateTransects(CheckTopology=False)

        # set up some flags for deciding on whether to reverse
        self.ReverseFlags = np.zeros(self.NoTransects)

        # check orientation relative to the sea for a 
        # intersect first transect with each set of lines and get orientation from bathy to shore
        # find intersection between transect line and shapefile lines
        for i, ThisTransect in enumerate(self.Transects):
            
            # get offshore points to measure distances
            #OffshorePoint = nearest_points(BathyLines, Point(ThisTransect.CoastNode.X, ThisTransect.CoastNode.Y))[0]
            TransectLine = LineString(((ThisTransect.StartNode.X,ThisTransect.StartNode.Y),(ThisTransect.EndNode.X,ThisTransect.EndNode.Y)))
            Intersections = TransectLine.intersection(BathyLines)
            
            # catch no intersections
            if Intersections.geom_type == "GeometryCollection":
                continue

            # check there arent multiple intersections
            # get first intersection if so
            if Intersections.geom_type is "MultiPoint":
                CoastPoint = Point(ThisTransect.CoastNode.X, ThisTransect.CoastNode.Y)
                Distances = [IntersectPoint.distance(CoastPoint) for IntersectPoint in Intersections]
                Index = Distances.index(min(Distances))
                Intersection = Intersections[Index]
                
            else:
                # check if this is a new endnode by intersecting with line from startnode to endnode
                Intersection = Intersections
                
            OffshoreNode = Node(Intersection.x,Intersection.y)
            Distance1 = OffshoreNode.get_Distance(ThisTransect.StartNode)
            Distance2 = OffshoreNode.get_Distance(ThisTransect.EndNode)
            
            if (Distance1 > Distance2):
                self.ReverseFlags[i] = 1

            else:
                self.ReverseFlags[i] = -1
        
        # get x and y to reverse lines
        NReverse = np.count_nonzero(self.ReverseFlags == 1)
        NXReverse = np.count_nonzero(self.ReverseFlags == -1)
        
        if NReverse > NXReverse:
            X, Y = self.get_XY()
            self.__init__(self.ID, X[::-1], Y[::-1], self.Contour, self.Year, self.Cell, self.SubCell, self.CMU)
            
            # regenerate transects
            self.GenerateTransects(CheckTopology=False)

    def GenerateTransectsBetweenContours(self, ContourShp1, ContourShp2, Spacing, Distance2Sea=5000., Distance2Land=8000., CheckTopology=True):

        """

        Generates regularly spaced transects along the coastline by 
        finding the nearest point in another line dataset and drawing
        connecting lines
    
        MDH, August 2019

        Parameters
        ----------

        ContourShp1 : string
            Name of a shapefile with the first line/contour to look for when
            drawing transects. This should be the line nearest to the coast
        ContourShp2: string
            Name of a shapefile wit hthe second line/contour to look for when
            drawing transects. This should be the offshore line.
        Spacing : float
            The distance between consecutive points along the CoastLines
            in map units, spatial units depend on units of the CoastLine read in,
            Should be [m]
        Distance2Land : float
            Distance in [m] to extent transects landward when looking for (Bathy
            with ContourShp1
        Distance2Sea : float
            Distance in [m] to extent transects offshore when looking for intersection
            with ContourShp2
        CheckTopology : bool
            Flag to check and correct overlapping transects
        """

        self.CheckLineOrientation(ContourShp1, ContourShp2)

        self.GenerateTransects(Spacing, Distance2Sea, Distance2Land, CheckTopology=False)

        # load the contour shapefile
        GDF = gp.read_file(ContourShp1)
        Lines = GDF['geometry']
        
        # make a multlinestring if there are multiple lines
        LineList = []
        for LineObj in Lines:
            if not LineObj:
                continue
            elif (LineObj.geom_type == "MultiLineString"):
                for ThisLine in LineObj:
                    LineList.append(ThisLine)
            elif (LineObj.geom_type == "LineString"):
                LineList.append(LineObj)
            else:
                sys.exit("problem reading lines")

        # catch situation where only one line
        if len(LineList) == 1:
            Lines1 = LineList[0]
        else:
            Lines1 = MultiLineString(LineList)

        # load the second contour shapefile
        GDF = gp.read_file(ContourShp2)
        Lines = GDF['geometry']
        
        # make a multlinestring if there are multiple lines
        LineList = []
        for LineObj in Lines:
            if not LineObj:
                continue
            elif (LineObj.geom_type == "MultiLineString"):
                for ThisLine in LineObj:
                    LineList.append(ThisLine)
            elif (LineObj.geom_type == "LineString"):
                LineList.append(LineObj)
            else:
                sys.exit("problem reading lines")

        # catch situation where only one line
        if len(LineList) == 1:
            Lines2 = LineList[0]
        else:
            Lines2 = MultiLineString(LineList)

        # get points to define initial transect line and make it nice and long
        #\ add if statement here
        #self.GenerateTransects(Spacing, Distance2Sea, Distance2Land, CheckTopology=False)

        # flag to note interesections
        CheckTopologyFlag = CheckTopology
        Intersections = True

        while Intersections:
            
            # intersect Transect with shapefile to find new end node of transect
            DeleteFlags = np.ones(len(self.Transects))

            for i, Transect in enumerate(self.Transects):
            
                # find intersection between transect line and shapefile lines
                Intersection = Transect.LineString.intersection(Lines2)
                
                # catch no intersections
                if Intersection.geom_type != "GeometryCollection":
                
                    # check there arent multiple intersections, if there are just get the nearest
                    if Intersection.geom_type is "MultiPoint":
                        StartPoint = Point(Transect.CoastNode.X, Transect.CoastNode.Y)
                        Distances = [IntersectPoint.distance(StartPoint) for IntersectPoint in Intersection]
                        Index = Distances.index(min(Distances))
                        Intersection = Intersection[Index]
                    
                    # set this as the new start node
                    NewStartNode = Node(Intersection.x,Intersection.y)
                
                else:
                    NewStartNode = Transect.StartNode

                # rebuild transect with new start node here.
                Transect.__init__(Transect.CoastNode, NewStartNode, Transect.EndNode, Transect.LineID, Transect.ID)

                # now do the same with the raw coastline data (i.e. the original contour)
                Intersection = Transect.LineString.intersection(Lines1)
                
                # catch no intersections
                if Intersection.geom_type != "GeometryCollection":
                    
                    # check there arent multiple intersections, if there are just get the nearest
                    if Intersection.geom_type is "MultiPoint":
                        StartPoint = Point(Transect.CoastNode.X, Transect.CoastNode.Y)
                        Distances = [IntersectPoint.distance(StartPoint) for IntersectPoint in Intersection]
                        Index = Distances.index(min(Distances))
                        Intersection = Intersection[Index]
                        
                    # set this as the new end node
                    NewEndNode = Node(Intersection.x,Intersection.y)

                    # reinitialise transect with new startnode and new endnode
                    Transect.__init__(Transect.CoastNode, NewStartNode, NewEndNode, Transect.LineID, Transect.ID)

                else:
                    DeleteFlags[i] = 0
                
            self.Transects = [Transect for n, Transect in enumerate(self.Transects) if DeleteFlags[n] == 1]
                
            # check for overlaps?
            # if CheckTopologyFlag:
            #    Intersections = self.CheckTransectTopology()
            #    CheckTopologyFlag = False
            #else:
            
            Intersections = False   

        if CheckTopology:
            self.DeleteOverlappingTransects()
        
        for i, Transect in enumerate(self.Transects):
            Transect.ID = str(i)

    def GenerateMidpointLineBetweenContours(self, ContourShp1, ContourShp2, Spacing, Distance2Sea=5000., Distance2Land=8000., CheckTopology=True):

        """

        Generates a line comprising points based on the midpoint of regulaarly spaced
        transects between two contours
    
        MDH, July 2020

        Parameters
        ----------

        ContourShp1 : string
            Name of a shapefile with the first line/contour to look for when
            drawing transects. This should be the line nearest to the coast
        ContourShp2: string
            Name of a shapefile wit hthe second line/contour to look for when
            drawing transects. This should be the offshore line.
        Spacing : float
            The distance between consecutive points along the CoastLines
            in map units, spatial units depend on units of the CoastLine read in,
            Should be [m]
        Distance2Land : float
            Distance in [m] to extent transects landward when looking for intersection
            with ContourShp1
        Distance2Sea : float
            Distance in [m] to extent transects offshore when looking for intersection
            with ContourShp2
        CheckTopology : bool
            Flag to check and correct overlapping transects
        """

        # flag here to check if transects already exist and if not call function
        if not self.Transects:
            self.GenerateTransectsBetweenContours(ContourShp1, ContourShp2, Spacing, Distance2Sea, Distance2Land, CheckTopology)

        # loop through transects and get midpoints as a new list of nodes
        Midpoints = [Transect.get_Midpoint() for Transect in self.Transects]
        X = [Midpoint.X for Midpoint in Midpoints]
        Y = [Midpoint.Y for Midpoint in Midpoints]

        # use Midpoints to initialise a new line
        self.__init__(self.ID, X, Y, Contour=None, Year=None, Cell = self.Cell, SubCell = self.SubCell, CMU = self.CMU)

    def ExtendTransectsToLineShp(self, LineShp, CheckTopology=False):
        
        """
        MDH, August 2020
        
        """
        
        # load the contour shapefile
        GDF = gp.read_file(LineShp)
        Lines = GDF['geometry']
        
        # make a multlinestring if there are multiple lines
        LineList = []
        for LineObj in Lines:
            if not LineObj:
                continue
            elif (LineObj.geom_type == "MultiLineString"):
                for ThisLine in LineObj:
                    LineList.append(ThisLine)
            elif (LineObj.geom_type == "LineString"):
                LineList.append(LineObj)
            else:
                sys.exit("problem reading lines")

        # catch situation where only one line
        if len(LineList) == 1:
            Lines = LineList[0]
        else:
            Lines = MultiLineString(LineList)

        # flag to note interesections
        Intersections = True

        while Intersections:
            
            # intersect Transect with shapefile to find new end node of transect
            DeleteFlags = np.ones(len(self.Transects))

            for i, Transect in enumerate(self.Transects):
                
                # copy transect and extend
                TransectCopy = Transect
                TransectCopy.ExtendTransect(1000.,1000.)
            
                # find intersection between transect line and shapefile lines
                Intersection = TransectCopy.LineString.intersection(Lines)
                
                # catch no intersections
                if Intersection.geom_type != "GeometryCollection":
                
                    # check there arent multiple intersections, if there are just get the nearest
                    if Intersection.geom_type is "MultiPoint":
                        StartPoint = Point(Transect.CoastNode.X, Transect.CoastNode.Y)
                        Distances = [IntersectPoint.distance(StartPoint) for IntersectPoint in Intersection]
                        Index = Distances.index(min(Distances))
                        Intersection = Intersection[Index]
                    
                    # set this as the new start node
                    NewStartNode = Node(Intersection.x,Intersection.y)
                
                else:
                    NewStartNode = Transect.StartNode

                # rebuild transect with new start node here.
                Transect.__init__(Transect.CoastNode, NewStartNode, Transect.EndNode, Transect.LineID, Transect.ID)

            self.Transects = [Transect for n, Transect in enumerate(self.Transects) if DeleteFlags[n] == 1]
                
            Intersections = False   

        if CheckTopology:
            self.DeleteOverlappingTransects()
        
        for i, Transect in enumerate(self.Transects):
            Transect.ID = str(i)
            
        
    def IntersectTransectsWithIntertidal(self, IntertidalPolyShp):

        """
        MDH, June 2020
        
        """

        # load the contour shapefile
        GDF = gp.read_file(IntertidalPolyShp)
        Polys = GDF['geometry']
        
        # make a multipolygon if there are multiple polys
        PolyList = []
        for PolyObj in Polys:
            if not PolyObj:
                continue
            elif (PolyObj.geom_type == "MultiPolygon"):
                for ThisPoly in PolyObj:
                    PolyList.append(ThisPoly)
            elif (PolyObj.geom_type == "Polygon"):
                PolyList.append(PolyObj)
            else:
                sys.exit("problem reading lines")

        # catch situation where only one poly
        if len(PolyList) == 1:
            Polys = PolyList[0]
        else:
            Polys = MultiPolygon(PolyList)

        # flag to note interesections
        Intersections = True

        # intersect Transect with shapefile to find new end node of transect
        DeleteFlags = np.ones(len(self.Transects))

        for i, Transect in enumerate(self.Transects):
        
            print("\tLine", self.ID, "Transect", i, "/", self.NoTransects)
            
            # find intersection between transect line and shapefile lines
            try:
                IntersectionLines = Transect.LineString.intersection(Polys)
            except:
                print(self.ID, Transect.ID)
                continue
            
            # catch no intersections
            if IntersectionLines.geom_type != "GeometryCollection":
            
                # check there arent multiple intersections, if there are just get the nearest
                if IntersectionLines.geom_type is "MultiLineString":
                    StartPoint = Point(Transect.CoastNode.X, Transect.CoastNode.Y)
                    
                    MidPoints = []
                    for IntersectLine in IntersectionLines:
                        X, Y = IntersectLine.coords.xy
                        MeanX = np.mean(X)
                        MeanY = np.mean(Y)
                        MidPoints.append(Point(MeanX,MeanY))
                    
                    Distances = [MidPoint.distance(StartPoint) for MidPoint in MidPoints]
                    Index = Distances.index(min(Distances))
                    ClosestLine = IntersectionLines[Index]
                    
                else:
                    ClosestLine = IntersectionLines
                
                # set this as the new start and end node
                #print(ClosestLine)
                X, Y = zip(*ClosestLine.coords)
                NewStartNode = Node(X[0], Y[0])
                NewEndNode = Node(X[-1], Y[-1])
            
                # reinitialise transect with new startnode and new endnode
                Transect.__init__(Transect.CoastNode, NewStartNode, NewEndNode, Transect.LineID, Transect.ID)

            else:
                DeleteFlags[i] = 0
            
        self.Transects = [Transect for n, Transect in enumerate(self.Transects) if DeleteFlags[n] == 1]
                
        for i, Transect in enumerate(self.Transects):
            Transect.ID = str(i)
            Transect.ExtendTransect(1., 1.)

    def CheckTransectTopology(self,ThinFactor=2):

        """
        Check for overlapping transects and correct by 
        setting new start/end points evenly spaced in the near shore 
        between non-intersecting transects, then delete any remaining overlapping

        MDH, January 2020

        """

        #print("\tChecking Transect Topology...")

        Intersections = True

        while Intersections:

            # empty array of bools for flagging intersections
            IntersectionsFlags = np.zeros(len(self.Transects))
            DeleteFlags = np.ones(len(self.Transects))
            
            # get a list of all transects
            LinesList = [Transect.LineString for Transect in self.Transects]

            # loope through transects
            # intersect each Transect with all the others to identify intersecting
            for i, Transect1 in enumerate(self.Transects):
                for j, Transect2 in enumerate(self.Transects):
                    
                    # catch identical lines
                    if i == j:
                        continue
                    
                    # otherwise check for intersection
                    if Transect1.LineString.intersects(Transect2.LineString):
                        IntersectionsFlags[i] = 1
                        
            # check for intersections
            if not IntersectionsFlags.any():
                return
            
            # get list of contiguous intersection groups
            IntersectionsFlags = np.insert(IntersectionsFlags, 0, 0)
            
            # get a list of the start and end points of contiguous barrier lines
            StartEndFlags = np.diff(IntersectionsFlags)

            # if last line finishes on a intersection flag the last element as the end 
            if StartEndFlags[StartEndFlags.nonzero()[0][-1]] == 1:
                StartEndFlags[-1] = -1

            StartList = np.argwhere(StartEndFlags == 1).flatten()
            EndList = np.argwhere(StartEndFlags == -1).flatten()

            if not len(StartList) == len(EndList):
                print("\tStart and End lists not the same length")
                print("\t\t", len(StartList),len(EndList))
                return

            for i in range(0,len(StartList)):
                
                # get non intersecting end point coordinates from adjacent transects
                StartX = (self.Transects[StartList[i]]).EndNode.X
                StartY = (self.Transects[StartList[i]]).EndNode.Y
                EndX = (self.Transects[EndList[i]]).EndNode.X
                EndY = (self.Transects[EndList[i]]).EndNode.Y

                # interpolate some new transect endpoints to avoid intersections
                InterpolatedX = np.linspace(StartX,EndX,EndList[i]-StartList[i])
                InterpolatedY = np.linspace(StartY,EndY,EndList[i]-StartList[i])
                
                # loop across groups of intersecting transects and re-initiate
                for j, Transect, in enumerate(self.Transects[StartList[i]:EndList[i]]):
                    
                    if j % ThinFactor:
                        DeleteFlags[StartList[i]+j] = 0

                    NewEndNode = Node(InterpolatedX[j], InterpolatedY[j])
                    Transect.__init__(Transect.CoastNode, Transect.StartNode, NewEndNode, Transect.LineID, Transect.ID)
            
            # resample transects after thinning sections with overlaps
            if (ThinFactor > 1):
                self.Transects = [Transect for i, Transect in enumerate(self.Transects) if DeleteFlags[i] == 1]

            return Intersections

    def DeleteOverlappingTransects(self):

        """
        Check for overlapping transects and correct by 
        deleting longest transects in the pair
        
        MDH, Feb 2020

        """

        print("\t" + str(self.__class__.__name__) + ": Deleting Overlapping transects..")

        # setup array of flags for marking deletions
        DeleteFlags = np.ones(len(self.Transects))

        # loope through transects
        # intersect each Transect with all the others to identify intersecting
        for i, Transect1 in enumerate(self.Transects):
            for j, Transect2 in enumerate(self.Transects):
                    
                # catch identical lines
                if i == j:
                    continue
                elif DeleteFlags[i] == 0:
                    continue
                elif DeleteFlags[j] == 0:
                    continue

                # otherwise check for intersection
                if Transect1.LineString.intersects(Transect2.LineString):
                    
                    # find the longest transect and flag to delete
                    if Transect1.Length > Transect2.Length:
                        DeleteFlags[i] = 0
                    else:
                        DeleteFlags[j] = 0

        # keep transects based on deletion flags
        self.Transects = [Transect for i, Transect in enumerate(self.Transects) if DeleteFlags[i] == 1]    
        
    def GeneratePoints(self, Spacing):
        """
        Generates regularly spaced points along the coastline

        MDH, August 2019

        Parameters
        ----------
        Spacing : float
            The distance between consecutive points along the CoastLines
            in map units, spatial units depend on units of the CoastLine read in,
            Should be [m]
              
        """

        # print("Line: Generating Transects perpendicular to the coast")
        
        # if rewriting Points, empty the Points and transects list
        if len(self.Points) != 0:
            self.Points = []
            self.Transects = []

        # Give each node unique ID
        PointCount = 0
        
        # Parameters for tracing along length
        CumulativeLength = 0.0
        NextPosition = Spacing

        # Track spacing and generate profile at desired distances
        for i in range(0, self.NoNodes):

            #Update the cumulative length of the line
            CumulativeLength += self.SegmentLength[i]

            # get orientation
            TempOrientation = self.Orientation[i]
            
            # Test to see if we're going to create a node
            while CumulativeLength > NextPosition:

                #calculate point for section
                DistanceToStepBack = CumulativeLength - NextPosition
                dX = DistanceToStepBack * np.sin( np.radians( TempOrientation ) )
                dY = DistanceToStepBack * np.cos( np.radians( TempOrientation ) )
                
                # find the point for the node along the line
                PointX = self.Nodes[i+1].X - dX
                PointY = self.Nodes[i+1].Y - dY

                self.Points.append(Node(PointX, PointY, ID=PointCount))

                # update to find next transect
                PointCount += 1
                NextPosition += Spacing
        
        # record number of transects
        self.NoPoints = PointCount

    def GetShorefaceSlope(self, BathyShp):
        
        """
        
        Finds shortest distance from coast to -10m bathy contour and calculates slope
        
        MDH, August 2020
        
        """
        
        # load the shp
        # load the contour shapefile
        GDF = gp.read_file(BathyShp)
        Lines = GDF['geometry']
        
        # make a multlinestring if there are multiple lines
        LineList = []
        for LineObj in Lines:
            if not LineObj:
                continue
            elif (LineObj.geom_type == "MultiLineString"):
                for ThisLine in LineObj:
                    LineList.append(ThisLine)
            elif (LineObj.geom_type == "LineString"):
                LineList.append(LineObj)
            else:
                sys.exit("problem reading lines")

        # catch situation where only one line
        if len(LineList) == 1:
            Lines = LineList[0]
        else:
            Lines = MultiLineString(LineList)
        
        
        for ThisTransect in self.Transects:
            
            # get shortest distance to bathy and associated point
            try:
                TempNode = ThisTransect.HistoricShorelinesPositions[-1][0]
            except:
                continue
            
            BasePoint = Point(TempNode.X, TempNode.Y)
            NearestPoint = nearest_points(Lines, BasePoint)[0]
            NearestPoint = Node(NearestPoint.x, NearestPoint.y)
            Distance = NearestPoint.get_Distance(ThisTransect.HistoricShorelinesPositions[-1][0])
            ThisTransect.ShorefaceDistance = Distance
            ThisTransect.ShorefaceDepth = ThisTransect.MHWS+10.
            ThisTransect.ShorefaceSlope = (ThisTransect.MHWS+10.)/Distance

    def ReverseLine(self):
        """
        Reverses the order of a line object
        
        MDH, June 2019
        """

        X, Y = self.get_XY()
        self.GenerateNodes(X[::-1], Y[::-1])

    def get_XY(self):
        """
        Returns X and Y coordinates as vector numpy arrays

        MDH, June 2019
        """
        X = []
        Y = []
        for Node in self.Nodes:
            X.append(Node.X)
            Y.append(Node.Y)
        
        return np.array(X), np.array(Y)
