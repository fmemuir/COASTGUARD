"""
Description of file goes here

Martin D. Hurst
University of Glasgow
June 2019

"""

import numpy as np

class Node:
    
    """
    Description of object goes here

    """
    
    def __init__(self, X, Y, Z=None, Dist=None, ID=None):
        
        self.X = X
        self.Y = Y
        self.Z = Z
        self.Dist = Dist
        self.ID = ID

        if not isinstance(self.X, float):
            print("PROBLEM")
            print(type(self.X))
            

    def __eq__(self,other):
        if (self.X == other.X) and (self.Y == other.Y):
            return True
        elif (abs(self.X-other.X) < 0.0001) and (abs(self.Y-other.Y) < 0.0001):
            print ("Close but no cigar!")
            return False
        else:
            return False
        
    def __str__(self):
        String = "Node Object\nX: %.2f\nY: %.2f\n" %(self.X, self.Y)
        return String

    def get_XY(self):
        return self.X, self.Y
    
    def get_XZ(self):
        return self.X, self.Z
    
    def get_Distance(self,OtherNode):
        return np.sqrt((self.X-OtherNode.X)**2.+(self.Y-OtherNode.Y)**2.)

    def get_Orientation(self,OtherNode):
        
        """
        
        Maybe this could be a more general function external to class?
        
        MDH
        
        """
        
        #calculate the spatial change
        dx = OtherNode.X - self.X
        dy = OtherNode.Y - self.Y
        
        # catch rare zero dy values
        if np.abs(dy) < 0.001:
            dy = 0.001

        #Calculate the orientation of the line
        #N.B. this will depend on where the start segment is
        #so that 270 is esseintially the same as 90 but depends
        #which end of the line the cycle starts at
        if dx > 0 and dy > 0:
            Orientation = np.degrees( np.arctan( dx / dy ) )
        elif dx > 0 and dy < 0:
            Orientation = 180.0 + np.degrees( np.arctan( dx / dy ) )
        elif dx < 0 and dy < 0:
            Orientation = 180.0 + np.degrees( np.arctan( dx / dy ) )
        elif dx < 0 and dy > 0:
            Orientation = 360 + np.degrees( np.arctan( dx / dy ) )
        else:
            import pdb
            pdb.set_trace()
            
        return Orientation
        