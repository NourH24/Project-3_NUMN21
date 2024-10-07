import numpy as np
import math

class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        
    def distance(self, point: 'Point'):
        """
        Calculates the distance between self and point
        
        Parameters:
            point: Point, the point we need to calculate the distance to.

        Returns:
            float, the distance between the two points
        """
        return math.sqrt((self.x - point.x)**2 + (self.y - point.y)**2)