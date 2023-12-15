import numpy as np
import math
def getCornerAngles(Ix,Iy,points):
    length, width = Ix.shape
    rows , cols = np.nonzero(points)
    angles = np.zeros((length,width))
    for i in range(len(rows)):
        x = cols[i]
        y = rows[i]
        angles[y,x] = math.atan2(Iy[y,x],Ix[y,x]) 
    return angles