import numpy as np
import math

def hough_circles_acc(edges,r):
    length,width = edges.shape
    H = np.zeros((width,length))

    rows , cols = np.nonzero(edges)
    theta = np.linspace(-180,178,180)
    n_theta = len(theta)
    for i in range(len(rows)):
        x = cols[i]
        y = rows[i]
        for j in range(n_theta):
            a = int(x-r*math.cos(math.radians(theta[j])))
            b = int(y-r*math.sin(math.radians(theta[j])))
            if a < width and b < length:
                H[a,b] += 1
    return H