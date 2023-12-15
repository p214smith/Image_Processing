import numpy as np

def getCornerPoints(img,neighborhood,th):
    length, width = img.shape
    values = np.zeros((length,width))
    threshold = np.amax(img) * th
    for i in range(length):
        for j in range(width):
            r_min = max(0,i-neighborhood)
            r_max = min(length,i+neighborhood)
            c_min = max(0,j-neighborhood)
            c_max = min(width,j+neighborhood)
            if img[i,j] > threshold and img[i,j] == np.amax(img[r_min:r_max,c_min:c_max]):
                values[i,j] = 1
    return values