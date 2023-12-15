import numpy as np
import cv2 as cv

def warp(img,U,V):
    m , n = img.shape
    X, Y = np.meshgrid(range(n),range(m))
    resultimage = cv.remap(img.astype(np.float32), X.astype(np.float32) -U.astype(np.float32), Y.astype(np.float32)- V.astype(np.float32), cv.INTER_LINEAR,cv.BORDER_REPLICATE)
    resultimage2 =cv.remap(img.astype(np.float32), X.astype(np.float32) -U.astype(np.float32), Y.astype(np.float32)- V.astype(np.float32), cv.INTER_NEAREST,cv.BORDER_REPLICATE)
    a = np.isnan(resultimage)
    i = np.where(a == True, 1, 0)
    i = np.argwhere(i)
    resultimage[i] = resultimage2[i]
    return resultimage2