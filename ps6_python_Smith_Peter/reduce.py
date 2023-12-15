import numpy as np
import cv2 as cv

def reduce(img):
    kernel = np.array([[1,4,6,4,1]])/16
    kernel = np.matmul(kernel.T,kernel)
    layer0 = cv.filter2D(img.astype(np.float32),-1,kernel)
    red1 = layer0[0::2,0::2]
    layer1 = cv.filter2D(red1.astype(np.float32),-1,kernel)
    red2 = layer1[0::2,0::2]
    layer2 = cv.filter2D(red2.astype(np.float32),-1,kernel)
    red3 = layer2[0::2,0::2]
    return layer0, red1, red2, red3
def reducer(img):
    kernel = np.array([[1,4,6,4,1]])/16
    kernel = np.matmul(kernel.T,kernel)
    layer0 = cv.filter2D(img.astype(np.float32),-1,kernel)
    red1 = layer0[0::2,0::2]
    return red1