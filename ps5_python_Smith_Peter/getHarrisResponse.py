import numpy as np
import cv2 as cv

def getHarrisResponse(Ix,Iy,sigma):
    length,width = Ix.shape
    Ix2 = np.multiply(Ix,Ix)
    IxIy = np.multiply(Ix,Iy)
    Iy2 = np.multiply(Iy,Iy)
    Ix2 = cv.GaussianBlur(Ix2,(5,5),2.5)
    IxIy = cv.GaussianBlur(IxIy,(5,5),2.5)
    Iy2 = cv.GaussianBlur(Iy2,(5,5),2.5)
    Response = np.zeros((length,width))
    for i in range(length):
        for j in range(width):
            H = np.array([[Ix2[i,j],IxIy[i,j]],[IxIy[i,j] ,Iy2[i,j]]])
            Response[i,j] =  np.linalg.det(H) - sigma*np.trace(H)**2
    return Response