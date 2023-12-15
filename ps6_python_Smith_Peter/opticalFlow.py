import numpy as np
import cv2 as cv
def opticalFlow(baseImage,newImage):
    x_kernal = np.array([[-1,1],[-1,1]])
    y_kernal = np.array([[-1,-1],[1,1]])
    t_kernal = np.array([[1,1],[1,1]])
    grad_x = cv.filter2D(baseImage.astype(np.float32),-1,x_kernal)
    grad_y = cv.filter2D(baseImage.astype(np.float32),-1,y_kernal)
    grad_t = cv.filter2D(baseImage.astype(np.float32),-1,t_kernal) - cv.filter2D(newImage.astype(np.float32),-1,t_kernal)

    Ix2 = np.multiply(grad_x,grad_x)
    IxIy = np.multiply(grad_x,grad_y)
    Iy2 = np.multiply(grad_y,grad_y)
    IxIt = np.multiply(grad_x,grad_t)
    IyIt = np.multiply(grad_y,grad_t)
    Ix2 = cv.boxFilter(Ix2,-1,(5,5),normalize=True)
    IxIy = cv.boxFilter(IxIy,-1,(5,5),normalize=True)
    Iy2 = cv.boxFilter(Iy2,-1,(5,5),normalize=True)
    negIxIt = - cv.boxFilter(IxIt,-1,(5,5),normalize=True)
    negIyIt = - cv.boxFilter(IyIt,-1,(5,5),normalize=True)
    U = np.zeros(baseImage.shape)
    V = np.zeros(baseImage.shape)
    for i in range(baseImage.shape[0]):
        for j in range(baseImage.shape[1]):
            det = np.array([[Ix2[i,j],IxIy[i,j]],[IxIy[i,j],Iy2[i,j]]])
            b = np.array([[negIxIt[i,j]],[negIyIt[i,j]]])
            p = np.dot(np.linalg.pinv(det),b)
            U[i,j] = p[0]
            V[i,j] = p[1]
    return U , V