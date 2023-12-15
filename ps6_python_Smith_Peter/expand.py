import numpy as np
import cv2 as cv
from reduce import reduce
def expand(img):
    g0, g1, g2, g3 = reduce(img)
    kernel = np.array([[.5,2.0,3.0,2.0,.5]])/4
    kernel = np.matmul(kernel.T,kernel)
    exp1 = np.zeros((g3.shape[0] *2,g3.shape[1] *2))
    exp1[::2,::2] = g3[:,:]
    exp1 = cv.filter2D(exp1.astype(np.float32),-1,kernel)
    if exp1.shape[0] > g2.shape[0]:
        exp1 = np.delete(exp1, -1, axis=0)
    if exp1.shape[1] > g2.shape[1]:
        exp1 = np.delete(exp1, -1, axis=1)
    lap1 = g2 - exp1
    exp2 = np.zeros((g2.shape[0] *2,g2.shape[1] *2))
    exp2[::2,::2] = g2[:,:]
    exp2 = cv.filter2D(exp2.astype(np.float32),-1,kernel)
    if exp2.shape[0] > g1.shape[0]:
        exp2 = np.delete(exp2, -1, axis=0)
    if exp2.shape[1] > g1.shape[1]:
        exp2 = np.delete(exp2, -1, axis=1)
    lap2 = g1 - exp2
    exp3 = np.zeros((g1.shape[0] *2,g1.shape[1] *2))
    exp3[::2,::2] = g1[:,:]
    exp3 = cv.filter2D(exp3.astype(np.float32),-1,kernel)
    if exp3.shape[0] > g0.shape[0]:
        exp3 = np.delete(exp3, -1, axis=0)
    if exp3.shape[1] > g0.shape[1]:
        exp3 = np.delete(exp3, -1, axis=1)
    lap3 = g0 - exp3
    return g3, lap1, lap2, lap3

def expander(img):
    kernel = np.array([[.5,2.0,3.0,2.0,.5]])/4
    kernel = np.matmul(kernel.T,kernel)
    exp1 = np.zeros((img.shape[0] *2,img.shape[1] *2))
    exp1[::2,::2] = img[:,:]
    exp1 = cv.filter2D(exp1.astype(np.float32),-1,kernel)
    return exp1