import numpy as np
import cv2 as cv
from warp import warp
from opticalFlow import opticalFlow
from reduce import reducer
from expand import expander
from copy import deepcopy


def HLK(img1, img2, n):
    k = n
    while k > 0:
        red1 = deepcopy(img1)
        red2 = deepcopy(img2)
        for i in range(k-1):
            red1 = reducer(red1)
            red2 = reducer(red2)
        if k == n:
            U = np.zeros(red1.shape,np.float32)
            V = np.zeros(red1.shape,np.float32)
        else:
            U = expander(U) * 2
            V = expander(V) * 2
        if U.shape[0] > red2.shape[0]:
            U = np.delete(U, -1, axis=0)
            V = np.delete(V,-1,axis=0)
        if V.shape[1] > red2.shape[1]:
            U = np.delete(U, -1, axis=1)
            V = np.delete(V, -1, axis=1)
        Wk = warp(red2,U,V)
        Du, Dv = opticalFlow(red1,Wk)
        U = U + Du
        V = V + Dv
        k = k - 1
    return U , V