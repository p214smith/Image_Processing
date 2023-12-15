import numpy as np
import cv2 as cv
import math
def hough_lines_draw(image,file,peaks,rho,theta):
    for i in range(len(peaks)):
        rh, th = peaks[i]
        a = math.cos(np.radians(theta[th]))
        x0 = rho[rh]*a
        b = math.sin(np.radians(theta[th]))
        y0 = rho[rh]*b
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv.line(image,(x1,y1),(x2,y2),(0,255,0),2)
    cv.imwrite(file,image)