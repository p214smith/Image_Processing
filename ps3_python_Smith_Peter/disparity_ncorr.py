import cv2
import numpy as np
def disparity_ncorr(L, R):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
    
    Params:
    L: Grayscale left image
    R: Grayscale right image, same size as L

    Returns: Disparity map, same size as L, R
    """

    # TODO: Your code here
    length, width = L.shape
    D = np.zeros((length,width),int)
    tplRows = 10
    tplCols = 10
    
    for r in range(length):
       
        for c in range(width):
            r_min = max(0,int(r)-tplRows)
            r_max = min(length,int(r)+tplRows)
            c_min = max(0,int(c))
            c_max = min(width,int(c)+tplCols)
            c_diff = c - c_min
            L_patch = L[int(r_min):int(r_max),int(c_min):int(c_max)]
            method = eval('cv2.TM_CCOEFF_NORMED')
            point = cv2.matchTemplate(R,L_patch,method)
            length1, width1 = L_patch.shape
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(point)
            D[r,c] = abs(c - (max_loc[0]))
    return D
            