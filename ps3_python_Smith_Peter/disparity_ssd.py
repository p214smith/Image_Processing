import numpy as np
import math

def disparity_ssd(L, R):
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
            l_length, l_width  = L_patch.shape
            min_x = math.inf
            best_X = 0
            for x in range(width):
                
                cr_min = max(0,int(x)- c_diff)
                
                cr_max = min(width,cr_min + l_width)
                if (cr_max - cr_min) < l_width:
                    cr_min = cr_max - l_width
                R_patch = R[int(r_min):int(r_max),int(cr_min):int(cr_max)]
                diff = L_patch.ravel() - R_patch.ravel()
                ssd = np.dot(diff,diff)
                if ssd < min_x:
                    min_x = ssd
                    best_X = x
            D[r,c] = abs(c - best_X)
    return D
