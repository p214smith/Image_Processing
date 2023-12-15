import numpy as np
import math
def hough_lines_acc(BW, Theta = np.linspace(-90,89,180),RhoResolution = 1):
    width, length = BW.shape
    D = np.ceil(np.sqrt((width-1)**2 + (length-1)**2))
    diagonal = RhoResolution* np.ceil(D/RhoResolution)
    n_rho = 2*(np.ceil(D/RhoResolution))+1
    n_theta = len(Theta)
    rho_values = np.linspace(-diagonal,diagonal,2*diagonal.astype(int))
    H = np.zeros((n_rho.astype(int),n_theta))
    rows , cols = np.nonzero(BW)
    
    for i in range(len(rows)):
        x = cols[i]
        y = rows[i]
        
        for j in range(n_theta):
            rho = round( x * math.cos(math.radians(Theta[j]))+ y * math.sin(math.radians(Theta[j]))) + diagonal.astype(int)
            H[rho,j] += 1
    return H , rho_values , Theta