from hough_circles_acc import hough_circles_acc
import numpy as np

def find_circles(BW, r_range,th= 1.5):
    z = len(r_range)
    length,width = BW.shape
    H = np.zeros((width,length,z))
    for i in range(len(r_range)):
        H[:,:,i] = hough_circles_acc(BW,r_range[i])
    centers = []
    radii = []
    vals = []
    length , width, depth = H.shape
    neighborhood = int(length/15)
    threshold = np.amax(H)/th
    i = 0
    new_max = np.amax(H)
    while  (new_max > threshold):
        holder = np.unravel_index(H.argmax(),H.shape)
        r_min = max(0,holder[0].astype(int)-neighborhood)
        r_max = min(length,holder[0].astype(int)+neighborhood)
        c_min = max(0,holder[1].astype(int)-neighborhood)
        c_max = min(width,holder[1]+neighborhood)
        H[int(r_min):int(r_max),int(c_min):int(c_max),holder[2]]=0
        vals.append(holder)
        
        holder = np.unravel_index(H.argmax(),H.shape)
        new_max = np.amax(H)
        i += 1
        diff = 8
        new_vals = []
        for x, y , r in vals:
            
            if all(abs(x-x0) > diff or abs(y - y0) > diff or abs(r - r0)> diff for x0,y0,r0 in new_vals):
                new_vals.append((x,y,r))
        for x, y , r in new_vals:
            centers.append((x,y))
            radii.append(r)
    return centers, radii
        