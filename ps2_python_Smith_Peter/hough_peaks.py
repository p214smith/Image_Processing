import numpy as np

def hough_peaks(H,n,th=2,parallel=False):
    peaks = []
    threshold = np.amax(H)/th
    length , width = H.shape
    neighborhood = int(length/50)
    for i in range(n):
        if np.amax(H) > threshold:
            r, c = np.unravel_index(H.argmax(),H.shape)
            peaks.append(np.unravel_index(H.argmax(),H.shape))
            r_min = max(0,int(r)-neighborhood)
            r_max = min(length,int(r)+neighborhood)
            c_min = max(0,int(c)-neighborhood)
            c_max = min(width,int(c)+neighborhood)
        
        H[int(r_min):int(r_max),int(c_min):int(c_max)]=0
    if parallel == True:
        for i in range(len(peaks)):
            r , c = peaks[i]
            for i in range(length):
                if H[i,c] > threshold/1.25:
                    peaks.append((i,c))
    return peaks