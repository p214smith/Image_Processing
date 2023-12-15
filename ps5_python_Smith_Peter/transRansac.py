import numpy as np

def transRansac(matches,pointsA,pointsB,tolerance):
    e = 0.0
    N = 17.0
    j = 0
    while e < 0.5 and j < 50:
        np.random.shuffle(matches)
        mat = matches[0]
        sam = matches[1:19]
        x1,y1 = pointsA[mat.queryIdx].pt
        x2,y2 = pointsB[mat.trainIdx].pt
        vec = (x2-x1,y2-y1)
        invals = 0.0
        for i in range(int(N)):
            x1,y1 = pointsA[sam[i].queryIdx].pt
            x2,y2 = pointsB[sam[i].trainIdx].pt
            vec1 = (x2-x1,y2-y1)
            diff = (vec1[0]-vec[0],vec1[1]-vec[1])
            
            if (abs(diff[0])< tolerance and abs(diff[1])<tolerance):
                invals += 1.0
        e = invals/N
        j += 1
    newMatches = []
    for i in range(len(matches)):
        x1,y1 = pointsA[matches[i].queryIdx].pt
        x2,y2 = pointsB[matches[i].trainIdx].pt
        vec1 = (x2-x1,y2-y1)
        diff = (vec1[0]-vec[0],vec1[1]-vec[1])
        if (abs(diff[0])< tolerance and abs(diff[1])<tolerance):
            newMatches.append(matches[i])
    return vec , newMatches , j
    