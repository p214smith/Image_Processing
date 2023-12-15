import numpy as np

def twoPointTransform(matches,pointsA,pointsB,tolerance):
    e = 0.0
    N = 72.0
    j = 0
    while e < 0.5 and j < 100:
        np.random.shuffle(matches)
        mat = matches[0:2]
        sam = matches[2:75]
        x1,y1 = pointsA[mat[0].queryIdx].pt
        x2,y2 = pointsA[mat[1].queryIdx].pt
        xp1,yp1 = pointsB[mat[0].trainIdx].pt
        xp2,yp2 = pointsB[mat[1].trainIdx].pt
        pvec = np.array([[xp1],[yp1],[xp2],[yp2]])
        aMatrix = np.array([[x1,y1,1,0],[y1,-x1,0,1],[x2,y2,1,0],[y2,-x2,0,1]])
        vec = np.dot(np.linalg.pinv(aMatrix),pvec)
        affine = np.array([[vec[0],vec[1],vec[2]],[-vec[1],vec[0],vec[3]]])
        affine = np.reshape(affine,(2,3))
        invals = 0.0
        for i in range(int(N)):
            x1,y1 = pointsA[sam[i].queryIdx].pt
            x2,y2 = pointsB[sam[i].trainIdx].pt
            xvec = np.array([[x1],[y1],[1]])
            proj = np.dot(affine,xvec)
            diff =( proj[0] - x2, proj[1] - y2)
            if (abs(diff[0])< tolerance and abs(diff[1])<tolerance):
                invals += 1.0
        e = invals/N
        
        j += 1
    newMatches = []
    for i in range(len(matches)):
        x1,y1 = pointsA[matches[i].queryIdx].pt
        x2,y2 = pointsB[matches[i].trainIdx].pt
        xvec = np.array([[x1],[y1],[1]])
        proj = np.dot(affine,xvec)
        diff =( proj[0] - x2, proj[1] - y2)
        if (abs(diff[0])< tolerance and abs(diff[1])<tolerance):
            newMatches.append(matches[i])
    return affine , newMatches , j