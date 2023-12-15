import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from getHarrisResponse import getHarrisResponse
from getCornerPoints import getCornerPoints
from getCornerAngles import getCornerAngles
from transRansac import transRansac
from twoPointTransform import twoPointTransform
simA = cv.imread("./input/simA.jpg",0)
simB = cv.imread("./input/simB.jpg",0)
transA = cv.imread("./input/transA.jpg",0)
transB = cv.imread("./input/transB.jpg",0)

simA_x_grad = cv.Sobel(simA,cv.CV_64F,1,0,ksize=3)
simA_y_grad = cv.Sobel(simA,cv.CV_64F,0,1,ksize=3)
scaled_simA_x_grad = cv.convertScaleAbs(simA_x_grad)
scaled_simA_y_grad = cv.convertScaleAbs(simA_y_grad)
simA_sideXside = np.zeros((480,1280),dtype=np.uint8)
simA_sideXside[:,0:640] = scaled_simA_x_grad
simA_sideXside[:,640:1280] = scaled_simA_y_grad
cv.imwrite("./output/ps5-1-a-2.png",simA_sideXside)

transA_x_grad = cv.Sobel(transA,cv.CV_64F,1,0,ksize=3)
transA_y_grad = cv.Sobel(transA,cv.CV_64F,0,1,ksize=3)
scaled_transA_x_grad = cv.convertScaleAbs(transA_x_grad)
scaled_transA_y_grad = cv.convertScaleAbs(transA_y_grad)
transA_sideXside = np.zeros((480,1280),dtype=np.uint8)
transA_sideXside[:,0:640] = scaled_transA_x_grad
transA_sideXside[:,640:1280] = scaled_transA_y_grad
cv.imwrite("./output/ps5-1-a-1.png",transA_sideXside)

simA_response = getHarrisResponse(simA_x_grad,simA_y_grad,0.04)
scaled_simA = ((simA_response - np.min(simA_response)) / (np.max(simA_response) - np.min(simA_response)))*255
cv.imwrite("./output/ps5-1-b-3.png",scaled_simA)

simB_x_grad = cv.Sobel(simB,cv.CV_64F,1,0,ksize=3)
simB_y_grad = cv.Sobel(simB,cv.CV_64F,0,1,ksize=3)
simB_response = getHarrisResponse(simB_x_grad,simB_y_grad,0.04)
scaled_simB = ((simB_response - np.min(simB_response)) / (np.max(simB_response) - np.min(simB_response)))*255
cv.imwrite("./output/ps5-1-b-4.png",scaled_simB)

transA_response = getHarrisResponse(transA_x_grad,transA_y_grad,0.04)
scaled_transA = ((transA_response - np.min(transA_response)) / (np.max(transA_response) - np.min(transA_response)))*255
cv.imwrite("./output/ps5-1-b-1.png",scaled_transA)

transB_x_grad = cv.Sobel(transB,cv.CV_64F,1,0,ksize=3)
transB_y_grad = cv.Sobel(transB,cv.CV_64F,0,1,ksize=3)
transB_response = getHarrisResponse(transB_x_grad,transB_y_grad,0.04)
scaled_transB = ((transB_response - np.min(transB_response)) / (np.max(transB_response) - np.min(transB_response)))*255
cv.imwrite("./output/ps5-1-b-2.png",scaled_transB)

simA_corners = getCornerPoints(simA_response,5,0.05)
rows , cols = np.nonzero(simA_corners)
simA = cv.imread("./input/simA.jpg")
print(len(rows))
for i in range(len(rows)):
    x = cols[i]
    y = rows[i]
    cv.circle(simA,(int(x),int(y)),2,(0,255,0),2)
cv.imwrite("./output/ps5-1-c-3.png",simA)

simB_corners = getCornerPoints(simB_response,5,0.045)
rows , cols = np.nonzero(simB_corners)
simB = cv.imread("./input/simB.jpg")
print(len(rows))
for i in range(len(rows)):
    x = cols[i]
    y = rows[i]
    cv.circle(simB,(int(x),int(y)),2,(0,255,0),2)
cv.imwrite("./output/ps5-1-c-4.png",simB)

transA_corners = getCornerPoints(transA_response,5,0.08)
rows , cols = np.nonzero(transA_corners)
transA = cv.imread("./input/transA.jpg")
print(len(rows))
for i in range(len(rows)):
    x = cols[i]
    y = rows[i]
    cv.circle(transA,(int(x),int(y)),2,(0,255,0),2)
cv.imwrite("./output/ps5-1-c-1.png",transA)

transB_corners = getCornerPoints(transB_response,5,0.06)
rows , cols = np.nonzero(transB_corners)
transB = cv.imread("./input/transB.jpg")
print(len(rows))
for i in range(len(rows)):
    x = cols[i]
    y = rows[i]
    cv.circle(transB,(int(x),int(y)),2,(0,255,0),2)
cv.imwrite("./output/ps5-1-c-2.png",transB)
transA = cv.imread("./input/transA.jpg")
transB = cv.imread("./input/transB.jpg")
rows , cols = np.nonzero(transA_corners)

transA_angles = getCornerAngles(transA_x_grad,transA_y_grad,transA_corners)
transA_siftfeat = cv.xfeatures2d.SIFT_create()
transA_sift = cv.SIFT_create()
pointsAt = []
for i in range(len(rows)):
    x = int(cols[i])
    y = int(rows[i])
    pointsAt.append(cv.KeyPoint(x,y,size=10,angle=transA_angles[y,x],octave=0))
transA_drawn = cv.drawKeypoints(transA,pointsAt,0,(0,255,0),flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
rows , cols = np.nonzero(transB_corners)
pointsTA, descriptorsTA = transA_sift.compute(transA,pointsAt)
transB_angles = getCornerAngles(transB_x_grad,transB_y_grad,transB_corners)
transB_sift = cv.SIFT_create()
pointsBt = []
for i in range(len(rows)):
    x = int(cols[i])
    y = int(rows[i])
    pointsBt.append(cv.KeyPoint(x,y,size=10,angle=transB_angles[y,x],octave=0))
transB_drawn = cv.drawKeypoints(transB,pointsBt,0,(0,255,0),flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
trans_sideXside = np.zeros((480,1280,3),dtype=np.uint8)
trans_sideXside[:,0:640,:] = transA_drawn
trans_sideXside[:,640:1280,:] = transB_drawn
cv.imwrite("./output/ps5-2-a-1.png",trans_sideXside)
pointsTB, descriptorsTB = transB_sift.compute(transB,pointsBt)
simA = cv.imread("./input/simA.jpg")
simB = cv.imread("./input/simB.jpg")
rows , cols = np.nonzero(simA_corners)

simA_angles = getCornerAngles(simA_x_grad,simA_y_grad,simA_corners)

simA_sift = cv.SIFT_create()
pointsAs = []
for i in range(len(rows)):
    x = int(cols[i])
    y = int(rows[i])
    pointsAs.append(cv.KeyPoint(x,y,size=10,angle=simA_angles[y,x],octave=0))
simA_drawn = cv.drawKeypoints(simA,pointsAs,0,(0,255,0),flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
rows , cols = np.nonzero(simB_corners)
pointsSA, descriptorSA = simA_sift.compute(simA,pointsAs)
simB_angles = getCornerAngles(simB_x_grad,simB_y_grad,simB_corners)
simB_sift = cv.SIFT_create()
pointsBs = []
for i in range(len(rows)):
    x = int(cols[i])
    y = int(rows[i])
    pointsBs.append(cv.KeyPoint(x,y,size=10,angle=simB_angles[y,x],octave=0))
simB_drawn = cv.drawKeypoints(simB,pointsBs,0,(0,255,0),flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
sim_sideXside = np.zeros((480,1280,3),dtype=np.uint8)
sim_sideXside[:,0:640,:] = simA_drawn
sim_sideXside[:,640:1280,:] = simB_drawn
cv.imwrite("./output/ps5-2-a-2.png",sim_sideXside)
pointsSB, descriptorSB = simB_sift.compute(simB,pointsBs)
transMatch = cv.BFMatcher()
simMatch = cv.BFMatcher()
transMatches = transMatch.match(descriptorsTA,descriptorsTB)
simMatches = simMatch.match(descriptorSA,descriptorSB)
simLines_sideXside = np.zeros((480,1280,3),dtype=np.uint8)
transLines_sideXside= np.zeros((480,1280,3),dtype=np.uint8)
simLines_sideXside[:,0:640,:] = simA
simLines_sideXside[:,640:1280,:] = simB
transLines_sideXside[:,0:640,:] = transA
transLines_sideXside[:,640:1280,:] = transB
for match in transMatches:
    x1,y1 = pointsTA[match.queryIdx].pt
    x2,y2 = pointsTB[match.trainIdx].pt
    cv.line(transLines_sideXside,(int(x1),int(y1)),(640+int(x2),int(y2)),(0,255,0),1)
cv.imwrite("./output/ps5-2-b-1.png",transLines_sideXside)

for match in simMatches:
    x1,y1 = pointsSA[match.queryIdx].pt
    x2,y2 = pointsSB[match.trainIdx].pt
    cv.line(simLines_sideXside,(int(x1),int(y1)),(640+int(x2),int(y2)),(0,255,0),1)
cv.imwrite("./output/ps5-2-b-2.png",simLines_sideXside)
Tvector , Tmatches , j= transRansac(np.asarray(transMatches),pointsTA,pointsTB,15)
print(Tvector, j,len(Tmatches))
transRansac_sideXside= np.zeros((480,1280,3),dtype=np.uint8)
transRansac_sideXside[:,0:640,:] = transA
transRansac_sideXside[:,640:1280,:] = transB
for match in Tmatches:
    x1,y1 = pointsTA[match.queryIdx].pt
    x2,y2 = pointsTB[match.trainIdx].pt
    cv.line(transRansac_sideXside,(int(x1),int(y1)),(640+int(x2),int(y2)),(0,255,0),1)
cv.imwrite("./output/ps5-3-a-1.png",transRansac_sideXside)
affine, Smatches,j = twoPointTransform(np.asarray(simMatches),pointsSA,pointsSB,5)
print(affine,j,len(Smatches))
simRansac_sideXside = np.zeros((480,1280,3),dtype=np.uint8)
simRansac_sideXside[:,0:640,:] = simA
simRansac_sideXside[:,640:1280,:] = simB
for match in Smatches:
    x1,y1 = pointsSA[match.queryIdx].pt
    x2,y2 = pointsSB[match.trainIdx].pt
    cv.line(simRansac_sideXside,(int(x1),int(y1)),(640+int(x2),int(y2)),(0,255,0),1)
cv.imwrite("./output/ps5-3-b-1.png",simRansac_sideXside)
print("percentage of Trans matches = ",str(len(Tmatches)/len(transMatches)*100))
print("percentage of Sim matches = ",str(len(Smatches)/len(simMatches)*100))
