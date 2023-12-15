# ps2
import os
import numpy as np
import cv2
from disparity_ncorr import disparity_ncorr
## 1-a
# Read images
L = cv2.imread(os.path.join('input', 'pair0-L.png'), 0) * (1.0 / 255.0)  # grayscale, [0, 1]
R = cv2.imread(os.path.join('input', 'pair0-R.png'), 0) * (1.0 / 255.0)

# Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
from disparity_ssd import disparity_ssd
D_L = disparity_ssd(L, R)
D_R = disparity_ssd(R, L)
D_L_max = np.amax(D_L)
image = ((D_L/D_L_max)*255).round().astype(np.uint8)
D_R_max = np.amax(D_R)
image1 = ((D_R/D_R_max)*255).round().astype(np.uint8)
cv2.imwrite('./Output/ps3-1-a-1.png',image)
cv2.imwrite('./Output/ps3-1-a-2.png',image1)

# TODO: Save output images (D_L as output/ps2-1-a-1.png and D_R as output/ps2-1-a-2.png)
# Note: They may need to be scaled/shifted before saving to show results properly

# TODO: Rest of your code here
L = cv2.imread(os.path.join('input', 'pair1-L.png'), 0) * (1.0 / 255.0)
R = cv2.imread(os.path.join('input', 'pair1-R.png'), 0) * (1.0 / 255.0)
D_L = disparity_ssd(L, R)
D_R = disparity_ssd(R, L)
D_L_max = np.amax(D_L)
image = ((D_L/D_L_max)*255).round().astype(np.uint8)
D_R_max = np.amax(D_R)
image1 = ((D_R/D_R_max)*255).round().astype(np.uint8)
cv2.imwrite('./Output/ps3-2-a-1.png',image)
cv2.imwrite('./Output/ps3-2-a-2.png',image1)
mean = 0 
sigma = 20
noise = np.zeros(L.shape)
cv2.randn(noise, mean, sigma)
noisy_L = cv2.add(L[:,:],noise[:,:])
D_L = disparity_ssd(noisy_L, R)
D_L_max = np.amax(D_L)
image = ((D_L/D_L_max)*255).round().astype(np.uint8)
cv2.imwrite('./Output/ps3-3-a-1.png',image)
D_R = disparity_ssd(R,noisy_L)
D_R_max = np.amax(D_R)
image1 = ((D_R/D_R_max)*255).round().astype(np.uint8)
cv2.imwrite('./Output/ps3-3-a-2.png',image1)
contrasted_L = L * 1.1
D_L = disparity_ssd(contrasted_L, R)
D_L_max = np.amax(D_L)
image = ((D_L/D_L_max)*255).round().astype(np.uint8)
cv2.imwrite('./Output/ps3-3-b-1.png',image)
D_R = disparity_ssd(R,contrasted_L)
D_R_max = np.amax(D_R)
image1 = ((D_R/D_R_max)*255).round().astype(np.uint8)
cv2.imwrite('./Output/ps3-3-b-2.png',image1)
L = cv2.imread(os.path.join('input', 'pair1-L.png'), 0)
R = cv2.imread(os.path.join('input', 'pair1-R.png'), 0)
D_L = disparity_ncorr(L,R)
D_L_max = np.amax(D_L)
image = ((D_L/D_L_max)*255).round().astype(np.uint8)
cv2.imwrite('./Output/ps3-4-a-1.png',image)
D_R = disparity_ncorr(R,L)
D_R_max = np.amax(D_R)
image1 = ((D_R/D_R_max)*255).round().astype(np.uint8)
cv2.imwrite('./Output/ps3-4-a-2.png',image1)

mean = 0
sigma = 20
noise = np.zeros(L.shape,np.uint8)
cv2.randn(noise, mean, sigma)
noisy_L = cv2.add(L[:,:],noise[:,:])
D_L = disparity_ncorr(noisy_L, R)
D_L_max = np.amax(D_L)
image = ((D_L/D_L_max)*255).round().astype(np.uint8)
cv2.imwrite('./Output/ps3-4-b-1.png',image)
D_R = disparity_ncorr(R,noisy_L)
D_R_max = np.amax(D_R)
image1 = ((D_R/D_R_max)*255).round().astype(np.uint8)
cv2.imwrite('./Output/ps3-4-b-2.png',image1)
contrasted_L = (L * 1.1).round().astype(np.uint8)
D_L = disparity_ncorr(contrasted_L, R)
D_L_max = np.amax(D_L)
image = ((D_L/D_L_max)*255).round().astype(np.uint8)
cv2.imwrite('./Output/ps3-4-c-1.png',image)
D_R = disparity_ncorr(R,contrasted_L)
D_R_max = np.amax(D_R)
image1 = ((D_R/D_R_max)*255).round().astype(np.uint8)
cv2.imwrite('./Output/ps3-4-d-2.png',image1)