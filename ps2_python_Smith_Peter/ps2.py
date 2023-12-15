import numpy as np
import cv2 as cv
from hough_lines_draw import hough_lines_draw 
from hough_circles_acc import hough_circles_acc
from matplotlib import pyplot as plt
from hough_lines_acc import hough_lines_acc
from hough_peaks import hough_peaks
from find_circles import find_circles


#1
img = cv.imread("./Input/ps2-input0.png")
edges = cv.Canny(img,100,200)
cv.imwrite('./Output/ps2-1-a-1.png',edges)

#2
H, rho, theta = hough_lines_acc(edges)
H_max = np.amax(H)
image = ((H/H_max)*255).round().astype(np.uint8)
cv.imwrite('./Output/ps2-2-a-1.png',image)
peaks = hough_peaks(H,10,20)
for i in range(len(peaks)):
    b,a =peaks[i]
    cv.circle(image,(int(a),int(b)),3,255,3)
cv.imwrite('./Output/ps2-2-b-1.png',image)
hough_lines_draw(img,"./output/ps2-2-c-1.png",peaks,rho,theta)

#3
img = cv.imread("./Input/ps2-input0-noise.png")
smoothed = cv.GaussianBlur(img,(9,9),2)
cv.imwrite('./Output/ps2-3-a-1.png',smoothed)
edges = cv.Canny(img,100,200)
cv.imwrite('./Output/ps2-3-b-1.png',edges)
edges_smoothed = cv.Canny(smoothed,100,200)
cv.imwrite('./Output/ps2-3-b-2.png',edges_smoothed)
H, rho, theta = hough_lines_acc(edges_smoothed)
H_max = np.amax(H)
image = ((H/H_max)*255).round().astype(np.uint8)
peaks = hough_peaks(H,10,3.5)
for i in range(len(peaks)):
    b,a =peaks[i]
    cv.circle(image,(int(a),int(b)),3,255,3)
cv.imwrite('./Output/ps2-3-c-1.png',image)
hough_lines_draw(img,"./output/ps2-3-c-2.png",peaks,rho,theta)

#4
img = cv.imread("./Input/ps2-input1.png",0)
smoothed = cv.GaussianBlur(img,(11,17),4)
cv.imwrite('./output/ps2-4-a-1.png',smoothed)
edges_smoothed = cv.Canny(smoothed,100,200)
cv.imwrite('./Output/ps2-4-b-1.png',edges_smoothed)
H, rho, theta = hough_lines_acc(edges_smoothed)
H_max = np.amax(H)
image = ((H/H_max)*255).round().astype(np.uint8)
peaks = hough_peaks(H,10,6.5)
for i in range(len(peaks)):
    b,a =peaks[i]
    cv.circle(image,(int(a),int(b)),3,255,3)
cv.imwrite('./Output/ps2-4-c-1.png',image)
imag = cv.cvtColor(img,cv.COLOR_GRAY2RGB)
hough_lines_draw(imag,"./output/ps2-4-c-2.png",peaks,rho,theta)

#5

img = cv.imread("./Input/ps2-input1.png",0)
smoothed = cv.GaussianBlur(img,(7,7),3)
cv.imwrite('./output/ps2-5-a-1.png',smoothed)
edges_smoothed = cv.Canny(smoothed,100,200)
cv.imwrite('./Output/ps2-5-a-2.png',edges_smoothed)
H = hough_circles_acc(edges_smoothed,20)
H_max = np.amax(H)
image = ((H/H_max)*255).round().astype(np.uint8)
peaks = hough_peaks(H,10,9)
imag = cv.cvtColor(img,cv.COLOR_GRAY2RGB)
for i in range(len(peaks)):
    x , y = peaks[i]
    cv.circle(imag,(int(x),int(y)),20,(0,255,0),2)
cv.imwrite('./Output/ps2-5-a-3.png',imag)
radi = np.linspace(20,50)
centers, radii = find_circles(edges_smoothed,np.linspace(20,50))
imag = cv.cvtColor(img,cv.COLOR_GRAY2RGB)
for i in range(len(radii)):
    x , y = centers[i]
    cv.circle(imag,(x,y),int(radi[radii[i]]),(0,255,0),2)
cv.imwrite('./Output/ps2-5-b-1.png',imag)

#6

img = cv.imread("./Input/ps2-input2.png",0)
smoothed = cv.GaussianBlur(img,(7,7),0)
edges_smoothed = cv.Canny(smoothed,50,100)
H, rho, theta = hough_lines_acc(edges_smoothed)
H_max = np.amax(H)
image = ((H/H_max)*255).round().astype(np.uint8)
peaks = hough_peaks(H,20,4)
imag = cv.cvtColor(smoothed,cv.COLOR_GRAY2RGB)
hough_lines_draw(imag,"./output/ps2-6-a-1.png",peaks,rho,theta)

img = cv.imread("./Input/ps2-input2.png",0)
smoothed = cv.GaussianBlur(img,(3,15),3)
edges_smoothed = cv.Canny(smoothed,100,100,L2gradient=True)
H, rho, theta = hough_lines_acc(edges_smoothed)
peaks = hough_peaks(H,20,2)
imag = cv.cvtColor(smoothed,cv.COLOR_GRAY2RGB)
hough_lines_draw(imag,"./output/ps2-6-c-1.png",peaks,rho,theta)

#7

img = cv.imread("./Input/ps2-input2.png",0)
smoothed = cv.GaussianBlur(img,(3,3),0)
edges_smoothed = cv.Canny(smoothed,75,100,L2gradient=True)
H, rho, theta = hough_lines_acc(edges_smoothed)
peaks = hough_peaks(H,3,2)
imag = cv.cvtColor(smoothed,cv.COLOR_GRAY2RGB)
radi = np.linspace(20,35,15)
centers, radii = find_circles(edges_smoothed,np.linspace(20,35,15))
imag = cv.cvtColor(smoothed,cv.COLOR_GRAY2RGB)
for i in range(len(radii)):
    x , y = centers[i]
    cv.circle(imag,(x,y),int(radi[radii[i]]),(0,255,0),2)
cv.imwrite('./Output/ps2-7-a-1.png',imag)

#8

img = cv.imread("./Input/ps2-input3.png",0)
smoothed = cv.GaussianBlur(img,(5,7),1.5)
edges_smoothed = cv.Canny(smoothed,75,100,L2gradient=True)
H, rho, theta = hough_lines_acc(edges_smoothed)
peaks = hough_peaks(H,3,2)
imag = cv.cvtColor(smoothed,cv.COLOR_GRAY2RGB)
radi = np.linspace(20,35,15)
centers, radii = find_circles(edges_smoothed,np.linspace(20,35,15),1.3)
imag = cv.cvtColor(img,cv.COLOR_GRAY2RGB)
for i in range(len(radii)):
    x , y = centers[i]
    cv.circle(imag,(x,y),int(radi[radii[i]]),(0,255,0),2)
H, rho, theta = hough_lines_acc(edges_smoothed)
peaks = hough_peaks(H,20,2)
hough_lines_draw(imag,"./output/ps2-8-a-1.png",peaks,rho,theta)
#cv.imwrite('./Output/ps2-8-a-1.png',imag)