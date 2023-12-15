from kmeans_multiple import kmeans_multiple
import numpy as np
import cv2 as cv
from scipy.spatial import distance
import copy

def segment_kmeans(image, K, iters, R):
    ids, means, ssd = kmeans_multiple(image,K,iters,R)
    i = 0
    j = 0
    image_size = image.shape
    while i < image_size[0]:
        j = 0
        while j < image_size[1]:
            image[i,j,:] = means[ids[i,j],:]
            j += 1
        i += 1
    return image
img = cv.imread("./Input/im3.png")
new_image = segment_kmeans(img,3,7,5)
cv.imwrite('./Output/img3_3_7_5.png',new_image)
img = cv.imread("./Input/im3.png")
new_image = segment_kmeans(img,5,7,5)
cv.imwrite('./Output/img3_5_7_5.png',new_image)
img = cv.imread("./Input/im3.png")
new_image = segment_kmeans(img,7,7,5)
cv.imwrite('./Output/img3_7_7_5.png',new_image)



img = cv.imread("./Input/im5.png")

new_image = segment_kmeans(img,7,15,5)
cv.imwrite('./Output/devins_truck.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,15,5)
# cv.imwrite('./Output/img3_5_15_5.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,15,5)
# cv.imwrite('./Output/img3_7_15_5.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,30,5)
# cv.imwrite('./Output/img3_3_30_5.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,30,5)
# cv.imwrite('./Output/img3_5_30_5.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,30,5)
# cv.imwrite('./Output/img3_7_30_5.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,7,15)
# cv.imwrite('./Output/img3_3_7_15.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,7,15)
# cv.imwrite('./Output/img3_5_7_15.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,7,15)
# cv.imwrite('./Output/img3_7_7_15.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,15,15)
# cv.imwrite('./Output/img3_3_15_15.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,15,15)
# cv.imwrite('./Output/img3_5_15_15.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,15,15)
# cv.imwrite('./Output/img3_7_15_15.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,30,15)
# cv.imwrite('./Output/img3_3_30_15.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,30,15)
# cv.imwrite('./Output/img3_5_30_15.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,30,15)
# cv.imwrite('./Output/img3_7_30_15.png',new_image)


# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,7,20)
# cv.imwrite('./Output/img3_3_7_20.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,7,20)
# cv.imwrite('./Output/img3_5_7_20.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,7,20)
# cv.imwrite('./Output/img3_7_7_20.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,15,20)
# cv.imwrite('./Output/img3_3_15_20.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,15,20)
# cv.imwrite('./Output/img3_5_15_20.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,15,20)
# cv.imwrite('./Output/img3_7_15_20.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,30,20)
# cv.imwrite('./Output/img3_3_30_20.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,30,20)
# cv.imwrite('./Output/img3_5_30_20.png',new_image)

# img = cv.imread("./Input/im3.png")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,30,20)
# cv.imwrite('./Output/img3_7_30_20.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,7,5)
# cv.imwrite('./Output/img2_3_7_5.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,7,5)
# cv.imwrite('./Output/img2_5_2_5.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,7,5)
# cv.imwrite('./Output/img2_7_2_5.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,15,5)
# cv.imwrite('./Output/img2_3_15_5.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,15,5)
# cv.imwrite('./Output/img2_5_15_5.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,15,5)
# cv.imwrite('./Output/img2_7_15_5.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,30,5)
# cv.imwrite('./Output/img2_3_30_5.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,30,5)
# cv.imwrite('./Output/img2_5_30_5.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,30,5)
# cv.imwrite('./Output/img2_7_30_5.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,7,15)
# cv.imwrite('./Output/img2_3_7_15.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,7,15)
# cv.imwrite('./Output/img2_5_7_15.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,7,15)
# cv.imwrite('./Output/img2_7_7_15.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,15,15)
# cv.imwrite('./Output/img2_3_15_15.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,15,15)
# cv.imwrite('./Output/img2_5_15_15.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,15,15)
# cv.imwrite('./Output/img2_7_15_15.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,30,15)
# cv.imwrite('./Output/img2_3_30_15.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,30,15)
# cv.imwrite('./Output/img2_5_30_15.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,30,15)
# cv.imwrite('./Output/img2_7_30_15.png',new_image)


# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,7,20)
# cv.imwrite('./Output/img2_3_7_20.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,7,20)
# cv.imwrite('./Output/img2_5_7_20.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,7,20)
# cv.imwrite('./Output/img2_7_7_20.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,15,20)
# cv.imwrite('./Output/img2_3_15_20.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,15,20)
# cv.imwrite('./Output/img2_5_15_20.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,15,20)
# cv.imwrite('./Output/img2_7_15_20.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,30,20)
# cv.imwrite('./Output/img2_3_30_20.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,30,20)
# cv.imwrite('./Output/img2_5_30_20.png',new_image)

# img = cv.imread("./Input/im2.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,30,20)
# cv.imwrite('./Output/img2_7_30_20.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,7,5)
# cv.imwrite('./Output/img1_3_7_5.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,7,5)
# cv.imwrite('./Output/img1_5_2_5.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,7,5)
# cv.imwrite('./Output/img1_7_2_5.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,15,5)
# cv.imwrite('./Output/img1_3_15_5.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,15,5)
# cv.imwrite('./Output/img1_5_15_5.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,15,5)
# cv.imwrite('./Output/img1_7_15_5.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,30,5)
# cv.imwrite('./Output/img1_3_30_5.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,30,5)
# cv.imwrite('./Output/img1_5_30_5.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,30,5)
# cv.imwrite('./Output/img1_7_30_5.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,7,15)
# cv.imwrite('./Output/img1_3_7_15.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,7,15)
# cv.imwrite('./Output/img1_5_7_15.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,7,15)
# cv.imwrite('./Output/img1_7_7_15.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,15,15)
# cv.imwrite('./Output/img1_3_15_15.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,15,15)
# cv.imwrite('./Output/img1_5_15_15.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,15,15)
# cv.imwrite('./Output/img1_7_15_15.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,30,15)
# cv.imwrite('./Output/img1_3_30_15.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,30,15)
# cv.imwrite('./Output/img1_5_30_15.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,30,15)
# cv.imwrite('./Output/img1_7_30_15.png',new_image)


# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,7,20)
# cv.imwrite('./Output/img1_3_7_20.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,7,20)
# cv.imwrite('./Output/img1_5_7_20.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,7,20)
# cv.imwrite('./Output/img1_7_7_20.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,15,20)
# cv.imwrite('./Output/img1_3_15_20.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,15,20)
# cv.imwrite('./Output/img1_5_15_20.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,15,20)
# cv.imwrite('./Output/img1_7_15_20.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,3,30,20)
# cv.imwrite('./Output/img1_3_30_20.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,5,30,20)
# cv.imwrite('./Output/img1_5_30_20.png',new_image)

# img = cv.imread("./Input/im1.jpg")
# image = cv.resize(img,(100,100),interpolation = cv.INTER_AREA)
# new_image = segment_kmeans(image,7,30,20)
# cv.imwrite('./Output/img1_7_30_20.png',new_image)




