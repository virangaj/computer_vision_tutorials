import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def Brute_Force_Matching(img1, img2, dist):
    # Initiate SIFT detector
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)

    # Apply ratio test - distance 0.75
    good = []
    for m,n in matches:
        if m.distance < dist*n.distance:
            good.append([m])
            
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img3


img1 = cv.imread('bird.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('bird in scene.jpg',cv.IMREAD_GRAYSCALE) # trainImage

img3=Brute_Force_Matching(img1, img2, 0.75)
img4=Brute_Force_Matching(img1, img2, 0.2)


## distance 0.75
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Distance = 0.75")
plt.imshow(img3)

plt.subplot(1,2,2)
plt.axis("off")
plt.title("Distance = 0.2")
plt.imshow(img4)

plt.show()
