import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("F://4th Yr//ML AI//sources//image2.jpg")
img_copy = img.copy()
img_copy1 = img.copy()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_copy = gray.copy()
gray_copy1 = gray.copy()
sift1 = cv.SIFT_create()


'''
SIFT::create	(	int 	nfeatures,
int 	nOctaveLayers,
double 	contrastThreshold,
double 	edgeThreshold,
double 	sigma,
int 	descriptorType 
)		
'''


sift2 = cv.SIFT_create(nfeatures=0, nOctaveLayers=5, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
sift3 = cv.SIFT_create(nfeatures=0, nOctaveLayers=5, contrastThreshold=0.04, edgeThreshold=20, sigma=1.6)

kp1 = sift1.detect(gray, None)
kp2 = sift2.detect(gray_copy, None)
kp3 = sift3.detect(gray_copy1, None)

img1 = cv.drawKeypoints(gray, kp1, img)
img2 = cv.drawKeypoints(gray_copy, kp2, img_copy)
img3 = cv.drawKeypoints(gray_copy1, kp3, img_copy1)

plt.subplot(131)
plt.title('SIFT_create() \nnfeatures = 0\nnOctaveLayers = 3\ncontrastThreshold = 0.04\nedgeThreshold = 10\nsigma = 1.6')
plt.axis('off')
plt.imshow(img1)

plt.subplot(132)
plt.title('nfeatures = 0\nnOctaveLayers = 5\ncontrastThreshold = 0.04\nedgeThreshold = 10\nsigma = 1.6')
plt.axis('off')
plt.imshow(img2)

plt.subplot(133)
plt.title('nfeatures = 0\nnOctaveLayers = 5\ncontrastThreshold = 0.04\nedgeThreshold = 20\nsigma = 1.6')
plt.axis('off')
plt.imshow(img3)

plt.show()
