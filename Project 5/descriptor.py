import numpy as np
import cv2 as cv

img = cv.imread(r"F://4th Yr//ML AI//sources//house_3.png")
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

mask_size = (5, 5)
mask = cv.getStructuringElement(cv.MORPH_RECT, mask_size)

sift = cv.SIFT_create()
kp = sift.detect(gray,None)
img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
kp, des = sift.detectAndCompute(gray,mask)

cv.imshow('sift_keypoints.jpg',img)
cv.waitKey(0)
cv.destroyAllWindows()


