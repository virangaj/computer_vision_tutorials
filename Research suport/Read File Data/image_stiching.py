import cv2
import numpy as np

img1 = cv2.imread("F://4th Yr//ML AI//sources//drone_1.jpg")
img2 = cv2.imread("F://4th Yr//ML AI//sources//drone_2.jpg")
desired_width = 500
desired_height = 300
image1 = cv2.resize(img1, (desired_width, desired_height))
image2 = cv2.resize(img2, (desired_width, desired_height))



# Load additional images if required
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
matches = matcher.match(descriptors1, descriptors2, None)
matches = sorted(matches, key=lambda x: x.distance)


good_matches = matches[:100]
points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

for i, match in enumerate(good_matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
homography, _ = cv2.findHomography(points2, points1, cv2.RANSAC)
result = cv2.warpPerspective(image2, homography, (image1.shape[1] + image2.shape[1], image1.shape[0]))
result[0:image1.shape[0], 0:image1.shape[1]] = image1
cv2.imshow('Stitched Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
