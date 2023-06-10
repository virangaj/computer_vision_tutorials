import cv2

# Load the two input images
image1 = cv2.imread("F://4th Yr//ML AI//sources//h_face2.jpg")
image2 = cv2.imread("F://4th Yr//ML AI//sources//h_face3.jpg")

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Create SIFT objects
sift = cv2.SIFT_create()

# Detect key points and compute descriptors for both images
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Create a Brute-Force Matcher
bf = cv2.BFMatcher()

# Perform matching
matches = bf.match(descriptors1, descriptors2)

# Sort the matches by distance (lower distance means better match)
matches = sorted(matches, key=lambda x: x.distance)

# Draw top matches
matching_result = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matching result
cv2.imshow('Matching Result', matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
