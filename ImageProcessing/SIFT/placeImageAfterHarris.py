import cv2
import numpy as np


def harris_corner_detection(image, window_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    harris_response = cv2.cornerHarris(gray, window_size, 3, 0.01)
    corner_threshold = 0.01 * harris_response.max()
    corner_points = np.where(harris_response > corner_threshold)

    keypoints = []
    descriptors = []
    max_descriptor_size = 16

    for y, x in zip(*corner_points):
        x = int(x)
        y = int(y)
        keypoint = cv2.KeyPoint(x, y, 5)
        keypoints.append(keypoint)
        descriptor = gray[y - 2:y + 2, x - 2:x + 2].flatten()

        if len(descriptor) < max_descriptor_size:
            descriptor = np.pad(descriptor, (0, max_descriptor_size - len(descriptor)), mode='constant')
        elif len(descriptor) > max_descriptor_size:
            descriptor = descriptor[:max_descriptor_size]

        descriptors.append(descriptor)

    return keypoints, descriptors


def match_descriptors(desc1, desc2):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    desc1 = np.array(desc1)
    desc2 = np.array(desc2)
    matches = matcher.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


image1 = cv2.imread("F://4th Yr//ML AI//sources//plane_1.jpeg")
image2 = cv2.imread("F://4th Yr//ML AI//sources//plane_2.jpeg")

window_size = 2

keypoints1, descriptors1 = harris_corner_detection(image1, window_size)
keypoints2, descriptors2 = harris_corner_detection(image2, window_size)

matches = match_descriptors(descriptors1, descriptors2)

# Find the transformation matrix using the matching keypoints
src_pts = np.float32([keypoints1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Warp image1 to align with image2
h, w = image1.shape[:2]
aligned_image = cv2.warpPerspective(image1, M, (w, h))

# Create a blank image with the same size as image2
output_height, output_width = image2.shape[:2]
output_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)

# Copy image2 to the output image
output_image = image2.copy()

# Overlay the aligned image on the output image
output_image[0:h, 0:w] = aligned_image

# Display the result
cv2.imshow("Aligned Image", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
