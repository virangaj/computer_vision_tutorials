import cv2
import numpy as np


def harris_corner_detection(image, window_size):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the Harris corner response
    harris_response = cv2.cornerHarris(gray, window_size, 3, 0.01)

    # Threshold the corner response to obtain corner points
    corner_threshold = 0.01 * harris_response.max()
    corner_points = np.where(harris_response > corner_threshold)
    # Create a descriptor for each corner point
    keypoints = []
    descriptors = []
    max_descriptor_size = 16  # Set the maximum descriptor size
    for y, x in zip(*corner_points):
        x = int(x)  # Convert x to integer
        y = int(y)  # Convert y to integer
        keypoint = cv2.KeyPoint(x, y, 5)  # Create a cv2.KeyPoint object
        keypoints.append(keypoint)
        descriptor = gray[y - 2:y + 2, x - 2:x + 2].flatten()
        # Pad or truncate the descriptor to a fixed size
        if len(descriptor) < max_descriptor_size:
            descriptor = np.pad(descriptor, (0, max_descriptor_size - len(descriptor)), mode='constant')
        elif len(descriptor) > max_descriptor_size:
            descriptor = descriptor[:max_descriptor_size]
        descriptors.append(descriptor)

    return keypoints, descriptors


def match_descriptors(desc1, desc2):
    # Create a brute-force matcher
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    desc1 = np.array(desc1)
    desc2 = np.array(desc2)
    # Match descriptors between two sets
    matches = matcher.match(desc1, desc2)
    # Sort the matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


# Load two images
image1 = cv2.imread("F://4th Yr//ML AI//sources//bike_1.png")
image2 = cv2.imread("F://4th Yr//ML AI//sources//bike_2.png")

windows_sizes = 2
# Perform Harris corner detection and descriptor extraction for both images
keypoints1, descriptors1 = harris_corner_detection(image1, windows_sizes)
keypoints2, descriptors2 = harris_corner_detection(image2, windows_sizes)

# Match descriptors between the two images
matches = match_descriptors(descriptors1, descriptors2)

# Draw the matched keypoints on the images
matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None)

# Display the matched image
cv2.imshow("Matched Keypoints", matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
