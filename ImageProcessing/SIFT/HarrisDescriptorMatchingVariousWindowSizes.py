import cv2
import numpy as np


def harris_corner_detection(image, window_size):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the Harris corner response
    harris_response = cv2.cornerHarris(gray, 2, 3, 0.2)

    # Threshold the corner response to obtain corner points
    corner_threshold = 0.01 * harris_response.max()
    corner_points = np.where(harris_response > corner_threshold)

    # Create a descriptor for each corner point
    descriptors = []
    max_descriptor_size = 16  # Set the maximum descriptor size

    for y, x in zip(*corner_points):
        descriptor = gray[y - 2:y + 2, x - 2:x + 2].flatten()  # Example descriptor: 4x4 patch flattened

        # Pad or truncate the descriptor to a fixed size
        if len(descriptor) < max_descriptor_size:
            descriptor = np.pad(descriptor, (0, max_descriptor_size - len(descriptor)), mode='constant')
        elif len(descriptor) > max_descriptor_size:
            descriptor = descriptor[:max_descriptor_size]

        descriptors.append(descriptor)

    return corner_points, descriptors


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


# Load the image
image = cv2.imread("F://4th Yr//ML AI//sources//plane_1.jpeg")

# Define the window sizes to test
window_sizes = [3, 5, 7, 9]

# Iterate over the window sizes and perform Harris corner detection
for window_size in window_sizes:
    # Perform Harris corner detection and descriptor extraction
    corner_points, descriptors = harris_corner_detection(image, window_size)

    # Match descriptors between the same image
    matches = match_descriptors(descriptors, descriptors)

    # Draw the matched keypoints on the image
    matched_image = cv2.drawMatches(image, np.transpose(corner_points), image, np.transpose(corner_points), matches, None)

    # Display the matched image
    cv2.imshow("Matched Keypoints (Window Size: {})".format(window_size), matched_image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
