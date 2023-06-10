
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the two images
img1 = cv2.imread('plane.jpeg')
img2 = cv2.imread('plane2.jpeg')

# Convert the images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Create SIFT object
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors using SIFT
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Create a Brute-Force Matcher
bf = cv2.BFMatcher()

# Harris corner detection with different window sizes
window_sizes = [2, 5, 100]  # Vary the window sizes as desired

fig, axes = plt.subplots(len(window_sizes), figsize=(8, 6))

for i, window_size in enumerate(window_sizes):
    # Perform Harris corner detection
    dst = cv2.cornerHarris(gray1, blockSize=window_size, ksize=3, k=0.04)

    # Threshold the corner response
    dst_thresh = 0.01 * dst.max()
    corners = np.argwhere(dst > dst_thresh)  # Get the coordinates of the detected corners

    # Extract descriptors using intensity-based local window
    descriptors_harris = []
    for corner in corners:
        x, y = corner.ravel()
        window = gray1[x - window_size: x + window_size + 1, y - window_size: y + window_size + 1]
        descriptors_harris.append(window.flatten())

    # Match descriptors using Brute-Force Matcher
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)  # Change the ratio 'k' as desired

    # Apply ratio test on matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:  # Change the ratio '0.7' as desired
            good_matches.append(m)

    # Draw matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Plot the matches
    axes[i].imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    axes[i].set_title(f'Harris (Window Size: {window_size})')
    axes[i].axis('off')

# Adjust the subplot spacing
plt.tight_layout()

# Display the plots
plt.show()
