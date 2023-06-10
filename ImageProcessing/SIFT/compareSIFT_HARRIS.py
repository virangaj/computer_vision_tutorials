import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_corners(image):
    # Convert the image to grayscale

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the Harris corner response
    corner_response = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

    # Threshold the corner response to obtain corner points
    corner_response_threshold = 0.01 * corner_response.max()
    corner_mask = corner_response > corner_response_threshold

    # Apply non-maximum suppression to eliminate weak corners
    corners = np.zeros_like(corner_response)
    corners[corner_mask] = corner_response[corner_mask]
    corners = cv2.dilate(corners, None)

    # Display the detected corners on the original image
    image_with_corners = image.copy()
    image_with_corners[corners > 0.01 * corners.max()] = [0, 0, 255]  # Draw red circles on corners
    image_with_corners = cv2.cvtColor(image_with_corners, cv2.COLOR_BGR2RGB)
    return image_with_corners


# Load the input image
image = cv2.imread("F://4th Yr//ML AI//sources//plane_1.jpg")

# scale image
image_scaleup = cv2.resize(image, (800, 800))
# Apply Harris corner detection
result = detect_corners(image)
result1 = detect_corners(image_scaleup)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

img_copy = image_scaleup.copy()
img_copy1 = image_scaleup.copy()
gray = cv2.cvtColor(image_scaleup, cv2.COLOR_BGR2GRAY)
gray_copy = gray.copy()
gray_copy1 = gray.copy()
sift1 = cv2.SIFT_create()
sift2 = cv2.SIFT_create(nfeatures=0, nOctaveLayers=5, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
sift3 = cv2.SIFT_create(nfeatures=0, nOctaveLayers=5, contrastThreshold=0.04, edgeThreshold=20, sigma=1.6)

kp1 = sift1.detect(gray, None)
kp2 = sift2.detect(gray_copy, None)
kp3 = sift3.detect(gray_copy1, None)

img1 = cv2.drawKeypoints(gray, kp1, image_scaleup)
img2 = cv2.drawKeypoints(gray_copy, kp2, img_copy)
img3 = cv2.drawKeypoints(gray_copy1, kp3, img_copy1)

# Display the original image and the result
plt.subplot(231)
plt.title('original')
plt.axis('off')
plt.imshow(image)
plt.subplot(232)
plt.title('Harris Corner Dtection')
plt.axis('off')
plt.imshow(result)

plt.subplot(233)
plt.title('Harris Corner Dtection for scaleup')
plt.axis('off')
plt.imshow(result1)

plt.subplot(234)
plt.title('SIFT_create() \nnfeatures = 0\nnOctaveLayers = 3\ncontrastThreshold = 0.04\nedgeThreshold = 10\nsigma = 1.6')
plt.axis('off')
plt.imshow(img1)

plt.subplot(235)
plt.title('nfeatures = 0\nnOctaveLayers = 5\ncontrastThreshold = 0.04\nedgeThreshold = 10\nsigma = 1.6')
plt.axis('off')
plt.imshow(img2)

plt.subplot(236)
plt.title('nfeatures = 0\nnOctaveLayers = 5\ncontrastThreshold = 0.04\nedgeThreshold = 20\nsigma = 1.6')
plt.axis('off')
plt.imshow(img3)
plt.show()