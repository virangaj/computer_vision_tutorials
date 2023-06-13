import cv2

# Load the images
image1 = cv2.imread("F://4th Yr//ML AI//sources//bike_1.png", 0)
image2 = cv2.imread("F://4th Yr//ML AI//sources//bike_2.png", 0)


# Create a simple descriptor for a feature in a 5x5 neighborhood

def get_key_points(img):
    # Create a SIFT object
    sift = cv2.xfeatures2d.SIFT_create()

    # Detect keypoints in the image
    keypoints1 = sift.detect(img, None)
    return keypoints1


def simple_descriptor(image, keypoint):
    x, y = keypoint.pt
    patch = image[int(y) - 2:int(y) + 3, int(x) - 2:int(x) + 3]
    return patch.flatten()  # Flatten the patch into a 1D array


# Extract features using the simple descriptor
keypoints1 = get_key_points(image1) # Obtain keypoints in image1 using a feature detection method
keypoints2 = get_key_points(image2)  # Obtain keypoints in image2 using the same feature detection method

descriptors1 = [simple_descriptor(image1, kp) for kp in keypoints1]
descriptors2 = [simple_descriptor(image2, kp) for kp in keypoints2]

# Use the SIFT descriptor implementation in OpenCV
sift = cv2.xfeatures2d.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)


# Perform matching using the simple method (SSD distance)
def simple_match(descriptor1, descriptor2):
    distance = sum((descriptor1 - descriptor2) ** 2)  # Compute the sum of squared differences
    return distance


# Match features using the simple method
matches = []
for i, desc1 in enumerate(descriptors1):
    best_match = None
    best_distance = float('inf')
    for j, desc2 in enumerate(descriptors2):
        distance = simple_match(desc1, desc2)
        if distance < best_distance:
            best_distance = distance
            best_match = j
    matches.append(cv2.DMatch(i, best_match, best_distance))

# Use the SIFT matching implementation in OpenCV
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)  # Perform 2-nearest neighbor matching

# Apply distance ratio test to filter matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:  # Adjust the distance ratio threshold as needed
        good_matches.append(m)
