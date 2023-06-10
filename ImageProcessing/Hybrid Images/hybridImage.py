import cv2

# Load and resize the two input sources to the same size
image1 = cv2.imread("F://4th Yr//ML AI//sources//h_face1.jpg")
image2 = cv2.imread("F://4th Yr//ML AI//sources//monkey_face1.jpg")

print(image1.shape)
print(image2.shape)

# Resize the sources to the same dimensions
width, height = 800, 800
# Adjust the dimensions as needed
cropped_image = image2[40:, 40:]

image1 = cv2.resize(image1, (width, height))
image2 = cv2.resize(cropped_image, (width, height))

# Apply a Gaussian blur to each image
sigma1 = 5
sigma2 = 15

image1_blur = cv2.GaussianBlur(image1, (0, 0), sigmaX=sigma1)
image2_blur = cv2.GaussianBlur(image2, (0, 0), sigmaX=sigma2)

# Subtract the blurred sources from the original sources to obtain the high-frequency components:
image1_high_freq = cv2.subtract(image1, image1_blur)
image2_high_freq = cv2.subtract(image2, image2_blur)

# Combine the low-frequency components of one image with the high-frequency components of the other image:
hybrid_image = cv2.add(image1_blur, image2_high_freq)

cv2.imshow('Hybrid Image', hybrid_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
