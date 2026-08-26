# import necessary libraries for image processing
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
# 
# Task 1: Image Pre Processing - Research and testing different image
# pre-processing techniques to improve the quality of images for
# further analysis.

# Load Image with OpenCV
image_path = Path(__file__).parent / "images" / "1.jpeg"
image = cv2.imread(str(image_path))

# Resize the image
# Resize the image of the digit to a specific size
# Were resizing to 28x28 pixels for digit recognition tasks (28x28 because it is a common size for MNIST dataset)
image = cv2.resize(image, (256, 256))

# Convert the image to grayscale
# Convert the image to grayscale to reduce the number of channels and simplify the data
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



# Note: used openCV instead of Pillow because it is a widely used library for image processing and computer vision tasks. Also OpenCV loads images as NumPy arrays natively, so it drops straight into TensorFlow or PyTorch with no conversion step.

# Noise Reduction
# Applying blur on this 28X28 image isn't necessary, but if we were to use a larger image, we could apply a Gaussian blur to reduce noise and improve the quality of the image.
# REF: https://www.geeksforgeeks.org/python/python-image-blurring-using-opencv/
# REF: https://docs.opencv.org/4.13.0/d4/d13/tutorial_py_filtering.html
# 

# Run 3 images through the same pre-processing steps and compare the results
# Compare all three images
def show(images, titles=None, cols=4):
	rows = (len(images) + cols - 1) // cols
	figure, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
	axes = axes.ravel() if len(images) > 1 else [axes]

	for axis in axes:
		axis.axis("off")

	for index, current_image in enumerate(images):
		if current_image.ndim == 2:
			axes[index].imshow(current_image, cmap="gray")
		else:
			axes[index].imshow(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB))
		if titles:
			axes[index].set_title(titles[index])

	plt.tight_layout()
	plt.show()


all_images = []
all_titles = []
for number in range(1, 4):
	current_path = Path(__file__).parent / "images" / f"{number}.jpeg"
	original_image = cv2.imread(str(current_path))
	if original_image is None:
		raise FileNotFoundError(f"Could not load image: {current_path}")

	resized_image = cv2.resize(original_image, (256, 256))
	current_gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
	blurred_image = cv2.GaussianBlur(current_gray_image, (5, 5), 0)
	all_images.extend([original_image, resized_image, current_gray_image, blurred_image])
	all_titles.extend([
		f"Image {number} original",
		f"Image {number} resized",
		f"Image {number} grayscale",
		f"Image {number} Gaussian blur",
	])

show(all_images, all_titles)
# ------------- Do this before Resizing -----------------
# blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# 




# REF: https://medium.com/@maahip1304/the-complete-guide-to-image-preprocessing-techniques-in-python-dca30804550c
