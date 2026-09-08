from pathlib import Path #builds file paths safely accross operating systems
import cv2 #OpenCV, used for reading, resizing, converting, thresholding and inverting images
import matplotlib.pyplot as plt #displays images
import numpy as np 

"""
COS30018 - Task 1 Image Preprocessing Experiment

Techniques tested:
1. Grayscale   
2. Otsu Thresholding
3. Inverted (MNIST-style)

what my code does
original image as input (digits) -> convert to grayscale -> convert to otsu threshold -> convert to Inverted -> compare all images in final output
"""


def show_results(images, titles, cols=4): # displays all processed images in a grid
    rows = (len(images) + cols - 1) // cols #to calculate how many rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = np.array(axes).reshape(-1) 

    for ax in axes:
        ax.axis("off")

    for i, img in enumerate(images):
        if img.ndim == 2:
            axes[i].imshow(img, cmap="gray")
        else:
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        axes[i].set_title(titles[i], fontsize=9)

    plt.tight_layout()
    plt.show()


# ---------------------------
# Process images
# ---------------------------
images_folder = Path(__file__).parent.parent / "images"

all_images = []
all_titles = []

print("\nImage Preprocessing Results")
print("-" * 40)

for number in range(1, 4):

    image_path = images_folder / f"{number}.jpeg"

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Could not load {image_path}")

    # Resize for consistent processing
    resized = cv2.resize(image, (256, 256))

    # Step 1: Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Step 2: Otsu Thresholding
    _, otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Step 3: Invert colours (MNIST style)
    inverted = cv2.bitwise_not(otsu)

    # Save images for comparison
    all_images.extend([resized, gray, otsu, inverted])
    all_titles.extend([
        f"Image {number}\nOriginal",
        "Grayscale",
        "Otsu",
        "Inverted"
    ])

    # Print useful information
    print(f"\nImage {number}")
    print(f"Brightness: {gray.mean():.2f}")
    print(f"Otsu threshold applied successfully.")

print("\nFinished processing all images.")

# Show comparison grid
show_results(all_images, all_titles)

#reference
#Learn Matplotlib in 30 Minutes - Python Matplotlib Tutorial: https://youtu.be/7Lc2AxiM17o?si=0CFUZduO3ohtBxSv
#Image Processing with OpenCV and Python: https://youtu.be/kSqxn6zGE0c?si=aJLOHnZN2-IBA7Z4
#The Complete Guide to Image Preprocessing Techniques in Python: https://medium.com/@maahip1304/the-complete-guide-to-image-preprocessing-techniques-in-python-dca30804550c