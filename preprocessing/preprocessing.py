"""
Task 1 - Image preprocessing

Turns a photo / scan / drawing of handwriting into what the models expect:
white ink on a black background, with each digit finally shrunk to a 28x28
MNIST-style picture

    preprocess(image)          whole image  -> clean black/white image (ink = 255)
    make_mnist_like(digit)     one digit    -> 28x28 float array (0..1), centred like MNIST

try it:  python preprocessing/preprocessing.py test_images/2026.jpg
"""
import sys

import cv2
import numpy as np


def to_grayscale(image):
    """Colour (BGR) -> grayscale. Already-grey images are returned unchanged"""
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def binarize(gray, method="adaptive"):
    """Grayscale -> black/white with the INK WHITE (255) and the paper BLACK (0)

    method:
      "fixed"     one threshold (127) for the whole image - simple, fails with shadows
      "otsu"      picks the best single threshold automatically - good for even lighting
      "adaptive"  a separate threshold for each small area - best for phone photos with shadows
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)          # smooth out camera noise first
    if method == "fixed":
        _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    elif method == "otsu":
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == "adaptive":
        # blockSize: size of the local area (odd number), C: how much darker than
        # the local average a pixel must be to count as ink
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, blockSize=31, C=10)
    else:
        raise ValueError(f"unknown method {method}")

    # If most of the image came out white, the input was light ink on a dark
    # background (like MNIST itself), so flip it back.
    if np.mean(binary) > 127:
        binary = 255 - binary
    return binary

def remove_specks(binary, min_area=20):
    """Delete tiny white blobs (dust, paper texture) smaller than min_area pixels."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for i in range(1, count):                            # label 0 is the background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def preprocess(image, method="adaptive"):
    """full preprocessing for one input image (a numpy array from cv2.imread)"""

    gray = to_grayscale(image)
    binary = binarize(gray, method)

    # min_area grows with the image so big phone photos lose more noise than small drawings
    min_area = max(10, int(gray.shape[0] * gray.shape[1] * 0.00002))
    return remove_specks(binary, min_area)

def make_mnist_like(digit):
    """One cropped digit (white on black) -> 28x28 float32 array, values 0..1

    MNIST digits are drawn thick, fitted into a 20x20 box and placed in a 28x28
    image so that their centre of mass is in the middle
    """
    ys, xs = np.nonzero(digit)
    if len(xs) == 0:
        return np.zeros((28, 28), dtype=np.float32)
    digit = digit[ys.min():ys.max() + 1, xs.min():xs.max() + 1]   # tight crop
    h, w = digit.shape

    # Thin pen strokes almost vanish when shrunk, so thicken them first
    # MNIST strokes are roughly 1/8 of the digit's height
    distance = cv2.distanceTransform(digit, cv2.DIST_L2, 3)
    stroke = 2 * np.percentile(distance[distance > 0], 90)
    target = max(h, w) / 8
    if stroke < target:
        size = int(target - stroke) | 1                  # odd kernel size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        digit = cv2.copyMakeBorder(digit, size, size, size, size, cv2.BORDER_CONSTANT, value=0)
        digit = cv2.dilate(digit, kernel)
        h, w = digit.shape

    # Resize so the longer side is 20 pixels, keeping the shape (a "1" stays thin)
    scale = 20.0 / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    small = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Paste into the middle of a 28x28 black image
    canvas = np.zeros((28, 28), dtype=np.uint8)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = small

    # Shift so the centre of mass is exactly in the middle
    moments = cv2.moments(canvas)
    if moments["m00"] > 0:

        cx, cy = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
        shift = np.float32([[1, 0, 14 - cx], [0, 1, 14 - cy]])
        canvas = cv2.warpAffine(canvas, shift, (28, 28))

    return canvas.astype(np.float32) / 255.0

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    path = sys.argv[1] if len(sys.argv) > 1 else "test_images/2026.jpg"
    image = cv2.imread(path)
    if image is None:
        sys.exit(f"Could not read {path}")
    gray = to_grayscale(image)
    methods = ["fixed", "otsu", "adaptive"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("grayscale")
    for ax, m in zip(axes[1:], methods):
        ax.imshow(preprocess(image, m), cmap="gray")
        ax.set_title(m)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()