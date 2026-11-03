"""
segmentation.py
================
Task 2 (Image Segmentation) for HNRS.

Method chosen: Connected Component Analysis.

Idea: after binarizing a multi-digit image, each separate blob of touching
white pixels is one digit (as long as digits don't touch each other). This
script finds each blob, crops it out, sorts left-to-right, and feeds each
crop through the Task 1 pipeline (my_preprocessing.py) to get 28x28 images
ready for the model.

Setup (only needs to be done once):
    pip install pillow numpy

Usage:
    python segmentation.py --image number.png --out_dir segmented
"""

import argparse
import os
import numpy as np
from PIL import Image, ImageDraw

import my_preprocessing as pre  # reuses grayscale / Otsu / resize_and_pad from Task 1


# --------------------------------------------------------------------- #
# Step 1: Connected component labeling (implemented manually with NumPy)
# --------------------------------------------------------------------- #
def label_components(foreground_mask):
    """
    Groups touching foreground pixels into numbered blobs.
    Returns:
        labels      - 2D array of same shape as foreground_mask, with each
                      pixel labeled with its blob number (0 = background)
        num_labels  - how many blobs were found 
    """
    h, w = foreground_mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    visited = ~foreground_mask  # background pixels count as already visited
    current_label = 0

    # the 8 neighbouring positions around a pixel (up/down/left/right + diagonals)
    neighbours = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                  (0, 1), (1, -1), (1, 0), (1, 1)]

    for start_y in range(h):
        for start_x in range(w):
            if visited[start_y, start_x]:
                continue

            current_label += 1
            stack = [(start_y, start_x)]
            visited[start_y, start_x] = True

            while stack:
                y, x = stack.pop()
                labels[y, x] = current_label
                for dy, dx in neighbours:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))

    return labels, current_label


# --------------------------------------------------------------------- #
# Step 2: Turn each labeled blob into a bounding box
# --------------------------------------------------------------------- #
def bounding_boxes(labels, num_labels, min_area=15):
    """
    For each blob number, finds the smallest rectangle that contains it.
    Blobs smaller than `min_area` pixels are dropped, since they are
    usually background noise/specks rather than real digits.

    Returns a list of (x0, y0, x1, y1) boxes, sorted left-to-right.
    """
    boxes = []
    for label_id in range(1, num_labels + 1):
        ys, xs = np.where(labels == label_id)
        if len(xs) < min_area:
            continue  # too small to be a real digit stroke
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        boxes.append((int(x0), int(y0), int(x1) + 1, int(y1) + 1))

    boxes.sort(key=lambda b: b[0])  # left-to-right, matches reading order
    return boxes


# --------------------------------------------------------------------- #
# Step 3: Crop each digit out, with a small margin, and prepare for the model
# --------------------------------------------------------------------- #
def crop_digits(binary_arr, boxes, margin=4):
    """Crops each bounding box out of the binary image, with a small margin."""
    h, w = binary_arr.shape
    crops = []
    for (x0, y0, x1, y1) in boxes:
        x0m = max(0, x0 - margin)
        y0m = max(0, y0 - margin)
        x1m = min(w, x1 + margin)
        y1m = min(h, y1 + margin)
        crops.append(binary_arr[y0m:y1m, x0m:x1m])
    return crops


# --------------------------------------------------------------------- #
# Full segmentation pipeline
# --------------------------------------------------------------------- #
def segment_image(image, method="otsu", min_area=15, margin=4):
    """
    Runs the whole Task 2 pipeline on one multi-digit image:
        grayscale -> binarize -> label blobs -> boxes -> crop -> 28x28

    Returns:
        boxes       - bounding box per digit, left-to-right
        digit_imgs  - list of 28x28 PIL images, one per digit, left-to-right
    """
    gray = pre.to_grayscale(image)
    binary = pre.BINARIZATION_METHODS[method](gray)

    # binarize functions output ink=0 (dark), background=255 (light) -
    # same convention as the rest of Task 1 pipeline - so foreground
    # pixels are the DARK ones here, not the white ones.
    foreground_mask = binary < 128

    labels, num_labels = label_components(foreground_mask)
    boxes = bounding_boxes(labels, num_labels, min_area=min_area)
    crops = crop_digits(binary, boxes, margin=margin)

    digit_imgs = [pre.resize_and_pad(crop, size=28) for crop in crops]
    return boxes, digit_imgs


# --------------------------------------------------------------------- #
# Visualization - draws boxes on the original + shows the crops in a row
# --------------------------------------------------------------------- #
def visualize(image, boxes, digit_imgs):
    """Builds one figure: the original image with boxes drawn, above the
    individual 28x28 crops it produced, in the order they were read."""
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    for (x0, y0, x1, y1) in boxes:
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)

    tile_h = 60
    strip = Image.new("RGB", (max(annotated.width, tile_h * len(digit_imgs)), tile_h),
                       (16, 16, 16))
    x = 0
    for d in digit_imgs:
        strip.paste(d.convert("RGB").resize((tile_h, tile_h)), (x, 0))
        x += tile_h

    out = Image.new("RGB", (max(annotated.width, strip.width),
                             annotated.height + strip.height), (16, 16, 16))
    out.paste(annotated, (0, 0))
    out.paste(strip, (0, annotated.height))
    return out


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="HNRS Task 2 - connected component segmentation")
    p.add_argument("--image", required=True, help="path to a multi-digit number image")
    p.add_argument("--method", default="otsu", choices=list(pre.BINARIZATION_METHODS))
    p.add_argument("--min_area", type=int, default=15,
                   help="blobs smaller than this many pixels are treated as noise")
    p.add_argument("--margin", type=int, default=4,
                   help="extra pixels of padding kept around each cropped digit")
    p.add_argument("--out_dir", default="segmented", help="folder to save results into")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    image = pre.load_image(args.image)

    boxes, digit_imgs = segment_image(image, method=args.method,
                                       min_area=args.min_area, margin=args.margin)

    print(f"found {len(digit_imgs)} digit(s)")

    for i, d in enumerate(digit_imgs):
        path = os.path.join(args.out_dir, f"digit_{i}.png")
        d.save(path)
        print(f"  digit {i} -> {path}")

    vis_path = os.path.join(args.out_dir, "segmentation_result.png")
    visualize(image, boxes, digit_imgs).save(vis_path)
    print(f"visualization -> {vis_path}")


if __name__ == "__main__":
    main()