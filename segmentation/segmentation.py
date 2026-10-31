"""
Task 2 - Image segmentation: cut a preprocessed image of a number into single digits

    segment(binary)  -  list of (box, digit28, line) in reading order:
                     lines top to bottom, each line left to right
                     box = (x, y, w, h) in the original image
                     digit28 = 28x28 MNIST-style array from make_mnist_like
                     line = 0 for the first line of writing, 1 for the next, ...

Three techniques are included so them can be compared:
    "components"  connected components  - each separate blob of ink is a digit
    "contours"    outer contours        - very similar, via cv2.findContours
    "projection"  vertical projection   - cut wherever a column has no ink at all

Try it:  python segmentation/segmentation.py test_images/2026.jpg
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))    # so "preprocessing" can be imported
from preprocessing.preprocessing import make_mnist_like, preprocess


# ---------- three ways to find the boxes ----------

def boxes_by_components(binary):
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return [tuple(stats[i, :4]) for i in range(1, count)]      # (x, y, w, h), skip background


def boxes_by_contours(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours]


def boxes_by_projection(binary):
    """Add up the ink in every column; runs of columns with ink are digits."""
    has_ink = binary.sum(axis=0) > 0
    boxes, start = [], None
    for x, ink in enumerate(list(has_ink) + [False]):         # extra False closes the last run
        if ink and start is None:
            start = x
        elif not ink and start is not None:
            rows = np.nonzero(binary[:, start:x].sum(axis=1))[0]
            boxes.append((start, rows.min(), x - start, rows.max() - rows.min() + 1))
            start = None
    return boxes


# ---------- clean-up rules ----------

def merge_overlapping(boxes, max_height):
    """Join boxes that sit above each other (a broken '5', the dots of a division sign),
    but only if the joined box is still about one character tall. Two digits on
    different lines also share columns, and joining those makes a box twice as tall."""
    merged = []
    for bx, by, bw, bh in sorted(boxes, key=lambda b: b[0]):
        for i, (x, y, w, h) in enumerate(merged):           # check every box, not just the last
            overlap = min(x + w, bx + bw) - max(x, bx)
            top, bottom = min(y, by), max(y + h, by + bh)
            if overlap > 0.5 * min(w, bw) and bottom - top <= 1.3 * max_height:
                left = min(x, bx)
                merged[i] = (left, top, max(x + w, bx + bw) - left, bottom - top)
                break
        else:
            merged.append((bx, by, bw, bh))
    return merged


def group_into_lines(boxes):
    """Reading order: lines top to bottom, each line left to right.
    A box joins a line when its vertical middle falls inside that line's top..bottom."""
    lines = []
    for box in sorted(boxes, key=lambda b: b[1] + b[3] / 2):
        x, y, w, h = box
        middle = y + h / 2
        for line in lines:
            if line["top"] <= middle <= line["bottom"]:
                line["boxes"].append(box)
                line["top"], line["bottom"] = min(line["top"], y), max(line["bottom"], y + h)
                break
        else:
            lines.append({"top": y, "bottom": y + h, "boxes": [box]})
    return [sorted(line["boxes"], key=lambda b: b[0]) for line in lines]


def crop_box(binary, labels, stats, box):
    """Cut out one box, keeping only blobs that lie fully inside it. A plain rectangle
    crop also picks up the edge of a neighbour (or a dropped speck) that pokes into it."""
    x, y, w, h = box
    piece = binary[y:y + h, x:x + w].copy()
    ids = labels[y:y + h, x:x + w]
    for i in np.unique(ids[ids > 0]):
        bx, by, bw, bh = stats[i, :4]
        if bx < x or by < y or bx + bw > x + w or by + bh > y + h:
            piece[ids == i] = 0
    return piece

def split_wide(binary, box, typical_height, min_ratio=1.3, max_neck=0.2):
    """Two touching digits make one wide box. Cut it at the column with the least ink.

    Only cut if the box is clearly wide (width > min_ratio x height) AND the ink gets
    thin somewhere in the middle (a "neck" with under max_neck of the fullest column).
    A single wide digit like a flat "2" has no neck, so it is left alone.
    """
    x, y, w, h = box
    if h < 0.5 * typical_height or w < min_ratio * h:     # not a tall-and-wide blob: leave it
        return [box]
    columns = binary[y:y + h, x:x + w].sum(axis=0).astype(float)
    left, right = int(w * 0.3), int(w * 0.7)               # only cut near the middle
    cut = left + int(np.argmin(columns[left:right]))
    if columns[cut] > max_neck * columns.max():            # no thin neck: probably one digit
        return [box]
    return (split_wide(binary, (x, y, cut, h), typical_height)
            + split_wide(binary, (x + cut, y, w - cut, h), typical_height))


# ---------- the main function ----------

def segment(binary, method="components", split_touching=True):
    """Return [(box, digit28, line), ...] in reading order."""
    if method == "components":
        boxes = boxes_by_components(binary)
    elif method == "contours":
        boxes = boxes_by_contours(binary)
    elif method == "projection":
        boxes = boxes_by_projection(binary)
    else:
        raise ValueError(f"unknown method {method}")
    if not boxes:
        return []

    # Drop anything touching the picture's edge: in a photo that is the table, the edge
    # of the paper or a shadow, not handwriting (Kept if it would remove everything)
    height, width = binary.shape
    inside = [(x, y, w, h) for x, y, w, h in boxes
              if x > 0 and y > 0 and x + w < width and y + h < height]
    if inside:
        boxes = inside

    #   Drop leftovers much smaller than the biggest mark (noise the preprocessing missed)
    biggest = max(w * h for _, _, w, h in boxes)
    boxes = [b for b in boxes if b[2] * b[3] > 0.01 * biggest]

    boxes = merge_overlapping(boxes, max_height=max(h for _, _, _, h in boxes))

    # Drop marks far smaller than a normal digit (a full stop, a stray dot): the model
    # would blow them up to 20x20 and read a filled circle as "0". Using the longer
    # side keeps a wide-but-flat minus sign
    typical_height = float(np.median([h for _, _, _, h in boxes]))
    boxes = [b for b in boxes if max(b[2], b[3]) >= 0.3 * typical_height]

    was_split = set()                                      # halves of touching digits share one blob
    if split_touching:
        new = []
        for b in boxes:
            parts = split_wide(binary, b, typical_height)
            if len(parts) > 1:
                was_split.update(parts)
            new += parts
        boxes = new

    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    results = []
    for line_number, line in enumerate(group_into_lines(boxes)):
        for box in line:
            x, y, w, h = box
            digit = binary[y:y + h, x:x + w] if box in was_split else crop_box(binary, labels, stats, box)
            results.append(((int(x), int(y), int(w), int(h)), make_mnist_like(digit), line_number))
    return results


# ---------- show the result when run directly ----------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    path = sys.argv[1] if len(sys.argv) > 1 else "test_images/2026.jpg"
    image = cv2.imread(path)
    if image is None:
        sys.exit(f"Could not read {path}")
    binary = preprocess(image)
    parts = segment(binary)
    print(f"Found {len(parts)} digit(s)")

    shown = image.copy()
    thickness = max(2, image.shape[1] // 250)            # thicker lines on big phone photos
    for ax, (_, digit) in zip(axes[1:], parts):
        cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 0, 255), thickness)
    fig, axes = plt.subplots(1, len(parts) + 1, figsize=(3 * (len(parts) + 1), 3))
    axes = np.atleast_1d(axes)
    axes[0].imshow(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB))
    for ax, (_, digit) in zip(axes[1:], parts):
        ax.imshow(digit, cmap="gray")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()