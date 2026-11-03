"""
my_preprocessing.py
====================
Task 1 (Image Preprocessing) for HNRS 
"""

import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


# --------------------------------------------------------------------- #
# Step 1: Load the image
# --------------------------------------------------------------------- #
def load_image(path):
    """Open an image file from disk. Works with png/jpg/etc."""
    return Image.open(path)


# --------------------------------------------------------------------- #
# Step 2: Grayscale
# --------------------------------------------------------------------- #
def to_grayscale(img):
    """Collapse colour channels into a single 0-255 brightness channel."""
    return img.convert("L")


# --------------------------------------------------------------------- #
# Step 3: Binarization - three methods, so we can compare them
# --------------------------------------------------------------------- #
def binarize_fixed(gray_img, threshold=127):
    """
    Simplest method: pick one brightness cutoff by hand.
    Anything brighter than `threshold` becomes white (255), rest becomes black (0).
    Weak point: a fixed number won't work well across images with different lighting.
    """
    arr = np.array(gray_img)
    return ((arr > threshold) * 255).astype(np.uint8)


def _otsu_threshold(arr):
    """
    Otsu's method: automatically finds the best cutoff by testing every
    possible threshold (0-255) and picking the one that best separates the
    image into two groups (ink vs background) - specifically, the one that
    maximises the difference between the two groups' average brightness.
    """
    hist, _ = np.histogram(arr.flatten(), bins=256, range=(0, 256))
    total = arr.size
    sum_total = np.dot(np.arange(256), hist)

    sum_bg = 0.0
    weight_bg = 0.0
    best_variance = 0.0
    best_threshold = 0

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        # how well-separated the two groups are at this threshold
        variance_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance_between > best_variance:
            best_variance = variance_between
            best_threshold = t

    return best_threshold


def binarize_otsu(gray_img):
    """Automatic threshold - no guessing needed, adapts to each image."""
    arr = np.array(gray_img)
    t = _otsu_threshold(arr)
    return ((arr > t) * 255).astype(np.uint8)


def binarize_adaptive(gray_img, block_size=15, offset=10):
    """
    Adaptive threshold - instead of one global cutoff, compares each pixel to
    the average brightness of the small neighbourhood around it. Handles
    uneven lighting (e.g. a shadow across part of the page) better than
    fixed or Otsu, but is slower since it's computed pixel by pixel.
    """
    arr = np.array(gray_img).astype(np.float32)
    pad = block_size // 2
    padded = np.pad(arr, pad, mode="reflect")
    local_mean = np.zeros_like(arr)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            local_mean[i, j] = padded[i:i + block_size, j:j + block_size].mean()

    return ((arr > (local_mean - offset)) * 255).astype(np.uint8)


BINARIZATION_METHODS = {
    "fixed": lambda g: binarize_fixed(g),
    "otsu": lambda g: binarize_otsu(g),
    "adaptive": lambda g: binarize_adaptive(g),
}


# --------------------------------------------------------------------- #
# Extra preprocessing techniques - for a richer comparison
# --------------------------------------------------------------------- #
def denoise_median(gray_img, size=3):
    """
    Median filter - replaces each pixel with the median of its neighbours.
    Good at removing salt-and-pepper speckle (e.g. dust/grain from a phone
    camera) while keeping edges reasonably sharp, unlike a plain blur.
    """
    return gray_img.filter(ImageFilter.MedianFilter(size=size))


def denoise_blur(gray_img, radius=1):
    """
    Gaussian blur - smooths out noise but also softens the strokes
    themselves. Included so it can be compared against median filtering.
    """
    return gray_img.filter(ImageFilter.GaussianBlur(radius=radius))


def equalize_histogram(gray_img):
    """
    Histogram equalization - spreads out the range of brightness values so
    the image uses the full 0-255 range. Helps when a photo is washed out
    or too dark, making the ink stand out more before thresholding.
    """
    return ImageOps.equalize(gray_img)


def edge_detect(gray_img):
    """
    Edge detection - highlights the outline/boundary of the digit's strokes
    rather than the filled shape. Not used in the final pipeline, but useful
    to show in a report as an alternative representation of the digit.
    """
    return gray_img.filter(ImageFilter.FIND_EDGES)


def invert(gray_img):
    """Flips black and white. Useful when ink and background polarity is unknown."""
    return ImageOps.invert(gray_img)


EXTRA_METHODS = {
    "equalized": equalize_histogram,
    "median denoise": denoise_median,
    "gaussian blur": denoise_blur,
    "edges": edge_detect,
    "inverted": invert,
}


# --------------------------------------------------------------------- #
# Step 4: Resize and pad to 28x28 (matches MNIST input size)
# --------------------------------------------------------------------- #
def resize_and_pad(binary_arr, size=28):
    """
    Resize the digit so its longest side fits inside `size` pixels (keeping
    its proportions - no stretching), then paste it centred onto a black
    size x size canvas. This avoids distorting thin digits like "1" into
    wide blobs.
    """
    img = Image.fromarray(binary_arr)
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("L", (size, size), color=0)
    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2
    canvas.paste(resized, (x0, y0))
    return canvas


# --------------------------------------------------------------------- #
# Step 5: Normalize for the model
# --------------------------------------------------------------------- #
def to_model_input(img28):
    """Convert the final 28x28 image into a float32 array scaled to [0, 1]."""
    arr = np.array(img28).astype(np.float32) / 255.0
    return arr[np.newaxis, ...]  # shape (1, 28, 28)


# --------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------- #
def preprocess(image, method="otsu", size=28):
    """Run every step in order. `method` picks which binarization to use."""
    gray = to_grayscale(image)
    binary = BINARIZATION_METHODS[method](gray)
    final28 = resize_and_pad(binary, size)
    return final28


# --------------------------------------------------------------------- #
# Comparison figure - for the report
# --------------------------------------------------------------------- #
def _labeled(img, text):
    """Add a small text label above an image tile."""
    img = img.convert("RGB") if img.mode != "RGB" else img
    bar = Image.new("RGB", (img.width, 18), (30, 30, 30))
    draw = ImageDraw.Draw(bar)
    draw.text((3, 3), text, fill=(255, 255, 255))
    out = Image.new("RGB", (img.width, img.height + 18))
    out.paste(bar, (0, 0))
    out.paste(img, (0, 18))
    return out


def _comparison_tiles(image):
    """Build the labeled tiles (grayscale + extras + binarization methods) for one image."""
    gray = to_grayscale(image)
    tiles = [_labeled(gray, "grayscale")]

    for name, fn in EXTRA_METHODS.items():
        tiles.append(_labeled(fn(gray), name))

    for name, fn in BINARIZATION_METHODS.items():
        binary = fn(gray)
        tiles.append(_labeled(Image.fromarray(binary), name))

    return tiles


def comparison_figure(image):
    """
    Side-by-side tile of every technique this script knows about, for ONE
    image: grayscale, the extra preprocessing effects (denoise/equalize/
    edges/inverted), and all three binarization methods.
    """
    tiles = _comparison_tiles(image)

    # wrap into rows of 4 so it stays readable instead of one giant strip
    cols = 4
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    tile_w = max(t.width for t in tiles)
    tile_h = max(t.height for t in tiles)

    grid = Image.new("RGB", (tile_w * cols, tile_h * len(rows)), (16, 16, 16))
    for r, row in enumerate(rows):
        for c, t in enumerate(row):
            grid.paste(t, (c * tile_w, r * tile_h))
    return grid


def multi_comparison_figure(images):
    """
    Same set of techniques as comparison_figure, but for SEVERAL images:
    one row per image, so you can see how each technique behaves across
    different numbers side by side.
    """
    rows = [_comparison_tiles(img) for img in images]

    tile_w = max(t.width for row in rows for t in row)
    tile_h = max(t.height for row in rows for t in row)
    rows = [[t.resize((tile_w, tile_h)) for t in row] for row in rows]

    n_cols = len(rows[0])
    grid = Image.new("RGB", (tile_w * n_cols, tile_h * len(rows)), (16, 16, 16))
    for r, row in enumerate(rows):
        for c, t in enumerate(row):
            grid.paste(t, (c * tile_w, r * tile_h))
    return grid


# --------------------------------------------------------------------- #
# Stages figure - shows ONE digit moving through each processing step
# --------------------------------------------------------------------- #
def _stage_tiles(image, method="otsu", size=28):
    """Build the 4 labeled tiles (original/grayscale/binarized/28x28) for one image."""
    original = image.convert("RGB")
    gray = to_grayscale(image)
    binary = BINARIZATION_METHODS[method](gray)
    binary_img = Image.fromarray(binary)
    final28 = resize_and_pad(binary, size)
    # scale the tiny 28x28 result up so it's actually visible next to the others
    final_display = final28.resize((original.width, original.height), Image.NEAREST)

    return [
        _labeled(original, "1: original"),
        _labeled(gray, "2: grayscale"),
        _labeled(binary_img, f"3: binarized ({method})"),
        _labeled(final_display, "4: resized to 28x28"),
    ]


def stages_figure(image, method="otsu", size=28):
    """
    Takes ONE image and shows it after each step of the pipeline, side by
    side: original -> grayscale -> binarized -> resized/padded to 28x28.
    """
    tiles = _stage_tiles(image, method, size)
    total_width = sum(t.width for t in tiles)
    height = max(t.height for t in tiles)
    grid = Image.new("RGB", (total_width, height), (16, 16, 16))
    x = 0
    for t in tiles:
        grid.paste(t, (x, 0))
        x += t.width
    return grid


def multi_stages_figure(images, method="otsu", size=28):
    """
    Takes SEVERAL images (e.g. different handwritten numbers) and builds a
    grid with one row per image, each row showing that image's own
    original -> grayscale -> binarized -> 28x28 steps. Handy for showing
    the pipeline works consistently across several different samples.
    """
    rows = [_stage_tiles(img, method, size) for img in images]

    # make every tile the same size so the grid lines up neatly
    tile_w = max(t.width for row in rows for t in row)
    tile_h = max(t.height for row in rows for t in row)
    rows = [[t.resize((tile_w, tile_h)) for t in row] for row in rows]

    grid = Image.new("RGB", (tile_w * 4, tile_h * len(rows)), (16, 16, 16))
    for r, row in enumerate(rows):
        for c, tile in enumerate(row):
            grid.paste(tile, (c * tile_w, r * tile_h))
    return grid


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="HNRS Task 1 - my preprocessing implementation")
    p.add_argument("--image", required=True, nargs="+",
                   help="path to one input image, or several separated by spaces")
    p.add_argument("--method", default="otsu", choices=list(BINARIZATION_METHODS))
    p.add_argument("--out", help="where to save the final 28x28 result (single image only)")
    p.add_argument("--compare", help="where to save the technique-comparison figure (works for one or many images)")
    p.add_argument("--stages", help="where to save the step-by-step figure (works for one or many images)")
    args = p.parse_args()

    images = [load_image(path) for path in args.image]

    if args.stages:
        if len(images) == 1:
            stages_figure(images[0], method=args.method).save(args.stages)
        else:
            multi_stages_figure(images, method=args.method).save(args.stages)
        print(f"stages figure -> {args.stages} ({len(images)} image(s))")

    if args.compare:
        if len(images) == 1:
            comparison_figure(images[0]).save(args.compare)
        else:
            multi_comparison_figure(images).save(args.compare)
        print(f"comparison figure -> {args.compare} ({len(images)} image(s))")

    if len(images) > 1:
        if args.out:
            print("Note: --out only works with a single --image; skipping.")
        return

    img = images[0]

    result = preprocess(img, method=args.method)
    if args.out:
        result.save(args.out)
        print(f"28x28 result -> {args.out}")

    arr = to_model_input(result)
    print(f"output shape={arr.shape} dtype={arr.dtype} range=[{arr.min():.2f},{arr.max():.2f}]")


if __name__ == "__main__":
    main()