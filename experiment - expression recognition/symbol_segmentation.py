"""
symbol_segmentation.py
========================

"""

import my_preprocessing as pre
import segmentation as seg


def segment_symbols(image, method="otsu", min_area=15, margin=4):
    """
    Segments a raw photo of a handwritten mathematical expression into individual symbols.
    Returns a list of bounding boxes and a list of 28x28 images of the symbols
    """
    gray = pre.to_grayscale(image)
    binary = pre.BINARIZATION_METHODS[method](gray)

    foreground_mask = binary < 128  # ink is the dark pixels, per your convention
    labels, num_labels = seg.label_components(foreground_mask)
    boxes = seg.bounding_boxes(labels, num_labels, min_area=min_area)
    crops = seg.crop_digits(binary, boxes, margin=margin)

    symbol_imgs = []
    for crop in crops:
        inverted_crop = 255 - crop  # the polarity fix described above
        symbol_imgs.append(pre.resize_and_pad(inverted_crop, size=28))

    return boxes, symbol_imgs
