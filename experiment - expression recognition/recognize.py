"""
Simplest possible way to use this: give it ONE photo of a whole
handwritten expression, get the recognized expression + computed answer
back. No manual splitting of the image needed - segmentation inside the
pipeline does that automatically.

Usage:
    python recognize.py path/to/your_photo.jpg
"""

import sys
from pipeline import recognize_expression_image


def main():
    if len(sys.argv) != 2:
        print("Usage: python recognize.py path/to/your_photo.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    out = recognize_expression_image(image_path)

    print(f"Image: {image_path}")
    print(f"Symbols found ({out['n_segments']}): {' '.join(out['symbols'])}")
    print(f"Recognized expression: {out['expression']}")
    print(f"Computed answer: {out['result']}")
    if out["error"]:
        print(f"Warning: {out['error']}")


if __name__ == "__main__":
    main()
