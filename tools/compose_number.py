"""
Build images of multi-digit numbers out of single-digit images.
This covers the assignment's "automatic creation of the image of a number from a
folder of images of individual digits", and makes the synthetic test set.

python tools/compose_number.py export        # 1. save MNIST test digits into data/digit_folders/0..9
python tools/compose_number.py 4721          # 2. make one image of 4721 -> composed_4721.png
python tools/compose_number.py testset 1000  # 3. make 1000 labelled test images

Images are black ink on white, like handwriting on paper.
"""
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from torchvision import datasets

DIGIT_ROOT = Path("data/digit_folders")          # data/digit_folders/0/*.png ... 9/*.png
TESTSET_DIR = Path("data/multidigit_synthetic")

def export_mnist_digits(per_digit=200):
    """Save MNIST *test* digits as PNG files, one folder per digit.
    We use the test split so these pictures were never seen during training."""
    test = datasets.MNIST("data", train=False, download=False)
    saved = {d: 0 for d in range(10)}
    
    for i in range(len(test)):
        label = int(test.targets[i])
        if saved[label] >= per_digit:
            continue
            
        folder = DIGIT_ROOT / str(label)
        folder.mkdir(parents=True, exist_ok=True)
        
        picture = 255 - test.data[i].numpy()     # black ink on white paper
        cv2.imwrite(str(folder / f"{i}.png"), picture)
        saved[label] += 1
        
    print(f"Saved {sum(saved.values())} digit images in {DIGIT_ROOT}/")

def compose_from_files(paths, rng=random):
    """Place the digit images side by side with random size, height and spacing.
    Returns a grayscale image, black ink on white."""
    pieces = []
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        ink = 255 - img if img.mean() > 127 else img  # work with white ink on black
        
        ys, xs = np.nonzero(ink > 30)
        ink = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]  # tight crop
        
        scale = rng.uniform(2.5, 3.5)            # 20px MNIST digit -> ~60px tall
        ink = cv2.resize(ink, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        pieces.append(ink)
        
    height = max(p.shape[0] for p in pieces) + 60
    width = sum(p.shape[1] for p in pieces) + 25 * len(pieces) + 60
    canvas = np.zeros((height, width), dtype=np.uint8)
    
    x = 30
    for p in pieces:
        y = (height - p.shape[0]) // 2 + rng.randint(-8, 8)  # random vertical wobble
        region = canvas[y:y + p.shape[0], x:x + p.shape[1]]
        np.maximum(region, p, out=region)                    # overlap-safe paste
        x += p.shape[1] + rng.randint(4, 25)                 # small gap
        
    canvas = canvas[:, :x + 30]
    return 255 - canvas                          # back to black on white

def compose_number(number, rng=random):
    """Make an image of number (a string like "4721") from random digit files."""
    paths = [rng.choice(list((DIGIT_ROOT / d).glob("*.png"))) for d in number]
    return compose_from_files(paths, rng)

def compose_from_folder(folder):
    """Join every image in folder in file-name order (1.png, 2.png, ...)."""
    paths = sorted(p for p in Path(folder).iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
    return compose_from_files(paths)

def make_test_set(count=1000, min_len=2, max_len=8, seed=0):
    """Save count random numbers as images, with the answers in labels.csv."""
    rng = random.Random(seed)                    # same seed = same test set every time
    TESTSET_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(TESTSET_DIR / "labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "number"])
        for i in range(count):
            length = rng.randint(min_len, max_len)
            number = "".join(rng.choice("0123456789") for _ in range(length))
            name = f"{i:04d}.png"
            cv2.imwrite(str(TESTSET_DIR / name), compose_number(number, rng))
            writer.writerow([name, number])
            
    print(f"Saved {count} images and labels.csv in {TESTSET_DIR}/")

if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "export"
    
    if command == "export":
        export_mnist_digits()
    elif command == "testset":
        make_test_set(int(sys.argv[2]) if len(sys.argv) > 2 else 1000)
    elif command.isdigit():
        cv2.imwrite(f"composed_{command}.png", compose_number(command))
        print(f"Saved composed_{command}.png")
    else:
        print(__doc__)