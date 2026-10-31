"""
Task 4 - the whole system in one function: image in, number out


    result = recognise(image, model)
    result["number"]       "2026"
    result["boxes"]        [(x, y, w, h), ...] one per digit, left to right
    result["confidences"]  [0.99, 0.97, ...]  how sure the model was about each digit
    result["digits"]       the 28x28 pictures the model actually saw
    result["binary"]       the preprocessed black/white image

try it:

    python pipeline.py test_images/2026.jpg          one photo, default model cnn_aug
    python pipeline.py test_images/2026.jpg cnn      one photo, another model
    python pipeline.py test_images                   every photo in the folder, every model
"""
import sys
from pathlib import Path

import cv2

from predictor import available_models, load_model, predict_digits
from preprocessing.preprocessing import preprocess
from segmentation.segmentation import segment


def recognise(image, model, threshold="adaptive", seg_method="components"):
    binary = preprocess(image, threshold)                  # task 1
    parts = segment(binary, seg_method)                    #  task 2
    boxes = [box for box, _ in parts]
    digits = [digit for _, digit in parts]
    labels, confidences = predict_digits(model, digits)    # task 3
    number = "".join(str(d) for d in labels)               # join left to right
    return {"number": number, "boxes": boxes, "labels": labels,
            "confidences": confidences, "digits": digits, "binary": binary}


def test_folder(folder):
    """run every model on every photo. File names are the true numbers, ex: 2026.jpg"""
    photos = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for name in available_models():
        model = load_model(name)
        right = 0
        answers = []
        for photo in photos:
            guess = recognise(cv2.imread(str(photo)), model)["number"]
            right += guess == photo.stem
            answers.append(f"{photo.stem}->{guess}")
        print(f"{name:10s} {right}/{len(photos)} right   " + "  ".join(answers))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "test_images/2026.jpg"
    if Path(path).is_dir():
        test_folder(path)
        sys.exit()
    model_name = sys.argv[2] if len(sys.argv) > 2 else "cnn_aug"
    image = cv2.imread(path)
    if image is None:
        sys.exit(f"Could not read {path}")
    result = recognise(image, load_model(model_name))
    print("Number:", result["number"])
    print("Confidence per digit:", [round(c, 2) for c in result["confidences"]])