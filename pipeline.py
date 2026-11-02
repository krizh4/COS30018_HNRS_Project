"""
Task 4 - the whole system in one function: image in, number out


    result = recognise(image, model)
    result["number"]       "2026"   (a second line of writing comes after a space: "64 710")
    result["boxes"]        [(x, y, w, h), ...] one per digit, in reading order
    result["confidences"]  [0.99, 0.97, ...]  how sure the model was about each digit
    result["digits"]       the 28x28 pictures the model actually saw
    result["binary"]       the preprocessed black/white image

try it:

    python pipeline.py test_images/2026.jpg          one photo, default model cnn_aug
    python pipeline.py test_images/2026.jpg cnn      one photo, another model
    python pipeline.py test_images                   every photo in the folder, every model
"""
import ast
import re
import sys
from itertools import product
from pathlib import Path

import cv2
import numpy as np

from predictor import available_models, load_model, predict_probs
from preprocessing.preprocessing import preprocess
from segmentation.segmentation import segment


NOT_A_DIGIT = 10      # the 11th class of cnn_junk: "this mark is not a digit"
MIN_CONFIDENCE = 0.8  # below this the model is guessing, so don't trust the number
MIN_CONFIDENCE_IN_SUM = 0.5  # a reading that is valid maths is extra evidence, so allow less sure marks
SIGNS = {11: "+", 12: "-", 13: "×", 14: "÷", 15: "(", 16: ")"}   # the extra classes of cnn_math


def label_text(label):
    """What to show for one model answer: a digit, a maths sign, or ✗ for junk"""
    if label == NOT_A_DIGIT:
        return "✗"
    return SIGNS.get(label, str(label))


LOOK_ALIKES = [1, 15, 16]          # "1", "(" and ")" are all one tall stroke
OPERATORS = [11, 12, 13, 14]       # + - × ÷


def is_sum(text):
    """True if text is a complete sum that can be calculated, like (3+4)×2"""
    text = re.sub(r"(?<!\d)0+(?=\d)", "", text.replace("×", "*").replace("÷", "/"))   # '07' -> '7'
    try:
        tree = ast.parse(text, mode="eval").body
    except SyntaxError:
        return False
    allowed = (ast.BinOp, ast.UnaryOp, ast.Constant, ast.operator, ast.unaryop)
    return all(isinstance(node, allowed) for node in ast.walk(tree))   # no "1(2)" function calls


def fix_look_alikes(labels, confidences, probs):
    """ the model often can't tell a 1 from a bracket, so let the maths decide
    In a plain number a bracket must be a 1. In a sum, try the 1 / ( / ) choices for the
    marks the model is unsure about, and keep the most likely one that can be calculated"""

    labels, confidences = list(labels), list(confidences)
    if len(probs) == 0 or probs.shape[1] <= max(LOOK_ALIKES):
        return labels, confidences   
                          # model has no brackets (ex: cnn_junk)
    look_alikes = [i for i, label in enumerate(labels) if label in LOOK_ALIKES]

    for i in look_alikes:                                  # sure it is one of the three look-alikes
        confidences[i] = float(probs[i][LOOK_ALIKES].sum())

    if not any(label in OPERATORS for label in labels):    # a plain number: no brackets
        for i in look_alikes:
            labels[i] = 1
        return labels, confidences
    
    text = lambda trial: "".join(label_text(t) for t in trial)
    if is_sum(text(labels)):
        return labels, confidences                         # already makes sense: keep the model's answer
    # only re-think the marks the model was unsure about (a long sum can have 10+ ones and
    # brackets, and trying 3 choices for every one of them would take far too long)

    unsure = sorted(look_alikes, key=lambda i: probs[i][labels[i]])[:8]
    best, best_score = None, -1.0
    for choice in product(LOOK_ALIKES, repeat=len(unsure)):
        trial = labels[:]
        for i, label in zip(unsure, choice):
            trial[i] = label
        score = float(np.prod([probs[i][label] for i, label in zip(unsure, choice)]))
        
        if score > best_score and is_sum(text(trial)):
            best, best_score = trial, score
    return (best if best is not None else labels), confidences


def recognise(image, model, threshold="adaptive", seg_method="components"):
    binary = preprocess(image, threshold)                  # task 1
    parts = segment(binary, seg_method)                    #  task 2
    boxes = [box for box, _, _ in parts]
    digits = [digit for _, digit, _ in parts]
    lines = [line for _, _, line in parts]
    probs = predict_probs(model, digits)                   # task 3
    labels, confidences = probs.argmax(axis=1).tolist(), probs.max(axis=1).tolist()
    labels, confidences = fix_look_alikes(labels, confidences, probs)
    number = ""                                            # join in reading order,
    
    for i, label in enumerate(labels):                     # with a space where a new line starts
        if i > 0 and lines[i] != lines[i - 1]:
            number += " "
        number += label_text(label)
    in_sum = any(label in OPERATORS for label in labels) and is_sum(number.replace(" ", ""))
    needed = MIN_CONFIDENCE_IN_SUM if in_sum else MIN_CONFIDENCE
    valid = NOT_A_DIGIT not in labels and min(confidences, default=1.0) >= needed
    if not valid:
        number = "Not a number"
    return {"number": number, "valid": valid, "boxes": boxes, "labels": labels,
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