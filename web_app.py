"""
Task 5 (web version) - the HNRS GUI in the browser, served by Flask on this computer only

    python web_app.py        then the browser opens at http://127.0.0.1:5000

The page is in web/ (index.html, style.css, app.js). This file only answers its requests,
using the team's existing functions:
    pipeline.recognise()   predictor.available_models() / load_model()   train.train()
"""
import ast
import base64
import csv
import operator
import os
import random
import re
import sys
import threading
import time
import webbrowser
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from pipeline import recognise
from predictor import available_models, load_model
from train import MODELS, train

# The folder with checkpoints/, data/ and web/. Inside a PyInstaller .exe that is the
# unpacked bundle (sys._MEIPASS); when run as a script it is the repo's top folder.
BASE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
os.chdir(BASE)

PORT = 5000
MNIST_DIR = Path("data/MNIST/raw")

app = Flask(__name__, static_folder=str(BASE / "web"), static_url_path="/static")
loaded = {}        # model name -> loaded model, so each one loads only once
mnist = None       # MNIST test digits, loaded the first time Compose needs them
job = None         # the current (or last) training run


# ---------- helpers ----------

def decode_image(data_url):
    """'data:image/png;base64,....' from the browser -> BGR numpy image."""
    raw = base64.b64decode(data_url.split(",", 1)[1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def encode_png(image):
    """numpy image -> 'data:image/png;base64,....' the browser can show."""
    return "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", image)[1]).decode()


def get_model(name):
    if name not in loaded:
        loaded[name] = load_model(name)
    return loaded[name]


OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def _evaluate(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_evaluate(node.operand)
    raise ValueError("only numbers and + - x / are allowed")


def calculate(text):
    """'12+7' -> '19' for the arithmetic extension. A plain number or anything invalid -> None.
    Never use eval() here: the text comes from a model's guesses, through a web request."""
    expr = text.replace("×", "*").replace("÷", "/").rstrip("=")
    expr = re.sub(r"(?<!\d)0+(?=\d)", "", expr)           # '07' -> '7' (Python rejects leading zeros)
    if not re.search(r"\d[+\-*/]", expr):
        return None                                        # no operator between numbers: nothing to do
    try:
        value = _evaluate(ast.parse(expr, mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError):
        return None
    return str(int(value)) if float(value).is_integer() else str(round(value, 6))


def recognise_json(image, settings, expected=None):
    """Run the team's pipeline and turn the result into something the browser can draw."""
    name = settings.get("model") or "cnn_aug"
    start = time.perf_counter()
    r = recognise(image, get_model(name), settings.get("threshold", "adaptive"),
                  settings.get("segmentation", "components"))
    ms = round((time.perf_counter() - start) * 1000)
    height, width = image.shape[:2]
    binary = r["binary"]
    if max(binary.shape) > 1200:                           # phone photos: send a smaller copy
        scale = 1200 / max(binary.shape)
        binary = cv2.resize(binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return {
        "number": r["number"],
        "answer": calculate(r["number"]),
        "labels": [str(label) for label in r["labels"]],
        "confidences": [round(c, 4) for c in r["confidences"]],
        "boxes": r["boxes"],
        "width": width, "height": height,                  # so the page can scale the boxes
        "digits": [encode_png((d * 255).astype(np.uint8)) for d in r["digits"]],
        "binary": encode_png(binary),
        "expected": expected,
        "correct": None if expected is None else r["number"] == expected,
        "model": name, "ms": ms,
    }


def load_mnist_test():
    """The 10,000 MNIST test pictures grouped by digit: {0: [28x28 arrays], 1: [...], ...}"""
    images = np.fromfile(MNIST_DIR / "t10k-images-idx3-ubyte", np.uint8, offset=16).reshape(-1, 28, 28)
    labels = np.fromfile(MNIST_DIR / "t10k-labels-idx1-ubyte", np.uint8, offset=8)
    return {d: list(images[labels == d]) for d in range(10)}


def load_digit_folder(folder):
    """A folder with sub-folders 0 ... 9 of single-digit images: {0: [paths], 1: [...], ...}"""
    found = {}
    for d in range(10):
        sub = Path(folder) / str(d)
        files = sub.glob("*") if sub.is_dir() else []
        found[d] = [p for p in files if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")]
    return found


def compose(number, source):
    """One random picture of each digit, side by side -> one image, dark ink on white paper."""
    tiles = []
    for ch in number:
        pick = random.choice(source[int(ch)])
        if isinstance(pick, Path):
            pick = cv2.imdecode(np.fromfile(pick, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if pick.mean() < 127:                                   # white ink on black (MNIST style)
            pick = 255 - pick                                   # -> dark ink on white, like paper
        pick = cv2.resize(pick, (round(pick.shape[1] * 28 / pick.shape[0]), 28))   # same height
        tiles += [pick, np.full((28, 4), 255, np.uint8)]        # digit + a small gap
    strip = np.hstack(tiles[:-1])
    return cv2.copyMakeBorder(strip, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)


# ---------- pages and API ----------

@app.get("/")
def page():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/models")
def api_models():
    names = available_models()
    return jsonify(models=names, default="cnn_aug" if "cnn_aug" in names else (names[0] if names else None),
                   trainable=list(MODELS))


@app.post("/api/recognise")
def api_recognise():
    body = request.get_json()
    if not available_models():
        return jsonify(error="No trained model yet: train one in the Train tab"), 400
    image = decode_image(body["image"])
    if image is None:
        return jsonify(error="That file isn't an image OpenCV can read"), 400
    return jsonify(recognise_json(image, body, body.get("expected")))


@app.post("/api/compose")
def api_compose():
    global mnist
    body = request.get_json()
    number = str(body.get("number", "")).strip()
    if not number.isdigit():
        return jsonify(error="Type digits only, e.g. 4719"), 400
    if body.get("source") == "folder":
        source = load_digit_folder(body.get("folder", ""))
    else:
        if mnist is None:
            mnist = load_mnist_test()
        source = mnist
    missing = sorted({d for d in number if not source[int(d)]})
    if missing:
        return jsonify(error="No pictures found for: " + ", ".join(missing)), 400
    if not available_models():
        return jsonify(error="No trained model yet: train one in the Train tab"), 400
    image = compose(number, source)
    return jsonify(recognise_json(image, body, number) | {"image": encode_png(image)})


@app.post("/api/train")
def api_train():
    global job
    if job is not None and job["thread"].is_alive():
        return jsonify(error="A model is already training"), 409
    body = request.get_json()
    name = body.get("model")
    if name not in MODELS:
        return jsonify(error=f"Unknown model {name}"), 400
    try:
        epochs, batch, lr = int(body["epochs"]), int(body["batch_size"]), float(body["lr"])
    except (KeyError, ValueError):
        return jsonify(error="Epochs, batch size and learning rate must be numbers"), 400
    this = {"name": name, "epochs": epochs, "started": time.time(), "accuracy": None, "error": None}

    def work():                                        # runs in the background thread
        try:
            this["accuracy"] = train(name, epochs=epochs, batch_size=batch, lr=lr)
            loaded.pop(name, None)                     # forget the old weights so the new ones load
        except Exception as error:                     # show the problem instead of failing silently
            this["error"] = str(error)

    this["thread"] = threading.Thread(target=work, daemon=True)
    this["thread"].start()
    job = this
    return jsonify(started=name)


@app.get("/api/train/status")
def api_train_status():
    if job is None:
        return jsonify(running=False, rows=[])
    running = job["thread"].is_alive()                 # check first, so the last epoch is never missed
    log = Path("results") / f"{job['name']}_log.csv"
    rows = []
    if log.exists() and log.stat().st_mtime >= job["started"] - 1:   # this run's log, not an old one
        with open(log, newline="") as f:
            rows = [row for row in csv.DictReader(f) if row.get("seconds")]
    return jsonify(running=running, name=job["name"], epochs=job["epochs"], rows=rows,
                   accuracy=job["accuracy"], error=job["error"])


if __name__ == "__main__":
    url = f"http://127.0.0.1:{PORT}"
    print(f"HNRS is running at {url}  (keep this window open; close it to stop)")
    threading.Timer(1.0, webbrowser.open, args=[url]).start()
    app.run(host="127.0.0.1", port=PORT)               # 127.0.0.1: only this computer can connect