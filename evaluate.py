"""
task 4 - Evaluate every trained model and the whole system.

    python evaluate.py            full evaluation
    python evaluate.py --quick    fast check

Everything is written to results/evaluation/:

    single_digit.csv              accuracy, per-digit accuracy and speed on the 10,000 MNIST test digits
    confusion_<model>.png         which digits each model mixes up
    multi_digit.csv               whole-number accuracy on generated numbers (2 - 6 digits) made of
                                  MNIST test digits, run through the full pipeline
    photos.csv                    every model on every real photo in test_images/
    threshold_comparison.csv      fixed / otsu / adaptive on real photos and badly lit pictures
    segmentation_comparison.csv   components / contours / projection on spaced and touching digits
    extension.csv                 written sums with + - x ÷ ( ) and the sum photos
    model_comparison.png          single-digit vs whole-number accuracy of every model
    training_curves.png           test accuracy after every epoch of every neural network
    method_comparison.png         threshold and segmentation methods side by side

"""
import argparse
import csv
import random
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")                                     # save charts to files, no window
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from models.symbols import FIRST_SIGN_LABEL, SIGNS, MathSymbols
import pipeline
from pipeline import label_text, recognise
from predictor import available_models, load_model, predict_probs
from segmentation.segmentation import segment
from train_classic import load_mnist
from web_app import calculate

OUT = Path("results") / "evaluation"
PHOTOS = Path("test_images")
SEED = 7                                                  # different from every training seed
THRESHOLDS = ["fixed", "otsu", "adaptive"]
SEGMENTATIONS = ["components", "contours", "projection"]


KNOWN_ANSWERS = {
    "test-math-expression01": "(1521×7692)-(121+321)",
    "test-math-expression02": "969+4×3÷5-6+10×105-1×8",
}

# Chart colours: one fixed colour per series, plus quiet ink for text and grid lines
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ---------- helpers ----------

def read_text(result):
    """The model's reading as text, even when the pipeline said "Not a number".
    (Whether the confidence check rejected it is counted separately)"""
    return "".join(label_text(label) for label in result["labels"])


def paper_strip(pictures, gap=4, scale=2):
    """28x28 white-on-black pictures -> one image of dark ink on white paper, side by side,
    like the Compose tab. A negative gap makes neighbours overlap (touching handwriting).
    The image is enlarged 2x: at 28 pixels per digit, thresholding destroys too much detail,
    while a phone photo has digits of 50+ pixels."""
    step = 28 + gap
    paper = np.full((48, 20 + step * (len(pictures) - 1) + 28), 255, np.uint8)
    for i, picture in enumerate(pictures):
        x = 10 + i * step
        paper[10:38, x:x + 28] = np.minimum(paper[10:38, x:x + 28], 255 - picture)   # ink on top of ink
    return cv2.resize(paper, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def uneven_light(image, rng):
    """Make a clean picture look like a badly lit photo: the paper fades from white to
    grey across the page (a shadow), plus camera noise"""
    h, w = image.shape
    angle = rng.uniform(0, 2 * np.pi)
    ramp = np.cos(angle) * np.linspace(-1, 1, w)[None, :] + np.sin(angle) * np.linspace(-1, 1, h)[:, None]
    ramp = (ramp - ramp.min()) / max(np.ptp(ramp), 1e-6)            # 0 .. 1 across the page
    darkest = rng.uniform(0.35, 0.55)                              # how dark the shadow gets
    light = 1 - (1 - darkest) * ramp
    noise = np.random.default_rng(rng.randrange(10 ** 6)).normal(0, 6, image.shape)
    return np.clip(image * light + noise, 0, 255).astype(np.uint8)


def save_csv(name, header, rows):
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  saved {OUT / name}")


def style(ax, title, ylabel):
    """Quiet axes so the data stands out"""
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def photo_answers():
    """{photo path: expected text} for every photo whose answer we know"""
    answers = {}
    for path in sorted(PHOTOS.iterdir()):
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        if path.stem.isdigit():
            answers[path] = path.stem                     # numbers - the file name is the answer
        elif path.stem in KNOWN_ANSWERS:
            answers[path] = KNOWN_ANSWERS[path.stem]
    return answers


def reads_signs(model):
    """True for cnn_math (17 classes), the only model that knows + - × ÷ ( )"""
    probs = predict_probs(model, [np.zeros((28, 28), np.float32)])
    return probs.shape[1] > FIRST_SIGN_LABEL


# ---------- 1. single digits (Task 3) ----------

def evaluate_single_digits(models):
    print("1. Single digits: 10,000 MNIST test digits")
    x, y = load_mnist(train=False)
    x = x.reshape(-1, 28, 28)
    rows, summary = [], {}
    for name, model in models.items():
        start = time.time()
        probs = np.concatenate([predict_probs(model, x[i:i + 1000]) for i in range(0, len(x), 1000)])
        ms = 1000 * (time.time() - start) / len(x)
        guess = probs.argmax(axis=1)
        accuracy = 100 * (guess == y).mean()
        per_digit = [100 * (guess[y == d] == d).mean() for d in range(10)]
        not_digit = int((guess >= 10).sum())              # said "junk" or a sign (cnn_junk / cnn_math)
        rows.append([name, round(accuracy, 2), *[round(a, 2) for a in per_digit], not_digit, round(ms, 3)])
        summary[name] = accuracy
        save_confusion(name, y, guess)
        print(f"  {name:9s} {accuracy:6.2f}%   {ms:.3f} ms per digit")
    save_csv("single_digit.csv",
             ["model", "accuracy"] + [f"digit_{d}" for d in range(10)] + ["called_not_digit", "ms_per_digit"], rows)
    return summary


def save_confusion(name, truth, guess):
    """10 rows (true digit) x 11 columns (guessed digit, or 'other' = junk / sign)"""
    matrix = np.zeros((10, 11), int)
    for t, g in zip(truth, guess):
        matrix[t, min(g, 10)] += 1
    off = matrix.astype(float)
    off[np.arange(10), np.arange(10)] = np.nan            # colour only the mistakes, the diagonal is huge
    fig, ax = plt.subplots(figsize=(6.4, 5))
    ax.imshow(off, cmap="Blues", vmin=0)
    for t in range(10):
        for g in range(11):
            if matrix[t, g]:
                ax.text(g, t, matrix[t, g], ha="center", va="center", fontsize=7,
                        color=INK if t != g else MUTED, fontweight="bold" if t == g else None)
    ax.set_xticks(range(11), [str(d) for d in range(10)] + ["other"])
    ax.set_yticks(range(10))
    ax.set_xlabel("guessed", color=MUTED)
    ax.set_ylabel("true digit", color=MUTED)
    ax.set_title(f"{name}: confusion matrix (10,000 MNIST test digits)", loc="left", fontsize=10)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"confusion_{name}.png", dpi=150)
    plt.close(fig)


# ---------- 2. whole numbers (Task 4) ----------

def make_numbers(count, gap=4):
    """`count` random numbers of 2-6 digits, each built from random MNIST test digits"""
    x, y = load_mnist(train=False)
    x = (x.reshape(-1, 28, 28) * 255).astype(np.uint8)
    by_digit = {d: x[y == d] for d in range(10)}
    rng = random.Random(SEED)
    numbers = []
    for _ in range(count):
        text = "".join(rng.choice("0123456789") for _ in range(rng.randint(2, 6)))
        image = paper_strip([by_digit[int(c)][rng.randrange(len(by_digit[int(c)]))] for c in text], gap)
        numbers.append((text, image))
    return numbers


def score(cases, model, threshold="adaptive", segmentation="components"):
    """Run the full pipeline on (expected text, image) cases. Returns percentages:
    exact (whole text right), count (right number of marks = segmentation right),
    marks (marks right, when the count is right), rejected (pipeline said "Not a number")"""
    exact = count_ok = marks_ok = marks_total = rejected = 0
    for expected, image in cases:
        result = recognise(image, model, threshold, segmentation)
        text = read_text(result)
        exact += text == expected
        rejected += not result["valid"]
        if len(text) == len(expected):
            count_ok += 1
            marks_ok += sum(a == b for a, b in zip(text, expected))
            marks_total += len(expected)
    n = max(len(cases), 1)
    return {"exact": 100 * exact / n, "count": 100 * count_ok / n,
            "marks": 100 * marks_ok / max(marks_total, 1), "rejected": 100 * rejected / n}


def evaluate_numbers(models, numbers):
    print(f"2. Whole numbers: {len(numbers)} generated numbers through the full pipeline")
    rows, summary = [], {}
    for name, model in models.items():
        s = score(numbers, model)
        rows.append([name, round(s["exact"], 2), round(s["count"], 2), round(s["marks"], 2), round(s["rejected"], 2)])
        summary[name] = s["exact"]
        print(f"  {name:9s} {s['exact']:6.2f}% numbers right   ({s['rejected']:.1f}% rejected as not sure)")
    save_csv("multi_digit.csv",
             ["model", "number_accuracy", "segmentation_count_correct", "digit_accuracy_when_count_correct",
              "rejected_by_confidence_check"], rows)
    return summary


# ---------- 3. real photos ----------

def evaluate_photos(models):
    print("3. Real photos in test_images/")
    rows = []
    answers = photo_answers()
    for name, model in models.items():
        right = tried = 0
        for path, expected in answers.items():
            if not expected.isdigit() and not reads_signs(model):
                continue                                  # only cnn_math can read sums
            result = recognise(cv2.imread(str(path)), model)
            text = read_text(result)
            rows.append([path.name, expected, name, text, text == expected, result["valid"]])
            right += text == expected
            tried += 1
        print(f"  {name:9s} {right}/{tried} photos right")
    save_csv("photos.csv", ["photo", "expected", "model", "read", "correct", "accepted"], rows)


# ---------- 4. preprocessing and segmentation methods (Tasks 1 and 2) ----------

def evaluate_methods(model_name, model, numbers):
    """Clean photos are easy for every method, so each comparison also uses a hard case:
    uneven lighting for the thresholds, touching digits for the segmentation"""
    print(f"4. Threshold and segmentation methods (model {model_name})")
    photos = [(expected, cv2.imread(str(path))) for path, expected in photo_answers().items() if expected.isdigit()]
    rng = random.Random(SEED)
    shadowed = [(text, uneven_light(image, rng)) for text, image in numbers]
    touching = make_numbers(len(numbers), gap=-10)       # neighbours overlap by 10 px

    threshold_rows = []
    for method in THRESHOLDS:
        on_photos = score(photos, model, threshold=method)
        on_shadow = score(shadowed, model, threshold=method)
        threshold_rows.append([method, round(on_photos["exact"], 1), round(on_shadow["count"], 1),
                               round(on_shadow["exact"], 1)])
        print(f"  threshold {method:10s} photos {on_photos['exact']:5.1f}%   uneven light {on_shadow['exact']:5.1f}%")
    save_csv("threshold_comparison.csv",
             ["threshold", "photos_number_accuracy", "uneven_light_count_correct",
              "uneven_light_number_accuracy"], threshold_rows)

    segmentation_rows = []
    for method in SEGMENTATIONS:
        on_photos = score(photos, model, segmentation=method)
        on_spaced = score(numbers, model, segmentation=method)
        on_touching = score(touching, model, segmentation=method)
        segmentation_rows.append([method, round(on_photos["exact"], 1), round(on_spaced["exact"], 1),
                                  round(on_touching["count"], 1), round(on_touching["exact"], 1)])
        print(f"  segmentation {method:10s} photos {on_photos['exact']:5.1f}%   spaced {on_spaced['exact']:5.1f}%"
              f"   touching {on_touching['exact']:5.1f}%")
    # What is the "split touching digits" rule worth? Switch it off and measure again
    pipeline.segment = lambda binary, method: segment(binary, method, split_touching=False)
    try:
        on_photos = score(photos, model)
        on_spaced = score(numbers, model)
        on_touching = score(touching, model)
    finally:
        pipeline.segment = segment                        # always switch it back on
    segmentation_rows.append(["components, no splitting", round(on_photos["exact"], 1), round(on_spaced["exact"], 1),
                              round(on_touching["count"], 1), round(on_touching["exact"], 1)])
    print(f"  components without splitting   touching {on_touching['exact']:5.1f}%")
    save_csv("segmentation_comparison.csv",
             ["segmentation", "photos_number_accuracy", "spaced_number_accuracy",
              "touching_count_correct", "touching_number_accuracy"], segmentation_rows)
    return threshold_rows, segmentation_rows


# ---------- 5. extension: sums ----------

def make_sums(count):
    """Random sums like 12+7, (3+4)×2 or 8÷2-1, drawn with MNIST test digits and
    generated sign pictures (a seed the sign model was never trained on)"""
    x, y = load_mnist(train=False)
    x = (x.reshape(-1, 28, 28) * 255).astype(np.uint8)
    by_digit = {d: x[y == d] for d in range(10)}
    signs = MathSymbols(200, seed=SEED)
    sign_pictures = {}
    for image, label in zip(signs.images.squeeze(1).numpy(), signs.labels):
        sign_pictures.setdefault(SIGNS[label - FIRST_SIGN_LABEL], []).append((image * 255).astype(np.uint8))

    rng = random.Random(SEED)
    number = lambda: str(rng.randint(1, 99))
    op = lambda: rng.choice(["+", "-", "×", "÷"])
    templates = [lambda: number() + op() + number(),
                 lambda: number() + op() + number() + op() + number(),
                 lambda: "(" + number() + op() + number() + ")" + op() + number(),
                 lambda: number() + op() + "(" + number() + op() + number() + ")"]
    sums = []
    for _ in range(count):
        text = rng.choice(templates)()
        pictures = [by_digit[int(c)][rng.randrange(len(by_digit[int(c)]))] if c.isdigit()
                    else rng.choice(sign_pictures[c]) for c in text]
        sums.append((text, paper_strip(pictures)))
    return sums


def evaluate_sums(models, count):
    math_models = {name: model for name, model in models.items() if reads_signs(model)}
    if not math_models:
        print("5. Extension: skipped (no cnn_math checkpoint)")
        return
    print(f"5. Extension: {count} generated sums + the sum photos")
    sums = make_sums(count)
    photos = [(expected, cv2.imread(str(path))) for path, expected in photo_answers().items()
              if not expected.isdigit()]
    rows = []
    for name, model in math_models.items():
        for label, cases in (("generated sums", sums), ("sum photos", photos)):
            read_right = answer_right = 0
            for expected, image in cases:
                text = read_text(recognise(image, model))
                read_right += text == expected
                answer_right += calculate(text) is not None and calculate(text) == calculate(expected)
            n = max(len(cases), 1)
            rows.append([name, label, len(cases), round(100 * read_right / n, 1), round(100 * answer_right / n, 1)])
            print(f"  {name} {label}: {100 * read_right / n:.1f}% read exactly, {100 * answer_right / n:.1f}% correct answer")
    save_csv("extension.csv", ["model", "test_set", "cases", "read_exactly", "answer_correct"], rows)


# ---------- 6. charts ----------

def chart_models(single, numbers):
    names = list(single)
    pos = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 0.9), 4))
    ax.bar(pos - 0.2, [single[n] for n in names], 0.38, color=BLUE, label="single digits (MNIST test)")
    ax.bar(pos + 0.2, [numbers[n] for n in names], 0.38, color=ORANGE, label="whole numbers (2-6 digits)")
    for i, n in enumerate(names):
        ax.text(i - 0.2, single[n] + 0.5, f"{single[n]:.1f}", ha="center", fontsize=7, color=MUTED)
        ax.text(i + 0.2, numbers[n] + 0.5, f"{numbers[n]:.1f}", ha="center", fontsize=7, color=MUTED)
    ax.set_xticks(pos, names)
    low = min(min(single.values()), min(numbers.values()))
    ax.set_ylim(max(0, low - 10), 101)
    style(ax, "Accuracy per model: one digit vs a whole number", "accuracy (%)")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png", dpi=150)
    plt.close(fig)


def chart_training():
    logs = sorted(Path("results").glob("*_log.csv"))
    if not logs:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for colour, path in zip(SERIES, logs):
        with open(path) as f:
            rows = list(csv.DictReader(f))
        epochs = [int(r["epoch"]) for r in rows]
        accuracy = [float(r["test_acc"]) for r in rows]
        name = path.stem.replace("_log", "")
        ax.plot(epochs, accuracy, color=colour, linewidth=2, marker="o", markersize=4, label=name)
    ax.set_xlabel("epoch", color=MUTED)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))   # whole epochs only
    style(ax, "Test accuracy after each training epoch", "test accuracy (%)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=150)
    plt.close(fig)


def chart_methods(threshold_rows, segmentation_rows):
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [3, 4]})
    names = [r[0] for r in threshold_rows]
    pos = np.arange(len(names))
    left.bar(pos - 0.2, [r[1] for r in threshold_rows], 0.38, color=BLUE, label="real photos")
    left.bar(pos + 0.2, [r[3] for r in threshold_rows], 0.38, color=ORANGE, label="uneven lighting")
    left.set_xticks(pos, names)
    style(left, "Threshold methods", "whole number right (%)")
    left.set_ylim(0, 122)
    left.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")

    names = [r[0] for r in segmentation_rows]
    pos = np.arange(len(names))
    right.bar(pos - 0.27, [r[1] for r in segmentation_rows], 0.26, color=BLUE, label="real photos")
    right.bar(pos, [r[2] for r in segmentation_rows], 0.26, color=AQUA, label="spaced digits")
    right.bar(pos + 0.27, [r[4] for r in segmentation_rows], 0.26, color=ORANGE, label="touching digits")
    right.set_xticks(pos, [n.replace(", ", "\n") for n in names], fontsize=8)
    style(right, "Segmentation methods", "whole number right (%)")
    right.set_ylim(0, 122)
    right.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "method_comparison.png", dpi=150)
    plt.close(fig)


# ---------- main ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the HNRS models and system")
    parser.add_argument("--quick", action="store_true", help="fewer generated numbers and sums")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    names = available_models()
    if not names:
        raise SystemExit("No trained models in checkpoints/ - train some first (python train.py cnn_aug)")
    models = {name: load_model(name) for name in names}
    count = 200 if args.quick else 1000

    single = evaluate_single_digits(models)
    numbers = make_numbers(count)
    whole = evaluate_numbers(models, numbers)
    evaluate_photos(models)
    best = next((n for n in ("cnn_junk", "cnn_aug", "cnn_math") if n in models), names[0])
    threshold_rows, segmentation_rows = evaluate_methods(best, models[best], numbers[:300])
    evaluate_sums(models, count // 2)

    chart_models(single, whole)
    chart_training()
    chart_methods(threshold_rows, segmentation_rows)
    print(f"Done. Tables and charts are in {OUT}/")