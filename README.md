# HNRS – Handwritten Number Recognition System

COS30018 Intelligent Systems | Swinburne University of Technology | Project Option B | Team Hy-Group-1

HNRS reads **handwritten multi-digit numbers** from a photo or a drawing. It also reads **simple handwritten sums** (digits with `+ − × ÷ ( )`) and calculates the answer. Everything runs in a local web app: you can draw, upload a photo, build a number from single-digit images, pick a model, change the preprocessing settings, and train models.

```
photo / drawing ---> 1. Preprocessing ---> 2. Segmentation ---> 3. Recognition ---> 4. Number / sum
                    grayscale, threshold,  one box per mark,    CNN gives a label   join in reading order,
                    fill, clean            28x28 MNIST-style    + confidence        check, calculate
```

---

## Contents

- [Features](#features)
- [Setup](#setup)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [The models](#the-models)
- [Training data](#training-data)
- [Using the web app](#using-the-web-app)
- [Command-line tools](#command-line-tools)
- [Project structure](#project-structure)
- [Results](#results)
- [Known limitations](#known-limitations)
- [Team and acknowledgements](#team-and-acknowledgements)

---

## Features

- **Multi-digit numbers** from phone photos, scans or a drawing canvas (for example `2026`, `4719`).
- **Several lines** of writing, read top to bottom with a space between lines.
- **Eight digit models to compare**: five neural networks (MLP, simple CNN, CNN with batch normalisation, CNN with data augmentation, a small Vision Transformer) and three classic machine-learning models (SVM, k-nearest neighbours, Random Forest).
- **Automatic evaluation**: one script tests every model on single digits, whole numbers, real photos and sums, compares the preprocessing and segmentation methods, and saves tables and charts.
- **"Not a number" detection**: scribbles and random marks are rejected instead of being forced into a digit.
- **Maths extension**: recognises `+ − × ÷ ( )` and calculates the result, for example `(3+4)×2 = 14`.
- **Image acquisition both ways** (as the brief asks): load an image file, or build a number automatically from a folder of single-digit images (or from MNIST).
- **GUI with settings**: choose the model, the threshold method and the segmentation method; see the boxes, the 28×28 pictures the model saw, and a confidence for every mark; train models with your own epochs, batch size and learning rate.

---

## Setup

You need Python 3.12 or newer. A GPU is optional: training uses an NVIDIA GPU when PyTorch can see one, and the CPU otherwise.

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
# With an NVIDIA GPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# Without a GPU:
pip install torch torchvision
pip install -r requirements.txt
```

If PowerShell refuses to run `Activate.ps1`, run this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**macOS / Linux:** the same steps, but activate with `source .venv/bin/activate`.

MNIST is already in `data/MNIST/raw`. If it's missing, torchvision downloads it the first time you train.

---

## Quick start

Trained weights (`checkpoints/`) are **not** stored in git; the team shares them on OneDrive. To make your own, train the models you want. Each one takes 1–2 minutes on a GPU.

```powershell
python train_classic.py      # SVM, k-NN and Random Forest (about 1 minute)
python train.py cnn_aug      # best plain digit model
python train.py cnn_junk     # digits + "not a digit"
python train.py cnn_math     # digits + "not a digit" + maths signs (extension)
```

Then start the app:

```powershell
python web_app.py
```

It opens **http://127.0.0.1:5000** in your browser. Press `Ctrl+C` in the terminal to stop it.

---

## How it works

### Task 1 – Preprocessing (`preprocessing/preprocessing.py`)

| Step | What it does | Why |
|---|---|---|
| Grayscale + Gaussian blur | Colour → gray, smooths camera noise | Thresholding works on one channel and dislikes noise |
| **Thresholding** (choose one) | `fixed` (127), `otsu` (automatic single threshold), `adaptive` (a separate threshold for each 31×31 area) | Adaptive is the default because phone photos have shadows and uneven light |
| Fill hollow strokes | Fills enclosed "holes" whose original colour is ink-dark | Adaptive thresholding keeps only the edges of very thick marker strokes; real holes (inside 0, 6, 8, 9) are paper-coloured and stay open |
| Remove specks | Deletes tiny blobs; the size limit grows with the image | Dust and paper texture |
| Light ink on dark? | Flips the image when needed | Ink is always white on black from here on, like MNIST |

`make_mnist_like()` turns each cut-out mark into an MNIST-style picture:
1. Thicken thin strokes (to about 1/8 of the height).
2. Fit the longer side into 20 px, keeping the shape.
3. Pad to 28×28.
4. Shift so the centre of mass sits in the middle.

### Task 2 – Segmentation (`segmentation/segmentation.py`)

Three techniques, chosen in the app:

- **`components`** (default): each connected blob of ink is one mark.
- **`contours`**: outer contours with OpenCV.
- **`projection`**: splits at empty columns (vertical projection).

Clean-up rules after that:

- Drop boxes touching the picture's edge (the table, the edge of the paper, shadows).
- Drop noise much smaller than the biggest mark, and dots much smaller than a normal digit. The longer side is used, so a flat minus sign survives.
- **Merge** boxes that sit above each other, such as a broken "5" or the dots of a ÷, but only if the result is still about one character tall.
- **Split** touching digits: a wide box is cut where the ink is thinnest, but only if there's a clear "neck".
- **Group into lines**: going left to right, each mark joins the line whose most recent mark it overlaps in height. Following the line mark by mark keeps sloping writing together, then lines are read top to bottom.
- Cut each box out keeping only the blobs fully inside it, so a neighbour's edge doesn't sneak in.

### Task 3 – Recognition (`predictor.py`, `models/`)

Each 28×28 picture goes through the chosen model. The model returns a probability for every class. The label is the most likely class, and the **confidence** is its probability. See [The models](#the-models).

### Task 4 – Building the number or the sum (`pipeline.py`, `web_app.py`)

1. **Join the labels in reading order**, with a space where a new line starts.
2. **Fix look-alikes (cnn_math only).** A "1", "(" and ")" can all be one tall stroke, so the maths decides:
   - In a plain number (no operator), a bracket is always read as 1.
   - In a sum, if the reading isn't valid maths, the 1 / ( / ) choices for the least certain marks are tried. The most likely choice that gives a valid sum is kept.
3. **Validity check.** The result is **"Not a number"** if any mark is junk (✗), or if any mark's confidence is below **80%**. In a reading that is a valid sum, the limit is **50%**, because valid maths is extra evidence that the reading is right.
4. **Calculate (extension).** If there's an operator, `calculate()` works out the answer with a small safe parser. It never uses `eval()`, because the text comes from model guesses.

---

## The models

The neural networks are trained with Adam (learning rate 0.001), cross-entropy loss, batch size 128, 5 epochs and seed 42. These are the defaults in `train.py`, and you can change them in the Train tab. The classic models are trained with scikit-learn by `train_classic.py` and saved as `.joblib` files; the app loads them like any other model.

| Name | Network | Classes | Parameters | Trained on | Purpose |
|---|---|---|---|---|---|
| `mlp` | Fully connected: 784 → 256 → 256 → 10 | 10 | 269k | MNIST | Baseline with no idea of image shape |
| `cnn` | LeNet-style: 2 conv blocks + 2 dense layers | 10 | 422k | MNIST | Basic CNN |
| `cnn_bn` | Deeper CNN: 4 conv layers with batch normalisation, dropout 0.3 | 10 | 871k | MNIST | Best accuracy on clean MNIST |
| `cnn_aug` | Same network as `cnn_bn` | 10 | 871k | MNIST + random rotate / shift / scale | Copes better with real, messy handwriting |
| `vit` | Tiny Vision Transformer: 16 patches of 7×7, 4 encoder layers, 4 heads, CLS token | 10 | 139k | MNIST | Week 8 transformer approach, for comparison |
| `cnn_junk` | `cnn_aug` network with 11 outputs | 0–9 + **not a digit** | 871k | MNIST (also in pipeline style) + junk | Rejects scribbles: says "Not a number" |
| `cnn_math` | `cnn_aug` network with 17 outputs | 0–9, not a digit, `+ − × ÷ ( )` | 873k | MNIST (also in pipeline style) + junk without dashes + maths signs | Extension: reads and calculates sums |
| `svm` | Support vector machine (RBF kernel) after PCA to 50 features, with calibrated probabilities | 10 | – | 20,000 MNIST digits (an SVM gets very slow with more) | Strongest classic ML baseline |
| `knn` | k-nearest neighbours (k = 3) after PCA to 50 features | 10 | – | MNIST | Classic baseline: "looks most like these training digits" |
| `rf` | Random Forest, 100 trees | 10 | – | MNIST | Classic baseline: many voting decision trees |

**Class numbers:**

| Label | Meaning |
|---|---|
| 0–9 | Digits |
| 10 | Not a digit |
| 11–16 | `+`, `−`, `×`, `÷`, `(`, `)` |

**Which model to use:**
- **Numbers:** `cnn_junk`. It's as accurate as `cnn_aug` on digits and rejects junk.
- **Sums:** `cnn_math`.
- **Comparing techniques in the report:** all of them (`evaluate.py` does this for you)

---

## Training data

| Data | Size | Where it comes from |
|---|---|---|
| MNIST | 60,000 train / 10,000 test | `data/MNIST/raw` (torchvision) |
| Augmented MNIST | Same pictures, changed every epoch | `cnn_aug`, `cnn_junk`, `cnn_math`: rotate ±10°, shift 10%, scale 0.85–1.15 |
| Pipeline-style MNIST | 60,000 / 10,000 | MNIST digits sent through `make_mnist_like()`, so the model knows what real cut-out digits look like. Without these, the model learned the shortcut "processed = junk" and rejected real photo digits. |
| Junk ("not a digit") | 12,000 / 2,000 | `models/junk.py`: random scribbles, smooth loops, cursive curls, flat dashes, filled blobs, crossed lines, coils. `cnn_math` leaves out the dashes, because for it a flat line is a minus sign. |
| Maths signs | 48,000 / 8,000 | `models/symbols.py`: hand-drawn-style `+ − × ÷ ( )` with random size, slant, wobble, pen thickness (thin pen to thick marker) and blobs where the pen lands. Brackets get twice as many examples, because they look like a "1". |


---

## Using the web app

| Area | What you can do |
|---|---|
| **Settings bar** | Model, threshold method (`adaptive` / `otsu` / `fixed`), segmentation method (`components` / `contours` / `projection`), show the preprocessed black-and-white image |
| **Draw** | Write with the mouse or a pen. It recognises when you lift the pen, or press Recognise. You can change the pen size. |
| **Photo** | Choose or drop a photo. If the file is named after the number (for example `2026.jpg`), the result says **Correct** or **Wrong**. |
| **Compose** | Type a number, and the app builds an image from random single digits: from the MNIST test set, or from a folder with sub-folders `0`–`9` (default `data/digit_folders`). Press Random for a random number. |
| **Train** | Pick a model, epochs, batch size and learning rate. Watch the loss and accuracy per epoch. The app asks before replacing an existing checkpoint. |
| **Result** | The number (and `= answer` for a sum), a box and label on every mark, the 28×28 picture the model saw for each mark, and its confidence (red = under 80%). |

**Tips for best results:**
- Leave a small gap between characters.
- Make brackets clearly curved.
- For ÷, draw a dash with one dot above and one below.
- Use good, even light for photos.

---

## Command-line tools

Run these from the project root, with the virtual environment active.

```powershell
python train.py cnn_math                         # train one model -> checkpoints/, results/
python train_classic.py                          # train SVM, k-NN and Random Forest
python evaluate.py                               # test everything -> results/evaluation/ (a few minutes)
python evaluate.py --quick                       # smaller, faster version
python pipeline.py test_images/2026.jpg cnn_junk # recognise one photo
python pipeline.py test_images                   # every model on every photo (file name = answer)
python preprocessing/preprocessing.py test_images/2026.jpg   # show the preprocessing steps
python segmentation/segmentation.py test_images/2026.jpg     # show the boxes and 28x28 digits
python -m models.junk                            # preview junk pictures
python -m models.symbols                         # preview maths-sign pictures
python export_datasets.py                        # junk + sign datasets in MNIST format
```

`my_processing.py` is a standalone preprocessing experiment, built with PIL. It compares binarisation and denoising methods side by side and isn't used by the app.

---

## Project structure

```
├── web_app.py                  Flask server: the GUI's API, compose, training in the background
├── web/                        The page: index.html, app.js, style.css
├── pipeline.py                 recognise(): image in -> number/sum out (tasks 1-4)
├── predictor.py                Load models and get probabilities for 28x28 pictures
├── train.py                    Train, test and save any neural network; the MODELS list
├── train_classic.py            Train the classic models: SVM, k-NN, Random Forest
├── evaluate.py                 Test every model and the whole system; tables and charts for the report
├── preprocessing/
│   └── preprocessing.py        Grayscale, thresholds, hollow-stroke fill, specks, make_mnist_like()
├── segmentation/
│   └── segmentation.py         components / contours / projection + clean-up rules + reading order
├── models/
│   ├── cnn.py                  SimpleCNN, BetterCNN, data loaders (MNIST + extra data)
│   ├── mlp.py                  MLP
│   ├── vit.py                  TinyViT
│   ├── junk.py                 "Not a digit" generator + pipeline-style MNIST
│   └── symbols.py              Maths-sign generator
├── export_datasets.py          Save the junk and sign datasets as MNIST IDX files
├── my_processing.py            Standalone preprocessing experiments
├── test_images/                Real photos named after their answer (for testing)
├── results/                    <model>_log.csv (loss and accuracy per epoch), classic_results.csv,
│   └── evaluation/             everything evaluate.py measures: CSV tables and PNG charts
├── data/MNIST/raw/             MNIST in its original IDX format
├── checkpoints/                Trained weights (not in git, shared on OneDrive)
└── requirements.txt
```

---

## Results

`python evaluate.py` measures everything below and more, and saves it in `results/evaluation/`:

| File | What it shows |
|---|---|
| `single_digit.csv`, `confusion_<model>.png` | Every model on the 10,000 MNIST test digits: accuracy, per-digit accuracy, speed, and which digits get mixed up |
| `multi_digit.csv`, `model_comparison.png` | Whole numbers (2–6 digits, made of MNIST test digits) through the full pipeline. One wrong digit makes the whole number wrong, so this is lower than single-digit accuracy |
| `photos.csv` | Every model on every photo in `test_images/` |
| `threshold_comparison.csv` | fixed / Otsu / adaptive on the photos and on badly lit pictures (a shadow across the page) |
| `segmentation_comparison.csv` | components / contours / projection, and components without the "split touching digits" rule, on spaced and touching digits |
| `extension.csv` | Written sums and the sum photos: read exactly? correct answer? |
| `training_curves.png`, `method_comparison.png` | Charts for the report |

Final test accuracy after 5 epochs, from `results/<model>_log.csv`:

| Model | Test accuracy | Notes |
|---|---|---|
| `mlp` | 97.76% | MNIST test set |
| `cnn` | 98.86% | MNIST test set |
| `cnn_bn` | **99.26%** | MNIST test set |
| `cnn_aug` | 99.12% | MNIST test set |
| `vit` | 97.72% | MNIST test set |
| `cnn_junk` | 99.07% | MNIST + pipeline-style MNIST + junk test pictures |
| `cnn_math` | 99.17% | MNIST + pipeline-style MNIST + junk + maths-sign test pictures |
| `svm`, `knn`, `rf` | see `results/classic_results.csv` | MNIST test set |

**Real photos:** with `adaptive` thresholding, `cnn_aug`, `cnn_junk` and `cnn_math` read all the number photos in `test_images/` correctly. Run `python pipeline.py test_images` to check every model.

**Threshold comparison:** all three methods handle the clean photos, but with uneven lighting `adaptive` stays far ahead of `fixed` and `otsu` (see `threshold_comparison.csv`). That's why `adaptive` is the default.

**Extension:** `test-math-expression01.jpeg` reads `(1521×7692)-(121+321) = 11699090`, and the sloping `test-math-expression02.jpeg` reads `969+4×3÷5-6+10×105-1×8 = 2007.4`.

The accuracy for `cnn_junk` and `cnn_math` is measured on a bigger test set that includes written pictures, so it isn't directly comparable with the MNIST-only rows.

---

## Known limitations

- **Touching characters** that have no thin "neck" between them can't be split. Leave small gaps.
- **"1" versus brackets** is ambiguous by nature. The look-alike rule helps in sums, but very straight brackets may still be read as 1.
- **Maths signs and junk are synthetic.** Real handwritten sign samples would improve `cnn_math`.
- **Some scribbles really are digit-shaped** (a loopy "3"), so no model can reject every scribble. `cnn_junk` rejects more scribbles than `cnn_math`.
- **The 80% confidence cutoff** occasionally rejects a messy but real digit (about 1% of MNIST digits).
- **Very large pen sizes** in the Draw tab make neighbouring characters touch.

---

## Team and acknowledgements

**Team Hy-Group-1:** Dasun, Krishmika, Shehan, Randhi.

- MNIST dataset: Yann LeCun, Corinna Cortes and Christopher J.C. Burges.
- Built with PyTorch, torchvision, OpenCV, NumPy, scikit-learn, Matplotlib and Flask.