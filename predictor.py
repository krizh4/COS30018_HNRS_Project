"""
One way to load and use ANY trained model - pytorch (.pt) or scikit-learn (.joblib)
The evaluation scripts, the pipeline and the GUI all use these three functions


    available_models()            names of every trained model in checkpoints/
    load_model(name)              load one of them
    predict_digits(model, digits) digits = list of 28x28 arrays (0..1) ----> (labels, confidences)
"""
from pathlib import Path

import joblib
import numpy as np
import torch

from train import MODELS, load_trained_model

CHECKPOINT_DIR = Path("checkpoints")


def available_models():
    """Trained models the GUI can offer (experiment runs starting with hp_ are skipped)"""
    names = [p.stem for p in sorted(CHECKPOINT_DIR.glob("*.pt")) if p.stem in MODELS]
    names += [p.stem for p in sorted(CHECKPOINT_DIR.glob("*.joblib"))]
    return names


def load_model(name):
    joblib_file = CHECKPOINT_DIR / f"{name}.joblib"
    if joblib_file.exists():
        return joblib.load(joblib_file)          # a scikit-learn model
    return load_trained_model(name)              # a pytorch model


def predict_digits(model, digits):
    """Return (labels, confidences): the guessed digit and how sure the model is (0..1)"""
    x = np.asarray(digits, dtype=np.float32)
    if len(x) == 0:
        return [], []
    if isinstance(model, torch.nn.Module):
        with torch.no_grad():
            scores = model(torch.from_numpy(x).unsqueeze(1))     # add the channel dimension
            probs = torch.softmax(scores, dim=1).numpy()          # scores ---> probabilities
    else:
        probs = model.predict_proba(x.reshape(len(x), -1))       # sklearn wants flat rows of 784
    return probs.argmax(axis=1).tolist(), probs.max(axis=1).tolist()