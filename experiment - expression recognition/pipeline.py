"""
End-to-end pipeline for ONE expression image:
  segment (Task 2 code) -> classify each symbol -> build expression
  string (with parentheses) -> evaluate safely
"""

import os
import numpy as np
from tensorflow import keras

import my_preprocessing as pre
from symbol_segmentation import segment_symbols
from symbol_dataset import LABEL_NAMES
from expression_parser import recognize_and_evaluate, ExpressionError

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "symbol_cnn.keras")


def recognize_expression_image(image_path: str, model=None, model_path: str = DEFAULT_MODEL_PATH,
                                method: str = "otsu", confidence_threshold: float = 0.0):
    """
    Returns a dict with:
      symbols       - predicted symbol characters, in order
      confidences   - softmax confidence per symbol
      expression    - assembled expression string, e.g. "(2347*6581)-(342*18)"
      result        - evaluated numeric result, or None if evaluation failed
      error         - error message if any, else None
      n_segments    - how many symbols segmentation found
    """
    if model is None:
        model = keras.models.load_model(model_path)

    image = pre.load_image(image_path)
    boxes, symbol_imgs = segment_symbols(image, method=method)

    symbols, confidences = [], []
    for sym_img in symbol_imgs:
        arr = np.array(sym_img).astype("float32") / 255.0
        batch = np.expand_dims(arr, axis=(0, -1))
        probs = model.predict(batch, verbose=0)[0]
        label = int(np.argmax(probs))
        confidences.append(float(probs[label]))
        symbols.append(LABEL_NAMES[label])

    result_dict = {
        "symbols": symbols,
        "confidences": confidences,
        "n_segments": len(symbol_imgs),
        "expression": None,
        "result": None,
        "error": None,
    }

    low_conf = [s for s, c in zip(symbols, confidences) if c < confidence_threshold]
    if low_conf:
        result_dict["error"] = f"Low-confidence symbol(s) below threshold: {low_conf}"

    try:
        expr, value = recognize_and_evaluate(symbols)
        result_dict["expression"] = expr
        result_dict["result"] = value
    except ExpressionError as e:
        result_dict["error"] = (result_dict["error"] + "; " if result_dict["error"] else "") + str(e)

    return result_dict
