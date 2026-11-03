"""
Test script for the expression recognition pipeline.
Run this after training the model to see how well it performs on a few test images.
The test images are in the test_images/ folder.

"""

import os
from tensorflow import keras
from pipeline import recognize_expression_image, DEFAULT_MODEL_PATH
from expression_parser import safe_eval

TEST_CASES = [
    # (filename in test_images/, true expression as a string)
    ("ex1.PNG", "(2347*6581)-(342*18)"),   
    
]

TEST_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "test_images")


def run_tests():
    if not os.path.exists(DEFAULT_MODEL_PATH):
        print(f"No trained model found at {DEFAULT_MODEL_PATH}.")
        print("Run 'python train.py' first.")
        return

    model = keras.models.load_model(DEFAULT_MODEL_PATH)

    n_correct_expr = 0
    n_correct_result = 0
    n_total = 0

    print(f"{'File':<14} {'True expr':<26} {'Recognized':<26} {'True val':<14} {'Pred val':<14} {'Match?'}")
    print("-" * 110)

    for filename, true_expr in TEST_CASES:
        path = os.path.join(TEST_IMAGES_DIR, filename)
        if not os.path.exists(path):
            print(f"{filename:<14} (file not found - skipping)")
            continue

        n_total += 1
        true_value = safe_eval(true_expr)
        out = recognize_expression_image(path, model=model)

        pred_expr = out["expression"] or "".join(out["symbols"])
        pred_value = out["result"]

        expr_match = (pred_expr == true_expr)
        result_match = (pred_value is not None and abs(pred_value - true_value) < 1e-6)

        if expr_match:
            n_correct_expr += 1
        if result_match:
            n_correct_result += 1

        status = "OK" if result_match else ("PARTIAL" if expr_match else "WRONG")
        if out["error"]:
            status += f"  [{out['error']}]"

        print(f"{filename:<14} {true_expr:<26} {pred_expr:<26} {true_value:<14} "
              f"{pred_value if pred_value is not None else '-':<14} {status}")

    if n_total:
        print("\n=== Summary ===")
        print(f"Exact expression match: {n_correct_expr}/{n_total} ({n_correct_expr/n_total:.1%})")
        print(f"Correct final answer:   {n_correct_result}/{n_total} ({n_correct_result/n_total:.1%})")
    else:
        print("No test images found. Add photos to test_images/ and update TEST_CASES.")


if __name__ == "__main__":
    run_tests()
