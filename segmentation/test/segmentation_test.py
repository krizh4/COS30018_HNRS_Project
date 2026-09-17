import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def segment_handwritten_digits(image_path="test_number.jpeg"):
    if not os.path.exists(image_path):
        print(f"Error: Cannot find '{image_path}'")
        return

    # 1. Load image
    img = cv2.imread(image_path)
    h_img, w_img = img.shape[:2]

    # 2. Preprocess: Grayscale + Gaussian Blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Adaptive Thresholding (White ink on Black background)
    # Automatically handles shadows and uneven light across the page
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,  # Block size
        10,  # Constant subtracted from mean
    )

    # Optional: Light morphological closing to solidify pen strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 4. Find External Contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 5. Filter out noise specks AND the entire page border
    valid_boxes = []
    total_area = h_img * w_img

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        box_area = w * h

        # Reject tiny dots (noise) and giant boxes (>60% of the entire photo)
        if 200 < box_area < (total_area * 0.6) and h > 20:
            valid_boxes.append((x, y, w, h))

    # 6. Sort from Left to Right
    valid_boxes = sorted(valid_boxes, key=lambda b: b[0])
    print(f"Detected {len(valid_boxes)} valid digit(s).")

    os.makedirs("cropped_digits", exist_ok=True)
    output_vis = img.copy()

    # 7. Crop, Pad to Square, and Save
    for i, (x, y, w, h) in enumerate(valid_boxes):
        cv2.rectangle(output_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            output_vis,
            str(i + 1),
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        # Crop isolated character
        digit_crop = thresh[y : y + h, x : x + w]

        # Pad into a square box (preserves aspect ratio for MNIST)
        size = max(w, h) + 20
        square_crop = np.zeros((size, size), dtype=np.uint8)
        x_offset = (size - w) // 2
        y_offset = (size - h) // 2
        square_crop[
            y_offset : y_offset + h, x_offset : x_offset + w
        ] = digit_crop

        # Resize to standard 28x28
        digit_28x28 = cv2.resize(
            square_crop, (28, 28), interpolation=cv2.INTER_AREA
        )

        crop_path = f"cropped_digits/digit_{i+1}.png"
        cv2.imwrite(crop_path, digit_28x28)
        print(f"Saved: {crop_path}")

    # 8. Visual Verification Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title(f"Segmented Digits Found: {len(valid_boxes)}")
    plt.imshow(cv2.cvtColor(output_vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Thresholded Mask (White on Black)")
    plt.imshow(thresh, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("segmentation_output.png")
    plt.show()


if __name__ == "__main__":
    segment_handwritten_digits("test_number.jpeg")