import cv2
import numpy as np
from collections import defaultdict

def detect_colors_from_pil(pil_img,
                           center_roi_fraction=0.2,
                           lesion_diff_threshold=40,
                           color_distance_threshold=50,
                           pixel_fraction_threshold=0.01,
                           morph_kernel_size=5):
    """
    Color analysis function that accepts a PIL image and returns present colors and score.
    """
    rgb_img = np.array(pil_img.convert("RGB"))
    H, W = rgb_img.shape[:2]

    # --- Center ROI ---
    center_x, center_y = W // 2, H // 2
    roi_size = int(min(W, H) * center_roi_fraction)
    x0 = max(0, center_x - roi_size // 2)
    x1 = min(W, center_x + roi_size // 2)
    y0 = max(0, center_y - roi_size // 2)
    y1 = min(H, center_y + roi_size // 2)
    center_roi = rgb_img[y0:y1, x0:x1]

    lesion_ref = np.mean(center_roi.reshape(-1, 3), axis=0) if center_roi.size > 0 else np.mean(rgb_img.reshape(-1, 3), axis=0)

    float_img = rgb_img.astype(np.float32)
    dist_map = np.linalg.norm(float_img - lesion_ref.astype(np.float32), axis=2)
    lesion_mask = (dist_map < lesion_diff_threshold).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel)
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(lesion_mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        lesion_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    else:
        lesion_mask = np.zeros_like(lesion_mask)

    # --- Color detection ---
    ref_colors_rgb = {
        "blue-gray": (90, 105, 120),
        "white": (245, 240, 235),
        "black": (35, 30, 30),
        "red": (220, 90, 85),
        "light brown": (190, 160, 120),
        "dark brown": (120, 80, 50)
    }

    lesion_pixels = rgb_img[lesion_mask == 255]
    total_lesion_pixels = max(1, len(lesion_pixels))

    color_counts = defaultdict(int)
    for pixel in lesion_pixels:
        distances = {name: np.linalg.norm(pixel - np.array(rgb)) for name, rgb in ref_colors_rgb.items()}
        closest_color = min(distances, key=distances.get)
        if distances[closest_color] <= color_distance_threshold:
            color_counts[closest_color] += 1

    present_colors = [c for c, count in color_counts.items() if count / total_lesion_pixels >= pixel_fraction_threshold]
    score = len(present_colors) * 0.5

    return present_colors, score
