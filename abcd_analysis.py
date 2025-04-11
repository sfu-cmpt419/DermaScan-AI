import os
import cv2
import numpy as np
import pandas as pd

def compute_asymmetry(mask):
    """Improved asymmetry calculation using major axis alignment."""
    # Find principal axes using PCA
    coords = np.column_stack(np.where(mask > 0)).astype(np.float64)  # Convert to float
    if len(coords) < 2:
        return 0.0
    
    mean = np.mean(coords, axis=0)
    coords -= mean  # Now works with float coordinates
    
    cov = np.cov(coords, rowvar=False)
    _, vecs = np.linalg.eigh(cov)
    vx, vy = vecs[:, -1]  # Principal component
    
    # Rotate points to align with principal axis
    theta = np.arctan2(vy, vx)
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
    rotated_coords = np.dot(coords, rot_mat)
    
    # Split along principal axis
    min_val = np.min(rotated_coords[:, 0])
    max_val = np.max(rotated_coords[:, 0])
    split_point = (min_val + max_val) / 2
    
    left = rotated_coords[rotated_coords[:, 0] < split_point]
    right = rotated_coords[rotated_coords[:, 0] >= split_point]
    
    # Calculate asymmetry
    area_total = len(coords)
    area_diff = abs(len(left) - len(right))
    return area_diff / area_total if area_total > 0 else 0.0

# Rest of the code remains the same as previous optimized version
# (keep the same main(), compute_border_irregularity(), and compute_diameter() functions)

def compute_border_irregularity(mask):
    """Improved border irregularity calculation with convex hull comparison."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    
    # Calculate original and convex hull perimeters
    perimeter = cv2.arcLength(contour, True)
    hull_perimeter = cv2.arcLength(hull, True)
    
    # Normalized difference between actual and convex hull perimeter
    return (perimeter - hull_perimeter) / hull_perimeter if hull_perimeter > 0 else 0.0

def compute_diameter(mask, target_size=(512, 512), min_px=50, max_px=530):
    # Resize the mask to the desired target size 
    resized_mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    
    # Extract coordinates of white pixels (non-zero)
    coords = np.column_stack(np.where(resized_mask > 0)).astype(np.int32)
    if len(coords) < 2:
        return 0.0, 0.0  # raw_diameter, normalized_score

    # Compute the convex hull of the lesion pixels
    hull = cv2.convexHull(coords)
    hull_points = hull.squeeze()  # Expected shape: (N, 2)
    
    # Check that the hull has at least two points
    if hull_points.ndim < 2 or len(hull_points) < 2:
        return 0.0, 0.0

    # Compute the maximum Euclidean distance between any two points of the convex hull
    max_dist = 0.0
    for i in range(len(hull_points)):
        for j in range(i + 1, len(hull_points)):
            dist = np.linalg.norm(hull_points[i] - hull_points[j])
            if dist > max_dist:
                max_dist = dist
    
    # Normalize the diameter score
    clamped_dist = max(min(max_dist, max_px), min_px)
    normalized_score = (clamped_dist - min_px) / (max_px - min_px)

    return round(max_dist, 2), round(normalized_score, 3)


def main():
    segmented_folder = "./segmented"  # Update with your path
    image_files = [f for f in os.listdir(segmented_folder) if f.endswith(".png")]
    
    results = []
    
    for image_file in image_files:
        image_path = os.path.join(segmented_folder, image_file)
        mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"Warning: Could not read {image_file}")
            continue
        
        # Binarize mask
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        
        # Check if mask is empty
        if np.max(mask) == 0:
            print(f"Warning: Empty mask in {image_file}")
            results.append({"Image": image_file, "Asymmetry (A)": 0, "Border (B)": 0, "Diameter (D)": 0})
            continue
        
        # Compute features
        A = compute_asymmetry(mask)
        B = compute_border_irregularity(mask)
        D = compute_diameter(mask)
        
        results.append({"Image": image_file, "Asymmetry (A)": A, "Border (B)": B, "Diameter (D)": D})
        print(f"Processed {image_file}: A={A:.4f}, B={B:.4f}, D={D:.2f} px")

    df = pd.DataFrame(results)
    csv_path = "./ABCD_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to {csv_path}")

if __name__ == "__main__":
    main()