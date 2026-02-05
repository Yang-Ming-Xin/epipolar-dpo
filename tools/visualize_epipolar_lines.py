"""
Epipolar Line Visualization Tool

Visualizes epipolar geometry between two frames:
- Computes fundamental matrix using Kornia (same as epipolar-dpo metrics)
- Draws epipolar lines for selected matched points
- Displays Sampson error for each point

Usage:
    python visualize_epipolar_lines.py --img1 frame1.png --img2 frame2.png --output epipolar_viz.png
    
    Or for a directory:
    python visualize_epipolar_lines.py --dir /path/to/images --output_dir /path/to/output
"""

import cv2
import numpy as np
import argparse
import os
import glob
import re
import torch

# Kornia imports (same as epipolar-dpo metrics)
from kornia.geometry.epipolar import find_fundamental, sampson_epipolar_distance


def natural_sort_key(s):
    """Sort strings with embedded numbers naturally."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


def extract_and_match_sift(img1, img2, ratio_thresh=0.75, min_matches=20):
    """
    Extract SIFT features and match between two images.
    Same logic as epipolar-dpo metrics.
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
    
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)
    
    if desc1 is None or desc2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None, None, kp1, kp2, []
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < min_matches:
        return None, None, kp1, kp2, good_matches
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    return pts1, pts2, kp1, kp2, good_matches


def compute_fundamental_and_sampson(pts1, pts2):
    """
    Compute fundamental matrix and Sampson distances using Kornia.
    Exactly matches epipolar-dpo/metrics implementation.
    """
    # Convert to tensors for Kornia - shape (B, N, 2)
    points1_tensor = torch.from_numpy(pts1).float().unsqueeze(0)
    points2_tensor = torch.from_numpy(pts2).float().unsqueeze(0)
    
    # Kornia's find_fundamental
    F_matrix = find_fundamental(points1_tensor, points2_tensor)
    
    if F_matrix is None or torch.isnan(F_matrix).any():
        return None, None
    
    # Compute Sampson distances
    sampson_dist_squared = sampson_epipolar_distance(
        points1_tensor, points2_tensor, F_matrix, squared=True
    )
    sampson_dist = torch.sqrt(sampson_dist_squared + 1e-8).squeeze()
    
    return F_matrix.squeeze(0).numpy(), sampson_dist.numpy()


def compute_epipolar_line(F, pt, img_shape):
    """
    Compute epipolar line in image 2 for a point in image 1.
    
    l = F @ [x, y, 1]^T
    Line equation: l[0]*x + l[1]*y + l[2] = 0
    """
    pt_homogeneous = np.array([pt[0], pt[1], 1.0])
    line = F @ pt_homogeneous
    
    # Normalize line
    line = line / (np.sqrt(line[0]**2 + line[1]**2) + 1e-8)
    
    # Compute line endpoints at image boundaries
    h, w = img_shape[:2]
    
    # Solve for y at x=0 and x=w
    if abs(line[1]) > 1e-8:
        y_at_x0 = -line[2] / line[1]
        y_at_xw = -(line[0] * w + line[2]) / line[1]
        x1, y1 = 0, int(y_at_x0)
        x2, y2 = w, int(y_at_xw)
    else:
        # Vertical line
        x1 = x2 = int(-line[2] / line[0])
        y1, y2 = 0, h
    
    return (x1, y1), (x2, y2)


def visualize_epipolar_lines(img1, img2, pts1, pts2, F_matrix, sampson_distances, 
                              output_path=None, num_points=8, title="", outlier_thresh=40.0):
    """
    Visualize epipolar lines for selected matched points.
    
    - Left: Image 1 with selected points colored by Sampson error
    - Right: Image 2 with corresponding epipolar lines and matched points
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Create side-by-side canvas
    max_h = max(h1, h2)
    canvas = np.zeros((max_h, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1.copy()
    canvas[:h2, w1:w1+w2] = img2.copy()
    
    # Select points to visualize (spread across error range)
    n_pts = len(pts1)
    if n_pts <= num_points:
        selected_indices = list(range(n_pts))
    else:
        # Sample uniformly from sorted error indices
        sorted_indices = np.argsort(sampson_distances)
        step = n_pts // num_points
        selected_indices = sorted_indices[::step][:num_points]
    
    # Color map: green (low error) to red (high error)
    max_error = max(sampson_distances[selected_indices])
    min_error = min(sampson_distances[selected_indices])
    error_range = max_error - min_error + 1e-8
    
    colors = []
    for idx in selected_indices:
        error = sampson_distances[idx]
        # Normalize error to [0, 1]
        norm_error = (error - min_error) / error_range
        # Green to Red gradient
        r = int(255 * norm_error)
        g = int(255 * (1 - norm_error))
        b = 0
        colors.append((b, g, r))  # BGR
    
    # Draw epipolar lines and points
    for i, idx in enumerate(selected_indices):
        pt1 = pts1[idx]
        pt2 = pts2[idx]
        error = sampson_distances[idx]
        color = colors[i]
        
        # Draw point in image 1
        cv2.circle(canvas, (int(pt1[0]), int(pt1[1])), 8, color, -1)
        cv2.circle(canvas, (int(pt1[0]), int(pt1[1])), 10, (255, 255, 255), 2)
        
        # Compute and draw epipolar line in image 2
        line_pt1, line_pt2 = compute_epipolar_line(F_matrix, pt1, img2.shape)
        cv2.line(canvas, 
                 (line_pt1[0] + w1, line_pt1[1]), 
                 (line_pt2[0] + w1, line_pt2[1]), 
                 color, 2)
        
        # Draw corresponding point in image 2
        cv2.circle(canvas, (int(pt2[0]) + w1, int(pt2[1])), 8, color, -1)
        cv2.circle(canvas, (int(pt2[0]) + w1, int(pt2[1])), 10, (255, 255, 255), 2)
        
        # Add Sampson error label near point in image 2
        label_pos = (int(pt2[0]) + w1 + 15, int(pt2[1]) + 5)
        cv2.putText(canvas, f"{error:.2f}px", label_pos, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(canvas, f"{error:.2f}px", label_pos, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Compute detailed statistics
    mean_error = float(np.mean(sampson_distances))
    median_error = float(np.median(sampson_distances))
    max_error = float(np.max(sampson_distances))
    min_error_stat = float(np.min(sampson_distances))
    std_error = float(np.std(sampson_distances))
    outlier_count = int(np.sum(sampson_distances > outlier_thresh))
    
    # Add title and statistics to image
    stats_text = f"Mean: {mean_error:.2f}px | Median: {median_error:.2f}px | Max: {max_error:.2f}px | Outliers(>{outlier_thresh}px): {outlier_count}"
    if title:
        stats_text = f"{title} | {stats_text}"
    
    cv2.putText(canvas, stats_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Add color legend
    legend_x = w1 + w2 - 200
    cv2.putText(canvas, "Error Legend:", (legend_x, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.rectangle(canvas, (legend_x, 70), (legend_x + 100, 90), (0, 255, 0), -1)
    cv2.putText(canvas, "Low", (legend_x + 105, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.rectangle(canvas, (legend_x, 95), (legend_x + 100, 115), (0, 0, 255), -1)
    cv2.putText(canvas, "High", (legend_x + 105, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    if output_path:
        cv2.imwrite(output_path, canvas)
        print(f"Saved epipolar visualization to: {output_path}")
    
    return canvas


def visualize_outliers(img1, img2, pts1, pts2, F_matrix, sampson_distances, 
                       output_path=None, outlier_thresh=40.0, title=""):
    """
    Visualize only the outlier points (high Sampson error) with their epipolar lines.
    
    Outliers are drawn in bright magenta with X markers for easy identification.
    Shows the distance from each point to its epipolar line.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Find outlier indices
    outlier_mask = sampson_distances > outlier_thresh
    outlier_indices = np.where(outlier_mask)[0]
    
    if len(outlier_indices) == 0:
        print(f"  No outliers found (threshold: {outlier_thresh}px)")
        return None
    
    # Create side-by-side canvas with darker overlay to highlight outliers
    max_h = max(h1, h2)
    canvas = np.zeros((max_h, w1 + w2, 3), dtype=np.uint8)
    # Darken original images to make outliers stand out
    canvas[:h1, :w1] = (img1.copy() * 0.5).astype(np.uint8)
    canvas[:h2, w1:w1+w2] = (img2.copy() * 0.5).astype(np.uint8)
    
    # Sort outliers by error (highest first)
    sorted_outlier_indices = outlier_indices[np.argsort(sampson_distances[outlier_indices])[::-1]]
    
    # Color gradient for outliers: yellow (lower outlier) to magenta (extreme outlier)
    outlier_errors = sampson_distances[sorted_outlier_indices]
    min_outlier_err = outlier_errors.min()
    max_outlier_err = outlier_errors.max()
    error_range = max_outlier_err - min_outlier_err + 1e-8
    
    for i, idx in enumerate(sorted_outlier_indices):
        pt1 = pts1[idx]
        pt2 = pts2[idx]
        error = sampson_distances[idx]
        
        # Color: yellow -> magenta gradient based on error severity
        norm_error = (error - min_outlier_err) / error_range
        b = int(255 * norm_error)
        g = int(255 * (1 - norm_error))
        r = 255
        color = (b, g, r)  # BGR
        
        # Draw X marker in image 1 (more visible than circle for outliers)
        x1, y1 = int(pt1[0]), int(pt1[1])
        cv2.line(canvas, (x1-12, y1-12), (x1+12, y1+12), color, 3)
        cv2.line(canvas, (x1-12, y1+12), (x1+12, y1-12), color, 3)
        cv2.circle(canvas, (x1, y1), 15, (255, 255, 255), 2)
        
        # Compute and draw epipolar line in image 2
        line_pt1, line_pt2 = compute_epipolar_line(F_matrix, pt1, img2.shape)
        cv2.line(canvas, 
                 (line_pt1[0] + w1, line_pt1[1]), 
                 (line_pt2[0] + w1, line_pt2[1]), 
                 color, 2, cv2.LINE_AA)
        
        # Draw X marker in image 2
        x2, y2 = int(pt2[0]) + w1, int(pt2[1])
        cv2.line(canvas, (x2-12, y2-12), (x2+12, y2+12), color, 3)
        cv2.line(canvas, (x2-12, y2+12), (x2+12, y2-12), color, 3)
        cv2.circle(canvas, (x2, y2), 15, (255, 255, 255), 2)
        
        # Draw connection line between matched points (to show mismatch)
        cv2.line(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        
        # Add error label with background for visibility
        label = f"#{i+1}: {error:.1f}px"
        label_pos = (x2 + 20, y2 + 5)
        # Background rectangle
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (label_pos[0]-2, label_pos[1]-th-2), 
                     (label_pos[0]+tw+2, label_pos[1]+4), (0, 0, 0), -1)
        cv2.putText(canvas, label, label_pos, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Add header
    header = f"OUTLIERS (>{outlier_thresh}px): {len(outlier_indices)} points | Max: {max_outlier_err:.1f}px"
    if title:
        header = f"{title} | {header}"
    cv2.putText(canvas, header, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Add legend
    legend_x = w1 + w2 - 250
    cv2.putText(canvas, "Outlier Severity:", (legend_x, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.rectangle(canvas, (legend_x, 70), (legend_x + 100, 90), (0, 255, 255), -1)
    cv2.putText(canvas, "Lower", (legend_x + 105, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.rectangle(canvas, (legend_x, 95), (legend_x + 100, 115), (255, 0, 255), -1)
    cv2.putText(canvas, "Extreme", (legend_x + 105, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    if output_path:
        cv2.imwrite(output_path, canvas)
        print(f"Saved outlier visualization to: {output_path}")
    
    return canvas


def process_image_pair(img1_path, img2_path, output_dir, pair_name="pair", num_points=8, outlier_thresh=40.0):
    """Process a single pair of images."""
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print(f"Error: Could not read images {img1_path} or {img2_path}")
        return None
    
    # Extract and match
    pts1, pts2, kp1, kp2, good_matches = extract_and_match_sift(img1, img2)
    
    if pts1 is None or pts2 is None:
        print(f"Error: Not enough matches for {pair_name}")
        return None
    
    # Compute fundamental matrix and Sampson distances
    F_matrix, sampson_distances = compute_fundamental_and_sampson(pts1, pts2)
    
    if F_matrix is None:
        print(f"Error: Could not compute fundamental matrix for {pair_name}")
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize epipolar lines
    output_path = os.path.join(output_dir, f"{pair_name}_epipolar.png")
    visualize_epipolar_lines(img1, img2, pts1, pts2, F_matrix, sampson_distances,
                              output_path, num_points, pair_name, outlier_thresh)
    
    # Visualize outliers separately if any exist
    outlier_output_path = os.path.join(output_dir, f"{pair_name}_outliers.png")
    visualize_outliers(img1, img2, pts1, pts2, F_matrix, sampson_distances,
                       outlier_output_path, outlier_thresh, pair_name)
    
    # Print detailed statistics
    mean_error = float(np.mean(sampson_distances))
    median_error = float(np.median(sampson_distances))
    max_error = float(np.max(sampson_distances))
    min_error = float(np.min(sampson_distances))
    std_error = float(np.std(sampson_distances))
    outlier_count = int(np.sum(sampson_distances > outlier_thresh))
    inlier_rate = float(np.mean(sampson_distances <= 5.0))
    
    # Inlier-only statistics (after removing outliers)
    inlier_mask = sampson_distances <= outlier_thresh
    inlier_distances = sampson_distances[inlier_mask]
    if len(inlier_distances) > 0:
        inlier_min = float(np.min(inlier_distances))
        inlier_max = float(np.max(inlier_distances))
        inlier_mean = float(np.mean(inlier_distances))
        inlier_median = float(np.median(inlier_distances))
        inlier_std = float(np.std(inlier_distances))
    else:
        inlier_min = inlier_max = inlier_mean = inlier_median = inlier_std = 0.0
    
    print(f"\n=== {pair_name} Sampson Error Statistics ===")
    print(f"Total matches: {len(pts1)}")
    print(f"--- All Points ---")
    print(f"  Min:    {min_error:.4f} px")
    print(f"  Max:    {max_error:.4f} px")
    print(f"  Mean:   {mean_error:.4f} px")
    print(f"  Median: {median_error:.4f} px")
    print(f"  Std:    {std_error:.4f} px")
    print(f"--- After Removing Outliers (>{outlier_thresh}px) ---")
    print(f"  Count:  {len(inlier_distances)} ({100*len(inlier_distances)/len(pts1):.1f}%)")
    print(f"  Min:    {inlier_min:.4f} px")
    print(f"  Max:    {inlier_max:.4f} px")
    print(f"  Mean:   {inlier_mean:.4f} px")
    print(f"  Median: {inlier_median:.4f} px")
    print(f"  Std:    {inlier_std:.4f} px")
    print(f"--- Summary ---")
    print(f"  Outliers (>{outlier_thresh}px): {outlier_count} ({100*outlier_count/len(pts1):.1f}%)")
    print(f"  Inliers (<=5px): {int(inlier_rate*len(pts1))} ({inlier_rate*100:.1f}%)")
    
    return {
        'match_count': len(pts1),
        'min_error': min_error,
        'max_error': max_error,
        'mean_error': mean_error,
        'median_error': median_error,
        'std_error': std_error,
        'outlier_count': outlier_count,
        'inlier_rate': inlier_rate,
        # Inlier-only stats
        'inlier_count': len(inlier_distances),
        'inlier_min': inlier_min,
        'inlier_max': inlier_max,
        'inlier_mean': inlier_mean,
        'inlier_median': inlier_median,
        'inlier_std': inlier_std,
    }


def process_directory(img_dir, output_dir, sampling_rate=1, num_points=8, outlier_thresh=40.0):
    """Process all consecutive frame pairs in a directory."""
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(img_dir, ext)))
    
    image_files = sorted(image_files, key=natural_sort_key)
    
    if len(image_files) < 2:
        print(f"Error: Need at least 2 images, found {len(image_files)}")
        return
    
    print(f"Found {len(image_files)} images in {img_dir}")
    print(f"Sampling rate: {sampling_rate}")
    
    sampled_files = image_files[::sampling_rate]
    print(f"Processing {len(sampled_files)} sampled frames")
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_stats = []
    for i in range(len(sampled_files) - 1):
        img1_path = sampled_files[i]
        img2_path = sampled_files[i + 1]
        pair_name = f"pair_{i:03d}"
        
        stats = process_image_pair(img1_path, img2_path, output_dir, pair_name, num_points, outlier_thresh)
        if stats:
            all_stats.append(stats)
    
    # Print summary
    if all_stats:
        print(f"\n{'='*60}")
        print(f"EPIPOLAR SUMMARY for {img_dir}")
        print(f"{'='*60}")
        print(f"Total pairs processed: {len(all_stats)}")
        print(f"Avg matches:        {np.mean([s['match_count'] for s in all_stats]):.1f}")
        print(f"Avg Min error:      {np.mean([s['min_error'] for s in all_stats]):.4f} px")
        print(f"Avg Max error:      {np.mean([s['max_error'] for s in all_stats]):.4f} px")
        print(f"Avg Mean error:     {np.mean([s['mean_error'] for s in all_stats]):.4f} px")
        print(f"Avg Median error:   {np.mean([s['median_error'] for s in all_stats]):.4f} px")
        print(f"Total outliers (>{outlier_thresh}px): {sum([s['outlier_count'] for s in all_stats])}")
        print(f"Avg inlier rate:    {np.mean([s['inlier_rate'] for s in all_stats])*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Visualize epipolar lines with Sampson error')
    
    # Mode 1: Single pair
    parser.add_argument('--img1', type=str, help='Path to first image')
    parser.add_argument('--img2', type=str, help='Path to second image')
    
    # Mode 2: Directory
    parser.add_argument('--dir', type=str, help='Directory containing image sequence')
    
    # Common arguments
    parser.add_argument('--output_dir', type=str, default='epipolar_viz', 
                        help='Output directory for visualizations')
    parser.add_argument('--sampling_rate', type=int, default=1, 
                        help='Process every Nth frame (for directory mode)')
    parser.add_argument('--num_points', type=int, default=8, 
                        help='Number of points to visualize epipolar lines for')
    parser.add_argument('--outlier_thresh', type=float, default=40.0, 
                        help='Threshold (in pixels) for counting outliers')
    parser.add_argument('--pair_name', type=str, default='epipolar', 
                        help='Name prefix for output files (for single pair mode)')
    
    args = parser.parse_args()
    
    if args.img1 and args.img2:
        process_image_pair(args.img1, args.img2, args.output_dir, args.pair_name, args.num_points, args.outlier_thresh)
    elif args.dir:
        process_directory(args.dir, args.output_dir, args.sampling_rate, args.num_points, args.outlier_thresh)
    else:
        print("Please provide either --img1 and --img2 for a single pair, or --dir for a sequence")
        parser.print_help()


if __name__ == "__main__":
    main()
