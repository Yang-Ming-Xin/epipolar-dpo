"""
Keypoint Match Visualization Script

This script visualizes SIFT keypoint matches between two frames to help diagnose
whether blur causes feature extraction to focus on irrelevant regions.

Usage:
    python visualize_keypoint_matches.py --img1 frame1.png --img2 frame2.png --output matches.png
    
    Or for a directory of image sequences:
    python visualize_keypoint_matches.py --dir /path/to/images --output_dir /path/to/output
"""

import cv2
import numpy as np
import argparse
import os
import glob
import re


def natural_sort_key(s):
    """Sort strings with embedded numbers naturally."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


def extract_and_match_sift(img1, img2, ratio_thresh=0.75):
    """
    Extract SIFT features and match between two images.
    
    Returns:
        pts1: Matched keypoint coordinates in image 1
        pts2: Matched keypoint coordinates in image 2
        kp1: All keypoints in image 1
        kp2: All keypoints in image 2
        good_matches: List of good matches
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
    
    # Create SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect and compute
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)
    
    if desc1 is None or desc2 is None or len(kp1) < 2 or len(kp2) < 2:
        return None, None, kp1, kp2, []
    
    # BFMatcher with KNN
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    
    # Apply ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
    
    # Extract matched point coordinates
    if len(good_matches) > 0:
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    else:
        pts1, pts2 = None, None
    
    return pts1, pts2, kp1, kp2, good_matches


def visualize_matches(img1, img2, kp1, kp2, good_matches, output_path=None, title=""):
    """
    Visualize keypoint matches between two images.
    
    Creates a side-by-side visualization with:
    - All detected keypoints (small circles)
    - Matched keypoints connected by lines
    """
    # Draw matches
    match_img = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        good_matches, None,
        matchColor=(0, 255, 0),      # Green for matches
        singlePointColor=(255, 0, 0), # Blue for unmatched keypoints
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    # Add statistics text
    h, w = match_img.shape[:2]
    stats_text = f"Keypoints: {len(kp1)} vs {len(kp2)} | Matches: {len(good_matches)}"
    if title:
        stats_text = f"{title} | {stats_text}"
    
    cv2.putText(match_img, stats_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    if output_path:
        cv2.imwrite(output_path, match_img)
        print(f"Saved visualization to: {output_path}")
    
    return match_img


def visualize_keypoint_heatmap(img, keypoints, output_path=None, title="Keypoint Distribution"):
    """
    Create a heatmap showing where keypoints are concentrated.
    This helps identify if keypoints are clustered in certain regions.
    """
    h, w = img.shape[:2]
    
    # Create heatmap
    heatmap = np.zeros((h, w), dtype=np.float32)
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(heatmap, (x, y), 20, 1.0, -1)
    
    # Normalize and apply colormap
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    heatmap = (heatmap / (heatmap.max() + 1e-8) * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay on original image
    overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
    
    # Add title
    cv2.putText(overlay, f"{title} | Total: {len(keypoints)} keypoints", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    if output_path:
        cv2.imwrite(output_path, overlay)
        print(f"Saved heatmap to: {output_path}")
    
    return overlay


def process_image_pair(img1_path, img2_path, output_dir, pair_name="pair"):
    """Process a single pair of images."""
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print(f"Error: Could not read images {img1_path} or {img2_path}")
        return None
    
    # Extract and match
    pts1, pts2, kp1, kp2, good_matches = extract_and_match_sift(img1, img2)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Visualize matches
    match_output = os.path.join(output_dir, f"{pair_name}_matches.png")
    visualize_matches(img1, img2, kp1, kp2, good_matches, match_output, pair_name)
    
    # 2. Visualize keypoint heatmaps for each image
    heatmap1_output = os.path.join(output_dir, f"{pair_name}_heatmap_frame1.png")
    heatmap2_output = os.path.join(output_dir, f"{pair_name}_heatmap_frame2.png")
    visualize_keypoint_heatmap(img1, kp1, heatmap1_output, f"{pair_name} Frame 1")
    visualize_keypoint_heatmap(img2, kp2, heatmap2_output, f"{pair_name} Frame 2")
    
    # Print statistics
    print(f"\n=== {pair_name} Statistics ===")
    print(f"Frame 1 keypoints: {len(kp1)}")
    print(f"Frame 2 keypoints: {len(kp2)}")
    print(f"Matched pairs: {len(good_matches)}")
    if len(kp1) > 0 and len(kp2) > 0:
        print(f"Match rate: {len(good_matches) / min(len(kp1), len(kp2)) * 100:.1f}%")
    
    return {
        'kp1_count': len(kp1),
        'kp2_count': len(kp2),
        'match_count': len(good_matches)
    }


def process_directory(img_dir, output_dir, sampling_rate=1):
    """Process all consecutive frame pairs in a directory."""
    # Find all images
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
    
    # Sample frames
    sampled_files = image_files[::sampling_rate]
    print(f"Processing {len(sampled_files)} sampled frames")
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_stats = []
    for i in range(len(sampled_files) - 1):
        img1_path = sampled_files[i]
        img2_path = sampled_files[i + 1]
        pair_name = f"pair_{i:03d}"
        
        stats = process_image_pair(img1_path, img2_path, output_dir, pair_name)
        if stats:
            all_stats.append(stats)
    
    # Print summary
    if all_stats:
        print(f"\n{'='*50}")
        print(f"SUMMARY for {img_dir}")
        print(f"{'='*50}")
        print(f"Total pairs processed: {len(all_stats)}")
        print(f"Avg keypoints (frame 1): {np.mean([s['kp1_count'] for s in all_stats]):.1f}")
        print(f"Avg keypoints (frame 2): {np.mean([s['kp2_count'] for s in all_stats]):.1f}")
        print(f"Avg matches: {np.mean([s['match_count'] for s in all_stats]):.1f}")


def main():
    parser = argparse.ArgumentParser(description='Visualize SIFT keypoint matches')
    
    # Mode 1: Single pair of images
    parser.add_argument('--img1', type=str, help='Path to first image')
    parser.add_argument('--img2', type=str, help='Path to second image')
    
    # Mode 2: Directory of images
    parser.add_argument('--dir', type=str, help='Directory containing image sequence')
    
    # Common arguments
    parser.add_argument('--output_dir', type=str, default='keypoint_viz', help='Output directory for visualizations')
    parser.add_argument('--sampling_rate', type=int, default=1, help='Process every Nth frame (for directory mode)')
    parser.add_argument('--pair_name', type=str, default='matches', help='Name prefix for output files (for single pair mode)')
    
    args = parser.parse_args()
    
    if args.img1 and args.img2:
        # Process single pair
        process_image_pair(args.img1, args.img2, args.output_dir, args.pair_name)
    elif args.dir:
        # Process directory
        process_directory(args.dir, args.output_dir, args.sampling_rate)
    else:
        print("Please provide either --img1 and --img2 for a single pair, or --dir for a sequence")
        parser.print_help()


if __name__ == "__main__":
    main()
