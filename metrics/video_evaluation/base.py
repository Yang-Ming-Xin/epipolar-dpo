import os
from typing import Tuple, Dict, Any
from abc import ABC, abstractmethod
import re
import glob

import cv2
from tqdm import tqdm


class BaseEvaluator(ABC):
    """
    Abstract base class for video evaluation.
    """

    def __init__(self, sampling_rate: int = 1):
        """
        Initialize the evaluator.

        Args:
            sampling_rate: Process every Nth frame (default: 1 = process all frames)
        """
        self.sampling_rate = sampling_rate

    @classmethod
    def from_config(cls, config):
        """Create an evaluator from a configuration dictionary."""
        pass

    def evaluate_video(self, video_path):
        """
        Evaluate a video file or image sequence by processing frames and computing metrics.

        Args:
            video_path: Path to MP4 video file or directory containing image sequence

        Returns:
            Dictionary with evaluation metrics
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Path not found: {video_path}")

        # Helper for natural sorting
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower()
                    for text in re.split('([0-9]+)', s)]

        # Check if input is directory (image sequence) or file (video)
        if os.path.isdir(video_path):
            # Load image sequence
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(video_path, ext)))
            
            image_files = sorted(image_files, key=natural_sort_key)
            
            if not image_files:
                raise ValueError(f"No images found in directory: {video_path}")
            
            # Read first image to get dimensions
            first_frame = cv2.imread(image_files[0])
            if first_frame is None:
                raise ValueError(f"Could not read first image: {image_files[0]}")
            
            frame_count = len(image_files)
            fps = 10.0  # Default FPS for image sequences
            height, width = first_frame.shape[:2]
            
            frames_iterator = ((i, cv2.imread(img_path)) for i, img_path in enumerate(image_files))
            is_video = False
        else:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")

            # Get video info
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            def video_frame_generator():
                idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    yield idx, frame
                    idx += 1
            
            frames_iterator = video_frame_generator()
            is_video = True

        print(f"Video: {os.path.basename(video_path)}")
        print(f"Dimensions: {width}x{height}, {frame_count} frames, {fps} fps")
        print(f"Processing every {self.sampling_rate} frame(s)")

        # Initialize metrics storage
        frame_metrics = []

        # Process frames
        pbar = tqdm(total=frame_count)

        for frame_idx, frame in frames_iterator:
            if frame is None:
                pbar.update(1)
                continue

            # Only process frames according to sampling rate
            if frame_idx % self.sampling_rate == 0:
                # Compute metrics for this frame
                try:
                    metrics = self.compute_metrics(frame)
                    if metrics is not None:
                        frame_metrics.append(metrics)
                except Exception as e:
                    print(f"Error processing frame {frame_idx}: {e}")
                    continue

            pbar.update(1)

        # Release resources
        if is_video:
            cap.release()
        pbar.close()

        main_metric, all_metrics = self.aggregate_metrics(frame_metrics)

        return main_metric, all_metrics

    @abstractmethod
    def compute_metrics(self, frame):
        """
        Compute metrics for a single frame.

        Args:
            frame: The video frame (numpy array in BGR format)

        Returns:
            Dictionary with metrics for this frame
        """
        pass

    @property
    @abstractmethod
    def name(self):
        """Return the name of this evaluator."""
        pass

    @abstractmethod
    def aggregate_metrics(self, frame_metrics) -> Tuple[float, Dict[str, Any]]:
        """
        Aggregate per-frame metrics into final metrics.

        Args:
            frame_metrics: List of dictionaries with per-frame metrics

        Returns:
            Dictionary with final aggregated metrics
        """
        pass
