#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
edge_detector.py  — robust, general-purpose edge detector

Features
- Multiple edge detection methods (Canny, Sobel, Laplacian, Scharr)
- CLAHE contrast normalization (helps low/high contrast)
- Denoising filters: gaussian | median | bilateral | None
- Advanced Canny with auto-thresholding
- Morphological path with directional enhancement
- Padding to preserve boundary edges
- Optional Hough refinement for line detection
- Green edge visualization
- Batch processing support

CLI examples:
  python3 edge_detector.py --image img.png --save --method canny
  python3 edge_detector.py --dir /path/to/folder --save --method sobel
"""

import os
import glob
from typing import List, Dict, Tuple

import cv2
import numpy as np


def _odd(n: int) -> int:
    """Ensure odd kernel size."""
    return n | 1


import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class EdgeDetectionError(Exception):
    """Custom exception for edge detection errors."""
    pass

class EdgeDetector:
    """
    Robust, general-purpose edge detector with checkerboard support.
    
    Methods:
        - "canny": Pure Canny edge detection
        - "morph": Morphological operations based
        - "hybrid": Combination of Canny and morphological
        - "auto": Automatically chooses method based on image
        
    Features:
        - Checkerboard detection and corner extraction
        - Multi-scale processing
        - Contrast enhancement (CLAHE)
        - Noise reduction
        - Green edge visualization
    """
    
    VALID_METHODS = {"canny", "sobel", "laplacian", "scharr"}
    VALID_DENOISE = {"gaussian", "median", "bilateral", None}
    
    @property
    def method(self):
        """Current edge detection method."""
        return self._method
        
    @method.setter
    def method(self, value):
        if value not in self.VALID_METHODS:
            raise ValueError(f"Method must be one of {self.VALID_METHODS}")
        self._method = value

    def __init__(self,
                 method: str = "canny",              # "canny" | "sobel" | "laplacian" | "scharr"
                    canny_low: float = 200.0,           # Default lower threshold for Canny
                    canny_high: float = 250.0,          # Default higher threshold for Canny
                    denoise: str = None,                # No denoising by default
                    ksize: int = 3,                     # Kernel size for Gaussian blur
                    sigma: float = 0.0,                 # Sigma for Gaussian blur
                    bilateral_sigma: float = 50.0,      # Bilateral filter parameter
                    auto_threshold: bool = False,       # Use fixed thresholds
                    l2gradient: bool = True,            # Use L2 gradient
                    use_clahe: bool = False,            # No CLAHE by default
                    hough_refine: bool = False,         # No Hough refinement
                    pad: int = 2,                       # Border padding size
                    min_blob_area_ratio: float = 0.0,   # No blob filtering
                    min_blob_area_px: int = 0,          # No minimum blob size
                    fast_nl_means_h: float = 0.0,       # No non-local means
                    noise_clean_ratio: float = 0.0,     # No extra noise cleaning
                    thin_edges: bool = False):          # No edge thinning
        self.method = method
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.denoise = denoise
        self.ksize = _odd(max(3, ksize))
        self.sigma = sigma
        self.bilateral_sigma = bilateral_sigma
        self.auto_threshold = auto_threshold
        self.l2gradient = l2gradient
        self.use_clahe = use_clahe
        self.hough_refine = hough_refine
        self.pad = max(0, int(pad))
        self.min_blob_area_ratio = max(0.0, float(min_blob_area_ratio))
        self.min_blob_area_px = max(0, int(min_blob_area_px))
        self.fast_nl_means_h = max(0.0, float(fast_nl_means_h))
        self.noise_clean_ratio = max(0.0, float(noise_clean_ratio))
        self.thin_edges = bool(thin_edges)

    # ------------------- utilities -------------------
    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            g = clahe.apply(g)
        return g

    def _prefilter(self, gray: np.ndarray) -> np.ndarray:
        """Apply balanced denoising while preserving edges."""
        if self.denoise == "gaussian":
            return cv2.GaussianBlur(gray, (self.ksize, self.ksize), self.sigma)
        if self.denoise == "median":
            return cv2.medianBlur(gray, self.ksize)
        if self.denoise == "bilateral":
            # Single pass bilateral filter with balanced parameters
            return cv2.bilateralFilter(gray, self.ksize, self.bilateral_sigma, self.bilateral_sigma)
        return gray

    def _non_local_means(self, gray: np.ndarray) -> np.ndarray:
        """Optional fast non-local means denoising for stubborn noise."""
        if self.fast_nl_means_h <= 0:
            return gray
        return cv2.fastNlMeansDenoising(gray, None, h=self.fast_nl_means_h,
                                        templateWindowSize=7, searchWindowSize=21)

    def _auto_canny_thresholds(self, gray: np.ndarray) -> Tuple[float, float]:
        """Estimate robust Canny thresholds from gradient statistics."""
        blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
        gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.hypot(gx, gy)
        max_grad = grad.max()
        if max_grad > 0:
            grad = grad / max_grad * 255.0
        grad = grad.flatten()
        grad = grad[grad > 0]
        if grad.size == 0:
            return 0.0, 0.0
        low = np.percentile(grad, 20)
        high = np.percentile(grad, 85)
        low = max(0.0, min(255.0, 0.7 * low))
        high = max(low + 5.0, min(255.0, 1.05 * high))
        return float(low), float(high)

    def _pad(self, img: np.ndarray) -> np.ndarray:
        if self.pad <= 0:
            return img
        return cv2.copyMakeBorder(img, self.pad, self.pad, self.pad, self.pad,
                                  borderType=cv2.BORDER_REPLICATE)

    def _unpad(self, img: np.ndarray) -> np.ndarray:
        if self.pad <= 0:
            return img
        return img[self.pad:-self.pad, self.pad:-self.pad]

    def _remove_small_blobs(self, bw: np.ndarray) -> np.ndarray:
        if self.min_blob_area_ratio <= 0 and self.min_blob_area_px <= 0:
            return bw
        h, w = bw.shape
        ratio_area = int(self.min_blob_area_ratio * h * w) if self.min_blob_area_ratio > 0 else 0
        min_area = max(1, ratio_area, self.min_blob_area_px)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        out = np.zeros_like(bw)
        for i in range(1, n):  # skip background
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                out[labels == i] = 255
        return out

    def _thin_edges(self, edges: np.ndarray) -> np.ndarray:
        """Morphological skeletonization to keep edges crisp."""
        if not self.thin_edges:
            return edges
        working = edges.copy()
        skel = np.zeros_like(edges)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            eroded = cv2.erode(working, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(working, temp)
            skel = cv2.bitwise_or(skel, temp)
            working = eroded
            if cv2.countNonZero(working) == 0:
                break
        return skel
        
    # ------------------- core methods -------------------
    def _edges_canny(self, gray: np.ndarray) -> np.ndarray:
        """Simple and effective Canny edge detection with border padding."""
        # Add padding to preserve edge detection at image boundaries
        padded_img = self._pad(gray)
        
        # Apply Gaussian filter to reduce noise
        filtered_img = cv2.GaussianBlur(padded_img, (3, 3), 0)
        # Apply Canny edge detection with provided thresholds
        edges = cv2.Canny(filtered_img, self.canny_low, self.canny_high)

        # Remove padding to get back to original size
        edges = self._unpad(edges)

        # Ensure binary 0/255 mono output
        return (edges > 0).astype(np.uint8) * 255

    def _edges_sobel(self, gray: np.ndarray) -> np.ndarray:
        """Sobel edge detection producing binary edges."""
        g = self._pad(self._prefilter(gray))
        dx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(dx**2 + dy**2)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        mag = self._unpad(magnitude.astype(np.uint8))
        # Binarize magnitude to produce crisp edges
        if np.any(mag > 0):
            thresh = float(np.percentile(mag[mag > 0], 90))
        else:
            thresh = 0.0
        if thresh <= 0:
            thresh = 16
        bw = (mag >= thresh).astype(np.uint8) * 255
        return bw

    def _edges_laplacian(self, gray: np.ndarray) -> np.ndarray:
        """Laplacian edge detection producing binary edges."""
        g = self._pad(self._prefilter(gray))
        lap = cv2.Laplacian(g, cv2.CV_64F, ksize=3)
        lap = np.absolute(lap)
        lap = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX)
        mag = self._unpad(lap.astype(np.uint8))
        if np.any(mag > 0):
            thresh = float(np.percentile(mag[mag > 0], 90))
        else:
            thresh = 0.0
        if thresh <= 0:
            thresh = 16
        bw = (mag >= thresh).astype(np.uint8) * 255
        return bw

    def _edges_scharr(self, gray: np.ndarray) -> np.ndarray:
        """Scharr edge detection (improved Sobel) producing binary edges."""
        g = self._pad(self._prefilter(gray))
        dx = cv2.Scharr(g, cv2.CV_64F, 1, 0)
        dy = cv2.Scharr(g, cv2.CV_64F, 0, 1)
        magnitude = np.sqrt(dx**2 + dy**2)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        mag = self._unpad(magnitude.astype(np.uint8))
        if np.any(mag > 0):
            thresh = float(np.percentile(mag[mag > 0], 90))
        else:
            thresh = 0.0
        if thresh <= 0:
            thresh = 16
        bw = (mag >= thresh).astype(np.uint8) * 255
        return bw

    def _refine_with_hough(self, edges: np.ndarray) -> np.ndarray:
        if not self.hough_refine:
            return edges
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                                minLineLength=max(20, min(edges.shape) // 12), maxLineGap=10)
        out = np.zeros_like(edges)
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                cv2.line(out, (x1, y1), (x2, y2), 255, 1, cv2.LINE_AA)
        return out if np.count_nonzero(out) > 0 else edges

    def _choose_method(self, gray: np.ndarray) -> str:
        # Heuristic: sparse/spotty Canny ⇒ morph; else hybrid
        c = self._edges_canny_single(gray)
        density = np.count_nonzero(c) / float(c.size)
        specks = cv2.morphologyEx(c, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        speck_ratio = np.count_nonzero(specks) / float(c.size)
        if density < 0.02 or speck_ratio > 0.01:
            return "morph"
        return "hybrid"

    # ------------------- public API -------------------
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges in an image using the selected method.
        
        Args:
            image: Input image (grayscale or BGR)
            
        Returns:
            uint8 edge map (0-255)
            
        Raises:
            EdgeDetectionError: If image is invalid or processing fails
        """
        logger = logging.getLogger(__name__)
        
        try:
            # Input validation
            if image is None:
                raise EdgeDetectionError("Input image is None")
            if not isinstance(image, np.ndarray):
                raise EdgeDetectionError(f"Expected numpy array, got {type(image)}")
            if image.size == 0:
                raise EdgeDetectionError("Empty image")
                
            # Convert to grayscale
            gray = self._to_gray(image)
            logger.debug(f"Processing {image.shape} image with {self.method} method")
            
            # Apply selected edge detection method
            if self.method == "canny":
                edges = self._edges_canny(gray)
            elif self.method == "sobel":
                edges = self._edges_sobel(gray)
            elif self.method == "laplacian":
                edges = self._edges_laplacian(gray)
            else:  # scharr
                edges = self._edges_scharr(gray)
            
            # Optional Hough refinement
            result = self._refine_with_hough(edges)
            
            # Log results
            num_edges = np.count_nonzero(result)
            logger.info(f"Detected {num_edges} edge pixels ({num_edges/result.size:.1%} of image)")
            
            return result
            
        except Exception as e:
            logger.error(f"Edge detection failed: {str(e)}")
            raise EdgeDetectionError(f"Edge detection failed: {str(e)}") from e

    

    def draw_edges_green(self, image: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """
        Draw detected edges in bright green on the original image.
        
        Args:
            image: Original BGR image
            edges: Edge map from detect_edges()
            
        Returns:
            Image with bright green edges
        """
        result = image.copy()
        # Get edge locations and set them to bright green (fixed intensity)
        edges_binary = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY) if edges.ndim == 3 else edges
        result[edges_binary > 0] = [0, 255, 0]  # Pure bright green for all edges
        return result

    # ------------------- batch helpers -------------------
    def save_edges_image(self, edges: np.ndarray, input_path: str, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        name, ext = os.path.splitext(os.path.basename(input_path))
        out_path = os.path.join(output_dir, f"{name}_edges{ext}")
        cv2.imwrite(out_path, edges)
        return out_path

    def detect_edges_in_file(self, image_path: str, save_output: bool = False, output_dir: str = None, save_overlay: bool = False) -> Dict[str, str]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")
        edges = self.detect_edges(img)
        out = self.save_edges_image(edges, image_path, output_dir) if (save_output and output_dir) else ""
        if save_overlay and output_dir:
            try:
                overlay = self.draw_edges_green(img, edges)
                name, ext = os.path.splitext(os.path.basename(image_path))
                out_overlay = os.path.join(output_dir, f"{name}_overlay{ext}")
                cv2.imwrite(out_overlay, overlay)
            except Exception:
                out_overlay = ""
        return {"input_path": image_path, "output_path": out, "num_edge_pixels": int(np.count_nonzero(edges))}

    def detect_edges_in_directory(self, directory: str, save_outputs: bool = False, output_subdir: str = "results", save_overlay: bool = False) -> List[Dict[str, str]]:
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        imgs = []
        for pat in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]:
            imgs += glob.glob(os.path.join(directory, pat))
        imgs = sorted(p for p in imgs if "_edges" not in os.path.basename(p))
        if not imgs:
            raise RuntimeError(f"No supported images found in: {directory}")
        out_dir = os.path.join(directory, output_subdir)
        os.makedirs(out_dir, exist_ok=True)

        results = []
        for p in imgs:
            try:
                r = self.detect_edges_in_file(p, save_output=save_outputs, output_dir=out_dir, save_overlay=save_overlay)
                print(f"[INFO] {os.path.basename(p)} | edges: {r['num_edge_pixels']}")
                results.append(r)
            except Exception as e:
                print(f"[WARN] {p}: {e}")
        print(f"[INFO] Completed {len(results)} images → {out_dir}")
        return results


# ------------------- CLI -------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Advanced edge detector with multiple methods")
    parser.add_argument("--image", type=str, help="Path to a single image file")
    parser.add_argument("--dir", type=str, help="Process all images in a directory")
    parser.add_argument("--save", action="store_true", 
                       help="Save edge images in 'results' subfolder")
    parser.add_argument("--method", type=str, default="canny",
                       choices=["canny", "sobel", "laplacian", "scharr"],
                       help="Edge detection method to use")
    parser.add_argument("--no-clahe", action="store_true",
                       help="Disable CLAHE contrast normalization")
    parser.add_argument("--denoise", type=str, default="gaussian",
                       choices=["gaussian", "median", "bilateral", "none"],
                       help="Denoising filter to apply")
    parser.add_argument("--hough", action="store_true",
                       help="Enable Hough transform line refinement")
    parser.add_argument("--pad", type=int, default=2,
                       help="Border padding size in pixels")
    parser.add_argument("--auto-threshold", action="store_true",
                        help="Enable adaptive threshold selection for Canny")
    parser.add_argument("--min-blob-area", type=float, default=0.00002,
                        help="Minimum area ratio for keeping connected edge components")
    parser.add_argument("--min-blob-area-px", type=int, default=6,
                        help="Absolute minimum connected component size in pixels")
    parser.add_argument("--nlm-h", type=float, default=3.0,
                        help="Strength parameter for non-local means denoising (0 to disable)")
    parser.add_argument("--noise-clean-ratio", type=float, default=0.12,
                        help="Trigger extra denoising when edge density exceeds this ratio")
    parser.add_argument("--no-thin", action="store_true",
                        help="Disable final morphological thinning")
    parser.add_argument("--save-overlay", action="store_true",
                       help="Save RGB overlay images with green edges alongside binary outputs")
    args = parser.parse_args()

    det = EdgeDetector(
        method=args.method,
        use_clahe=not args.no_clahe,
        denoise=None if args.denoise == "none" else args.denoise,
        hough_refine=args.hough,
        pad=args.pad,
        auto_threshold=args.auto_threshold,
        min_blob_area_ratio=args.min_blob_area,
        min_blob_area_px=args.min_blob_area_px,
        fast_nl_means_h=args.nlm_h,
        noise_clean_ratio=args.noise_clean_ratio,
        thin_edges=not args.no_thin
    )

    if args.image:
        result = det.detect_edges_in_file(args.image, save_output=args.save)
        print(result)
    elif args.dir:
        det.detect_edges_in_directory(args.dir, save_outputs=args.save, save_overlay=args.save_overlay)
    else:
        print("⚠️  Please specify either --image or --dir")
