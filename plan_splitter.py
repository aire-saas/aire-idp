"""
Architectural Plan Splitter
===========================
Splits architectural PDFs into two parts:
- Left: The actual floor plan
- Right: Info panel (legend, architect, bauherr, project info)

Uses multiple detection methods to find the split point reliably.
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import cv2
from typing import Tuple, Optional, List
import logging

# OCR support for scanned PDFs
try:
    import pytesseract
    # Configure Tesseract path for Windows
    if sys.platform == 'win32':
        tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class PlanSplitter:
    """
    Splits architectural plans into floor plan and info panel.
    Uses multiple detection strategies for robustness.
    """
    
    # Keywords typically found in the info panel (German architectural terms)
    # Including common variations and misspellings
    INFO_KEYWORDS = [
        # Primary identifiers (highest confidence)
        'ARCHITEKT', 'ARCHITECT', 'ARCHITEKTEN', 'ARCHITEKTUR',
        'BAUHERR', 'BAUHER', 'BAUHERRN', 'BAUHERREN',
        'LEGENDE', 'LEGEND',
        # Secondary identifiers
        'PROJEKT', 'PROJECT', 'BAUVORHABEN', 'VORHABEN',
        'WERKPLANUNG', 'PLANUNG', 'PLANUNGSSTAND',
        'GEMARKUNG', 'GEMEINDE', 'GESAMTFLÄCHE',
        'MASSTAB', 'MASSSTAB', 'MAßSTAB', 'M 1:',
        'DATUM', 'DATE', 'ERSTELLT',
        'GEZEICHNET', 'GEZ.', 'GEPR.', 'GEPRÜFT', 'GEPRUFT',
        'INDEX', 'ÄNDERUNG', 'ANDERUNG', 'REVISION',
        'FLUR', 'FLURSTÜCK', 'FLURSTUCK', 'FLURST',
        'PLANINHALT', 'PLANNUMMER', 'PLAN-NR', 'PLANKOPF',
        'ANSICHT', 'SCHNITT', 'FASSADE',
        'BÄUERLE', 'BAUERLE', 'BAEUERLE',
        # Table headers often found
        'BAUHERR:', 'ARCHITEKT:', 'PROJEKT:', 'PLANER',
        'STATIK', 'TRAGWERK', 'HAUSTECHNIK',
        'FREIGABE', 'GENEHMIGUNG', 'STATUS'
    ]
    
    # High-priority keywords that almost always indicate info panel
    PRIMARY_KEYWORDS = ['ARCHITEKT', 'ARCHITECT', 'BAUHERR', 'BAUHER', 'LEGENDE', 'LEGEND']
    
    def __init__(self, dpi: int = 150):
        """
        Initialize the splitter.
        
        Args:
            dpi: Resolution for PDF to image conversion. Higher = more accurate but slower.
        """
        self.dpi = dpi
        
    def pdf_to_image(self, pdf_path: str) -> np.ndarray:
        """Convert first page of PDF to numpy array (BGR format for OpenCV)."""
        doc = fitz.open(pdf_path)
        page = doc[0]
        
        # Render at specified DPI
        zoom = self.dpi / 72  # 72 is default PDF DPI
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to numpy array
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        # Convert RGB to BGR for OpenCV
        if pix.n == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:  # RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
        doc.close()
        return img
    
    def detect_vertical_lines(self, img: np.ndarray, min_length_ratio: float = 0.5) -> List[int]:
        """
        Detect strong vertical lines in the image.
        Returns x-coordinates of detected lines, sorted from right to left.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Only search in right 50% of image
        search_start = width // 2
        roi = gray[:, search_start:]
        
        # Edge detection
        edges = cv2.Canny(roi, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        min_line_length = int(height * min_length_ratio)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                                minLineLength=min_line_length, maxLineGap=20)
        
        vertical_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is nearly vertical (within 2 degrees)
                if abs(x2 - x1) < 5:  # Nearly vertical
                    x_pos = (x1 + x2) // 2 + search_start  # Adjust for ROI offset
                    vertical_lines.append(x_pos)
        
        # Remove duplicates (lines within 20 pixels of each other)
        vertical_lines = sorted(set(vertical_lines), reverse=True)
        filtered = []
        for x in vertical_lines:
            if not filtered or abs(x - filtered[-1]) > 20:
                filtered.append(x)
                
        return filtered
    
    def find_separator_line(self, img: np.ndarray, keyword_x: int) -> Optional[Tuple[int, float]]:
        """
        Find the nearest vertical separator line to the LEFT of the keyword region.
        Scans from the keyword boundary leftward to find the FIRST significant vertical line.
        
        Args:
            img: Input image
            keyword_x: X coordinate where keywords start (info panel left edge)
            
        Returns:
            Tuple of (x-coordinate, strength score) or None if not found
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # PHASE 0: Direct pixel scan - finds the VERY FIRST vertical edge
        # This is most reliable for finding the nearest line
        logger.info(f"Phase 0: Direct scan for nearest edge from keywords at x={keyword_x}")
        result = self._scan_for_nearest_vertical_edge(gray, keyword_x, min_coverage=0.08)
        if result:
            return result
        
        # PHASE 1: Hough detection in narrow zone (backup)
        narrow_right = keyword_x
        narrow_left = max(int(width * 0.60), keyword_x - int(width * 0.10))
        
        logger.info(f"Phase 1: Hough detection in NARROW zone from x={narrow_right} to x={narrow_left}")
        
        result = self._detect_lines_in_zone(gray, narrow_left, narrow_right, height, width, sensitive=True)
        if result:
            return result
        
        # PHASE 2: Wider search zone
        wide_right = keyword_x
        wide_left = max(int(width * 0.50), keyword_x - int(width * 0.25))
        
        logger.info(f"Phase 2: Searching WIDER zone from x={wide_right} to x={wide_left}")
        
        result = self._detect_lines_in_zone(gray, wide_left, wide_right, height, width, sensitive=False)
        if result:
            return result
        
        # Fallback: intensity-based detection
        return self._find_nearest_edge_by_intensity(gray, wide_left, wide_right, keyword_x)
    
    def _detect_lines_in_zone(self, gray: np.ndarray, search_left: int, search_right: int, 
                               height: int, width: int, sensitive: bool = False) -> Optional[Tuple[int, float]]:
        """
        Detect vertical lines in a specific zone.
        
        Args:
            sensitive: If True, use lower thresholds to detect fainter/shorter lines
        """
        roi = gray[:, search_left:search_right]
        roi_width = search_right - search_left
        
        if roi_width <= 10:
            return None
        
        # Edge detection - lower threshold if sensitive mode
        canny_low = 20 if sensitive else 30
        canny_high = 80 if sensitive else 100
        edges = cv2.Canny(roi, canny_low, canny_high, apertureSize=3)
        
        # Detect vertical lines with various sensitivities
        all_lines = []
        
        # More sensitive detection for lines close to keywords
        if sensitive:
            min_ratios = [0.5, 0.3, 0.2, 0.1, 0.05]  # Accept very short lines
            hough_threshold = 25  # Lower threshold
        else:
            min_ratios = [0.6, 0.4, 0.25, 0.15]
            hough_threshold = 40
        
        for min_ratio in min_ratios:
            min_length = int(height * min_ratio)
            max_gap = int(height * 0.20)  # Allow larger gaps for dashed lines
            
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=hough_threshold,
                                    minLineLength=min_length, maxLineGap=max_gap)
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if abs(x2 - x1) < 12:  # Nearly vertical (allow slight angle)
                        x_pos = (x1 + x2) // 2 + search_left
                        line_length = abs(y2 - y1)
                        all_lines.append((x_pos, line_length))
        
        if not all_lines:
            return None
        
        # Group lines that are close together (within 15 pixels)
        all_lines.sort(key=lambda x: -x[0])  # Sort RIGHT to LEFT (descending)
        
        groups = []
        current_group = [all_lines[0]]
        
        for i in range(1, len(all_lines)):
            if current_group[-1][0] - all_lines[i][0] < 15:
                current_group.append(all_lines[i])
            else:
                groups.append(current_group)
                current_group = [all_lines[i]]
        groups.append(current_group)
        
        # Find the RIGHTMOST valid group (nearest to keywords)
        for group in groups:
            avg_x = sum(l[0] for l in group) / len(group)
            max_length = max(l[1] for l in group)
            total_length = sum(l[1] for l in group)
            num_segments = len(group)
            coverage = min(1.0, total_length / height)
            
            # In sensitive mode, accept weaker signals
            if sensitive:
                is_valid = (max_length > height * 0.15) or (coverage > 0.10) or (num_segments >= 2)
            else:
                is_valid = (max_length > height * 0.30) or (coverage > 0.20) or (num_segments >= 3)
            
            if is_valid:
                result_x = int(avg_x)
                logger.info(f"Found NEAREST separator line at x={result_x} ({result_x/width:.1%}), "
                           f"coverage={coverage:.1%}, segments={num_segments}, max_length={max_length/height:.1%}")
                return result_x, coverage
        
        return None
    
    def _find_nearest_edge_by_intensity(self, gray: np.ndarray, search_left: int, search_right: int, keyword_x: int) -> Optional[Tuple[int, float]]:
        """
        Fallback method: find the nearest vertical edge by analyzing column intensity changes.
        Scans from keyword_x leftward to find the first strong edge.
        """
        height = gray.shape[0]
        width = gray.shape[1]
        
        # Scan from RIGHT to LEFT (from keywords toward the plan)
        for x in range(search_right - 2, search_left, -1):
            # Calculate vertical edge strength at this column
            diff_left = np.abs(gray[:, x].astype(float) - gray[:, x-1].astype(float))
            diff_right = np.abs(gray[:, x].astype(float) - gray[:, x+1].astype(float))
            
            # Count strong edges (pixels with high gradient)
            strong_edges = np.sum((diff_left > 40) | (diff_right > 40))
            coverage = strong_edges / height
            
            # If we find a column with significant vertical edge, that's our separator
            if coverage > 0.15:  # At least 15% of height has strong edge
                logger.info(f"Found edge by intensity at x={x} (coverage: {coverage:.1%})")
                return x, coverage
        
        return None
    
    def _scan_for_nearest_vertical_edge(self, gray: np.ndarray, keyword_x: int, min_coverage: float = 0.08) -> Optional[Tuple[int, float]]:
        """
        Directly scan from keyword boundary leftward to find the FIRST vertical edge.
        This catches thin lines that Hough transform might miss.
        
        Args:
            gray: Grayscale image
            keyword_x: Starting x position (keyword boundary)
            min_coverage: Minimum fraction of height that must have edge pixels
        """
        height, width = gray.shape
        
        # Don't search more than 15% of width to the left
        search_limit = max(int(width * 0.60), keyword_x - int(width * 0.15))
        
        logger.info(f"Direct scan from x={keyword_x} leftward to x={search_limit}")
        
        # Calculate edge strength for each column
        # Using Sobel for better edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.abs(sobelx)
        
        # Scan from RIGHT to LEFT
        for x in range(keyword_x - 5, search_limit, -1):
            if x < 1 or x >= width - 1:
                continue
            
            # Get edge strength at this column
            col_edge = sobelx[:, x]
            
            # Count pixels with strong vertical edge
            threshold = 50  # Edge strength threshold
            strong_edge_pixels = np.sum(col_edge > threshold)
            coverage = strong_edge_pixels / height
            
            # Also check if there's a continuous vertical segment
            binary_edge = (col_edge > threshold).astype(np.uint8)
            
            # Find runs of edge pixels
            runs = []
            in_run = False
            run_start = 0
            for y in range(height):
                if binary_edge[y] and not in_run:
                    in_run = True
                    run_start = y
                elif not binary_edge[y] and in_run:
                    in_run = False
                    runs.append(y - run_start)
            if in_run:
                runs.append(height - run_start)
            
            # Check if we have a significant vertical line
            max_run = max(runs) if runs else 0
            max_run_ratio = max_run / height
            
            # Accept if: decent coverage OR a long continuous segment
            if coverage > min_coverage or max_run_ratio > 0.25:
                logger.info(f"Direct scan found edge at x={x} ({x/width:.1%}), "
                           f"coverage={coverage:.1%}, longest_segment={max_run_ratio:.1%}")
                return x, max(coverage, max_run_ratio)
        
        return None
    
    def analyze_column_density(self, img: np.ndarray, num_columns: int = 100) -> Tuple[np.ndarray, int]:
        """
        Analyze content density in vertical columns.
        Returns density array and suggested split point.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Threshold to get binary image (content vs background)
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Calculate column width
        col_width = width // num_columns
        densities = []
        
        for i in range(num_columns):
            start_x = i * col_width
            end_x = min((i + 1) * col_width, width)
            col = binary[:, start_x:end_x]
            density = np.sum(col) / (col.shape[0] * col.shape[1] * 255)
            densities.append(density)
        
        densities = np.array(densities)
        
        # Look for significant change in density pattern in right portion
        # The info panel typically has different density characteristics
        right_portion = densities[num_columns // 2:]
        
        # Calculate rolling variance to find transition
        window = 5
        variances = []
        for i in range(len(right_portion) - window):
            var = np.var(right_portion[i:i+window])
            variances.append(var)
        
        # Find the leftmost significant variance change
        if variances:
            threshold = np.mean(variances) + np.std(variances)
            for i, var in enumerate(variances):
                if var > threshold:
                    split_col = (num_columns // 2 + i) * col_width
                    return densities, split_col
        
        # Fallback: use 75% of width
        return densities, int(width * 0.75)
    
    def detect_text_regions(self, pdf_path: str) -> List[Tuple[float, float, float, float, str]]:
        """
        Extract text regions from PDF with their bounding boxes.
        Returns list of (x0, y0, x1, y1, text) tuples.
        Falls back to OCR for scanned PDFs without embedded text.
        """
        doc = fitz.open(pdf_path)
        page = doc[0]
        
        # Get page dimensions
        page_width = page.rect.width
        page_height = page.rect.height
        
        text_regions = []
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if block["type"] == 0:  # Text block
                bbox = block["bbox"]
                text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text += span.get("text", "") + " "
                text = text.strip().upper()
                if text:
                    text_regions.append((bbox[0], bbox[1], bbox[2], bbox[3], text))
        
        doc.close()
        
        # If no text found, try OCR
        if not text_regions and OCR_AVAILABLE:
            logger.info("No embedded text found, using OCR...")
            text_regions = self._detect_text_with_ocr(pdf_path, page_width, page_height)
        elif not text_regions and not OCR_AVAILABLE:
            logger.warning("No embedded text found and OCR not available. Install pytesseract for OCR support.")
        
        return text_regions, page_width, page_height
    
    def _detect_text_with_ocr(self, pdf_path: str, page_width: float, page_height: float) -> List[Tuple[float, float, float, float, str]]:
        """
        Use OCR to detect text in scanned PDFs.
        Only scans the right portion of the image where the info panel is expected.
        """
        if not OCR_AVAILABLE:
            return []
        
        # Convert PDF to image
        img = self.pdf_to_image(pdf_path)
        height, width = img.shape[:2]
        
        # Scale factors
        scale_x = page_width / width
        scale_y = page_height / height
        
        # Only OCR the right 45% of the image (where info panel is)
        right_start = int(width * 0.55)
        roi = img[:, right_start:]
        
        # Convert to RGB for pytesseract
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Run OCR with bounding box data
        try:
            ocr_data = pytesseract.image_to_data(roi_rgb, lang='deu+eng', output_type=pytesseract.Output.DICT)
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return []
        
        text_regions = []
        n_boxes = len(ocr_data['text'])
        
        for i in range(n_boxes):
            text = ocr_data['text'][i].strip().upper()
            conf = int(ocr_data['conf'][i])
            
            # Only include confident detections with actual text
            if conf > 30 and text and len(text) > 2:
                x = ocr_data['left'][i] + right_start  # Adjust for ROI offset
                y = ocr_data['top'][i]
                w = ocr_data['width'][i]
                h = ocr_data['height'][i]
                
                # Convert to PDF coordinates
                x0 = x * scale_x
                y0 = y * scale_y
                x1 = (x + w) * scale_x
                y1 = (y + h) * scale_y
                
                text_regions.append((x0, y0, x1, y1, text))
                
        logger.info(f"OCR detected {len(text_regions)} text regions")
        return text_regions
    
    def find_info_panel_boundary_by_keywords(self, pdf_path: str, img_width: int) -> Tuple[Optional[int], float, Optional[int]]:
        """
        Find the left boundary of the info panel using keyword detection.
        Only looks for keywords in the RIGHT portion of the document.
        
        Returns:
            Tuple of (x-coordinate in image space, confidence score, leftmost_keyword_x)
            The leftmost_keyword_x is the exact x position of the leftmost key keyword found.
        """
        text_regions, page_width, page_height = self.detect_text_regions(pdf_path)
        
        # Scale factor from PDF coordinates to image coordinates
        scale = img_width / page_width
        
        # Only consider text in the right 40% of the document
        right_boundary = page_width * 0.60
        
        # KEY keywords to find the leftmost one
        KEY_KEYWORDS = ['LEGENDE', 'ARCHITEKT', 'BAUHERR', 'PROJEKT', 'WERKPLANUNG']
        
        # Find all key keywords with their exact positions
        key_keyword_positions = []  # List of (x, keyword_name)
        primary_x_coords = []
        secondary_x_coords = []
        
        for x0, y0, x1, y1, text in text_regions:
            # Skip if not in the right portion
            if x0 < right_boundary:
                continue
                
            img_x0 = int(x0 * scale)
            
            # Check for KEY keywords (to find the leftmost one)
            for keyword in KEY_KEYWORDS:
                if keyword in text:
                    key_keyword_positions.append((img_x0, keyword))
                    logger.info(f"Found KEY keyword '{keyword}' at x={img_x0} ({img_x0/img_width:.1%})")
                    break
            
            # Also track primary keywords for confidence
            for keyword in self.PRIMARY_KEYWORDS:
                if keyword in text:
                    primary_x_coords.append(img_x0)
                    break
            else:
                for keyword in self.INFO_KEYWORDS:
                    if keyword in text:
                        secondary_x_coords.append(img_x0)
                        break
        
        # Find the LEFTMOST key keyword
        leftmost_keyword_x = None
        if key_keyword_positions:
            leftmost_entry = min(key_keyword_positions, key=lambda x: x[0])
            leftmost_keyword_x = leftmost_entry[0]
            logger.info(f"LEFTMOST key keyword is '{leftmost_entry[1]}' at x={leftmost_keyword_x} ({leftmost_keyword_x/img_width:.1%})")
        
        # Determine boundary and confidence
        if primary_x_coords:
            leftmost = min(primary_x_coords)
            confidence = 5.0
            logger.info(f"Using primary keyword boundary at x={leftmost}")
            return max(0, leftmost - 15), confidence, leftmost_keyword_x
        elif secondary_x_coords:
            leftmost = min(secondary_x_coords)
            confidence = 2.5
            return max(0, leftmost - 10), confidence, leftmost_keyword_x
        
        return None, 0.0, leftmost_keyword_x
    
    def detect_frame_boundary(self, img: np.ndarray) -> Optional[int]:
        """
        Detect the drawing frame boundary.
        Many architectural plans have a frame around the drawing area.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Look for vertical edges using Sobel
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.abs(sobelx)
        
        # Analyze right portion
        search_start = int(width * 0.5)
        roi = sobelx[:, search_start:]
        
        # Sum along vertical axis to find strong vertical edges
        col_sums = np.sum(roi, axis=0)
        
        # Find peaks (strong vertical edges)
        threshold = np.mean(col_sums) + 2 * np.std(col_sums)
        peaks = np.where(col_sums > threshold)[0]
        
        if len(peaks) > 0:
            # Take the leftmost significant peak in the search region
            return peaks[0] + search_start
        
        return None
    
    def find_split_point(self, pdf_path: str, img: np.ndarray) -> int:
        """
        Find the optimal split point using a two-step approach:
        1. First, locate the info panel region using keywords (ARCHITEKT, BAUHERR, LEGENDE)
        2. Then, find the best existing vertical separator line to the LEFT of the keywords
        
        Returns x-coordinate for the split (snapped to an existing line when possible).
        """
        height, width = img.shape[:2]
        
        # STEP 1: Locate info panel using keywords
        keyword_boundary, keyword_confidence, leftmost_keyword_x = self.find_info_panel_boundary_by_keywords(pdf_path, width)
        
        if keyword_boundary and keyword_confidence >= 3.0:
            ratio = keyword_boundary / width
            logger.info(f"Keywords indicate info panel starts at x={keyword_boundary} ({ratio:.1%} of width)")
            
            # Use the leftmost keyword position if available (more precise)
            search_from = leftmost_keyword_x if leftmost_keyword_x else keyword_boundary
            if leftmost_keyword_x:
                logger.info(f"Using LEFTMOST keyword position x={leftmost_keyword_x} ({leftmost_keyword_x/width:.1%}) for line search")
            
            # STEP 2: Find the NEAREST separator line to the LEFT of the leftmost keyword
            separator = self.find_separator_line(img, search_from)
            
            if separator:
                sep_x, sep_score = separator
                sep_ratio = sep_x / width
                
                # Validate: separator should be to the left of the leftmost keyword
                if sep_x < search_from:
                    logger.info(f"✓ Snapping to NEAREST separator line at x={sep_x} ({sep_ratio:.1%} of width)")
                    return sep_x
                else:
                    logger.warning(f"Separator at x={sep_x} is not to the left of leftmost keyword at x={search_from}, using keyword boundary")
            else:
                logger.warning("No separator line found, using keyword boundary directly")
            
            # Fallback to keyword boundary with small margin
            return max(0, search_from - 5)
        
        # STEP 1b: No strong keyword detection - fall back to other methods
        logger.warning("No strong keyword detection, falling back to line detection")
        
        # Try to find separator line assuming info panel is around 80% mark
        estimated_info_start = int(width * 0.82)
        separator = self.find_separator_line(img, estimated_info_start)
        if separator:
            sep_x, sep_score = separator
            if sep_score > 0.15:  # Reasonable coverage
                logger.info(f"Using separator line at x={sep_x} ({sep_x/width:.1%} of width)")
                return sep_x
        
        # Try standard vertical line detection
        vertical_lines = self.detect_vertical_lines(img, min_length_ratio=0.4)
        valid_lines = [x for x in vertical_lines if 0.60 <= x/width <= 0.88]
        
        if valid_lines:
            # Pick the rightmost valid line (closest to info panel)
            best_line = max(valid_lines)
            logger.info(f"Using vertical line at x={best_line} ({best_line/width:.1%} of width)")
            return best_line
        
        # Last resort fallback
        fallback = int(width * 0.78)
        logger.warning(f"Using fallback split at x={fallback} (78% of width)")
        return fallback
    
    def split_plan(self, pdf_path: str, output_dir: str = None) -> Tuple[str, str]:
        """
        Split a PDF plan into floor plan and info panel images.
        
        Args:
            pdf_path: Path to the input PDF
            output_dir: Directory for output images. If None, uses same directory as input.
            
        Returns:
            Tuple of (floor_plan_path, info_panel_path)
        """
        pdf_path = Path(pdf_path)
        if output_dir is None:
            output_dir = pdf_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = pdf_path.stem
        
        logger.info(f"Processing: {pdf_path.name}")
        
        # Convert PDF to image
        img = self.pdf_to_image(str(pdf_path))
        height, width = img.shape[:2]
        logger.info(f"Image size: {width}x{height}")
        
        # Find split point
        split_x = self.find_split_point(str(pdf_path), img)
        
        # Split the image
        floor_plan = img[:, :split_x]
        info_panel = img[:, split_x:]
        
        # Save outputs
        floor_plan_path = output_dir / f"{base_name}_floor_plan.png"
        info_panel_path = output_dir / f"{base_name}_info_panel.png"
        
        cv2.imwrite(str(floor_plan_path), floor_plan)
        cv2.imwrite(str(info_panel_path), info_panel)
        
        logger.info(f"Saved floor plan: {floor_plan_path}")
        logger.info(f"Saved info panel: {info_panel_path}")
        
        return str(floor_plan_path), str(info_panel_path)
    
    def split_plan_with_preview(self, pdf_path: str, output_dir: str = None) -> Tuple[str, str, str]:
        """
        Split a plan and also save a preview image showing the split line.
        
        Returns:
            Tuple of (floor_plan_path, info_panel_path, preview_path)
        """
        pdf_path = Path(pdf_path)
        if output_dir is None:
            output_dir = pdf_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = pdf_path.stem
        
        logger.info(f"Processing: {pdf_path.name}")
        
        # Convert PDF to image
        img = self.pdf_to_image(str(pdf_path))
        height, width = img.shape[:2]
        logger.info(f"Image size: {width}x{height}")
        
        # Find split point
        split_x = self.find_split_point(str(pdf_path), img)
        
        # Create preview with split line
        preview = img.copy()
        cv2.line(preview, (split_x, 0), (split_x, height), (0, 0, 255), 3)
        
        # Split the image
        floor_plan = img[:, :split_x]
        info_panel = img[:, split_x:]
        
        # Save outputs
        floor_plan_path = output_dir / f"{base_name}_floor_plan.png"
        info_panel_path = output_dir / f"{base_name}_info_panel.png"
        preview_path = output_dir / f"{base_name}_preview.png"
        
        cv2.imwrite(str(floor_plan_path), floor_plan)
        cv2.imwrite(str(info_panel_path), info_panel)
        cv2.imwrite(str(preview_path), preview)
        
        logger.info(f"Saved floor plan: {floor_plan_path}")
        logger.info(f"Saved info panel: {info_panel_path}")
        logger.info(f"Saved preview: {preview_path}")
        
        return str(floor_plan_path), str(info_panel_path), str(preview_path)


def process_directory(input_dir: str, output_dir: str = None, dpi: int = 150):
    """Process all PDFs in a directory."""
    splitter = PlanSplitter(dpi=dpi)
    input_path = Path(input_dir)
    
    pdf_files = list(input_path.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    for pdf_file in pdf_files:
        try:
            splitter.split_plan_with_preview(str(pdf_file), output_dir)
        except Exception as e:
            logger.error(f"Failed to process {pdf_file.name}: {e}")


def main():
    """Command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Split architectural plans into floor plan and info panel")
    parser.add_argument("input", help="Input PDF file or directory")
    parser.add_argument("-o", "--output", help="Output directory (default: same as input)")
    parser.add_argument("--dpi", type=int, default=150, help="Resolution for processing (default: 150)")
    parser.add_argument("--preview", action="store_true", help="Generate preview images showing split line")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_dir():
        process_directory(str(input_path), args.output, args.dpi)
    elif input_path.is_file() and input_path.suffix.lower() == ".pdf":
        splitter = PlanSplitter(dpi=args.dpi)
        if args.preview:
            splitter.split_plan_with_preview(str(input_path), args.output)
        else:
            splitter.split_plan(str(input_path), args.output)
    else:
        logger.error(f"Invalid input: {args.input}")
        sys.exit(1)


if __name__ == "__main__":
    main()

