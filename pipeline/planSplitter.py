
import PIL
from PIL.Image import Image as PILImage
from pathlib import Path
import numpy as np
import json
import cv2
from typing import Tuple, Optional, List, Dict, Any
import logging
from process import PDFProcessor, PageContent

import base64

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
# KEY keywords to find the leftmost one
KEY_KEYWORDS = ['LEGENDE', 'ARCHITEKT', 'BAUHERR', 'PROJEKT', 'WERKPLANUNG']
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
PRIMARY_KEYWORDS = ['ARCHITEKT', 'ARCHITECT', 'BAUHERR', 'BAUHER', 'LEGENDE', 'LEGEND', 'PROJEKT', 'WERKPLANUNG']
class PageComponents:
    def __init__(self, 
                 pageno : int, 
                 infoPanel : PILImage | None = None, 
                 plan : PILImage | None = None,
                 json : str | None = None,
                 openAIGrouping : Dict[str, Any] | None = None):
        self.pageno = pageno
        self.infoPanel = infoPanel
        self.plan = plan
        self.json = json
        self.openAIGrouping = openAIGrouping 

class ProcessedPDF:
    def __init__(
        self,
       
        processedPages : list[PageComponents] | None = None
    ):  
       
        self.processedPages = processedPages

class PlanSplitter:
    def __init__(self, pdfProcessor: PDFProcessor):
        self.pdfProcessor = pdfProcessor
       
    def split_plan(self):   
        for pageContent in self.pdfProcessor.pages_content:
            page_image = pageContent.page_image
            split_x = self.find_split_point(pageContent = pageContent)
            pageContent.info_panel = page_image.crop((split_x, 0, page_image.width, page_image.height))
            pageContent.split_x = split_x
            #plan, x , y = self.crop_to_largest_connected_component(np.array(page_image.crop((0, 0, split_x, page_image.height))))
            x = 0
            y = 0
            plan = np.array(page_image.crop((0, 0, split_x, page_image.height)))
            #print("cropped x,y:", x,y)
            #plan = cv2.cvtColor(plan, cv2.COLOR_BGR2RGB)
            pageContent.plan = PIL.Image.fromarray(plan)
            pageContent.offset = (x, y)

    def find_split_point(self, pageContent: PageContent) -> int :
        """
        Find the optimal split point using a two-step approach:
        1. First, locate the info panel region using keywords (ARCHITEKT, BAUHERR, LEGENDE)
        2. Then, find the best existing vertical separator line to the LEFT of the keywords
        
        Returns x-coordinate for the split (snapped to an existing line when possible).
        """
        height = pageContent.page_image.height
        width = pageContent.page_image.width
        # STEP 1: Locate info panel using keywords
        #print("page content text:", pageContent.text)
        keyword_boundary, keyword_confidence, leftmost_keyword_x = self.find_info_panel_boundary_by_keywords(extractedTextJson= pageContent.text, pageWidth= width)

        if keyword_boundary and keyword_confidence >= 3.0:
            ratio = keyword_boundary / width
            
            
            # Use the leftmost keyword position if available (more precise)
            search_from = leftmost_keyword_x if leftmost_keyword_x else keyword_boundary
            #print("search from:", search_from) 
            # STEP 2: Find the NEAREST separator line to the LEFT of the leftmost keyword
            separator = self.find_separator_line(np.array(pageContent.page_image), int(search_from))
            
            if separator:
                sep_x, sep_score = separator
                sep_ratio = sep_x / width
                
                # Validate: separator should be to the left of the leftmost keyword
                if sep_x < search_from:

                    return sep_x
            
            # Fallback to keyword boundary with small margin

            return max(0, search_from - 5)
        
        # Try to find separator line assuming info panel is around 80% mark
        estimated_info_start = int(width * 0.82)
        separator = self.find_separator_line(np.array(pageContent.page_image), estimated_info_start)
        if separator:
            sep_x, sep_score = separator
            if sep_score > 0.15:  # Reasonable coverage
                return sep_x
        
        # Try standard vertical line detection
        vertical_lines = self.detect_vertical_lines(np.array(pageContent.page_image), min_length_ratio=0.4)
        valid_lines = [x for x in vertical_lines if 0.60 <= x/width <= 0.88]
        
        if valid_lines:
            # Pick the rightmost valid line (closest to info panel)
            best_line = max(valid_lines)
            return best_line
        
        # Last resort fallback
        fallback = int(width * 0.78)
        return fallback
       
    def find_info_panel_boundary_by_keywords(self, extractedTextJson: Any, pageWidth: int):
                
            
            right_boundary = pageWidth * 0.60

            # Find all key keywords with their exact positions
            key_keyword_positions = []  
            primary_x_coords = []
            secondary_x_coords = []
           # print(extractedTextJson)
            for item in extractedTextJson["text_regions"]:
                x1 = item["image_coordinates"]["x0"]
                if (x1 < right_boundary): 
                    continue
                y1 = item["image_coordinates"]["y0"]
                x2 = item["image_coordinates"]["x1"]
                y2 = item["image_coordinates"]["y1"]
                for keyword in KEY_KEYWORDS:
                    if (keyword in item["text"]):
                        key_keyword_positions.append((x1, keyword))
                        logger.info(f"Found KEY keyword '{keyword}' at x={x1} " )
                        break
                # Also track primary keywords for confidence
                for keyword in PRIMARY_KEYWORDS:
                    if keyword in item["text"]:
                        primary_x_coords.append(x1)
                        break
                    
                else:
                    for keyword in INFO_KEYWORDS:
                        if keyword in item["text"]:
                            secondary_x_coords.append(x1)
                            break
        
            # Find the LEFTMOST key keyword
            leftmost_keyword_x = None
            if key_keyword_positions:
                leftmost_entry = min(key_keyword_positions, key=lambda x: x[0])
                leftmost_keyword_x = leftmost_entry[0]
                logger.info(f"LEFTMOST key keyword is '{leftmost_entry[1]}' at x={leftmost_keyword_x} ")
                
            
            # Determine boundary and confidence
            if primary_x_coords:
                leftmost = min(primary_x_coords)
                confidence = 5.0
                return max(0, leftmost - 15), confidence, leftmost_keyword_x
            elif secondary_x_coords:
                leftmost = min(secondary_x_coords)
                confidence = 2.5
                return max(0, leftmost - 10), confidence, leftmost_keyword_x
        
            return None, 0.0, leftmost_keyword_x        

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
        logger.info(f"Phase 2: Searching WIDER zone from x={wide_right} to x={wide_left}")
        if result:
            return result
        
        # PHASE 2: Wider search zone
        wide_right = keyword_x
        wide_left = max(int(width * 0.50), keyword_x - int(width * 0.25))
        
        
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
        #print("search limit:", search_limit)
        #print("keyword x:", keyword_x) 
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
                return x, max(coverage, max_run_ratio)
        
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

    def crop_to_largest_connected_component(self, image):
        """
        Process an image by converting to grayscale, eroding, thresholding,
        extracting the largest connected component, and cropping to its bounding box.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image (can be color or grayscale)
        
        Returns:
        --------
        numpy.ndarray
            Cropped original image containing only the bounding box of the largest component
        """
        
        # Step 1: Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 2: Erode with 5x5 kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        eroded = cv2.erode(gray, kernel, iterations=1)
        
        # Step 3: Threshold with binary inverse and Otsu
        _, threshold = cv2.threshold(
            eroded, 
            0, 
            255, 
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # Step 4: Extract the largest connected component
        # Use connected components with statistics
        num_components, labels, stats, centroids = cv2.connectedComponentsWithStats(
            threshold, 
            connectivity=8
        )
        
        if num_components <= 1:
            # No components found (only background), return original image
            return image.copy()
        
        # Find the largest component (skip background which is label 0)
        largest_component_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        
        # Get the bounding box of the largest component
        x, y, width, height, area = stats[largest_component_label]
        
        # Step 5: Crop the original image to the bounding box
        cropped = image[y:y+height, x:x+width]
        
        return cropped, x , y
    def make_json_serializable(self, obj):
        """
        Convert non-JSON-serializable objects (like bytes) to JSON-serializable format.
        Recursively processes dictionaries and lists.
        """
        if isinstance(obj, bytes):
            # Convert bytes to base64 string
            return base64.b64encode(obj).decode('utf-8')
        elif isinstance(obj, dict):
            # Recursively process dictionary values
            return {key: self.make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            # Recursively process list items
            return [self.make_json_serializable(item) for item in obj]
        else:
            # Return other types as-is
            return obj
    def Visualize(self, output_folder="output"):
        output_dir = Path(output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        for page in self.pdfProcessor.pages_content:
            page_dir = output_dir / f"page_{page.page_number:03d}"
            page_dir.mkdir(exist_ok=True)

            page.page_image.save(page_dir / "page_image.png")
            
            # Convert to JSON-serializable format (handles bytes objects)
            serializable_text = self.make_json_serializable(page.text)
            
            with open(page_dir / "page_text.json", "w", encoding="utf-8") as f:
                json.dump(serializable_text, f, indent=4, ensure_ascii=False)

            logger.info(f"Saved page {page.page_number}")
            
            # Save extracted images
            for idx, img in enumerate(page.images):
                img.save(page_dir / f"extracted_image_{idx + 1:02d}.png")
            page.info_panel.save(page_dir / "info_panel.png")
            page.plan.save(page_dir / "plan.png")
        


def main():
    """Command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Split architectural plans into floor plan and info panel")
    parser.add_argument("input", help="Input PDF file")
    parser.add_argument("-o", "--output", help="Output directory (default: same as input)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    try:
        pdfProcessor = PDFProcessor(input_path)
        pdfProcessor.process()
        planSplitter = PlanSplitter(pdfProcessor= pdfProcessor)
        planSplitter.split_plan()
        planSplitter.Visualize()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise e

if __name__ == "__main__":
    main()