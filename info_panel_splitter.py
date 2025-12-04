"""
Info Panel Group Splitter
=========================
Splits the info panel image into individual groups (ARCHITEKT, BAUHERR, LEGENDE, etc.)
Uses hierarchical box detection to find group boundaries.
"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
import logging

# OCR support
try:
    import pytesseract
    if sys.platform == 'win32':
        tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class InfoPanelSplitter:
    """
    Splits info panel images into individual group sections.
    Uses hierarchical box detection to handle internal divisions.
    """
    
    # Keywords to identify groups
    GROUP_KEYWORDS = {
        'ARCHITEKT': ['ARCHITEKT', 'ARCHITECT', 'ARCHITEKTEN', 'PLANER'],
        'BAUHERR': ['BAUHERR', 'BAUHERRN', 'BAUHERREN', 'AUFTRAGGEBER'],
        'LEGENDE': ['LEGENDE', 'LEGEND', 'ZEICHENERKLÄRUNG', 'ZEICHENERKLARUNG'],
        'PROJEKT': ['PROJEKT', 'PROJECT', 'BAUVORHABEN', 'VORHABEN', 'OBJEKTBEZEICHNUNG'],
        'PLANINFO': ['PLANINHALT', 'PLANNUMMER', 'PLAN-NR', 'PLANKOPF', 'WERKPLANUNG', 'PLANSTAND'],
        'INDEX': ['INDEX', 'ÄNDERUNG', 'ANDERUNG', 'REVISION', 'ÄNDERUNGEN'],
        'LAGEPLAN': ['LAGEPLAN', 'ÜBERSICHT', 'UBERSICHT', 'LAGE'],
        'MASSSTAB': ['MASSTAB', 'MASSSTAB', 'MAßSTAB', 'SCALE'],
        'NORDPFEIL': ['NORD', 'NORTH', 'NORDPFEIL'],
    }
    
    def __init__(self, min_box_area_ratio: float = 0.003, merge_threshold: int = 3):
        """
        Initialize the splitter.
        
        Args:
            min_box_area_ratio: Minimum box area as ratio of total image area
            merge_threshold: Distance threshold for merging nearby boxes (smaller = less merging)
        """
        self.min_box_area_ratio = min_box_area_ratio
        self.merge_threshold = merge_threshold
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load an image from file."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return img
    
    def detect_lines(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect horizontal and vertical lines in the image.
        Returns binary masks for horizontal and vertical lines.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Threshold to get binary image
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width // 20, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height // 20))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        return horizontal_lines, vertical_lines
    
    def detect_boxes(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect rectangular boxes in the image.
        Returns list of (x, y, w, h) tuples.
        """
        height, width = img.shape[:2]
        min_area = int(height * width * self.min_box_area_ratio)
        
        # Get lines
        h_lines, v_lines = self.detect_lines(img)
        
        # Combine lines to find boxes
        combined = cv2.add(h_lines, v_lines)
        
        # Use smaller dilation to avoid over-connecting
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        combined = cv2.dilate(combined, kernel, iterations=1)
        
        # Find contours with hierarchy to understand nesting
        contours, hierarchy = cv2.findContours(combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out boxes that are too thin (likely just lines)
            if w < 50 or h < 40:
                continue
            
            # Filter out boxes that span almost the entire width (likely the outer frame)
            if w > width * 0.90:
                continue
            
            # Also filter boxes that span most of the height (outer frame)
            if h > height * 0.85:
                continue
            
            boxes.append((x, y, w, h))
        
        return boxes
    
    def detect_boxes_alternative(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Alternative box detection using edge detection and contours.
        Better for detecting individual group boxes.
        """
        height, width = img.shape[:2]
        min_area = int(height * width * self.min_box_area_ratio)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges slightly to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for contour in contours:
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's roughly rectangular (4 corners)
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                if area < min_area:
                    continue
                
                # Filter thin boxes and outer frame
                if w < 50 or h < 40:
                    continue
                if w > width * 0.90 or h > height * 0.85:
                    continue
                
                boxes.append((x, y, w, h))
        
        return boxes
    
    def boxes_overlap(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int], threshold: int = 0) -> bool:
        """Check if two boxes overlap or are within threshold distance."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Expand boxes by threshold
        x1 -= threshold
        y1 -= threshold
        w1 += 2 * threshold
        h1 += 2 * threshold
        
        # Check overlap
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)
    
    def box_contains(self, outer: Tuple[int, int, int, int], inner: Tuple[int, int, int, int], margin: int = 5) -> bool:
        """Check if outer box contains inner box."""
        ox, oy, ow, oh = outer
        ix, iy, iw, ih = inner
        
        return (ix >= ox - margin and 
                iy >= oy - margin and 
                ix + iw <= ox + ow + margin and 
                iy + ih <= oy + oh + margin)
    
    def merge_overlapping_boxes(self, boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Merge boxes that overlap or are very close together."""
        if not boxes:
            return []
        
        merged = list(boxes)
        changed = True
        
        while changed:
            changed = False
            new_merged = []
            used = set()
            
            for i, box1 in enumerate(merged):
                if i in used:
                    continue
                
                current = box1
                for j, box2 in enumerate(merged):
                    if i == j or j in used:
                        continue
                    
                    if self.boxes_overlap(current, box2, self.merge_threshold):
                        # Merge boxes
                        x1, y1, w1, h1 = current
                        x2, y2, w2, h2 = box2
                        
                        new_x = min(x1, x2)
                        new_y = min(y1, y2)
                        new_w = max(x1 + w1, x2 + w2) - new_x
                        new_h = max(y1 + h1, y2 + h2) - new_y
                        
                        current = (new_x, new_y, new_w, new_h)
                        used.add(j)
                        changed = True
                
                new_merged.append(current)
                used.add(i)
            
            merged = new_merged
        
        return merged
    
    def filter_nested_boxes(self, boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """
        Remove boxes that are contained within other boxes.
        Keeps only the outermost boxes (group boundaries).
        """
        if not boxes:
            return []
        
        # Sort by area (largest first)
        sorted_boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        
        outermost = []
        for box in sorted_boxes:
            is_nested = False
            for outer in outermost:
                if self.box_contains(outer, box):
                    is_nested = True
                    break
            
            if not is_nested:
                outermost.append(box)
        
        return outermost
    
    def detect_groups(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect group boxes using layout-based detection.
        
        Step 1: Detect sections based on visual layout (boxes, tables)
        Step 2: Keywords are used later for naming, not boundary detection
        
        Returns list of (x, y, w, h) for each group.
        """
        height, width = img.shape[:2]
        
        # PRIMARY: Layout-based section detection
        sections = self.detect_sections_by_layout(img)
        
        if len(sections) >= 2:
            logger.info(f"Layout detection found {len(sections)} sections")
            return sections
        
        # FALLBACK: Keyword-anchored segmentation  
        segments = self.segment_by_keywords(img)
        if len(segments) >= 2:
            logger.info(f"Keyword segmentation found {len(segments)} groups")
            return segments
        
        # LAST RESORT: Simple horizontal segmentation
        segments = self.segment_by_horizontal_lines(img)
        if len(segments) >= 2:
            return segments
        
        logger.warning("No groups detected")
        return []
    
    def detect_sections_by_layout(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect sections based on visual layout.
        - Identifies table regions (keep as single units)
        - Identifies graphic regions (plans, drawings)
        - Merges title boxes with their content below
        """
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Step 1: Detect horizontal lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(width * 0.7), 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)
        
        # Step 2: Find all horizontal line positions
        all_h_lines = self._find_all_horizontal_lines(h_lines, width, min_coverage=0.5)
        logger.info(f"Found {len(all_h_lines)} horizontal lines")
        
        # Step 3: Identify special regions
        table_regions = self._identify_table_regions(all_h_lines, height)
        logger.info(f"Found {len(table_regions)} table regions")
        
        graphic_regions = self._identify_graphic_regions(img, all_h_lines)
        logger.info(f"Found {len(graphic_regions)} graphic regions")
        
        # Combine special regions
        special_regions = table_regions + graphic_regions
        
        # Step 4: Find section boundaries
        section_dividers = self._find_section_dividers(all_h_lines, special_regions)
        logger.info(f"Found {len(section_dividers)} section dividers")
        
        # Step 5: Create initial sections
        sections = []
        all_boundaries = [0] + section_dividers + [height]
        all_boundaries = sorted(set(all_boundaries))
        
        for i in range(len(all_boundaries) - 1):
            y1 = all_boundaries[i]
            y2 = all_boundaries[i + 1]
            h = y2 - y1
            
            if h >= 30:
                sections.append((0, y1, width, h))
        
        # Step 6: Merge title boxes with their content BELOW
        sections = self._merge_titles_with_content(sections, img)
        
        # Step 7: Vertical splitting for non-table sections
        # Tables should NOT be split vertically
        sections = self._apply_vertical_splitting(sections, img, table_regions, graphic_regions)
        
        # Step 8: Final merge of tiny sections
        sections = self._merge_tiny_sections(sections, min_height=50)
        
        logger.info(f"Layout detection found {len(sections)} sections")
        return sections
    
    def _apply_vertical_splitting(self, sections: List[Tuple[int, int, int, int]], 
                                   img: np.ndarray,
                                   table_regions: List[Tuple[int, int]],
                                   graphic_regions: List[Tuple[int, int]]) -> List[Tuple[int, int, int, int]]:
        """
        Apply vertical splitting to sections that have side-by-side boxes.
        Tables and graphics are NOT split vertically.
        Title+Content pairs (title on left, content on right) are kept together.
        """
        height, width = img.shape[:2]
        new_sections = []
        
        for section in sections:
            x, y, w, h = section
            
            # Check if this section overlaps with a table or graphic region
            is_table_or_graphic = False
            for region_start, region_end in table_regions + graphic_regions:
                overlap_start = max(y, region_start)
                overlap_end = min(y + h, region_end)
                if overlap_end > overlap_start:
                    overlap = overlap_end - overlap_start
                    if overlap > h * 0.5:
                        is_table_or_graphic = True
                        break
            
            if is_table_or_graphic:
                new_sections.append(section)
                continue
            
            # Try to find vertical dividers in this section
            vertical_splits = self._find_vertical_dividers(img, section)
            
            if vertical_splits and len(vertical_splits) == 1:
                split_x = vertical_splits[0]
                left_width = split_x - x
                right_width = x + w - split_x
                
                # Check if BOTH sides have substantial content (not just a title)
                left_has_content = self._has_substantial_content(img, (x, y, left_width, h))
                right_has_content = self._has_substantial_content(img, (split_x, y, right_width, h))
                
                if left_has_content and right_has_content:
                    # Both sides have real content - do split
                    new_sections.append((x, y, left_width, h))
                    new_sections.append((split_x, y, right_width, h))
                    logger.info(f"Vertical split: both sides have content (left={left_width}, right={right_width})")
                else:
                    # One side is just a title/empty - keep together
                    logger.info(f"Keeping together: left_content={left_has_content}, right_content={right_has_content}")
                    new_sections.append(section)
                    continue
            
            elif vertical_splits and len(vertical_splits) == 2:
                # Three-way split - check if any part is just a title
                x1, x2 = vertical_splits
                widths = [x1 - x, x2 - x1, x + w - x2]
                
                # If all parts are substantial, split
                if all(ww >= w * 0.2 for ww in widths):
                    new_sections.append((x, y, widths[0], h))
                    new_sections.append((x1, y, widths[1], h))
                    new_sections.append((x2, y, widths[2], h))
                    logger.info(f"Three-way vertical split: {widths}")
                else:
                    new_sections.append(section)
            else:
                new_sections.append(section)
        
        return new_sections
    
    def _has_substantial_content(self, img: np.ndarray, section: Tuple[int, int, int, int]) -> bool:
        """
        Check if a section has substantial content (real info, not just a title).
        Returns True if:
        - Has multiple lines of text, OR
        - Has more than 3 words, OR
        - Has text that's NOT just a single title keyword
        """
        x, y, w, h = section
        
        # Skip very small sections
        if w < 80 or h < 40:
            logger.debug(f"Section too small: {w}x{h}")
            return False
        
        logger.debug(f"Checking content in section: x={x}, y={y}, w={w}, h={h}")
        
        # Crop section
        section_img = img[y:y+h, x:x+w]
        
        try:
            # Run OCR on the section (use eng as fallback if deu not available)
            try:
                ocr_data = pytesseract.image_to_data(section_img, lang='deu', output_type=pytesseract.Output.DICT)
            except:
                ocr_data = pytesseract.image_to_data(section_img, output_type=pytesseract.Output.DICT)
            
            # Count meaningful words (>1 char, confidence >10 - lowered for better detection)
            words = []
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip()
                conf = int(ocr_data['conf'][i])
                if conf > 10 and len(text) > 1:
                    words.append({
                        'text': text.upper(),
                        'y': ocr_data['top'][i]
                    })
            
            if len(words) == 0:
                logger.debug(f"No words found in section {w}x{h}")
                return False
            
            logger.debug(f"Section {w}x{h} has {len(words)} words: {[w['text'] for w in words[:5]]}")
            
            # Title keywords - if section contains ONLY one of these, it's just a label
            title_keywords = ['ABKÜRZUNG', 'ABKURZUNG', 'ABKÜRZUNGEN', 'ABKURZUNGEN',
                            'BEZEICHNUNG', 'TITEL', 'PROJEKT', 'INDEX', 'ARCHITEKT', 
                            'BAUHERR', 'BAUHERRN', 'LEGENDE', 'LAGEPLAN', 
                            'WERKPLANUNG', 'DATUM', 'AKTENZEICHEN']
            
            # If only 1 word and it's a title keyword - not substantial
            if len(words) == 1:
                word = words[0]['text']
                for kw in title_keywords:
                    if kw in word or word in kw:
                        logger.debug(f"Single title word detected: {word}")
                        return False
                # Single word but not a title - could be content
                return True
            
            # Multiple words - check if they span multiple lines
            y_positions = set()
            for word in words:
                y_line = word['y'] // 20  # Group within 20 pixels
                y_positions.add(y_line)
            
            # If multiple lines, it's substantial content
            if len(y_positions) >= 2:
                return True
            
            # Single line with multiple words - check if it's just "TITLE:" pattern
            all_text = ' '.join([w['text'] for w in words])
            for kw in title_keywords:
                # If just "KEYWORD" or "KEYWORD:" - not substantial
                if all_text.strip().rstrip(':') == kw:
                    return False
            
            # Multiple words on same line that's not just a title
            return len(words) >= 2
            
        except Exception as e:
            logger.debug(f"OCR error in content check: {e}")
            return False
    
    def _find_vertical_dividers(self, img: np.ndarray, section: Tuple[int, int, int, int]) -> List[int]:
        """
        Find vertical divider lines within a section.
        Only returns dividers when there are EXACTLY 1-2 clear full-height dividers
        creating 2-3 distinct content areas. More than that suggests a table/grid.
        """
        x, y, w, h = section
        
        # Skip small sections
        if w < 200 or h < 80:
            return []
        
        # Crop section
        section_img = img[y:y+h, x:x+w]
        gray = cv2.cvtColor(section_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Detect vertical lines that span MOST of the section height (>85%)
        min_line_height = int(h * 0.85)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_line_height))
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)
        
        # Find x-coordinates of strong vertical lines
        col_sums = np.sum(v_lines, axis=0)
        threshold = h * 255 * 0.7  # Must cover 70% of height
        
        dividers = []
        in_line = False
        line_start = 0
        
        for col_x, val in enumerate(col_sums):
            if val > threshold and not in_line:
                in_line = True
                line_start = col_x
            elif val <= threshold and in_line:
                in_line = False
                line_center = (line_start + col_x) // 2 + x
                
                # Only consider lines that create substantial sections (>100px on each side)
                if line_center > x + 100 and line_center < x + w - 100:
                    dividers.append(line_center)
        
        # Remove dividers that are too close together (need at least 150px between them)
        if dividers:
            filtered = [dividers[0]]
            for d in dividers[1:]:
                if d - filtered[-1] > 150:
                    filtered.append(d)
            dividers = filtered
        
        # IMPORTANT: If we found more than 2 dividers, it's likely a table/grid - don't split
        if len(dividers) > 2:
            logger.debug(f"Found {len(dividers)} vertical lines - likely a table, not splitting")
            return []
        
        # Only return dividers if they create meaningful sections
        if len(dividers) == 1:
            # Check that both resulting sections would be substantial
            left_width = dividers[0] - x
            right_width = x + w - dividers[0]
            if left_width >= 100 and right_width >= 100:
                return dividers
        elif len(dividers) == 2:
            # Check all three sections would be substantial
            widths = [dividers[0] - x, dividers[1] - dividers[0], x + w - dividers[1]]
            if all(w >= 80 for w in widths):
                return dividers
        
        return []
    
    def _identify_graphic_regions(self, img: np.ndarray, h_lines: List[int]) -> List[Tuple[int, int]]:
        """
        Identify regions that contain graphics/plans (not text or tables).
        Graphics have dense line content but not structured like tables.
        """
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        graphic_regions = []
        
        # Look at regions between horizontal lines
        boundaries = [0] + h_lines + [height]
        
        for i in range(len(boundaries) - 1):
            y1 = boundaries[i]
            y2 = boundaries[i + 1]
            h = y2 - y1
            
            # Skip small regions
            if h < 100:
                continue
            
            # Analyze this region
            region = gray[y1:y2, :]
            
            # Check for graphic characteristics
            if self._is_graphic_region(region):
                graphic_regions.append((y1, y2))
                logger.info(f"Graphic region: y={y1} to {y2}")
        
        return graphic_regions
    
    def _is_graphic_region(self, region: np.ndarray) -> bool:
        """
        Check if a region contains a graphic/plan.
        Graphics have:
        - High edge density (many lines)
        - Lines in various directions (not just horizontal/vertical grid)
        - Less text content
        """
        height, width = region.shape[:2]
        
        if height < 80 or width < 80:
            return False
        
        # Edge detection
        edges = cv2.Canny(region, 50, 150)
        
        # Calculate edge density
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Detect lines in various directions using Hough
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                                minLineLength=20, maxLineGap=10)
        
        if lines is None:
            return False
        
        # Count diagonal lines (not horizontal or vertical)
        diagonal_count = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            
            # A line is diagonal if it's not mostly horizontal or vertical
            if dx > 10 and dy > 10:
                diagonal_count += 1
        
        # Graphics typically have:
        # - High edge density (> 5%)
        # - Multiple diagonal lines
        has_high_edge_density = edge_density > 0.05
        has_diagonal_lines = diagonal_count > 5
        
        return has_high_edge_density and has_diagonal_lines
    
    def _merge_titles_with_content(self, sections: List[Tuple[int, int, int, int]], 
                                    img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Merge title boxes with their content boxes BELOW.
        A title box is small and contains a header keyword.
        """
        if len(sections) <= 1:
            return sections
        
        height = img.shape[0]
        merged = []
        skip_next = False
        
        for i, section in enumerate(sections):
            if skip_next:
                skip_next = False
                continue
            
            x, y, w, h = section
            
            # Check if this is a small title box
            is_title_box = h < 80  # Small height
            
            if is_title_box and i + 1 < len(sections):
                # Check if it contains a header keyword
                section_img = img[y:y+h, x:x+w]
                keyword = self._find_keyword_in_region(section_img)
                
                if keyword:
                    # This is a title - merge with the next section (content below)
                    next_section = sections[i + 1]
                    nx, ny, nw, nh = next_section
                    
                    # Create merged section
                    merged_h = (ny + nh) - y
                    merged.append((x, y, w, merged_h))
                    logger.info(f"Merged title '{keyword}' with content below: y={y} to {ny + nh}")
                    skip_next = True
                    continue
            
            merged.append(section)
        
        return merged
    
    def _find_keyword_in_region(self, region_img: np.ndarray) -> Optional[str]:
        """Find if a region contains a header keyword."""
        if not OCR_AVAILABLE:
            return None
        
        try:
            rgb = cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB)
            text = pytesseract.image_to_string(rgb, lang='deu+eng').upper().strip()
        except:
            return None
        
        HEADER_KEYWORDS = ['LEGENDE', 'ARCHITEKT', 'BAUHERR', 'PROJEKT', 'INDEX', 
                          'WERKPLANUNG', 'LAGEPLAN', 'PLANINHALT', 'BEZEICHNUNG',
                          'STATIK', 'TRAGWERK', 'FACHPLANER', 'BAULEITUNG']
        
        for keyword in HEADER_KEYWORDS:
            if keyword in text:
                return keyword
        
        return None
    
    def _find_all_horizontal_lines(self, h_lines_mask: np.ndarray, width: int, min_coverage: float) -> List[int]:
        """Find y-positions of all horizontal lines."""
        row_sums = np.sum(h_lines_mask, axis=1)
        threshold = width * 255 * min_coverage
        
        lines = []
        in_line = False
        line_start = 0
        
        for y, val in enumerate(row_sums):
            if val > threshold and not in_line:
                in_line = True
                line_start = y
            elif val <= threshold and in_line:
                in_line = False
                lines.append((line_start + y) // 2)
        
        return lines
    
    def _identify_table_regions(self, h_lines: List[int], img_height: int) -> List[Tuple[int, int]]:
        """
        Identify regions that are tables.
        Uses multiple criteria:
        1. Multiple rows with similar heights (evenly spaced lines)
        2. Rows that are not too tall (< 80 pixels typically)
        3. Merge adjacent table-like regions
        """
        if len(h_lines) < 3:
            return []
        
        # Calculate gaps between consecutive lines
        gaps = []
        for i in range(len(h_lines) - 1):
            gaps.append((h_lines[i], h_lines[i + 1], h_lines[i + 1] - h_lines[i]))
        
        # Identify table-like rows (small height, typically < 80 pixels)
        table_rows = []
        for start_y, end_y, gap in gaps:
            if 20 <= gap <= 80:  # Typical table row height
                table_rows.append((start_y, end_y))
        
        if len(table_rows) < 2:
            return []
        
        # Merge consecutive table rows into table regions
        table_regions = []
        current_start = table_rows[0][0]
        current_end = table_rows[0][1]
        
        for i in range(1, len(table_rows)):
            row_start, row_end = table_rows[i]
            
            # Check if this row is adjacent to the current region
            # Allow small gaps (up to 20 pixels) between table sections
            if row_start <= current_end + 20:
                # Extend current region
                current_end = row_end
            else:
                # Save current region if it has enough rows
                region_rows = self._count_rows_in_region(gaps, current_start, current_end)
                if region_rows >= 3:
                    table_regions.append((current_start, current_end))
                    logger.info(f"Table region: y={current_start} to {current_end} ({region_rows} rows)")
                
                # Start new region
                current_start = row_start
                current_end = row_end
        
        # Don't forget the last region
        region_rows = self._count_rows_in_region(gaps, current_start, current_end)
        if region_rows >= 3:
            table_regions.append((current_start, current_end))
            logger.info(f"Table region: y={current_start} to {current_end} ({region_rows} rows)")
        
        # Merge nearby table regions (might be parts of same table with slight gaps)
        table_regions = self._merge_nearby_table_regions(table_regions)
        
        return table_regions
    
    def _count_rows_in_region(self, gaps: List[Tuple[int, int, int]], start: int, end: int) -> int:
        """Count the number of rows within a region."""
        count = 0
        for row_start, row_end, gap in gaps:
            if row_start >= start and row_end <= end:
                count += 1
        return count + 1  # +1 because gaps = rows - 1
    
    def _merge_nearby_table_regions(self, regions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merge table regions that are close together (likely same table)."""
        if len(regions) <= 1:
            return regions
        
        merged = [regions[0]]
        
        for i in range(1, len(regions)):
            current_start, current_end = regions[i]
            prev_start, prev_end = merged[-1]
            
            # If regions are within 100 pixels, merge them
            if current_start - prev_end < 100:
                merged[-1] = (prev_start, current_end)
                logger.info(f"Merged table regions: {prev_start}-{prev_end} + {current_start}-{current_end}")
            else:
                merged.append((current_start, current_end))
        
        return merged
    
    def _find_section_dividers(self, all_lines: List[int], table_regions: List[Tuple[int, int]]) -> List[int]:
        """
        Find lines that are section dividers (not part of tables).
        """
        dividers = []
        
        for line in all_lines:
            # Check if this line is inside any table region
            is_in_table = False
            for table_start, table_end in table_regions:
                if table_start <= line <= table_end:
                    is_in_table = True
                    break
            
            if not is_in_table:
                dividers.append(line)
        
        # Also add the boundaries of table regions as potential dividers
        for table_start, table_end in table_regions:
            if table_start not in dividers:
                dividers.append(table_start)
            if table_end not in dividers:
                dividers.append(table_end)
        
        return sorted(set(dividers))
    
    def _find_strong_horizontal_lines(self, h_lines_mask: np.ndarray, width: int) -> List[int]:
        """Find y-positions of strong horizontal lines that span most of the width."""
        row_sums = np.sum(h_lines_mask, axis=1)
        threshold = width * 255 * 0.5  # Must cover at least 50% of width
        
        lines = []
        in_line = False
        line_start = 0
        
        for y, val in enumerate(row_sums):
            if val > threshold and not in_line:
                in_line = True
                line_start = y
            elif val <= threshold and in_line:
                in_line = False
                line_center = (line_start + y) // 2
                lines.append(line_center)
        
        # Filter lines that are too close together (within 10 pixels)
        filtered = []
        for line in lines:
            if not filtered or line - filtered[-1] > 10:
                filtered.append(line)
        
        return filtered
    
    def _is_table_section(self, section_img: np.ndarray) -> bool:
        """
        Detect if a section contains a table (grid structure).
        Tables have multiple internal horizontal AND vertical lines.
        """
        height, width = section_img.shape[:2]
        
        if height < 50:
            return False
        
        gray = cv2.cvtColor(section_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Detect internal horizontal lines (shorter than full width)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(width * 0.3), 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)
        
        # Detect internal vertical lines
        min_v_length = max(20, int(height * 0.15))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_v_length))
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)
        
        # Count horizontal and vertical lines
        h_line_count = self._count_lines(h_lines, axis=0)  # Count distinct y positions
        v_line_count = self._count_lines(v_lines, axis=1)  # Count distinct x positions
        
        # It's a table if it has multiple internal lines in both directions
        is_table = h_line_count >= 2 and v_line_count >= 1
        
        if is_table:
            logger.debug(f"Table detected: {h_line_count} h-lines, {v_line_count} v-lines")
        
        return is_table
    
    def _count_lines(self, lines_mask: np.ndarray, axis: int) -> int:
        """Count the number of distinct lines along an axis."""
        line_sums = np.sum(lines_mask, axis=axis)
        threshold = lines_mask.shape[1 - axis] * 255 * 0.1
        
        count = 0
        in_line = False
        
        for val in line_sums:
            if val > threshold and not in_line:
                in_line = True
                count += 1
            elif val <= threshold:
                in_line = False
        
        return count
    
    def _detect_subsections(self, section_img: np.ndarray, y_offset: int, width: int) -> List[Tuple[int, int, int, int]]:
        """
        Detect subsections within a section if there are clear dividers.
        Returns empty list if no clear subsections found.
        """
        height = section_img.shape[0]
        
        if height < 100:  # Too small to subdivide
            return []
        
        gray = cv2.cvtColor(section_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Look for strong horizontal divider lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(width * 0.6), 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)
        
        row_sums = np.sum(h_lines, axis=1)
        threshold = width * 255 * 0.4
        
        dividers = []
        in_line = False
        line_start = 0
        
        for y, val in enumerate(row_sums):
            if val > threshold and not in_line:
                in_line = True
                line_start = y
            elif val <= threshold and in_line:
                in_line = False
                dividers.append((line_start + y) // 2)
        
        # Only create subsections if we have clear dividers
        if len(dividers) < 1:
            return []
        
        # Filter dividers that are too close
        filtered_dividers = [dividers[0]]
        min_gap = max(40, height * 0.1)
        for d in dividers[1:]:
            if d - filtered_dividers[-1] >= min_gap:
                filtered_dividers.append(d)
        
        if len(filtered_dividers) < 1:
            return []
        
        # Create subsections
        subsections = []
        boundaries = [0] + filtered_dividers + [height]
        
        for i in range(len(boundaries) - 1):
            y1 = boundaries[i]
            y2 = boundaries[i + 1]
            h = y2 - y1
            if h >= 30:
                subsections.append((0, y_offset + y1, width, h))
        
        return subsections
    
    def _merge_tiny_sections(self, sections: List[Tuple[int, int, int, int]], 
                              min_height: int) -> List[Tuple[int, int, int, int]]:
        """Merge very small sections with their neighbors."""
        if len(sections) <= 1:
            return sections
        
        # Sort by y position
        sections = sorted(sections, key=lambda s: s[1])
        
        merged = []
        for section in sections:
            x, y, w, h = section
            
            if h < min_height and merged:
                # Merge with previous section
                prev = merged[-1]
                new_h = (y + h) - prev[1]
                merged[-1] = (prev[0], prev[1], prev[2], new_h)
            else:
                merged.append(section)
        
        return merged
    
    def segment_by_keywords(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Segment by finding header keywords and bounding them with horizontal lines.
        
        1. Find all horizontal lines
        2. Use OCR to find header keywords (ARCHITEKT, BAUHERR, LEGENDE, etc.)
        3. For each keyword, find the nearest line ABOVE it as the top boundary
        4. The bottom boundary is the line before the next keyword's top boundary
        """
        if not OCR_AVAILABLE:
            logger.warning("OCR not available for keyword-based segmentation")
            return []
        
        height, width = img.shape[:2]
        
        # Step 1: Find all horizontal lines
        horizontal_lines = self._find_horizontal_lines(img)
        logger.info(f"Found {len(horizontal_lines)} horizontal lines")
        
        if len(horizontal_lines) < 2:
            return []
        
        # Add boundaries
        all_lines = [0] + horizontal_lines + [height]
        all_lines = sorted(set(all_lines))
        
        # Step 2: Find header keywords with OCR
        keywords_with_positions = self._find_header_keywords_with_positions(img)
        logger.info(f"Found {len(keywords_with_positions)} header keywords")
        
        if not keywords_with_positions:
            return []
        
        # Sort keywords by y position (top to bottom)
        keywords_with_positions.sort(key=lambda k: k[1])
        
        # Step 3: For each keyword, find bounds
        segments = []
        
        for i, (keyword, keyword_y, keyword_x) in enumerate(keywords_with_positions):
            # Find nearest line ABOVE this keyword
            top_line = 0
            for line_y in all_lines:
                if line_y < keyword_y:
                    top_line = line_y
                else:
                    break
            
            # Find bottom boundary: line before next keyword's top boundary
            # or end of image if this is the last keyword
            if i + 1 < len(keywords_with_positions):
                next_keyword_y = keywords_with_positions[i + 1][1]
                # Find line just before the next keyword
                bottom_line = height
                for line_y in all_lines:
                    if line_y < next_keyword_y:
                        bottom_line = line_y
                    else:
                        break
            else:
                bottom_line = height
            
            # Create segment
            if bottom_line > top_line + 30:  # Minimum height
                segments.append((0, top_line, width, bottom_line - top_line))
                logger.info(f"  {keyword}: y={top_line} to {bottom_line}")
        
        return segments
    
    def _find_horizontal_lines(self, img: np.ndarray) -> List[int]:
        """Find y-coordinates of horizontal lines."""
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Detect horizontal lines (at least 70% of width)
        min_line_length = int(width * 0.70)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_length, 1))
        horizontal_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        
        # Find y-coordinates
        row_sums = np.sum(horizontal_mask, axis=1)
        threshold = width * 255 * 0.4
        
        lines = []
        in_line = False
        line_start = 0
        
        for y, val in enumerate(row_sums):
            if val > threshold and not in_line:
                in_line = True
                line_start = y
            elif val <= threshold and in_line:
                in_line = False
                lines.append((line_start + y) // 2)
        
        if in_line:
            lines.append((line_start + height) // 2)
        
        return lines
    
    def _find_header_keywords_with_positions(self, img: np.ndarray) -> List[Tuple[str, int, int]]:
        """
        Find header keywords (ARCHITEKT, BAUHERR, etc.) and their positions.
        Uses two methods:
        1. Look for keywords in left-aligned positions (traditional headers)
        2. Look for keywords that are ALONE on a line (subtitle style)
        
        Returns list of (keyword, y_position, x_position).
        """
        if not OCR_AVAILABLE:
            return []
        
        height, width = img.shape[:2]
        
        # Convert to RGB for OCR
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        try:
            ocr_data = pytesseract.image_to_data(img_rgb, lang='deu+eng', output_type=pytesseract.Output.DICT)
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return []
        
        # Header keywords to look for
        HEADER_KEYWORDS = ['LEGENDE', 'ARCHITEKT', 'BAUHERR', 'PROJEKT', 'INDEX', 
                          'WERKPLANUNG', 'LAGEPLAN', 'PLANINHALT', 'BEZEICHNUNG',
                          'STATIK', 'TRAGWERK', 'FACHPLANER', 'BAULEITUNG']
        
        # METHOD 1: Find keywords in left-aligned positions
        found_keywords = []
        n_boxes = len(ocr_data['text'])
        
        for i in range(n_boxes):
            text = ocr_data['text'][i].strip().upper()
            conf = int(ocr_data['conf'][i])
            
            if conf < 40 or not text:
                continue
            
            for keyword in HEADER_KEYWORDS:
                if text == keyword or text.startswith(keyword):
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    h = ocr_data['height'][i]
                    
                    # Left-aligned check (in left 60% of image)
                    if x <= width * 0.6:
                        center_y = y + h // 2
                        found_keywords.append((keyword, center_y, x, 'left_aligned'))
                    break
        
        # METHOD 2: Find keywords that are ALONE on a line
        # Group OCR results by line (similar y position)
        lines = self._group_ocr_by_lines(ocr_data, height)
        
        for line_y, line_words in lines.items():
            # Check if this line contains only a single keyword (or keyword + colon)
            if len(line_words) <= 2:  # 1-2 words on the line
                line_text = ' '.join([w['text'].upper() for w in line_words]).strip()
                line_text_clean = line_text.replace(':', '').strip()
                
                for keyword in HEADER_KEYWORDS:
                    if line_text_clean == keyword or line_text.startswith(keyword):
                        # This keyword is alone on its line - likely a header!
                        x = line_words[0]['left']
                        found_keywords.append((keyword, line_y, x, 'alone_on_line'))
                        logger.info(f"Found standalone header '{keyword}' at y={line_y}")
                        break
        
        # Sort by y position
        found_keywords.sort(key=lambda k: k[1])
        
        # Remove duplicates - keep only one keyword per vertical zone
        unique = []
        min_spacing = max(60, height * 0.02)  # At least 2% of height or 60 pixels
        
        for kw in found_keywords:
            keyword, y, x, method = kw
            
            is_duplicate = False
            for existing in unique:
                if abs(y - existing[1]) < min_spacing:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append((keyword, y, x))
                logger.info(f"Header '{keyword}' at y={y}, x={x} [{method}]")
        
        return unique
    
    def _group_ocr_by_lines(self, ocr_data: dict, img_height: int) -> Dict[int, List[dict]]:
        """
        Group OCR results by line (words with similar y positions).
        Returns dict mapping line_y -> list of word dicts.
        """
        n_boxes = len(ocr_data['text'])
        words = []
        
        for i in range(n_boxes):
            text = ocr_data['text'][i].strip()
            conf = int(ocr_data['conf'][i])
            
            if conf < 30 or not text:
                continue
            
            words.append({
                'text': text,
                'left': ocr_data['left'][i],
                'top': ocr_data['top'][i],
                'height': ocr_data['height'][i],
                'center_y': ocr_data['top'][i] + ocr_data['height'][i] // 2
            })
        
        # Sort by y position
        words.sort(key=lambda w: w['center_y'])
        
        # Group into lines (words within 15 pixels vertically)
        lines = {}
        line_threshold = 15
        
        for word in words:
            y = word['center_y']
            
            # Find existing line or create new
            matched_line = None
            for line_y in lines.keys():
                if abs(y - line_y) < line_threshold:
                    matched_line = line_y
                    break
            
            if matched_line is not None:
                lines[matched_line].append(word)
            else:
                lines[y] = [word]
        
        return lines
    
    def segment_by_horizontal_lines(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Segment the info panel by detecting STRONG horizontal divider lines.
        Uses keyword detection to merge sections that belong together.
        """
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Threshold
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Detect STRONG horizontal lines (must span at least 80% of width)
        # These are more likely to be group dividers, not table row dividers
        min_line_length = int(width * 0.80)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_length, 2))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        
        # Find y-coordinates of horizontal lines
        row_sums = np.sum(horizontal_lines, axis=1)
        threshold = width * 255 * 0.5  # Line must cover 50% of width
        
        line_rows = []
        in_line = False
        line_start = 0
        
        for y, val in enumerate(row_sums):
            if val > threshold and not in_line:
                in_line = True
                line_start = y
            elif val <= threshold and in_line:
                in_line = False
                line_center = (line_start + y) // 2
                line_rows.append(line_center)
        
        if in_line:
            line_rows.append((line_start + height) // 2)
        
        logger.info(f"Found {len(line_rows)} strong horizontal lines")
        
        # If too many lines, try to filter to only the strongest ones
        if len(line_rows) > 12:
            # Only keep lines with significant gaps between them
            filtered_lines = self._filter_close_lines(line_rows, min_gap=int(height * 0.05))
            line_rows = filtered_lines
            logger.info(f"After filtering close lines: {len(line_rows)} lines")
        
        # Add image boundaries
        boundaries = [0] + line_rows + [height]
        boundaries = sorted(set(boundaries))
        
        # Create initial segments
        raw_segments = []
        for i in range(len(boundaries) - 1):
            y1 = boundaries[i]
            y2 = boundaries[i + 1]
            raw_segments.append((0, y1, width, y2 - y1))
        
        # Merge small segments with their neighbors
        segments = self._merge_small_segments(raw_segments, img, min_height=80)
        
        logger.info(f"Final: {len(segments)} segments")
        return segments
    
    def _filter_close_lines(self, lines: List[int], min_gap: int) -> List[int]:
        """Remove lines that are too close together."""
        if not lines:
            return []
        
        filtered = [lines[0]]
        for line in lines[1:]:
            if line - filtered[-1] >= min_gap:
                filtered.append(line)
        
        return filtered
    
    def _merge_small_segments(self, segments: List[Tuple[int, int, int, int]], 
                               img: np.ndarray, min_height: int) -> List[Tuple[int, int, int, int]]:
        """
        Merge small segments that are likely part of a larger group.
        Uses keyword detection to identify group headers.
        """
        if not segments:
            return []
        
        height = img.shape[0]
        width = img.shape[1]
        
        # First pass: identify segments with keywords (group headers)
        segment_types = []
        for seg in segments:
            x, y, w, h = seg
            group_type = self.identify_group_by_keywords(img, seg)
            segment_types.append(group_type)
        
        # Second pass: merge segments
        merged = []
        i = 0
        while i < len(segments):
            current_seg = segments[i]
            current_type = segment_types[i]
            x, y, w, h = current_seg
            
            # If this segment has a keyword header, merge following segments 
            # until we hit another keyword header or a large segment
            if current_type != 'UNKNOWN':
                # Look ahead and merge content segments
                j = i + 1
                while j < len(segments):
                    next_seg = segments[j]
                    next_type = segment_types[j]
                    next_h = next_seg[3]
                    
                    # Stop if we hit another header or a large segment
                    if next_type != 'UNKNOWN' or next_h > min_height * 2:
                        break
                    
                    # Merge
                    h = (next_seg[1] + next_seg[3]) - y
                    j += 1
                
                merged.append((x, y, w, h))
                i = j
            else:
                # For unknown segments, check if they're very small
                if h < min_height and i > 0 and len(merged) > 0:
                    # Merge with previous segment
                    prev = merged[-1]
                    new_h = (y + h) - prev[1]
                    merged[-1] = (prev[0], prev[1], prev[2], new_h)
                else:
                    merged.append(current_seg)
                i += 1
        
        # Filter out very small segments
        final = [seg for seg in merged if seg[3] >= 30]
        
        return final
    
    def _remove_duplicate_boxes(self, boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Remove boxes that are essentially duplicates (very similar position and size)."""
        if not boxes:
            return []
        
        unique = []
        for box in boxes:
            is_duplicate = False
            for existing in unique:
                # Check if boxes are very similar
                x1, y1, w1, h1 = box
                x2, y2, w2, h2 = existing
                
                # Similar position and size?
                if (abs(x1 - x2) < 20 and abs(y1 - y2) < 20 and 
                    abs(w1 - w2) < 30 and abs(h1 - h2) < 30):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(box)
        
        return unique
    
    def identify_group_by_keywords(self, img: np.ndarray, box: Tuple[int, int, int, int]) -> str:
        """
        Identify what type of group this box contains.
        Strategy:
        1. Look at the FIRST word/line - usually the section title
        2. If first word is not a keyword, scan all content for keywords
        3. Use the most prominent keyword found
        """
        if not OCR_AVAILABLE:
            return 'UNKNOWN'
        
        x, y, w, h = box
        height, width = img.shape[:2]
        
        # Add small padding
        pad = 5
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(width - x, w + 2 * pad)
        h = min(height - y, h + 2 * pad)
        
        # Crop the region
        roi = img[y:y+h, x:x+w]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Get OCR with positions
        try:
            ocr_data = pytesseract.image_to_data(roi_rgb, lang='deu+eng', output_type=pytesseract.Output.DICT)
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return 'UNKNOWN'
        
        # Primary keywords (section titles) - ordered by priority
        # BAUHERR must come before PROJEKT to avoid misidentification
        PRIMARY_KEYWORDS = {
            # High priority - main section titles
            'BAUHERR': ['BAUHERR', 'BAUHERRN', 'BAUHER', 'AUFTRAGGEBER', 'BAUHERRIN'],
            'ARCHITEKT': ['ARCHITEKT', 'ARCHITECT', 'ARCHITEKTEN', 'ARCHITEK'],
            'LEGENDE': ['LEGENDE', 'LEGEND'],
            'WERKPLANUNG': ['WERKPLANUNG', 'PLANUNG', 'WERKPLAN'],
            'BEZEICHNUNG': ['BEZEICHNUNG', 'TITEL', 'OBJEKTBEZEICHNUNG'],
            'INDEX': ['INDEX', 'ÄNDERUNG', 'REVISION', 'ANDERUNG'],
            'LAGEPLAN': ['LAGEPLAN', 'ÜBERSICHT', 'LAGE', 'UBERSICHT'],
            'ABKUERZUNGEN': ['ABKÜRZUNG', 'ABKURZUNG', 'ABK.', 'ABKÜRZ'],
            'STATIK': ['STATIK', 'TRAGWERK', 'TRAGWERKSPLANER'],
            'PLANINFO': ['PLANINHALT', 'PLANNUMMER', 'PLAN-NR', 'PLANNR'],
            # Lower priority - secondary section titles
            'AKTENZEICHEN': ['AKTENZEICHEN', 'AKTENZ', 'AZ:'],
            'PROJEKT-NR': ['PROJEKT-NR', 'PROJEKTNR', 'PROJEKT-NUMMER', 'PROJEKTNUMMER', 'PRJ-NR'],
            'DATUM': ['DATUM', 'DATE', 'ERSTELLDATUM'],
            # Lowest priority - generic
            'PROJEKT': ['PROJEKT', 'PROJECT', 'BAUVORHABEN'],  # Last to avoid false matches
        }
        
        # Collect words with their positions
        words = []
        n_boxes = len(ocr_data['text'])
        for i in range(n_boxes):
            text = ocr_data['text'][i].strip().upper()
            conf = int(ocr_data['conf'][i])
            if conf > 15 and text and len(text) > 1:  # Lower threshold for better detection
                words.append({
                    'text': text,
                    'y': ocr_data['top'][i],
                    'x': ocr_data['left'][i],
                    'h': ocr_data['height'][i]
                })
        
        if not words:
            return 'UNKNOWN'
        
        # Sort by position (top to bottom, left to right)
        words.sort(key=lambda w: (w['y'], w['x']))
        
        # Strategy 1: Check the VERY FIRST word - highest priority
        first_word = words[0]['text']
        logger.debug(f"First word in section: '{first_word}', top-3: {[w['text'] for w in words[:3]]}")
        
        for group_name, keywords in PRIMARY_KEYWORDS.items():
            for kw in keywords:
                # Exact match or starts with keyword
                if first_word == kw or first_word.startswith(kw):
                    logger.info(f"First word '{first_word}' -> {group_name}")
                    return group_name
                # Keyword is contained in first word
                if kw in first_word and len(kw) >= 4:
                    logger.info(f"First word '{first_word}' contains {kw} -> {group_name}")
                    return group_name
        
        # Strategy 2: Check first 3 words in the top area
        top_y = words[0]['y']
        top_words = [w for w in words if w['y'] < top_y + 40][:3]
        
        for word_info in top_words:
            word = word_info['text']
            for group_name, keywords in PRIMARY_KEYWORDS.items():
                for kw in keywords:
                    if kw in word or word in kw:
                        return group_name
        
        # Strategy 3: Scan all content for any keyword
        all_text = ' '.join([w['text'] for w in words])
        
        for group_name, keywords in PRIMARY_KEYWORDS.items():
            for kw in keywords:
                if kw in all_text:
                    return group_name
        
        return 'UNKNOWN'
    
    def split_info_panel(self, image_path: str, output_dir: str = None) -> List[str]:
        """
        Split an info panel image into individual group images.
        
        Args:
            image_path: Path to the info panel image
            output_dir: Directory for output images. If None, uses same directory.
            
        Returns:
            List of output file paths
        """
        image_path = Path(image_path)
        if output_dir is None:
            output_dir = image_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = image_path.stem.replace('_info_panel', '')
        
        logger.info(f"Processing info panel: {image_path.name}")
        
        # Load image
        img = self.load_image(str(image_path))
        height, width = img.shape[:2]
        logger.info(f"Image size: {width}x{height}")
        
        # Detect groups
        groups = self.detect_groups(img)
        
        if not groups:
            logger.warning("No groups detected")
            return []
        
        # Sort groups by position (top to bottom, left to right)
        groups.sort(key=lambda b: (b[1], b[0]))
        
        output_paths = []
        group_counts = {}
        
        for i, box in enumerate(groups):
            x, y, w, h = box
            
            # Identify group type
            group_type = self.identify_group_by_keywords(img, box)
            
            # Handle duplicate group names
            if group_type in group_counts:
                group_counts[group_type] += 1
                group_suffix = f"_{group_counts[group_type]}"
            else:
                group_counts[group_type] = 1
                group_suffix = ""
            
            # Crop the group
            group_img = img[y:y+h, x:x+w]
            
            # Save
            output_name = f"{base_name}_group_{group_type}{group_suffix}.png"
            output_path = output_dir / output_name
            cv2.imwrite(str(output_path), group_img)
            
            logger.info(f"Saved group '{group_type}' ({w}x{h}) -> {output_name}")
            output_paths.append(str(output_path))
        
        return output_paths
    
    def split_with_preview(self, image_path: str, output_dir: str = None) -> Tuple[List[str], str]:
        """
        Split info panel and also save a preview showing detected groups.
        
        Returns:
            Tuple of (list of group paths, preview path)
        """
        image_path = Path(image_path)
        if output_dir is None:
            output_dir = image_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = image_path.stem.replace('_info_panel', '')
        
        logger.info(f"Processing info panel: {image_path.name}")
        
        # Load image
        img = self.load_image(str(image_path))
        height, width = img.shape[:2]
        logger.info(f"Image size: {width}x{height}")
        
        # Detect groups
        groups = self.detect_groups(img)
        
        # Create preview
        preview = img.copy()
        colors = [
            (0, 0, 255),    # Red
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
            (128, 0, 255),  # Purple
            (255, 128, 0),  # Orange
        ]
        
        output_paths = []
        group_counts = {}
        
        # Sort groups by position
        groups.sort(key=lambda b: (b[1], b[0]))
        
        for i, box in enumerate(groups):
            x, y, w, h = box
            color = colors[i % len(colors)]
            
            # Draw rectangle on preview
            cv2.rectangle(preview, (x, y), (x + w, y + h), color, 3)
            
            # Identify and label
            group_type = self.identify_group_by_keywords(img, box)
            
            # Handle duplicates
            if group_type in group_counts:
                group_counts[group_type] += 1
                group_suffix = f"_{group_counts[group_type]}"
            else:
                group_counts[group_type] = 1
                group_suffix = ""
            
            # Add label to preview
            label = f"{group_type}{group_suffix}"
            cv2.putText(preview, label, (x + 5, y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Crop and save group
            group_img = img[y:y+h, x:x+w]
            output_name = f"{base_name}_group_{group_type}{group_suffix}.png"
            output_path = output_dir / output_name
            cv2.imwrite(str(output_path), group_img)
            
            logger.info(f"Saved group '{label}' ({w}x{h}) -> {output_name}")
            output_paths.append(str(output_path))
        
        # Save preview
        preview_path = output_dir / f"{base_name}_groups_preview.png"
        cv2.imwrite(str(preview_path), preview)
        logger.info(f"Saved preview: {preview_path}")
        
        return output_paths, str(preview_path)


def process_directory(input_dir: str, output_dir: str = None):
    """Process all info panel images in a directory."""
    splitter = InfoPanelSplitter()
    input_path = Path(input_dir)
    
    # Find all info panel images
    info_panels = list(input_path.glob("*_info_panel.png"))
    logger.info(f"Found {len(info_panels)} info panel images")
    
    for panel_path in info_panels:
        try:
            splitter.split_with_preview(str(panel_path), output_dir)
        except Exception as e:
            logger.error(f"Failed to process {panel_path.name}: {e}")


def main():
    """Command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Split info panel into individual groups")
    parser.add_argument("input", help="Input info panel image or directory")
    parser.add_argument("-o", "--output", help="Output directory (default: same as input)")
    parser.add_argument("--preview", action="store_true", help="Generate preview showing detected groups")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    splitter = InfoPanelSplitter()
    
    if input_path.is_dir():
        process_directory(str(input_path), args.output)
    elif input_path.is_file():
        if args.preview:
            splitter.split_with_preview(str(input_path), args.output)
        else:
            splitter.split_info_panel(str(input_path), args.output)
    else:
        logger.error(f"Invalid input: {args.input}")
        sys.exit(1)


if __name__ == "__main__":
    main()

