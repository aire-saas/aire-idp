import PIL
import cv2
import numpy as np
from typing import Any, Tuple, List, Dict, Optional
import random
import json
from PIL import Image

# Maximum opening size (in pixels) to be ignored when detecting closed areas
threshold = 30
# Scale: pixels per meter (117 pixels = 1 meter)
PIXELS_PER_METER = 117


def detect_rooms_from_image(image: Image.Image, detections: List[Dict]):
    """
    Detect rooms from a floor plan image and detections.
    
    Args:
        image: PIL Image of floor plan
        detections: List of detection boxes with coordinates
        
    Returns:
        Tuple of (output, output_data)
    """
    # Convert PIL image to numpy array (BGR format for OpenCV)
    image_cv = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
    
    # Detect rooms using the existing detect_rooms function
    output, room_count, output_data = detect_rooms(image_cv, detections, 50)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    
    return output, output_data


def load_detections_from_detections(detections: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
    Extract doors, windows, stairs from detections list.
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Tuple of (doors, windows, stairs, aufzugs)
    """
    doors = []
    windows = []
    stairs = []
    aufzugs = []
    #print("detections :", detections)
    print("detections length :", len(detections))
    print("detections 0", detections[0])

    for box in detections:
        class_name = box.get('class_name', '').lower()
        source = box.get('source', '').lower()
        bbox = {
            'xmin': box.get('xmin', 0),
            'ymin': box.get('ymin', 0),
            'xmax': box.get('xmax', 0),
            'ymax': box.get('ymax', 0),
            'confidence': box.get('confidence', 0)
        }
        
        if class_name == 'door' or source == 'door':
            doors.append(bbox)
        elif class_name == 'window' or source == 'window':
            windows.append(bbox)
        elif 'stair' in class_name or 'treppe' in class_name or source == 'stairs':
            stairs.append(bbox)
        elif 'aufzug' in class_name or 'elevator' in class_name or 'lift' in class_name or source == 'aufzug':
            aufzugs.append(bbox)
    
    return doors, windows, stairs, aufzugs


def pixels_to_meters(pixels: float) -> float:
    """Convert pixels to meters."""
    return pixels / PIXELS_PER_METER


def pixels_sq_to_meters_sq(pixels_sq: float) -> float:
    """Convert square pixels to square meters."""
    return pixels_sq / (PIXELS_PER_METER ** 2)


def get_bbox_center(bbox: Dict) -> Tuple[float, float]:
    """Get the center point of a bounding box."""
    cx = (bbox['xmin'] + bbox['xmax']) / 2
    cy = (bbox['ymin'] + bbox['ymax']) / 2
    return cx, cy


def bbox_to_points(bbox: Dict) -> List[Tuple[float, float]]:
    """Get corner and edge midpoints of a bounding box for proximity checking."""
    xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    return [
        (xmin, ymin),  # top-left
        (xmax, ymin),  # top-right
        (xmin, ymax),  # bottom-left
        (xmax, ymax),  # bottom-right
        (cx, ymin),    # top-center
        (cx, ymax),    # bottom-center
        (xmin, cy),    # left-center
        (xmax, cy),    # right-center
        (cx, cy)       # center
    ]


def bboxes_overlap(bbox1: Dict, bbox2: Dict) -> bool:
    """Check if two bounding boxes overlap."""
    return not (bbox1['xmax'] < bbox2['xmin'] or 
                bbox1['xmin'] > bbox2['xmax'] or 
                bbox1['ymax'] < bbox2['ymin'] or 
                bbox1['ymin'] > bbox2['ymax'])


def merge_bboxes(bboxes: List[Dict]) -> Dict:
    """Merge multiple bounding boxes into one that covers all of them."""
    if not bboxes:
        return None
    
    xmin = min(b['xmin'] for b in bboxes)
    ymin = min(b['ymin'] for b in bboxes)
    xmax = max(b['xmax'] for b in bboxes)
    ymax = max(b['ymax'] for b in bboxes)
    avg_confidence = sum(b.get('confidence', 0) for b in bboxes) / len(bboxes)
    
    return {
        'xmin': xmin,
        'ymin': ymin,
        'xmax': xmax,
        'ymax': ymax,
        'confidence': avg_confidence
    }


def group_overlapping_bboxes(bboxes: List[Dict]) -> List[List[Dict]]:
    """Group bounding boxes that overlap into clusters."""
    if not bboxes:
        return []
    
    # Create a list to track which group each bbox belongs to
    n = len(bboxes)
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Union overlapping bboxes
    for i in range(n):
        for j in range(i + 1, n):
            if bboxes_overlap(bboxes[i], bboxes[j]):
                union(i, j)
    
    # Group by parent
    groups = {}
    for i in range(n):
        p = find(i)
        if p not in groups:
            groups[p] = []
        groups[p].append(bboxes[i])
    
    return list(groups.values())


def bboxes_adjacent(bbox1: Dict, bbox2: Dict, margin: int = 30) -> bool:
    """Check if two bounding boxes are adjacent (close to each other within margin)."""
    # Expand bbox1 by margin and check if it overlaps with bbox2
    expanded = {
        'xmin': bbox1['xmin'] - margin,
        'ymin': bbox1['ymin'] - margin,
        'xmax': bbox1['xmax'] + margin,
        'ymax': bbox1['ymax'] + margin
    }
    return bboxes_overlap(expanded, bbox2)


def group_adjacent_bboxes(bboxes: List[Dict], margin: int = 30) -> List[List[Dict]]:
    """Group bounding boxes that are adjacent (within margin) into clusters."""
    if not bboxes:
        return []
    
    # Create a list to track which group each bbox belongs to
    n = len(bboxes)
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Union adjacent bboxes
    for i in range(n):
        for j in range(i + 1, n):
            if bboxes_adjacent(bboxes[i], bboxes[j], margin):
                union(i, j)
    
    # Group by parent
    groups = {}
    for i in range(n):
        p = find(i)
        if p not in groups:
            groups[p] = []
        groups[p].append(bboxes[i])
    
    return list(groups.values())


def remove_wall_overlap_from_bbox(bbox: Dict, wall_mask: np.ndarray) -> List[Dict]:
    """
    Remove the parts of a bounding box that overlap with walls.
    Returns a list of sub-bboxes that don't overlap with walls.
    When a wall divides the bbox, only keep the largest part.
    """
    x1, y1 = int(bbox['xmin']), int(bbox['ymin'])
    x2, y2 = int(bbox['xmax']), int(bbox['ymax'])
    
    rows, cols = wall_mask.shape
    x1 = max(0, min(x1, cols - 1))
    x2 = max(0, min(x2, cols - 1))
    y1 = max(0, min(y1, rows - 1))
    y2 = max(0, min(y2, rows - 1))
    
    if x1 >= x2 or y1 >= y2:
        return []
    
    # Extract the region of interest from wall mask
    roi = wall_mask[y1:y2, x1:x2]
    
    # Invert wall mask (walls are white in binary, we want non-wall areas)
    non_wall = (roi == 0).astype(np.uint8) * 255
    
    # Find contours of non-wall areas
    contours, _ = cv2.findContours(non_wall, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # If no contours found, check if mostly non-wall
        wall_pixels = np.sum(roi > 0)
        total_pixels = roi.size
        if total_pixels > 0 and wall_pixels / total_pixels < 0.5:
            return [bbox]
        return []
    
    # Get all valid bboxes from contours
    result_bboxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        #if area < 100:  # Skip very small areas
        #   continue
        
        rx, ry, rw, rh = cv2.boundingRect(contour)
        result_bboxes.append({
            'xmin': x1 + rx,
            'ymin': y1 + ry,
            'xmax': x1 + rx + rw,
            'ymax': y1 + ry + rh,
            'confidence': bbox.get('confidence', 0),
            'area': area
        })
    
    if not result_bboxes:
        return []
    
    # If wall divides the bbox into multiple parts, keep only the largest one
    if len(result_bboxes) > 1:
        # Sort by area and keep only the largest
        result_bboxes.sort(key=lambda b: b.get('area', 0), reverse=True)
        largest = result_bboxes[0]
        # Remove the 'area' key before returning
        del largest['area']
        return [largest]
    
    # Remove the 'area' key
    if 'area' in result_bboxes[0]:
        del result_bboxes[0]['area']
    
    return result_bboxes


def is_opening_adjacent_to_room(bbox: Dict, room_mask: np.ndarray, margin: int = 15) -> bool:
    """
    Check if a door/window bounding box is adjacent to (touching or very close to) a room.
    Uses multiple points from the bbox to check proximity to the room boundary.
    """
    rows, cols = room_mask.shape
    points = bbox_to_points(bbox)
    
    for px, py in points:
        # Check in a small neighborhood around each point
        for dx in range(-margin, margin + 1, 5):
            for dy in range(-margin, margin + 1, 5):
                check_x = int(px + dx)
                check_y = int(py + dy)
                
                # Bounds check
                if 0 <= check_x < cols and 0 <= check_y < rows:
                    if room_mask[check_y, check_x] > 0:
                        return True
    
    return False


def assign_openings_to_rooms(doors: List[Dict], windows: List[Dict], 
                              room_masks: List[np.ndarray], margin: int = 15) -> Tuple[List[List[Dict]], List[List[Dict]]]:
    """
    Assign doors and windows to rooms based on proximity to room boundaries.
    A door/window can belong to multiple rooms (e.g., a door between two rooms).
    """
    num_rooms = len(room_masks)
    doors_per_room = [[] for _ in range(num_rooms)]
    windows_per_room = [[] for _ in range(num_rooms)]
    
    # Assign doors to rooms
    for door in doors:
        for room_idx, room_mask in enumerate(room_masks):
            if is_opening_adjacent_to_room(door, room_mask, margin):
                doors_per_room[room_idx].append(door)
    
    # Assign windows to rooms
    for window in windows:
        for room_idx, room_mask in enumerate(room_masks):
            if is_opening_adjacent_to_room(window, room_mask, margin):
                windows_per_room[room_idx].append(window)
    
    return doors_per_room, windows_per_room


def is_touching_border(mask: np.ndarray, labels: np.ndarray, label: int) -> bool:
    """
    Check if a labeled region touches any border of the image.
    Returns True if the region touches the image border.
    """
    rows, cols = mask.shape
    
    if np.any(labels[0, :] == label):
        return True
    if np.any(labels[rows - 1, :] == label):
        return True
    if np.any(labels[:, 0] == label):
        return True
    if np.any(labels[:, cols - 1] == label):
        return True
    
    return False


def is_covered_by_walls_on_sides(bbox: Dict, wall_mask: np.ndarray, min_sides: int = 3, wall_coverage_threshold: float = 0.5) -> bool:
    """
    Check if a bounding box is covered by walls on at least min_sides sides.
    
    Args:
        bbox: Bounding box with xmin, ymin, xmax, ymax
        wall_mask: Binary mask where walls are white (255)
        min_sides: Minimum number of sides that must be covered by walls
        wall_coverage_threshold: Minimum percentage of each side that must be covered by wall
    
    Returns:
        True if the bbox is covered by walls on at least min_sides sides
    """
    x1, y1 = int(bbox['xmin']), int(bbox['ymin'])
    x2, y2 = int(bbox['xmax']), int(bbox['ymax'])
    
    rows, cols = wall_mask.shape
    x1 = max(0, min(x1, cols - 1))
    x2 = max(0, min(x2, cols - 1))
    y1 = max(0, min(y1, rows - 1))
    y2 = max(0, min(y2, rows - 1))
    
    if x1 >= x2 or y1 >= y2:
        return False
    
    margin = 5  # Pixels to check outside the bbox for walls
    sides_covered = 0
    
    # Check top side
    top_y = max(0, y1 - margin)
    if top_y < y1:
        top_region = wall_mask[top_y:y1, x1:x2]
        if top_region.size > 0:
            wall_pixels = np.sum(top_region > 0)
            coverage = wall_pixels / top_region.size
            if coverage >= wall_coverage_threshold:
                sides_covered += 1
    
    # Check bottom side
    bottom_y = min(rows, y2 + margin)
    if bottom_y > y2:
        bottom_region = wall_mask[y2:bottom_y, x1:x2]
        if bottom_region.size > 0:
            wall_pixels = np.sum(bottom_region > 0)
            coverage = wall_pixels / bottom_region.size
            if coverage >= wall_coverage_threshold:
                sides_covered += 1
    
    # Check left side
    left_x = max(0, x1 - margin)
    if left_x < x1:
        left_region = wall_mask[y1:y2, left_x:x1]
        if left_region.size > 0:
            wall_pixels = np.sum(left_region > 0)
            coverage = wall_pixels / left_region.size
            if coverage >= wall_coverage_threshold:
                sides_covered += 1
    
    # Check right side
    right_x = min(cols, x2 + margin)
    if right_x > x2:
        right_region = wall_mask[y1:y2, x2:right_x]
        if right_region.size > 0:
            wall_pixels = np.sum(right_region > 0)
            coverage = wall_pixels / right_region.size
            if coverage >= wall_coverage_threshold:
                sides_covered += 1
    
    return sides_covered >= min_sides


# ----------------- Rectangle decomposition helpers (added) -----------------
def decompose_room_to_rectangles(room_mask: np.ndarray) -> List[Dict]:
    """
    Decompose a room shape into rectangles and return their bounding boxes.
    Uses a greedy approach to find maximal rectangles within the room area.
    """
    rectangles = []
    remaining_mask = room_mask.copy()
    
    rect_id = 0
    max_iterations = 20  # Limit iterations
    min_area_threshold = 50000  # Minimum rectangle area
    
    while rect_id < max_iterations:
        # Check remaining area
        remaining_area = cv2.countNonZero(remaining_mask)
        if remaining_area < min_area_threshold:
            break
        
        # Find the largest inscribed rectangle
        best_rect = find_largest_inscribed_rectangle(remaining_mask)
        
        if best_rect is None:
            break
        
        rx, ry, rw, rh = best_rect
        
        # Skip very small rectangles
        if rw < 10 or rh < 10 or rw * rh < min_area_threshold:
            break
        
        # Calculate area and perimeter
        area_pixels = rw * rh
        perimeter_pixels = 2 * (rw + rh)
        area_meters = pixels_sq_to_meters_sq(area_pixels)
        perimeter_meters = pixels_to_meters(perimeter_pixels)
        
        rectangles.append({
            "rect_id": rect_id,
            "bbox": {
                "x": int(rx),
                "y": int(ry),
                "width": int(rw),
                "height": int(rh)
            },
            "area_pixels": int(area_pixels),
            "area_meters_sq": round(area_meters, 2),
            "perimeter_pixels": int(perimeter_pixels),
            "perimeter_meters": round(perimeter_meters, 2)
        })
        rect_id += 1
        
        # Remove this rectangle area from remaining mask
        cv2.rectangle(remaining_mask, (rx, ry), (rx + rw, ry + rh), 0, -1)
    
    # Fallback: use bounding box if no rectangles found
    if not rectangles:
        contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            area_pixels = w * h
            perimeter_pixels = 2 * (w + h)
            rectangles.append({
                "rect_id": 0,
                "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                "area_pixels": int(area_pixels),
                "area_meters_sq": round(pixels_sq_to_meters_sq(area_pixels), 2),
                "perimeter_pixels": int(perimeter_pixels),
                "perimeter_meters": round(pixels_to_meters(perimeter_pixels), 2)
            })
    
    return rectangles


def find_largest_inscribed_rectangle(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Find the largest rectangle that fits entirely within the white area of the mask.
    Uses optimized histogram-based approach.
    """
    # Convert mask to binary (0 and 1)
    binary = (mask > 0).astype(np.uint8)
    
    if binary.sum() == 0:
        return None
    
    rows, cols = binary.shape
    
    # Downsample for large images to speed up
    scale = 1
    if rows > 500 or cols > 500:
        scale = max(rows, cols) // 500 + 1
        binary = binary[::scale, ::scale]
        rows, cols = binary.shape
    
    # Build height histogram for each row using vectorized operations
    heights = np.zeros((rows, cols), dtype=np.int32)
    heights[0] = binary[0]
    
    for i in range(1, rows):
        heights[i] = np.where(binary[i] == 1, heights[i - 1] + 1, 0)
    
    # Find largest rectangle in histogram for each row
    best_area = 0
    best_rect = None
    
    
    for i in range(rows):
        rect = largest_rectangle_in_histogram(heights[i], i)
        if rect is not None:
            x, y, w, h = rect
            area = w * h
            if area > best_area:
                best_area = area
                best_rect = rect
    
    # Scale back up
    if best_rect is not None and scale > 1:
        x, y, w, h = best_rect
        best_rect = (x * scale, y * scale, w * scale, h * scale)
    
    return best_rect


def largest_rectangle_in_histogram(heights: np.ndarray, current_row: int) -> Tuple[int, int, int, int]:
    """
    Find the largest rectangle in a histogram.
    Returns (x, y, width, height) of the rectangle.
    """
    # Convert to Python list for indexing performance
    heights_list = heights.tolist()
    n = len(heights_list)
    if n == 0:
        return None
    
    stack = []
    best_area = 0
    best_rect = None
    
    i = 0
    while i < n:
        if not stack or heights_list[i] >= heights_list[stack[-1]]:
            stack.append(i)
            i += 1
        else:
            top = stack.pop()
            h = heights_list[top]
            w = i if not stack else i - stack[-1] - 1
            x = 0 if not stack else stack[-1] + 1
            area = h * w
            
            if area > best_area:
                best_area = area
                best_rect = (x, current_row - h + 1, w, h)
    
    while stack:
        top = stack.pop()
        h = heights_list[top]
        w = n if not stack else n - stack[-1] - 1
        x = 0 if not stack else stack[-1] + 1
        area = h * w
        
        if area > best_area:
            best_area = area
            best_rect = (x, current_row - h + 1, w, h)
    
    return best_rect


# ----------------- End rectangle helpers -----------------


def detect_rooms(image: np.ndarray, json_data: Any = None, threshold_value: int = None) -> Tuple[np.ndarray, int, Dict]:
    """
    Detect rooms in a floor plan image.
    
    Args:
        image: numpy array (BGR format) of the floor plan image
        json_data: Optional JSON data or dict with door/window/stairs/aufzug detections
        threshold_value: Maximum opening size to ignore when detecting closed areas
    
    Returns:
        Tuple of (annotated_image, room_count, output_data) where output_data contains rooms, stairs, and aufzugs
    """
    global threshold
    if threshold_value is not None:
        threshold = threshold_value
    
    # Load doors, windows, stairs, and aufzugs from detections if provided
    doors = []
    windows = []
    stairs = []
    aufzugs = []
    #print("1")
    if json_data is not None:
        if isinstance(json_data, list):
            doors, windows, stairs, aufzugs = load_detections_from_detections(json_data)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply binary threshold to get walls (assuming walls are dark)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Apply morphological closing to close small gaps/openings
    kernel_size = threshold * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Wall mask (walls are white/255)
    wall_mask = closed
    
    # Invert to get rooms as white areas
    rooms_mask = cv2.bitwise_not(closed)

    #print("2")
    # Process stairs: remove wall overlaps and group adjacent ones (even with wall between)
    processed_stairs = []
    for stair in stairs:
        non_wall_parts = remove_wall_overlap_from_bbox(stair, wall_mask)
        processed_stairs.extend(non_wall_parts)
    #print("3")
    # Group adjacent stairs (within margin, even if wall between them)
    stair_groups = group_adjacent_bboxes(processed_stairs, margin=50)
    merged_stairs = []
    for group in stair_groups:
        merged = merge_bboxes(group)
        if merged:
            merged['parts'] = group  # Keep track of individual parts
            merged_stairs.append(merged)
    #print("4")
    # Process aufzugs: remove wall overlaps and group overlapping ones
    processed_aufzugs = []
    for aufzug in aufzugs:
        non_wall_parts = remove_wall_overlap_from_bbox(aufzug, wall_mask)
        processed_aufzugs.extend(non_wall_parts)
    #print("5")
    # Filter aufzugs: keep only those covered by walls on at least 3 sides
    valid_aufzugs = []
    for aufzug in processed_aufzugs:
        if is_covered_by_walls_on_sides(aufzug, wall_mask, min_sides=3):
            valid_aufzugs.append(aufzug)
    #print("6")
    # Group overlapping aufzugs
    aufzug_groups = group_overlapping_bboxes(valid_aufzugs)
    merged_aufzugs = []
    for group in aufzug_groups:
        merged = merge_bboxes(group)
        if merged:
            merged['parts'] = group
            merged_aufzugs.append(merged)
    #print("7")
    # Find connected components (each room is a separate component)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(rooms_mask, connectivity=8)
    
    # Create output image (copy of original)
    output = image.copy()
    
    # Generate distinct colors for each room (excluding background label 0)
    colors = generate_distinct_colors(num_labels - 1)
    #
    # print("8")
    # First pass: collect all valid room masks
    room_masks = []
    room_labels = []
    room_stats = []
    room_centroids_list = []
    
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        
        # Skip very small areas (noise)
        #if area < 100:
        #    continue
        
        # Skip regions that touch the image border (outer boundary)
        if is_touching_border(rooms_mask, labels, label):
            continue
        
        room_mask = (labels == label).astype(np.uint8) * 255
        room_masks.append(room_mask)
        room_labels.append(label)
        room_stats.append(stats[label])
        room_centroids_list.append(centroids[label])
    #print("9")
    # Assign doors and windows to rooms using detection data
    doors_per_room = [[] for _ in range(len(room_masks))]
    windows_per_room = [[] for _ in range(len(room_masks))]
    stairs_per_room = [[] for _ in range(len(room_masks))]
    aufzugs_per_room = [[] for _ in range(len(room_masks))]
    
    if json_data is not None:
        if doors or windows:
            #rint("9aa")
            doors_per_room, windows_per_room = assign_openings_to_rooms(doors, windows, room_masks)
            #print("9a")
        # Assign merged stairs and aufzugs to rooms
        for stair in merged_stairs:
            for room_idx, room_mask in enumerate(room_masks):
                #print("9bb")
                if is_opening_adjacent_to_room(stair, room_mask, margin=20):
                    stairs_per_room[room_idx].append(stair)
                    #print("9b")
                    break  # Each stair belongs to one location
        for aufzug in merged_aufzugs:
            for room_idx, room_mask in enumerate(room_masks):
                if is_opening_adjacent_to_room(aufzug, room_mask, margin=20):
                    aufzugs_per_room[room_idx].append(aufzug)
                   # print("9c")
                    break
   #print("10")
    # Store room information for JSON
    rooms_info = []
    
    # Track room index for actual rooms (excluding stairs/aufzug areas)
    actual_room_idx = 0
    #print("11")
    # Second pass: process each room
    for room_idx, (room_mask, label, stat, centroid) in enumerate(zip(room_masks, room_labels, room_stats, room_centroids_list)):
        area = stat[cv2.CC_STAT_AREA]
        
        # Get stairs and aufzugs for this room
        room_stairs = stairs_per_room[room_idx]
        room_aufzugs = aufzugs_per_room[room_idx]
        
        # Check if this room contains stairs or aufzug - skip it as a room
        has_stairs = len(room_stairs) > 0
        has_aufzug = len(room_aufzugs) > 0
        
        if has_stairs or has_aufzug:
            # Don't count as room, don't color - will draw stairs/aufzug later
            continue
        
        # Get room bounding box
        x = int(stat[cv2.CC_STAT_LEFT])
        y = int(stat[cv2.CC_STAT_TOP])
        w = int(stat[cv2.CC_STAT_WIDTH])
        h = int(stat[cv2.CC_STAT_HEIGHT])
        
        # Calculate room perimeter from contour
        contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter_pixels = 0
        if contours:
            perimeter_pixels = cv2.arcLength(max(contours, key=cv2.contourArea), True)
        
        # Convert to real-world measurements
        area_meters = pixels_sq_to_meters_sq(area)
        perimeter_meters = pixels_to_meters(perimeter_pixels)
       # print("11b")
        # Get doors and windows for this room
        room_doors = doors_per_room[room_idx]
        room_windows = windows_per_room[room_idx]

        doors_length = {
            i: pixels_to_meters(
                max(bbox['xmax'] - bbox['xmin'], bbox['ymax'] - bbox['ymin'])
            )
            for i, bbox in enumerate(room_doors)
        }
        windows_length = {
            i: pixels_to_meters(
                max(bbox['xmax'] - bbox['xmin'], bbox['ymax'] - bbox['ymin'])
            )
            for i, bbox in enumerate(room_windows)
        }

        # Decompose the room into rectangles and include in JSON
        rectangles = decompose_room_to_rectangles(room_mask)
      #  print("11c")
        # Store room info (without rectangles)
        room_info = {
            "room_id": actual_room_idx,
            "area_pixels": int(area),
            "area_meters_sq": round(area_meters, 2),
            "perimeter_pixels": round(perimeter_pixels, 2),
            "perimeter_meters": round(perimeter_meters, 2),
            "bounding_box": {
                "x": x,
                "y": y,
                "width": w,
                "height": h
            },
            "centroid": {
                "x": float(centroid[0]),
                "y": float(centroid[1])
            },
            "doors_count": len(room_doors),
            "doors_lengths": doors_length,
            "windows_count": len(room_windows),
            "windows_lengths": windows_length,
            # New: rectangles decomposition
            "rectangles_count": len(rectangles),
            "rectangles": rectangles
        }
        rooms_info.append(room_info)
        
        # Use normal color for rooms
        color = colors[actual_room_idx % len(colors)] if len(colors) > 0 else (200, 200, 200)
        
        # Apply color to this room with transparency
        colored_overlay = np.zeros_like(image)
        colored_overlay[labels == label] = color
        
        # Blend with original image
        alpha = 0.5
        output = np.where(
            np.stack([labels == label] * 3, axis=-1),
            cv2.addWeighted(output, 1 - alpha, colored_overlay, alpha, 0),
            output
        )
       # print("11d")
        # Add room ID label at centroid with area and perimeter
        cx, cy = int(centroid[0]), int(centroid[1])
        label_text = f"Room {actual_room_idx}"
        area_text = f"{area_meters:.2f}m2"
        perim_text = f"P:{perimeter_meters:.2f}m"
        doors_text = f"D:{len(room_doors)} W:{len(room_windows)}"
        cv2.putText(output, label_text, (cx - 40, cy - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(output, label_text, (cx - 40, cy - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(output, area_text, (cx - 30, cy - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(output, area_text, (cx - 30, cy - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(output, perim_text, (cx - 30, cy + 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(output, perim_text, (cx - 30, cy + 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(output, doors_text, (cx - 30, cy + 24), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(output, doors_text, (cx - 30, cy + 24), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Optionally: draw rectangle decompositions (uncomment if desired)
        # for rect in rectangles:
        #     rx = rect['bbox']['x']
        #     ry = rect['bbox']['y']
        #     rw = rect['bbox']['width']
        #     rh = rect['bbox']['height']
        #     cv2.rectangle(output, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)
        
        actual_room_idx += 1
    #print("12")
    # Draw stairs - draw each individual part (not merged rectangle)
    stairs_info = []
    for idx, stair in enumerate(merged_stairs):
        parts = stair.get('parts', [stair])  # Get individual parts, or use merged if no parts
        
        # Draw each part
        for part in parts:
            x1, y1 = int(part['xmin']), int(part['ymin'])
            x2, y2 = int(part['xmax']), int(part['ymax'])
            
            # Draw filled rectangle with transparency
            overlay = output.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)  # Red fill
            cv2.addWeighted(overlay, 0.4, output, 0.6, 0, output)
            # Draw border
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red border
        
        # Add label above the first part box
        first_part = parts[0]
        x1, y1 = int(first_part['xmin']), int(first_part['ymin'])
        x2, y2 = int(first_part['xmax']), int(first_part['ymax'])
        cx = (x1 + x2) // 2
        label_y = y1 - 10  # Position above the box
        if label_y < 20:  # If too close to top, put it inside the box
            label_y = y1 + 20
        cv2.putText(output, "STAIRS", (cx - 30, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(output, "STAIRS", (cx - 30, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Calculate total area from parts
        total_area = 0
        parts_info = []
        for part in parts:
            pw = int(part['xmax']) - int(part['xmin'])
            ph = int(part['ymax']) - int(part['ymin'])
            part_area = pw * ph
            total_area += part_area
            parts_info.append({
                "xmin": int(part['xmin']),
                "ymin": int(part['ymin']),
                "xmax": int(part['xmax']),
                "ymax": int(part['ymax']),
                "width": pw,
                "height": ph,
                "area_pixels": part_area
            })
        
        stairs_info.append({
            "stairs_id": idx,
            "bounding_box": {
                "xmin": x1,
                "ymin": y1,
                "xmax": x2,
                "ymax": y2,
                "width": x2 - x1,
                "height": y2 - y1
            },
            "parts": parts_info,
            "total_area_pixels": total_area,
            "total_area_meters_sq": round(pixels_sq_to_meters_sq(total_area), 2),
            "confidence": round(stair.get('confidence', 0), 3)
        })
  
    # Draw merged aufzug bounding boxes (green) directly on the image
    aufzugs_info = []
    for idx, aufzug in enumerate(merged_aufzugs):
        x1, y1 = int(aufzug['xmin']), int(aufzug['ymin'])
        x2, y2 = int(aufzug['xmax']), int(aufzug['ymax'])
        
        # Draw filled rectangle with transparency
        overlay = output.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)  # Green fill
        cv2.addWeighted(overlay, 0.4, output, 0.6, 0, output)
        # Draw border
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green border
        # Add label
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.putText(output, "AUFZUG", (cx - 30, cy), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(output, "AUFZUG", (cx - 30, cy), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add to aufzugs info
        aufzug_width = x2 - x1
        aufzug_height = y2 - y1
        aufzug_area = aufzug_width * aufzug_height
        aufzugs_info.append({
            "aufzug_id": idx,
            "bounding_box": {
                "xmin": x1,
                "ymin": y1,
                "xmax": x2,
                "ymax": y2,
                "width": aufzug_width,
                "height": aufzug_height
            },
            "area_pixels": aufzug_area,
            "area_meters_sq": round(pixels_sq_to_meters_sq(aufzug_area), 2),
            "confidence": round(aufzug.get('confidence', 0), 3)
        })
    
    # Prepare output data
    output_data = {
        "rooms": rooms_info,
        "stairs": stairs_info,
        "aufzugs": aufzugs_info
    }

    return output, len(rooms_info), output_data

def generate_distinct_colors(n: int) -> list:
    """Generate n distinct colors using HSV color space."""
    colors = []
    for i in range(n):
        hue = int(180 * i / n) if n > 0 else 0  # OpenCV uses 0-180 for hue
        color = cv2.cvtColor(np.uint8([[[hue, 255, 200]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in color))
    return colors
def save_rooms_json(output_data: Dict, output_path: str):
    """Save room, stairs, and aufzug information to a JSON file."""
    data = {
        "total_rooms": len(output_data.get("rooms", [])),
        "total_stairs": len(output_data.get("stairs", [])),
        "total_aufzugs": len(output_data.get("aufzugs", [])),
        "scale": {
            "pixels_per_meter": PIXELS_PER_METER,
            "description": "117 pixels = 1 meter"
        },
        "rooms": output_data.get("rooms", []),
        "stairs": output_data.get("stairs", []),
        "aufzugs": output_data.get("aufzugs", [])
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def print_rooms_summary(output_data: Dict):
    """Print a summary of all detected rooms, stairs, and aufzugs."""
    rooms = output_data.get("rooms", [])
    stairs = output_data.get("stairs", [])
    aufzugs = output_data.get("aufzugs", [])
    
    for room in rooms:
        print(f"  Room {room['room_id']}: {room['area_meters_sq']}m², "
              f"P:{room['perimeter_meters']}m, {room['doors_count']} doors, "
              f"{room['windows_count']} windows, {room.get('rectangles_count', 0)} rects")
    
    for stair in stairs:
        print(f"  Stairs {stair['stairs_id']}: {stair['total_area_meters_sq']}m²")
    
    for aufzug in aufzugs:
        print(f"  Aufzug {aufzug['aufzug_id']}: {aufzug['area_meters_sq']}m²")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python detect_rooms.py <image_path> [json_path] [threshold]")
        print("  image_path: Path to the floor plan image")
        print("  json_path: Optional path to JSON with door/window detections")
        print("  threshold: Maximum opening size to ignore (default: 10)")
        sys.exit(1)
    
    image_path = sys.argv[1]
    json_path = None
    
    if len(sys.argv) >= 3:
        # Check if second arg is a json file or a number (threshold)
        if sys.argv[2].endswith('.json'):
            json_path = sys.argv[2]
        else:
            threshold = int(sys.argv[2])
    
    if len(sys.argv) >= 4:
        threshold = int(sys.argv[3])
    
    print(f"Processing image: {image_path}")
    if json_path:
        print(f"Using JSON detections: {json_path}")
    print(f"Using threshold: {threshold}")
    
    try:
        result_image, num_rooms, output_data = detect_rooms(image_path, json_path, threshold)
        print(f"Detected {num_rooms} rooms, {len(output_data.get('stairs', []))} stairs, {len(output_data.get('aufzugs', []))} aufzugs")
        
        # Print summary
        print_rooms_summary(output_data)
        
        # Save the result image
        output_path = image_path.rsplit('.', 1)[0] + "_rooms_detected.png"
        cv2.imwrite(output_path, result_image)
        print(f"Result saved to: {output_path}")
        
        # Save the JSON file
        json_output_path = image_path.rsplit('.', 1)[0] + "_rooms.json"
        save_rooms_json(output_data, json_output_path)
        print(f"Room info JSON saved to: {json_output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
