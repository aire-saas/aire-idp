import cv2
import numpy as np
import math
import json
from collections import defaultdict
import copy
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import os
import random
from typing import Any

class WallFeaturesExtractor:
    def __init__(
        self,
        plan: np.ndarray | None = None,
        infoPanel: np.ndarray | None = None,
        infoPanel_x: int | None = None,
        rooms: Any | None = None,
        texts: Any | None = None,
    ):
        self.plan = plan
        self.infoPanel = infoPanel
        self.infoPanel_x = infoPanel_x
        self.rooms = rooms
        self.texts = texts
# region Legend Extraction
    def extract_legend(self, min_y = 350, min_width_ratio = 1/2, angle_tolerance_deg = 1):
        # Load image
        img = np.array(self.infoPanel)
        img= cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if img is None:
            raise ValueError("Could not load image")

        height, width = img.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines using Probabilistic Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=int(width * min_width_ratio),
            maxLineGap=10
        )

        if lines is None:
            raise ValueError("No lines detected")

        valid_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Compute angle in degrees
            angle = math.degrees(math.atan2((y2 - y1), (x2 - x1)))

            # Check near-horizontal condition
            if abs(angle) <= angle_tolerance_deg:
                line_length = abs(x2 - x1)

                # Check length and vertical position
                if line_length >= width * min_width_ratio and min(y1, y2) >= min_y:
                    avg_y = int((y1 + y2) / 2)
                    valid_lines.append((avg_y, x1, y1, x2, y2))

        if not valid_lines:
            raise ValueError("No valid horizontal line found")

        # Sort by Y position (top-most first)
        valid_lines.sort(key=lambda x: x[0])

        # Take the first qualifying line
        first_line_y = valid_lines[0][0]

        
        # Extract the image from that line downward
        extracted = img[:first_line_y, :]

        
        return extracted, first_line_y       

    def preprocess(self, img, json_data, y_threshold , x_threshhold, erosion_kernel_size=3):
        # Convert to BGR if grayscale
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        legend_texts = {}

        # Iterate over JSON text entries
        for idx, entry in enumerate(json_data):
            coords = entry.get("image_coordinates", {})
            x0 = int(coords.get("x0"))
            y0 = int(coords.get("y0"))
            x1 = int(coords.get("x1", 0))
            y1 = int(coords.get("y1", 0))
           


            # Filter text above y_threshold
            if y1 <= y_threshold and x0 >= x_threshhold:
                
                new_entry = copy.deepcopy(entry)

                # --- shift x coordinates ---
                new_entry["image_coordinates"]["x0"] = x0 - x_threshhold
                new_entry["image_coordinates"]["x1"] = x1 - x_threshhold

                legend_texts[idx] = new_entry

                # Draw white rectangle to "erase" the text
                cv2.rectangle(img, ((x0 - x_threshhold), y0), ((x1 - x_threshhold), y1), (255, 255, 255), thickness= -1)
       
        # Erode the image to clean up edges
        kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
        processed_img = cv2.erode(img, kernel, iterations=1)
        return processed_img , legend_texts

    def extract_rectangles(self, img, min_area=500, draw=True):
        """
        Detects rectangles in the image and optionally draws them.

        Args:
            image_path (str): Path to the input image.
            min_area (int): Minimum area of rectangles to consider.
            draw (bool): If True, draw rectangles on the image.

        Returns:
            result_img (numpy.ndarray): Image with rectangles drawn.
            rectangles (list): List of rectangle coordinates (x, y, w, h).
        """

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Blur and threshold for better contour detection
        #blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rectangles = []

        for cnt in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # Only keep quadrilaterals with sufficient area
            if len(approx) == 4 and cv2.contourArea(approx) > min_area:
                x, y, w, h = cv2.boundingRect(approx)
                rectangles.append((x, y, w, h))
                if draw:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            elif len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 100), 2)
            else:             
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(img, (x, y), (x + w, y + h), (220, 0, 100), 2)
        result_img = img if draw else None
        return result_img, rectangles

    def filter_rectangles_by_common_size(
        self,
        rectangles,
        size_tolerance_ratio=0.15,
        min_count=3
    ):
        """
        Filters rectangles by detecting a common rectangle size and
        returning rectangles belonging to the largest size group
        (in terms of number of elements).

        Args:
            rectangles (list): List of (x, y, w, h)
            size_tolerance_ratio (float): Allowed size variation (e.g. 0.1 = ±10%)
            min_count (int): Minimum number of rectangles required to define a common size

        Returns:
            filtered_rectangles (list): Rectangles belonging to the largest size group
            reference_size (tuple | None): (w, h) of detected common size
        """

        if not rectangles:
            return [], None

        # Group rectangles by approximate size
        size_groups = defaultdict(list)

        for rect in rectangles:
            _, _, w, h = rect
            matched = False
            for (gw, gh) in size_groups.keys():
                if (
                    abs(w - gw) <= gw * size_tolerance_ratio and
                    abs(h - gh) <= gh * size_tolerance_ratio
                ):
                    size_groups[(gw, gh)].append(rect)
                    matched = True
                    break
            if not matched:
                size_groups[(w, h)].append(rect)

        # Select the group with the largest number of elements (>= min_count)
        reference_group = None
        max_count = 0
        for (w, h), group in size_groups.items():
            if len(group) >= min_count:
                if len(group) > max_count:
                    max_count = len(group)
                    reference_group = (w, h)

        # No valid group found
        if reference_group is None:
            return rectangles, None

        # Return rectangles in the selected group
        filtered_rectangles = size_groups[reference_group]
        
        return filtered_rectangles, reference_group

    def choose_global_side_and_extract(
        self,
        rectangles,
        text_dict,
        y_overlap_ratio=0.1
    ):
        """
        Determines whether LEFT or RIGHT text dominates globally,
        then extracts the chosen-side word for each rectangle.

        Returns:
            chosen_side (str): "left" or "right"
            result (dict): {
                rect_index: {
                    "rectangle": (x,y,w,h),
                    "text": str | None,
                    "text_box": dict | None
                }
            }
        """

        def vertical_overlap(rect, word):
            ry, rh = rect[1], rect[3]
            wy0, wy1 = word["y0"], word["y1"]
            overlap = min(ry + rh, wy1) - max(ry, wy0)
            return overlap > 0 and overlap / min(rh, wy1 - wy0) >= y_overlap_ratio

        rect_info = {}
        right_only = 0
        left_only = 0
       
        # ---------- pass 1: analyze rectangles ----------
        for i, rect in enumerate(rectangles):
            rx, ry, rw, rh = rect
            left_candidates = []
            right_candidates = []

            for entry in text_dict.values():
                coords = entry["image_coordinates"]

                if not vertical_overlap(rect, coords):
                  
                    continue

                # left
                if coords["x1"] < rx:
                    dist = rx - coords["x1"]
                    left_candidates.append((dist, entry))

                # right
                elif coords["x0"] > rx + rw:
                    dist = coords["x0"] - (rx + rw)
                    right_candidates.append((dist, entry))

            left_entries = self.get_all_min_distance_candidates(left_candidates)
            right_entries = self.get_all_min_distance_candidates(right_candidates)

            rect_info[i] = {
                "left": left_entries,
                "right": right_entries
            }





            if left_entries and not right_entries:
                left_only += 1
            elif right_entries and not left_entries:
                right_only += 1

        # ---------- global decision ----------
        chosen_side = "left" if left_only > right_only else "right"

        # ---------- pass 2: extract chosen side ----------
        results = {}

        for i, rect in enumerate(rectangles):
            chosen_entries = rect_info[i][chosen_side]

            results[i] = {
                "rectangle": rect,
                "text": " ".join(e["text"] for e in chosen_entries) if chosen_entries else None,
                "text_box": [e["image_coordinates"] for e in chosen_entries] if chosen_entries else None
            }

        return chosen_side, results
    def get_all_min_distance_candidates(self,candidates):
        """
        candidates: list of (distance, entry)
        returns: list of entry dicts with minimal distance
        """
        if not candidates:
            return []

        min_dist = min(d for d, _ in candidates)
        return [entry for d, entry in candidates if d == min_dist]

    def createFinalDict(self, legendImg, results):
        """
        Creates the final dictionary containing text and cropped rectangle images.

        Args:
            legendImg (numpy.ndarray): Image containing the legend area
            results (dict): Output from choose_global_side_and_extract

        Returns:
            dict: {
                idx: {
                    "text": str,
                    "image": numpy.ndarray
                }
            }
        """

        final_dict = {}

        img_h, img_w = legendImg.shape[:2]

        for idx, item in results.items():
            text = item.get("text")
            rect = item.get("rectangle")

            if text is None or rect is None:
                continue

            x, y, w, h = rect

            # --- safety bounds ---
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(img_w, x + w)
            y1 = min(img_h, y + h)

            if x1 <= x0 or y1 <= y0:
                continue

            # --- crop rectangle ---
            crop = legendImg[y0:y1, x0:x1].copy()

            final_dict[idx] = {
                "text": text,
                "image": crop
            }

        return final_dict
# end Region
# region material matching
    def process_with_final_dict2(
    self,
    rooms_dict: dict,
    final_dict: dict,
    output_json_path: str
):
        plan = self.plan
        plan = np.array(self.plan)
        
        if plan is None:
            raise ValueError("Plan image not loaded")
        plan = cv2.cvtColor(plan, cv2.COLOR_RGB2BGR)
        h, w = plan.shape[:2]
        target_w, target_h = max(200, w), max(100, h)
        if w < 200 or h < 100:
            padded = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
            padded[:h, :w] = plan
            plan = padded

        planV = plan.copy()
        outputImages = {}

        rooms = rooms_dict.get("rooms", [])
        if not isinstance(rooms, list):
            raise ValueError("'rooms' must be a list")

        total_walls = 0

        for room in rooms:
            walls = room.get("walls", {})

            if not isinstance(walls, dict):
                continue


            for wall_id, wall_data in walls.items():
                total_walls += 1
                matchedMaterials = {}

                xmin = int(wall_data["xmin"])
                ymin = int(wall_data["ymin"])
                xmax = int(wall_data["xmax"])
                ymax = int(wall_data["ymax"])

                wall_bbox = (xmin, ymin, xmax, ymax)

                cv2.rectangle(planV, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)

                for mat_idx, material in final_dict.items():
                    material_img = material["image"]

                    try:
                        is_match, score = self.wall_matches_material_debug(
                            plan,
                            wall_bbox,
                            material_img,
                            threshold=0.85
                        )
                        if is_match:
                            matchedMaterials[mat_idx] = score
                    except Exception as e:
                        print(f"Material {mat_idx} failed: {e}")
                        continue

                if matchedMaterials:
                    best_i = min(matchedMaterials, key=matchedMaterials.get)
                    detected_text = final_dict[best_i]["text"]
                    detected_score = float(matchedMaterials[best_i])
                else:
                    detected_text = "could not detect material"
                    detected_score = None

                wall_data["detected_material"] = detected_text

                if matchedMaterials:
                    if best_i not in outputImages:
                        outputImages[best_i] = np.ones_like(plan) * 255

                    out = outputImages[best_i]
                    out[ymin:ymax, xmin:xmax] = plan[ymin:ymax, xmin:xmax]
                    cv2.rectangle(out, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                    label = detected_text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 1
                    thickness = 2
                    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)

                    cv2.rectangle(out, (0, 0), (tw + 10, th + 10), (255, 255, 255), -1)
                    cv2.putText(
                        out,
                        label,
                        (5, th + 5),
                        font,
                        scale,
                        (0, 0, 0),
                        thickness,
                        cv2.LINE_AA
                    )

        # --------------------------------------------------
        # Save outputs
        # --------------------------------------------------
        os.makedirs("materialsOutput", exist_ok=True)
        for k, img in outputImages.items():
            cv2.imwrite(f"materialsOutput/{k + 1}.png", img)

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(rooms_dict, f, indent=4, ensure_ascii=False)

        cv2.imwrite("planV.png", planV)

        print(f"Processed {total_walls} walls")
        print(f"Updated output saved to: {output_json_path}")

    def process_with_final_dict(
        self,
        JsonWalls: str,
        final_dict: dict,
        output_json_path: str
        ):

        # --------------------------------------------------
        # Load plan image
        # --------------------------------------------------
        plan = self.plan
        if plan is None:
            raise ValueError("Plan image not loaded")

        # Ensure minimum size
        h, w = plan.shape[:2]
        target_w, target_h = max(200, w), max(100, h)
        if w < 200 or h < 100:
            padded = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
            padded[:h, :w] = plan
            plan = padded

        # --------------------------------------------------
        # Load wall JSON (FLAT LIST FORMAT)
        # --------------------------------------------------
        with open(JsonWalls, "r", encoding="utf-8") as f:
            wall_json = json.load(f)

        if not isinstance(wall_json, list):
            raise ValueError("Expected JSON to be a list of annotations")

        # Extract wall items and coordinates (ALIGNED)
        wall_items = []
        walls = []

        for item in wall_json:
            if item.get("class_name") == "wall":
                x1 = int(item.get("xmin", 0))
                y1 = int(item.get("ymin", 0))
                x2 = int(item.get("xmax", 0))
                y2 = int(item.get("ymax", 0))

                wall_items.append(item)
                walls.append((x1, y1, x2, y2))

        print(f"Detected {len(walls)} walls")


        planV = plan.copy()
        outputImages = {}


        for wall_idx, wall in enumerate(walls):
            matchedMaterials = {}

            x1, y1, x2, y2 = wall
            cv2.rectangle(planV, (x1, y1), (x2, y2), (0, 255, 0), 1)

            # Compare wall with each material
            for mat_idx, material in final_dict.items():
                material_img = material["image"]

                try:
                    is_match, score = self.wall_matches_material_debug(
                        plan,
                        wall,
                        material_img,
                        threshold= 0.85
                    )
                    if is_match:
                        matchedMaterials[mat_idx] = score
                except Exception as e:
                    print(f"Material {mat_idx} failed: {e}")
                    continue

            # Select best material
            if matchedMaterials:
                best_i = min(matchedMaterials, key=matchedMaterials.get)
                detected_text = final_dict[best_i]["text"]
                detected_score = float(matchedMaterials[best_i])
            else:
                detected_text = "could not detect material"
                detected_score = None

            # --------------------------------------------------
            # Update JSON WALL ITEM (THIS WAS THE BUG)
            # --------------------------------------------------
            wall_items[wall_idx]["detected_material"] = detected_text
            if detected_score is not None:
                wall_items[wall_idx]["detected_material_score"] = detected_score

            # --------------------------------------------------
            # Output visualization per material
            # --------------------------------------------------
            if matchedMaterials:
                if best_i not in outputImages:
                    outputImages[best_i] = np.ones_like(plan) * 255

                out = outputImages[best_i]
                out[y1:y2, x1:x2] = plan[y1:y2, x1:x2]

                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = detected_text
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                thickness = 2
                (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)

                cv2.rectangle(out, (0, 0), (tw + 10, th + 10), (255, 255, 255), -1)
                cv2.putText(out, label, (5, th + 5),
                            font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

        # --------------------------------------------------
        # Save outputs
        # --------------------------------------------------
        os.makedirs("materialsOutput", exist_ok=True)
        for k, img in outputImages.items():
            cv2.imwrite(f"materialsOutput/{k + 1}.png", img)

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(wall_json, f, indent=4, ensure_ascii=False)

        cv2.imwrite("planV.png", planV)

        print(f"Processed {len(walls)} walls")
        print(f"Updated JSON saved to: {output_json_path}")

    def extract_wall_coordinates(self, json_file_path):
        """
        Extract just the coordinates of all wall bounding boxes
        
        Returns:
            List of tuples (xmin, ymin, xmax, ymax)
        """
        wall_coords = []
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            if item['class_name'] == 'wall':
                bbox = (int(item.get('xmin', 0)), int(item.get('ymin', 0)), int(item.get('xmax', 0)), int(item.get('ymax', 0)))
                wall_coords.append(bbox)
     
        return wall_coords
    
    def wall_matches_material_debug(
    self,
    plan_img,
    wall_bbox,
    material_img,
    threshold=0.65
):
   
        x0, y0, x1, y1 = wall_bbox
        wall = plan_img[y0:y1, x0:x1]

        material = material_img.copy()
        h, w = material.shape[:2]
        
        mat_color = self.lab_color_hist(material)
        

        mat_texture = self.lbp_texture(material)
        


        patches = self.sample_wall(wall, material)
        patches = [self.normalize_patch(p, w, h) for p in patches]


        color_scores = []
        texture_scores = []
        
        for i, patch in enumerate(patches):
            #print(f"\n--- PATCH {i} ---")

            wall_color = self.lab_color_hist(patch)
            wall_texture = self.lbp_texture(patch)

            c = self.chi2_distance(mat_color, wall_color)
            t = self.chi2_distance(mat_texture, wall_texture)

            #print(f"Color score   : {c:.3f}")
            #print(f"Texture score : {t:.3f}")

            color_scores.append(c)
            texture_scores.append(t)

        best_color = min(color_scores)
        best_texture = min(texture_scores)

        final_score = 0.5 * best_color + 0.5 * best_texture
        is_match = final_score < threshold

        #print("\n=== FINAL RESULT ===")
        #print(f"Best color score   : {best_color:.3f}")
        #print(f"Best texture score : {best_texture:.3f}")
        #print(f"Final score        : {final_score:.3f}")
        #print("MATCH" if is_match else "NO MATCH")

        return is_match, final_score

    def lab_color_hist(self, img, bins=(16, 16, 16)):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        hist = cv2.calcHist([lab], [0, 1, 2], None, bins,
                            [0, 256, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    def lbp_texture(self, img, num_points=24, radius=3):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, num_points, radius, method="uniform")
        bins = num_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins))
        hist = hist.astype(np.float32)
        hist /= hist.sum() + 1e-6
        return hist

    def chi2_distance(self, a, b, eps=1e-10):
        return 0.5 * np.sum(((a - b) ** 2) / (a + b + eps))

    def sample_wall(self, wall_img, material_img, max_patches=12):
        """
        wall_img: numpy array (H x W x C)
        material_img: numpy array (h x w x C)
        Returns: list of wall image patches (numpy arrays)
        """

        mheight, mwidth = material_img.shape[:2]
        wheight, wwidth = wall_img.shape[:2]

        # Compute areas
        marea = mwidth * mheight
        warea = wwidth * wheight

        # If wall is smaller than material → return whole wall
        if warea <= marea:
            return [wall_img]

        # Detect vertical wall
        vertical = wheight > wwidth

        # Rotate wall if vertical (so sampling is horizontal)
        if vertical:
            wall_img = cv2.rotate(wall_img, cv2.ROTATE_90_CLOCKWISE)
            wheight, wwidth = wall_img.shape[:2]

        # Patch size (material-sized)
        patch_w = min(mwidth, wwidth)
        patch_h = min(mheight, wheight)

        # How many patches fit horizontally
        n_patches_possible = wwidth // patch_w
        n_patches = min(max_patches, n_patches_possible)

        if n_patches == 0:
            patches = [wall_img]
        else:
            patches = []
            starts = np.linspace(0, wwidth - patch_w, n_patches_possible).astype(int)
            selected_starts = random.sample(list(starts), n_patches)

            for sx in selected_starts:
                patch = wall_img[0:patch_h, sx:sx + patch_w]
                patches.append(patch)

        # Rotate patches back if wall was vertical
        if vertical:
            patches = [cv2.rotate(p, cv2.ROTATE_90_COUNTERCLOCKWISE) for p in patches]

        print(f"Extracted {len(patches)} patches")
        return patches
    def normalize_patch(self, patch, w=64, h=64):
        return cv2.resize(patch, (w, h), interpolation=cv2.INTER_AREA)
# end region
     
    def run(self):
       
        legend, y_limit = self.extract_legend()
        text_regions = self.texts["text_regions"]
       
        
       
        preprocessedLegend, legend_text = self.preprocess(legend, text_regions, y_limit, self.infoPanel_x)
        
        _, rectangles = self.extract_rectangles(preprocessedLegend)
        
        filtered_rectangles, _ = self.filter_rectangles_by_common_size(rectangles)

        side, matched = self.choose_global_side_and_extract(filtered_rectangles, legend_text)

        final_dict = self.createFinalDict(legend, matched)
    
        self.process_with_final_dict2(rooms_dict= self.rooms, final_dict= final_dict, output_json_path="out.json")
        



def main():
    try:
        infoPanel = cv2.imread("4.png")
        plan = cv2.imread("plan.png")
        infoPanel_x = 3870
        with open('text_data.json', 'r') as json_file:
            data = json.load(json_file)
        #text_regions = data["text_regions"]
        wallFeaturesExtractor = WallFeaturesExtractor(plan, infoPanel, infoPanel_x,"walls3.json", data)
        
        wallFeaturesExtractor.run()
    except Exception as e:
        print(e)


def extract_from_first_horizontal_line(
    image_path,
    min_y=300,
    min_width_ratio=1/2,
    angle_tolerance_deg=1
):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image")

    height, width = img.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines using Probabilistic Hough Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=int(width * min_width_ratio),
        maxLineGap=10
    )

    if lines is None:
        raise ValueError("No lines detected")

    valid_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Compute angle in degrees
        angle = math.degrees(math.atan2((y2 - y1), (x2 - x1)))

        # Check near-horizontal condition
        if abs(angle) <= angle_tolerance_deg:
            line_length = abs(x2 - x1)

            # Check length and vertical position
            if line_length >= width * min_width_ratio and min(y1, y2) >= min_y:
                avg_y = int((y1 + y2) / 2)
                valid_lines.append((avg_y, x1, y1, x2, y2))

    if not valid_lines:
        raise ValueError("No valid horizontal line found")

    # Sort by Y position (top-most first)
    valid_lines.sort(key=lambda x: x[0])

    # Take the first qualifying line
    first_line_y = valid_lines[0][0]

    # Extract the image from that line downward
    extracted = img[:first_line_y, :]

    return extracted, first_line_y

def preprocess(img, original, json_data, y_threshold , x_threshhold, erosion_kernel_size=3):
    """
    Erases text in an image above a given y-coordinate using bounding boxes from JSON,
    applies erosion, and returns the processed image along with a legend of erased texts
    in the same JSON format.

    Args:
        image_path (str): Path to the input image.
        json_data (list): List of dictionaries with "text" and "image_coordinates".
        y_threshold (int or float): Y-coordinate threshold. Only text above this will be erased.
        erosion_kernel_size (int): Size of the kernel used for erosion.

    Returns:
        processed_img (numpy.ndarray): The processed image with text erased and eroded.
        legend_texts (dict): Dictionary of all JSON entries that were erased.
                             Format: {0: {...}, 1: {...}, ...}
    """
 

    # Convert to BGR if grayscale
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    legend_texts = {}

    # Iterate over JSON text entries
    for idx, entry in enumerate(json_data):
        coords = entry.get("image_coordinates", {})
        x0 = int(coords.get("x0"))
        y0 = int(coords.get("y0"))
        x1 = int(coords.get("x1", 0))
        y1 = int(coords.get("y1", 0))
        #print("x0" + str(x0) + "y0 " + str(y0) + "x1" + str(x1) + "y1 " + str(y1) )
        if (entry['text'] == 'Stahlbeton'):
                print("x0 ", x0)
                print("y1", y1)
                cond = (y1 <= y_threshold) and (x0 >= x_threshhold)
                print("yth", y_threshold)
                print("xth", x_threshhold)
                print("cond ", cond)

        # Filter text above y_threshold
        if y1 <= y_threshold and x0 >= x_threshhold:
            print("entry ", entry)
            new_entry = copy.deepcopy(entry)

            # --- shift x coordinates ---
            new_entry["image_coordinates"]["x0"] = x0 - x_threshhold
            new_entry["image_coordinates"]["x1"] = x1 - x_threshhold

            legend_texts[idx] = new_entry

            # Draw white rectangle to "erase" the text
            cv2.rectangle(img, ((x0 - x_threshhold), y0), ((x1 - x_threshhold), y1), (255, 255, 255), thickness= -1)

    # Erode the image to clean up edges
    kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
    processed_img = cv2.erode(img, kernel, iterations=1)
    #print(legend_texts)
    with open('legend.json', 'w') as json_file:
        json.dump(legend_texts, json_file, indent=4)  # indent=2 for 2 spaces
    return processed_img , legend_texts


def extract_rectangles(img, min_area=500, draw=True):
    """
    Detects rectangles in the image and optionally draws them.

    Args:
        image_path (str): Path to the input image.
        min_area (int): Minimum area of rectangles to consider.
        draw (bool): If True, draw rectangles on the image.

    Returns:
        result_img (numpy.ndarray): Image with rectangles drawn.
        rectangles (list): List of rectangle coordinates (x, y, w, h).
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur and threshold for better contour detection
    #blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []

    for cnt in contours:
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Only keep quadrilaterals with sufficient area
        if len(approx) == 4 and cv2.contourArea(approx) > min_area:
            x, y, w, h = cv2.boundingRect(approx)
            rectangles.append((x, y, w, h))
            if draw:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        elif len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 100), 2)
        else:             
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(img, (x, y), (x + w, y + h), (220, 0, 100), 2)
    result_img = img if draw else None
    return result_img, rectangles

def filter_rectangles_by_common_size(
    rectangles,
    size_tolerance_ratio=0.15,
    min_count=3
):
    """
    Filters rectangles by detecting a common rectangle size and
    returning rectangles belonging to the largest size group
    (in terms of number of elements).

    Args:
        rectangles (list): List of (x, y, w, h)
        size_tolerance_ratio (float): Allowed size variation (e.g. 0.1 = ±10%)
        min_count (int): Minimum number of rectangles required to define a common size

    Returns:
        filtered_rectangles (list): Rectangles belonging to the largest size group
        reference_size (tuple | None): (w, h) of detected common size
    """

    if not rectangles:
        return [], None

    # Group rectangles by approximate size
    size_groups = defaultdict(list)

    for rect in rectangles:
        _, _, w, h = rect
        matched = False
        for (gw, gh) in size_groups.keys():
            if (
                abs(w - gw) <= gw * size_tolerance_ratio and
                abs(h - gh) <= gh * size_tolerance_ratio
            ):
                size_groups[(gw, gh)].append(rect)
                matched = True
                break
        if not matched:
            size_groups[(w, h)].append(rect)

    # Select the group with the largest number of elements (>= min_count)
    reference_group = None
    max_count = 0
    for (w, h), group in size_groups.items():
        if len(group) >= min_count:
            if len(group) > max_count:
                max_count = len(group)
                reference_group = (w, h)

    # No valid group found
    if reference_group is None:
        return rectangles, None

    # Return rectangles in the selected group
    filtered_rectangles = size_groups[reference_group]
    print("fc" + str(len(filtered_rectangles)))
    return filtered_rectangles, reference_group

def rectangles_with_right_text_only(img,rectangles, text_dict, y_overlap_ratio=0.25):
    """
    Matches rectangles to the closest right-side word if no left-side word exists.

    Args:
        rectangles (list): List of (x, y, w, h)
        text_dict (dict): {
            id: {
                "text": str,
                "image_coordinates": {"x0","y0","x1","y1"}
            }
        }
        y_overlap_ratio (float): Minimum vertical overlap ratio

    Returns:
        dict: {
            rect_index: {
                "rectangle": (x,y,w,h),
                "text": str,
                "text_box": dict
            }
        }
    """

    results = {}

    for i, (rx, ry, rw, rh) in enumerate(rectangles):
        rect_y0 = ry
        rect_y1 = ry + rh

        has_left_text = False
        right_candidates = []

        for _, entry in text_dict.items():
            text = entry["text"]
            print("material text " + text)
            coords = entry["image_coordinates"]

            wx0 = coords["x0"]
            wy0 = coords["y0"]
            wx1 = coords["x1"]
            wy1 = coords["y1"]

            # --- vertical overlap check ---
            overlap = min(rect_y1, wy1) - max(rect_y0, wy0)
            min_height = min(rh, wy1 - wy0)

            if overlap <= 0 or overlap / min_height < y_overlap_ratio:
                continue

            # --- left text ---
            if wx1 < rx:
                has_left_text = True
                break

            # --- right text ---
            if wx0 > rx + rw:
                distance = wx0 - (rx + rw)
                right_candidates.append((distance, entry))

        # If no left text → choose closest right text
        if not has_left_text and right_candidates:
            right_candidates.sort(key=lambda x: x[0])
            closest = right_candidates[0][1]
            cv2.rectangle(img, (int(wx0), int(wy0)), (int(wx1), int(wy1)), (100,0,255), 2)
            results[i] = {
                "rectangle": (rx, ry, rw, rh),
                "text": closest["text"],
                "text_box": closest["image_coordinates"]
            }

    return img, results



def lab_color_hist(img, bins=(16, 16, 16)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([lab], [0, 1, 2], None, bins,
                        [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()



def lbp_texture(img, num_points=24, radius=3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, num_points, radius, method="uniform")
    bins = num_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins))
    hist = hist.astype(np.float32)
    hist /= hist.sum() + 1e-6
    return hist

def chi2_distance(a, b, eps=1e-10):
    return 0.5 * np.sum(((a - b) ** 2) / (a + b + eps))


def sample_wall(wall_img, material_img, max_patches=12):
    """
    wall_img: numpy array (H x W x C)
    material_img: numpy array (h x w x C)
    Returns: list of wall image patches (numpy arrays)
    """

    mheight, mwidth = material_img.shape[:2]
    wheight, wwidth = wall_img.shape[:2]

    # Compute areas
    marea = mwidth * mheight
    warea = wwidth * wheight

    # If wall is smaller than material → return whole wall
    if warea <= marea:
        return [wall_img]

    # Detect vertical wall
    vertical = wheight > wwidth

    # Rotate wall if vertical (so sampling is horizontal)
    if vertical:
        wall_img = cv2.rotate(wall_img, cv2.ROTATE_90_CLOCKWISE)
        wheight, wwidth = wall_img.shape[:2]

    # Patch size (material-sized)
    patch_w = min(mwidth, wwidth)
    patch_h = min(mheight, wheight)

    # How many patches fit horizontally
    n_patches_possible = wwidth // patch_w
    n_patches = min(max_patches, n_patches_possible)

    if n_patches == 0:
        patches = [wall_img]
    else:
        patches = []
        starts = np.linspace(0, wwidth - patch_w, n_patches_possible).astype(int)
        selected_starts = random.sample(list(starts), n_patches)

        for sx in selected_starts:
            patch = wall_img[0:patch_h, sx:sx + patch_w]
            patches.append(patch)

    # Rotate patches back if wall was vertical
    if vertical:
        patches = [cv2.rotate(p, cv2.ROTATE_90_COUNTERCLOCKWISE) for p in patches]

    print(f"Extracted {len(patches)} patches")
    return patches
         





def sample_wall_strips(
    wall_img,
    n_patches=5,
    strip_length=64,
    min_thickness=8
):
    h, w = wall_img.shape[:2]
    patches = []
    np.random.seed(0)

    vertical = h > w  # wall orientation

    for _ in range(n_patches):
        if vertical:
            thickness = min(w, min_thickness)
            if h < strip_length:
                continue
            y = np.random.randint(0, h - strip_length)
            patch = wall_img[y:y+strip_length, 0:thickness]
        else:
            thickness = min(h, min_thickness)
            if w < strip_length:
                continue
            x = np.random.randint(0, w - strip_length)
            patch = wall_img[0:thickness, x:x+strip_length]

        if patch.size > 0:
            patches.append(patch)

    return patches


def normalize_patch(patch, w=64, h=64):
    return cv2.resize(patch, (w, h), interpolation=cv2.INTER_AREA)

def plot_hist(h1, h2, title):
    plt.figure(figsize=(10, 3))
    plt.plot(h1, label="Material")
    plt.plot(h2, label="Wall Patch")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
def wall_matches_material_debug(
    plan_img,
    wall_bbox,
    material_img,
    threshold=1
):
   
    x0, y0, x1, y1 = wall_bbox
    wall = plan_img[y0:y1, x0:x1]

    material = material_img.copy()
    h, w = material.shape[:2]
    
    mat_color = lab_color_hist(material)
    

    mat_texture = lbp_texture(material)
    print("texture")

    #plot_hist(mat_color, mat_texture, f"Color Histogram – Patch {i}")
    #print("plot")

    patches = sample_wall(wall, material)
    patches = [normalize_patch(p, w, h) for p in patches]
    #print("ya zebi")
    #show_images("Wall + Strip Patches", [wall] + patches)

    color_scores = []
    texture_scores = []
    print("patches " + str(len(patches)))
    for i, patch in enumerate(patches):
        print(f"\n--- PATCH {i} ---")
        #visualize_lab(patch, f"Patch {i} LAB")
        #visualize_lbp(patch, f"Patch {i} LBP")

        wall_color = lab_color_hist(patch)
        wall_texture = lbp_texture(patch)

        c = chi2_distance(mat_color, wall_color)
        t = chi2_distance(mat_texture, wall_texture)

        print(f"Color score   : {c:.3f}")
        print(f"Texture score : {t:.3f}")

        color_scores.append(c)
        texture_scores.append(t)

    best_color = min(color_scores)
    best_texture = min(texture_scores)

    final_score = 0.5 * best_color + 0.5 * best_texture
    is_match = final_score < threshold

    print("\n=== FINAL RESULT ===")
    print(f"Best color score   : {best_color:.3f}")
    print(f"Best texture score : {best_texture:.3f}")
    print(f"Final score        : {final_score:.3f}")
    print("MATCH" if is_match else "NO MATCH")


    return is_match, final_score

def extract_bboxes(json_path, target_class_id=1):
    """
    Extracts bounding boxes from a list of dictionaries for a given class_id.

    Args:
        data (list of dict): The JSON data.
        target_class_id (int): The class_id to filter by (default is 1).

    Returns:
        list of tuples: Each tuple contains (xmin, ymin, xmax, ymax).
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    bboxes = []
    for item in data:
        if item.get("class_id") == target_class_id:
            bbox = (int(item.get('xmin', 0)), int(item.get('ymin', 0)), int(item.get('xmax', 0)), int(item.get('ymax', 0)))
            bboxes.append(bbox)
    print("bboxes " + str(len(bboxes)))
    return bboxes
def extract_wall_coordinates(json_file_path):
    """
    Extract just the coordinates of all wall bounding boxes
    
    Returns:
        List of tuples (xmin, ymin, xmax, ymax)
    """
    wall_coords = []
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        if item['class_name'] == 'wall':
            bbox = (int(item.get('xmin', 0)), int(item.get('ymin', 0)), int(item.get('xmax', 0)), int(item.get('ymax', 0)))
            wall_coords.append(bbox)
    print("walls " + str(len(wall_coords)))
    return wall_coords

def load_detections_from_json(json_path: str):
    walls = []
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    print("data length:", len(data))
    # Handle both single image and array of images format
    if isinstance(data, list):
        boxes_list = data[0].get('boxes', []) if data else []
    else:
        boxes_list = data.get('boxes', [])
    print("data length:", len(boxes_list))
    for box in boxes_list:
        class_name = box.get('class_name', '').lower()
        source = box.get('source', '').lower()
        bbox = (int(box.get('xmin', 0)), int(box.get('ymin', 0)), int(box.get('xmax', 0)), int(box.get('ymax', 0)))

              
        if class_name == 'wall' or source == 'wall':
            walls.append(bbox)
    
    return walls

# input is folder , output list of material images

def load_materials_list(materialsPath: str):
    materials = []

    for filename in os.listdir(materialsPath):
        file_path = os.path.join(materialsPath, filename)

        if not os.path.isfile(file_path):
            continue

        img = cv2.imread(file_path)
        if img is not None:  # ensures it's a valid image
             materials.append(img)

    return materials


def process(imagePath : str, JsonWalls : str, materialsPath: str):
    plan = cv2.imread(imagePath)
    walls = load_detections_from_json(json_path = JsonWalls)
    materials = load_materials_list(materialsPath)
    outputImages = {}
    for wall in walls:
        matchedMaterials = {}
        for i, material in enumerate(materials):
            try :
                    is_match , final_score = wall_matches_material_debug(plan , wall, material_img= material)
                    if (is_match):
                        matchedMaterials[i] = final_score
            except:
                    continue
        
        if matchedMaterials:
            best_i = min(matchedMaterials, key=matchedMaterials.get)
            if (not (best_i in outputImages)):
                outputImages[best_i] = np.ones_like(plan) * 255

            OutputImage = outputImages.get(best_i)
            OutputImage[wall[1] : wall[3], wall[0] : wall[2]]  = plan[wall[1] : wall[3], wall[0] : wall[2]]  
            cv2.rectangle(OutputImage, (wall[0], wall[1]), (wall[2], wall[3]), (0,255,0), 2)      
            
    for key, v in outputImages.items():
        cv2.imwrite("materialsOutput/" + str(key + 1) + ".png", v)


def process_with_final_dict(
    imagePath: str,
    JsonWalls: str,
    final_dict: dict,
    output_json_path: str
):
    # Load image
    plan = cv2.imread(imagePath)
    if plan is None:
        raise ValueError(f"Could not load image from {imagePath}")

    # Ensure image is at least 200x100
    h, w = plan.shape[:2]
    target_w, target_h = max(200, w), max(100, h)
    if w < 200 or h < 100:
        new_plan = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
        new_plan[0:h, 0:w] = plan
        plan = new_plan

    # Load walls JSON
    with open(JsonWalls, "r") as f:
        wall_json = json.load(f)

    # Normalize boxes access
    if isinstance(wall_json, list):
        boxes = wall_json[0].get("boxes", [])
    else:
        boxes = wall_json.get("boxes", [])

    # Get wall coordinates
    walls = extract_wall_coordinates(JsonWalls)

    outputImages = {}
    planV = plan.copy()
    for wall_idx, wall in enumerate(walls):
        matchedMaterials = {}
        cv2.rectangle(planV, (wall[0], wall[1]), (wall[2], wall[3]), (0,255,0), thickness= 1)

        # Check each material against this wall
        for mat_idx, material in final_dict.items():
            material_img = material["image"]
            try:
                is_match, final_score = wall_matches_material_debug(
                    plan,
                    wall,
                    material_img=material_img
                )
                if is_match:
                    matchedMaterials[mat_idx] = final_score
            except Exception as e:
                print(e)
                break

        # Select best match if exists
        if matchedMaterials:
            best_i = min(matchedMaterials, key=matchedMaterials.get)
            detected_text = final_dict[best_i]["text"]
        else:
            detected_text = "could not detect material"

        # Update JSON if box exists for this wall
        if wall_idx < len(boxes):
            boxes[wall_idx]["detected_material"] = detected_text

        # ---- IMAGE OUTPUT ----
        if matchedMaterials:
            if best_i not in outputImages:
                outputImages[best_i] = np.ones_like(plan) * 255

            OutputImage = outputImages[best_i]

            # Copy wall area
            OutputImage[wall[1]:wall[3], wall[0]:wall[2]] = \
                plan[wall[1]:wall[3], wall[0]:wall[2]]

            # Draw rectangle around wall
            cv2.rectangle(
                OutputImage,
                (wall[0], wall[1]),
                (wall[2], wall[3]),
                (0, 255, 0),
                2
            )

            # Draw material text on top-left
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            text_size, _ = cv2.getTextSize(detected_text, font, font_scale, thickness)
            text_w, text_h = text_size

            # Draw background rectangle for readability
            cv2.rectangle(
                OutputImage,
                (0, 0),
                (text_w + 10, text_h + 10),
                (255, 255, 255),
                -1
            )

            cv2.putText(
                OutputImage,
                detected_text,
                (5, text_h + 5),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                lineType=cv2.LINE_AA
            )

    # ---- SAVE OUTPUT IMAGES ----
    os.makedirs("materialsOutput", exist_ok=True)
    for key, img in outputImages.items():
        cv2.imwrite(f"materialsOutput/{key + 1}.png", img)

    # ---- SAVE UPDATED JSON ----
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(wall_json, f, indent=2, ensure_ascii=False)
    cv2.imwrite("planV.png", planV)
    print(f"Processed {len(walls)} walls. JSON saved to {output_json_path}")



if __name__ == "__main__":
    main()


