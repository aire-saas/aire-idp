import json
import argparse
from bisect import bisect_left

"""
assign_wall_thickness_and_length.py

Purpose
-------
This script post-processes wall detections from a floor-plan detection JSON file.
For each detected wall, it computes:

1. Physical wall thickness
2. Physical wall length

based on the bounding box dimensions and a provided pixel-to-meter scale.
Additionally, the script optionally adjusts ("snaps") the measured wall thickness
to a predefined standard thickness depending on the wall type.

The goal is to reduce small measurement errors caused by detection inaccuracies
while preserving physically realistic wall dimensions.

Overview of Processing
----------------------
For every detected bounding box representing a wall:

1. The bounding box width and height are computed in pixels.
2. The smaller dimension is interpreted as the wall thickness.
3. The larger dimension is interpreted as the wall length.
4. Both values are converted from pixels to meters using a user-provided
   pixels-per-meter (PPM) scale.
5. If a wall type is available and known:
   - The measured thickness is compared against a list of standard
     thicknesses defined for that wall type.
   - The nearest standard thickness is selected.
   - If the difference between the measured thickness and the nearest
     standard is less than 1 cm, the thickness is replaced by the
     standard value.
6. Otherwise, the calculated thickness is kept unchanged.

Important Rules
---------------
- If wall type is missing → use calculated thickness.
- If wall type is unknown → use calculated thickness.
- If difference to nearest standard ≥ 1 cm → use calculated thickness.
- Only small deviations (< 1 cm) are corrected.

Expected Input JSON Structure
-----------------------------
The input JSON must contain entries with bounding boxes, e.g.:

{
  "boxes": [
    {
      "source": "wall",
      "xmin": ...,
      "ymin": ...,
      "xmax": ...,
      "ymax": ...,
      "wall_type": "interior"
    }
  ]
}

A wall is detected when either:
    source == "wall"
or
    class_name == "wall"

Expected Output
---------------
The output JSON preserves the original structure and adds:

- wall_thickness_px
- wall_thickness_m_raw
- wall_thickness_m
- wall_thickness_m_snapped
- wall_thickness_standard_m
- wall_type_used
- wall_length_px
- wall_length_m

Units
-----
Pixel values remain in pixels.
Physical values are stored in meters.

Usage Example
-------------
python3 assign_wall_thickness_and_length.py \
    --input-json input.json \
    --output-json output.json \
    --ppm 117

where:
    ppm = pixels per meter

Assumptions
-----------
- Walls are axis-aligned bounding boxes.
- The smaller bounding box dimension corresponds to thickness.
- The larger bounding box dimension corresponds to length.
- The pixel-to-meter scale is constant across the image.

 
"""

STANDARD_THICKNESS_M ={

    # --- Masonry (Mauerwerk) ---
    "Mauerwerk_tragend_Ziegel":        [0.175, 0.20, 0.24, 0.30, 0.365, 0.42, 0.49],
    "Mauerwerk_tragend_Kalksandstein": [0.115, 0.175, 0.20, 0.24, 0.30],
    "Mauerwerk_tragend_Porenbeton":    [0.20, 0.24, 0.30, 0.365, 0.40],
    "Mauerwerk_nichttragend":          [0.075, 0.10, 0.115, 0.125, 0.15],
    "Wohnungstrennwand_Mauerwerk":     [0.20, 0.24, 0.30],

    # --- Concrete (Beton) ---
    "Stahlbetonwand":                  [0.18, 0.20, 0.24, 0.25, 0.30],
    "WU_Betonwand":                    [0.24, 0.30, 0.365, 0.40],
    "Kellerwand_Stahlbeton":           [0.24, 0.30, 0.365],
    "Brandwand_Beton":                 [0.24, 0.30, 0.365],

    # --- Drywall / Lightweight construction ---
    "Gipskarton_Einfachstaender":      [0.075, 0.10],
    "Gipskarton_Doppelstaender":       [0.125, 0.15, 0.175],
    "Installationswand_Trockenbau":    [0.15, 0.175, 0.20],
    "Wohnungstrennwand_Trockenbau":    [0.20, 0.25, 0.30],

    # --- Exterior wall systems ---
    "Aussenwand_Ziegel_mit_Daemmung":  [0.365, 0.42, 0.49],
    "Aussenwand_KS_WDVS":               [0.30, 0.365, 0.40],
    "Aussenwand_Stahlbeton_WDVS":       [0.30, 0.365, 0.40],
    "Vorsatzschale_Fassade":            [0.10, 0.12, 0.15],

    # --- Special functional walls ---
    "Brandwand_Mauerwerk":              [0.24, 0.30, 0.365],
    "Schallschutzwand":                 [0.20, 0.24, 0.30],
    "Installationswand_massiv":         [0.175, 0.20, 0.24],
    "Schachtwand":                      [0.115, 0.125, 0.15],
}


SNAP_TOLERANCE_M = 0.01  # 1 cm


def _nearest_standard(value_m: float, standards_sorted: list[float]) -> float:
    if not standards_sorted:
        return value_m

    i = bisect_left(standards_sorted, value_m)
    if i == 0:
        return standards_sorted[0]
    if i >= len(standards_sorted):
        return standards_sorted[-1]

    before = standards_sorted[i - 1]
    after = standards_sorted[i]
    return after if (after - value_m) < (value_m - before) else before


def _get_wall_type_or_none(box: dict) -> str | None:
    """
    Returns normalized wall type string if present, otherwise None.
    Adjust keys as needed to match your JSON.
    """
    wt = box.get("wall_type")
    if wt is None:
        wt = box.get("Wall_type")
    if wt is None:
        wt = box.get("wallType")
    if wt is None:
        return None
    wt = str(wt).strip().lower()
    return wt if wt else None


def assign_wall_thickness(input_json, output_json, pixels_per_meter):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    for image_entry in data:
        for box in image_entry.get("boxes", []):
            if box.get("source") != "wall":
                continue

            width_px = abs(box["xmax"] - box["xmin"])
            height_px = abs(box["ymax"] - box["ymin"])

            # thickness = smaller dimension
            thickness_px = min(width_px, height_px)
            thickness_m_raw = thickness_px / pixels_per_meter

            # length = larger dimension
            length_px = max(width_px, height_px)
            length_m = length_px / pixels_per_meter

            # --- snapping logic per your rules ---
            wall_type = _get_wall_type_or_none(box)

            # Default: use calculated thickness
            thickness_m = thickness_m_raw
            snapped = False
            nearest = None

            # Only attempt snapping if wall type exists AND we have standards for it
            if wall_type is not None and wall_type in STANDARD_THICKNESS_M:
                standards_sorted = sorted(STANDARD_THICKNESS_M[wall_type])
                nearest_candidate = _nearest_standard(thickness_m_raw, standards_sorted)
                diff = abs(thickness_m_raw - nearest_candidate)

                # Snap only if difference is LESS than 1cm, otherwise keep calculated
                if diff < SNAP_TOLERANCE_M:
                    thickness_m = nearest_candidate
                    snapped = True
                    nearest = nearest_candidate

            # Write results
            box["wall_thickness_px"] = thickness_px
            box["wall_thickness_m_raw"] = thickness_m_raw
            box["wall_thickness_m"] = thickness_m
            box["wall_thickness_m_snapped"] = snapped
            box["wall_thickness_standard_m"] = nearest  # None if not snapped
            box["wall_type_used"] = wall_type  # None if missing

            box["wall_length_px"] = length_px
            box["wall_length_m"] = length_m

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved wall thickness and length annotations to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assign wall thickness and length using pixel-to-meter scale, with type-based standard snapping"
    )
    parser.add_argument("--input-json", required=True, help="Input detection JSON")
    parser.add_argument("--output-json", required=True, help="Output JSON with wall thickness and length")
    parser.add_argument("--ppm", type=float, required=True, help="PIXELS_PER_METER")

    args = parser.parse_args()

    assign_wall_thickness(
        input_json=args.input_json,
        output_json=args.output_json,
        pixels_per_meter=args.ppm
    )
