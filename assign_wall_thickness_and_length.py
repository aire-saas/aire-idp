import json
import argparse


"""Purpose
Adds a physical thickness AND length to each detected wall in a floor-plan detection JSON.

Expected input
A JSON file containing detection results with bounding boxes, where walls are labeled with "source": "wall".

Expected output
A JSON file with the same structure as the input, but with additional wall thickness
and wall length fields added to each wall entry.

How to run it example:

python3 assign_wall_thickness_and_lenghth.py --input-json input.json --output-json output.json --ppm 117
"""


def assign_wall_thickness(
    input_json,
    output_json,
    pixels_per_meter
):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    for image_entry in data:
        for box in image_entry.get("boxes", []):
            if box.get("source") != "wall":
                continue

            width_px = abs(box["xmax"] - box["xmin"])
            height_px = abs(box["ymax"] - box["ymin"])

            # Wall thickness = smaller dimension
            thickness_px = min(width_px, height_px)
            thickness_m = thickness_px / pixels_per_meter

            # Wall length = larger dimension
            length_px = max(width_px, height_px)
            length_m = length_px / pixels_per_meter

            box["wall_thickness_px"] = thickness_px
            box["wall_thickness_m"] = thickness_m
            box["wall_length_px"] = length_px
            box["wall_length_m"] = length_m

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved wall thickness and length annotations to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assign wall thickness and length using pixel-to-meter scale"
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
