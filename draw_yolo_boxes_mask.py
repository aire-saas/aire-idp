#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

"""
This script runs a YOLO model on a floorplan image and produces a colored mask image that visualizes all detected objects as filled bounding boxes.
For large images, it can optionally split the image into overlapping tiles, run YOLO on each tile, shift each tile’s detections back into full-image coordinates, and then merge duplicate detections using class-wise non-maximum suppression.

After merging, it creates a blank (black) image with the same size as the original and draws each detection as a solid rectangle.
Each class ID is mapped to a deterministic BGR color, so objects of the same class always appear in the same color.

The final output is saved as a PNG mask (by default named <image>_mask.png), which contains only the filled detection regions and no overlay of the original floorplan.


"""

"""Run the script as following from the command line, replace the paths of the model and the floorplan image:


python3 draw_yolo_boxes_mask.py \
  --model /Users/albouchiayoub/PycharmProjects/aire-idp/Documentation/Week9/Model-trained-on-Augmented-German-DATASET/Model-german_ft_1024_aug/weights/best.pt \
  --image '/Users/albouchiayoub/PycharmProjects/aire-idp/Documentation/Week9/4/Screenshot 2026-01-15 at 14.57.39.png' \
  --device mps \
  --conf 0.10 \
  --tile 1024 \
  --overlap 256 \
  --iou_merge 0.5"""

def tile_positions(length: int, tile_size: int, stride: int):
    if length <= tile_size:
        return [0]
    pos = list(range(0, length - tile_size + 1, stride))
    if pos[-1] + tile_size < length:
        pos.append(length - tile_size)
    return pos


def iou_xyxy(a, b) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / (area_a + area_b - inter + 1e-9))


def nms_classwise(boxes, scores, classes, iou_thr):
    keep = []
    idxs = np.argsort(-scores)
    while idxs.size > 0:
        i = int(idxs[0])
        keep.append(i)
        rest = idxs[1:]
        if rest.size == 0:
            break

        survivors = []
        for j in rest:
            j = int(j)
            if classes[j] != classes[i]:
                survivors.append(j)
                continue
            if iou_xyxy(boxes[i], boxes[j]) <= iou_thr:
                survivors.append(j)

        idxs = np.array(survivors, dtype=int)

    return np.array(keep, dtype=int)


def class_color(class_id: int):
    """Deterministic BGR color per class id."""
    return (
        int((37 * (class_id + 1)) % 255),
        int((17 * (class_id + 1)) % 255),
        int((97 * (class_id + 1)) % 255),
    )


def draw_filled_boxes_mask(H, W, boxes, classes):
    """
    Returns an HxWx3 BGR mask (black background) with filled rectangles for detections.
    Each class gets a distinct color.
    """
    mask = np.zeros((H, W, 3), dtype=np.uint8)

    for (x1, y1, x2, y2), c in zip(boxes, classes):
        x1 = max(0, min(int(x1), W - 1))
        y1 = max(0, min(int(y1), H - 1))
        x2 = max(0, min(int(x2), W - 1))
        y2 = max(0, min(int(y2), H - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(mask, (x1, y1), (x2, y2), class_color(int(c)), thickness=-1)

    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--conf", type=float, default=0.25)

    # tiling + merge
    ap.add_argument("--tile", type=int, default=0, help="Tile size (e.g. 1024). 0 disables tiling.")
    ap.add_argument("--overlap", type=int, default=0, help="Tile overlap (e.g. 256).")
    ap.add_argument("--iou_merge", type=float, default=0.5, help="IoU threshold for merging overlapping detections.")

    # output
    ap.add_argument("--out", default=None, help="Output path for mask. Default: <image>_mask.png")
    args = ap.parse_args()

    model = YOLO(args.model)

    # Load image via PIL for tiling
    img_pil = Image.open(args.image).convert("RGB")
    W, H = img_pil.size

    all_boxes, all_scores, all_classes = [], [], []

    if args.tile and args.tile > 0:
        stride = args.tile - args.overlap
        if stride <= 0:
            raise ValueError("--overlap must be < --tile")

        xs = tile_positions(W, args.tile, stride)
        ys = tile_positions(H, args.tile, stride)

        for ty in ys:
            for tx in xs:
                tile = img_pil.crop((tx, ty, tx + args.tile, ty + args.tile))
                r = model.predict(
                    tile,
                    imgsz=args.tile,
                    conf=args.conf,
                    device=args.device,
                    verbose=False
                )[0]

                if r.boxes is None or len(r.boxes) == 0:
                    continue

                b = r.boxes.xyxy.cpu().numpy()
                s = r.boxes.conf.cpu().numpy()
                c = r.boxes.cls.cpu().numpy().astype(int)

                # Shift tile coords -> global coords
                b[:, [0, 2]] += tx
                b[:, [1, 3]] += ty

                all_boxes.append(b)
                all_scores.append(s)
                all_classes.append(c)

        if not all_boxes:
            boxes = np.empty((0, 4), dtype=float)
            classes = np.empty((0,), dtype=int)
        else:
            boxes = np.concatenate(all_boxes, axis=0)
            scores = np.concatenate(all_scores, axis=0)
            classes = np.concatenate(all_classes, axis=0)

            keep = nms_classwise(boxes, scores, classes, args.iou_merge)
            boxes, classes = boxes[keep], classes[keep]

    else:
        r = model.predict(img_pil, conf=args.conf, device=args.device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            boxes = np.empty((0, 4), dtype=float)
            classes = np.empty((0,), dtype=int)
        else:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)

            keep = nms_classwise(boxes, scores, classes, args.iou_merge)
            boxes, classes = boxes[keep], classes[keep]

    # Create filled mask (no overlay with original)
    mask = draw_filled_boxes_mask(H, W, boxes, classes)

    out_path = args.out or Path(args.image).with_name(Path(args.image).stem + "_mask.png")
    cv2.imwrite(str(out_path), mask)
    print(f"Saved mask: {out_path}")


if __name__ == "__main__":
    main()
