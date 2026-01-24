#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

"""This script runs a YOLO model on a floorplan image and saves a copy of the image with bounding boxes drawn on top of all detected objects.
For large images, it can optionally split the image into overlapping tiles, run YOLO on each tile, and then shift each tile’s detections back into full-image coordinates.

After collecting detections from all tiles (or from the full image if tiling is disabled), it merges duplicates using class-wise non-maximum suppression based on an IoU threshold.
This helps remove repeated boxes that come from overlapping tiles.

Finally, it converts the image to OpenCV format, draws colored rectangles (a deterministic color per class ID), and writes the annotated result to disk (defaulting to <image>_boxes.png)."""




""" Run the script as following from the command line, replace the paths of the model and the floorplan image:


python3 draw_yolo_boxes.py \
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


def draw_boxes(img_bgr, boxes, classes, thickness=2):
    out = img_bgr.copy()
    h, w = out.shape[:2]

    for (x1, y1, x2, y2), c in zip(boxes, classes):
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(0, min(int(x2), w - 1))
        y2 = max(0, min(int(y2), h - 1))

        color = (
            int((37 * (c + 1)) % 255),
            int((17 * (c + 1)) % 255),
            int((97 * (c + 1)) % 255),
        )
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--conf", type=float, default=0.25)

    # Tiling controls (like your working script)
    ap.add_argument("--tile", type=int, default=0, help="Tile size (e.g. 1024). 0 disables tiling.")
    ap.add_argument("--overlap", type=int, default=0, help="Tile overlap (e.g. 256).")
    ap.add_argument("--iou_merge", type=float, default=0.5, help="IoU threshold for merging overlapping detections.")

    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = YOLO(args.model)

    # Load image (use PIL for easy tiling, then convert to cv2 for drawing)
    img_pil = Image.open(args.image).convert("RGB")
    W, H = img_pil.size
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

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

                # shift tile detections back to global coords
                b[:, [0, 2]] += tx
                b[:, [1, 3]] += ty

                all_boxes.append(b)
                all_scores.append(s)
                all_classes.append(c)

        if not all_boxes:
            boxes = np.empty((0, 4))
            scores = np.empty((0,))
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
            boxes = np.empty((0, 4))
            classes = np.empty((0,), dtype=int)
        else:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)

            keep = nms_classwise(boxes, scores, classes, args.iou_merge)
            boxes, classes = boxes[keep], classes[keep]

    drawn = draw_boxes(img_bgr, boxes, classes)

    out_path = args.out or Path(args.image).with_name(Path(args.image).stem + "_boxes.png")
    cv2.imwrite(str(out_path), drawn)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
