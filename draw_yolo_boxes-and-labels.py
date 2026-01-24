#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import cv2
from ultralytics import YOLO
"""
This script runs a YOLO model on an input image and saves a copy of the image with the detected objects drawn as outlined bounding boxes.
It can process the full image in one pass or, for large images, split it into overlapping tiles, run YOLO on each tile, and shift tile detections back into full-image coordinates.

After collecting detections, it applies class-wise non-maximum suppression to remove duplicate overlapping boxes of the same class based on an IoU threshold.
It then draws rectangles on the original image using a deterministic color per class and writes the final annotated image to disk (defaulting to <image>_boxes.png), printing the number of detections and the output path.
"""

def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w - 1))
    y2 = max(0, min(int(round(y2)), h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union


def nms_classwise(boxes, scores, classes, iou_thr):
    keep = []
    idxs = np.argsort(-scores)
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        rest = idxs[1:]
        if rest.size == 0:
            break

        suppress = []
        for rpos, j in enumerate(rest):
            if classes[j] != classes[i]:
                continue
            if iou_xyxy(boxes[i], boxes[j]) > iou_thr:
                suppress.append(rpos)

        if suppress:
            mask = np.ones(rest.shape[0], dtype=bool)
            mask[suppress] = False
            rest = rest[mask]

        idxs = rest
    return np.array(keep, dtype=int)


def predict_full(model, img_bgr, conf, device):
    # IMPORTANT: do NOT pass "overlap" to Ultralytics predict()
    results = model.predict(img_bgr, conf=conf, device=device, verbose=False)
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return (
            np.zeros((0, 4), dtype=float),
            np.zeros((0,), dtype=float),
            np.zeros((0,), dtype=int),
        )


    boxes = r.boxes.xyxy.cpu().numpy().astype(float)
    scores = r.boxes.conf.cpu().numpy().astype(float)
    classes = r.boxes.cls.cpu().numpy().astype(int)
    return boxes, scores, classes


def predict_tiled(model, img_bgr, tile, overlap, conf, device):
    H, W = img_bgr.shape[:2]
    step = max(1, tile - overlap)

    all_boxes = []
    all_scores = []
    all_classes = []

    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            x1 = min(x0 + tile, W)
            y1 = min(y0 + tile, H)

            x0_eff = max(0, x1 - tile)
            y0_eff = max(0, y1 - tile)

            patch = img_bgr[y0_eff:y1, x0_eff:x1]

            boxes, scores, classes = predict_full(model, patch, conf=conf, device=device)
            if boxes.shape[0] == 0:
                continue

            boxes[:, [0, 2]] += x0_eff
            boxes[:, [1, 3]] += y0_eff

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_classes.append(classes)

    if not all_boxes:
        return (
            np.zeros((0, 4), dtype=float),
            np.zeros((0,), dtype=float),
            np.zeros((0,), dtype=int),
        )

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    classes = np.concatenate(all_classes, axis=0)
    return boxes, scores, classes


def draw_boxes(img_bgr, boxes, classes, thickness=2):
    out = img_bgr.copy()
    H, W = out.shape[:2]

    for (x1, y1, x2, y2), c in zip(boxes, classes):
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H)

        color = (
            int((37 * (c + 1)) % 255),
            int((17 * (c + 1)) % 255),
            int((97 * (c + 1)) % 255),
        )

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

    return out


def main():
    p = argparse.ArgumentParser(description="Draw YOLO bounding boxes on an input image (with optional tiling).")
    p.add_argument("--model", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--tile", type=int, default=0)
    p.add_argument("--overlap", type=int, default=0)
    p.add_argument("--iou_merge", type=float, default=0.5)
    args = p.parse_args()

    img_path = Path(args.image)
    out_path = Path(args.out) if args.out else img_path.with_name(img_path.stem + "_boxes.png")

    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise SystemExit(f"Could not read image: {img_path}")

    model = YOLO(args.model)

    if args.tile and args.tile > 0:
        boxes, scores, classes = predict_tiled(
            model, img_bgr, tile=args.tile, overlap=args.overlap, conf=args.conf, device=args.device
        )
    else:
        boxes, scores, classes = predict_full(model, img_bgr, conf=args.conf, device=args.device)

    if boxes.shape[0] > 0:
        keep = nms_classwise(boxes, scores, classes, args.iou_merge)
        boxes, classes = boxes[keep], classes[keep]

    drawn = draw_boxes(img_bgr, boxes, classes)

    if not cv2.imwrite(str(out_path), drawn):
        raise SystemExit(f"Failed to write output: {out_path}")

    print(f"Detections: {len(boxes)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
