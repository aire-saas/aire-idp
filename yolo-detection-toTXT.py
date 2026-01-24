#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

"""This script runs a YOLO object detection model on a floorplan image and exports the detected objects in both a visual and a textual form.
It supports large images by optionally splitting them into overlapping tiles, running detection on each tile, and merging all detections into global image coordinates using class-wise non-maximum suppression.

After merging, the script saves all detections to a text file in a simple, lossless format containing the class ID, confidence score, and absolute bounding box coordinates.
This makes the output easy to reuse for downstream processing, analysis, or debugging.

In addition, the script generates a mask image where each detected object is drawn as a filled rectangle, using a distinct color per class on a black background.
The final outputs are a detection text file and a colored mask image aligned with the original image dimensions."""


"""
Run example:

python3 yolo-detection-toTXT.py \
  --model /Users/albouchiayoub/PycharmProjects/aire-idp/Documentation/Week9/Model-trained-on-Augmented-German-DATASET/Model-german_ft_1024_aug/weights/best.pt \
  --image '/Users/albouchiayoub/PycharmProjects/aire-idp/Documentation/Week9/4/Screenshot 2026-01-15 at 14.57.39.png' \
  --device mps \
  --conf 0.10 \
  --tile 1024 \
  --overlap 256 \
  --iou_merge 0.5
"""


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


def save_detections_txt(txt_path: Path, boxes: np.ndarray, classes: np.ndarray, scores: np.ndarray):
    """
    Save detections in a simple, lossless format (global/full-image coordinates):
      <class_id> <confidence> <x1> <y1> <x2> <y2>
    """
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w") as f:
        for (x1, y1, x2, y2), cls, score in zip(boxes, classes, scores):
            f.write(
                f"{int(cls)} {float(score):.6f} "
                f"{float(x1):.2f} {float(y1):.2f} "
                f"{float(x2):.2f} {float(y2):.2f}\n"
            )


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
    ap.add_argument("--out_txt", default=None, help="Output path for detections txt. Default: <image>_detections.txt")
    args = ap.parse_args()

    model = YOLO(args.model)

    # Load image via PIL for tiling
    img_pil = Image.open(args.image).convert("RGB")
    W, H = img_pil.size

    all_boxes, all_scores, all_classes = [], [], []

    # Defaults if no detections
    boxes = np.empty((0, 4), dtype=float)
    classes = np.empty((0,), dtype=int)
    scores = np.empty((0,), dtype=float)

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

        if all_boxes:
            boxes = np.concatenate(all_boxes, axis=0)
            scores = np.concatenate(all_scores, axis=0)
            classes = np.concatenate(all_classes, axis=0)

            keep = nms_classwise(boxes, scores, classes, args.iou_merge)
            boxes = boxes[keep]
            scores = scores[keep]
            classes = classes[keep]

    else:
        r = model.predict(img_pil, conf=args.conf, device=args.device, verbose=False)[0]
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)

            keep = nms_classwise(boxes, scores, classes, args.iou_merge)
            boxes = boxes[keep]
            scores = scores[keep]
            classes = classes[keep]

    # Save detections to TXT (full-image coords, after merge)
    txt_path = Path(args.out_txt) if args.out_txt else Path(args.image).with_name(Path(args.image).stem + "_detections.txt")
    save_detections_txt(txt_path, boxes, classes, scores)
    print(f"Saved detections: {txt_path}")

    # Create filled mask (no overlay with original)
    mask = draw_filled_boxes_mask(H, W, boxes, classes)

    out_path = Path(args.out) if args.out else Path(args.image).with_name(Path(args.image).stem + "_mask.png")
    cv2.imwrite(str(out_path), mask)
    print(f"Saved mask: {out_path}")


if __name__ == "__main__":
    main()
