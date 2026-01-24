import argparse
import json
from typing import List, Dict, Any

import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO


"""This script detects elevators and stairs in a large floorplan image using a trained YOLO model, even when the image is too big to process in one pass.
It splits the image into overlapping tiles (size --tile, overlap --overlap), runs YOLO on each tile with a confidence threshold (--conf), then converts all tile detections back into full-image coordinates by adding each tile’s (tx, ty) offset.

After collecting detections from all tiles, it performs a global class-wise NMS merge (nms_per_class) using --iou_merge, so duplicate boxes from overlapping tiles are removed only within the same class (stairs won’t suppress elevators and vice versa).
It then saves the final merged detections to a JSON file (<out_base>_detections.json) containing class IDs, class names, confidences, and bounding boxes.

Finally, it draws the merged boxes onto the original image as an overlay: elevators and stairs get fixed colors (blue for elevator, red for stairs), each box is labeled with class name + confidence, and an optional extra draw threshold (--min_score_to_draw) can hide low-confidence boxes after merging.
The result is written as <out_base>_overlay.png."""


"""Run this code from the command line similar to this example:
    
     python3 yolo_to_elev_stairs.py \
  --model /Users/albouchiayoub/PycharmProjects/aire-idp/Documentation/Week10/stairs_elevator_yolov8s_640/weights/best.pt \
  --image "/Users/albouchiayoub/PycharmProjects/aire-idp/Final/Model-trained-on-Augmented-German-DATASET/Inference-Exmaples/3/70675-ARC-50006-F-I-A-GR-EG-H05.png" \
  --device mps \
  --conf 0.10 \
  --tile 1024 \
  --overlap 256 \
  --iou_merge 0.5 \
  --out_base result_elev_stairs"""

def tile_positions(length: int, tile_size: int, stride: int) -> List[int]:
    """Return start positions so last tile always reaches the end."""
    if length <= tile_size:
        return [0]
    pos = list(range(0, length - tile_size + 1, stride))
    if pos[-1] + tile_size < length:
        pos.append(length - tile_size)
    return pos


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU for boxes in xyxy format."""
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    return float(inter / (area_a + area_b - inter + 1e-9))


def nms_per_class(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_thr: float
) -> List[int]:
    """
    Greedy NMS, but only suppress boxes of the SAME class.
    Returns list of kept indices.
    """
    if boxes.size == 0:
        return []

    keep: List[int] = []
    idxs = np.argsort(-scores)  # descending by score

    while idxs.size > 0:
        i = int(idxs[0])
        keep.append(i)
        rest = idxs[1:]
        if rest.size == 0:
            break

        survivors: List[int] = []
        for j in rest:
            j = int(j)
            if classes[j] != classes[i]:
                survivors.append(j)
                continue
            if iou_xyxy(boxes[i], boxes[j]) <= iou_thr:
                survivors.append(j)
        idxs = np.array(survivors, dtype=int)

    return keep


# ----------------------------
# Drawing
# ----------------------------

def draw_boxes(
    image_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    names: dict,
    min_score_to_draw: float = 0.0
) -> np.ndarray:
    """
    Draw merged detections on the full image,
    using a different color per class.
    """
    out = image_bgr.copy()
    H, W = out.shape[:2]

    # --- fixed color palette (BGR)
    # extend if you add more classes
    CLASS_COLORS = {
        "elevator": (255, 0, 0),   # blue
        "stairs":   (0, 0, 255),   # red
    }

    DEFAULT_COLOR = (0, 255, 0)   # fallback (green)

    for b, s, c in zip(boxes_xyxy, scores, classes):
        if float(s) < float(min_score_to_draw):
            continue

        x1, y1, x2, y2 = [int(round(v)) for v in b.tolist()]
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W - 1, x2))
        y2 = max(0, min(H - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        cls_id = int(c)
        cls_name = names.get(cls_id, str(cls_id))
        color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)

        label = f"{cls_name} {float(s):.2f}"

        # bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # label background
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        th += baseline
        cv2.rectangle(
            out,
            (x1, max(0, y1 - th - 4)),
            (x1 + tw + 6, y1),
            color,
            -1
        )

        # label text
        cv2.putText(
            out,
            label,
            (x1 + 3, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return out


def save_detections_json(
    path: str,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    names: Dict[int, str]
) -> None:
    data: List[Dict[str, Any]] = []
    for b, s, c in zip(boxes_xyxy, scores, classes):
        cls_id = int(c)
        data.append({
            "class_id": cls_id,
            "class_name": names.get(cls_id, str(cls_id)),
            "confidence": float(s),
            "xyxy": [float(x) for x in b.tolist()],
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ----------------------------
# Main: tiled YOLO -> merge -> draw + json
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to best.pt")
    ap.add_argument("--image", required=True, help="Path to large floorplan image")

    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=256)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--iou_merge", type=float, default=0.5)
    ap.add_argument("--device", default="cpu", help="cpu, mps, 0, 0,1 ...")

    ap.add_argument("--out_base", default="result_elev_stairs")
    ap.add_argument("--min_score_to_draw", type=float, default=0.0, help="Extra draw threshold (after merge).")

    args = ap.parse_args()

    stride = args.tile - args.overlap
    if stride <= 0:
        raise ValueError("overlap must be < tile")

    model = YOLO(args.model)

    # Load image
    img_pil = Image.open(args.image).convert("RGB")
    W, H = img_pil.size

    xs = tile_positions(W, args.tile, stride)
    ys = tile_positions(H, args.tile, stride)

    all_boxes, all_scores, all_classes = [], [], []

    # --- tiled inference
    for ty in ys:
        for tx in xs:
            tile = img_pil.crop((tx, ty, tx + args.tile, ty + args.tile))
            r = model.predict(
                tile,
                imgsz=args.tile,
                conf=args.conf,
                iou=0.7,          # tile-level NMS inside ultralytics
                device=args.device,
                verbose=False
            )[0]

            if r.boxes is None or len(r.boxes) == 0:
                continue

            b = r.boxes.xyxy.cpu().numpy()
            s = r.boxes.conf.cpu().numpy()
            c = r.boxes.cls.cpu().numpy().astype(int)

            # shift from tile coords -> global coords
            b[:, [0, 2]] += tx
            b[:, [1, 3]] += ty

            all_boxes.append(b)
            all_scores.append(s)
            all_classes.append(c)

    if not all_boxes:
        print("Detections: 0 (no boxes).")
        return

    all_boxes = np.concatenate(all_boxes, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    all_classes = np.concatenate(all_classes, axis=0)

    # --- global merge across all tiles
    keep = nms_per_class(all_boxes, all_scores, all_classes, iou_thr=args.iou_merge)
    all_boxes = all_boxes[keep]
    all_scores = all_scores[keep]
    all_classes = all_classes[keep]

    print(f"Detections after merge: {len(all_boxes)}")

    # save JSON detections
    json_path = f"{args.out_base}_detections.json"
    save_detections_json(json_path, all_boxes, all_scores, all_classes, model.names)
    print(f"Saved detections JSON: {json_path}")

    # draw overlay
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    overlay = draw_boxes(
        image_bgr=img_bgr,
        boxes_xyxy=all_boxes,
        scores=all_scores,
        classes=all_classes,
        names=model.names,
        min_score_to_draw=args.min_score_to_draw
    )

    overlay_path = f"{args.out_base}_overlay.png"
    cv2.imwrite(overlay_path, overlay)
    print(f"Saved overlay image: {overlay_path}")


if __name__ == "__main__":
    main()
