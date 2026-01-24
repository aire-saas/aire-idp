import argparse
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO


"""This script takes a large floorplan image and uses a trained YOLO model to automatically detect structural elements such as walls, doors, and windows.
Because the image can be larger than what YOLO can process at once, it is split into overlapping tiles, and detections from all tiles are merged using non-maximum suppression to remove duplicates.

From the merged detections, the script builds a binary “closed wall” mask by drawing walls and optionally closing doors and windows, then dilating the result to seal small gaps.
This mask represents solid barriers, while the remaining pixels correspond to free space.

Finally, the script identifies rooms as connected components in the free space, filters out very small regions, and overlays each detected room with a unique color on the original image.
It saves both the wall mask and the final room-colored floorplan as output images."""


"""Run the script as following from the command line, replace the paths of the model and the floorplan image:



yolo_to_rooms.py \
  --model /Users/albouchiayoub/PycharmProjects/aire-idp/Documentation/Week9/Model-trained-on-Augmented-German-DATASET/Model-german_ft_1024_aug/weights/best.pt \
  --image '/Users/albouchiayoub/PycharmProjects/aire-idp/Documentation/Week9/4/Screenshot 2026-01-15 at 14.57.39.png' \
  --device mps \
  --conf 0.10 \
  --tile 1024 \
  --overlap 256 \
  --iou_merge 0.5 \
  --dilate 2 \
  --min_room_area 5000"""


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


def nms_per_class(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thr: float):
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
    return keep


# ----------------------------
# Build closed wall mask from YOLO boxes
# ----------------------------

def build_closed_wall_mask(
    W: int,
    H: int,
    boxes_xyxy: np.ndarray,
    classes: np.ndarray,
    names: dict,
    wall_thickness: int = 3,
    close_door_window: bool = True,
    close_thickness: int = 7,
    dilate: int = 3,
):
    """
    Returns a binary mask: walls/closed openings = 255, free space = 0.

    - walls: draw as filled rectangles (from YOLO wall boxes)
    - doors/windows: optionally draw as walls too (closing openings)
    - dilate: optional dilation to seal small gaps
    """
    mask = np.zeros((H, W), dtype=np.uint8)

    # map class name -> id (robust to ordering)
    name_to_id = {v: k for k, v in names.items()}
    wall_id = name_to_id.get("wall", None)
    door_id = name_to_id.get("door", None)
    window_id = name_to_id.get("window", None)

    def draw_rect(b, thickness_fill=0):
        x1, y1, x2, y2 = map(int, b)
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W - 1, x2))
        y2 = max(0, min(H - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return
        # filled rectangle
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

    # 1) draw walls
    if wall_id is not None:
        for b, c in zip(boxes_xyxy, classes):
            if int(c) == int(wall_id):
                draw_rect(b)

    # 2) close doors/windows by drawing them as walls too
    if close_door_window:
        for b, c in zip(boxes_xyxy, classes):
            if door_id is not None and int(c) == int(door_id):
                draw_rect(b)
            if window_id is not None and int(c) == int(window_id):
                draw_rect(b)

    # 3) Optional: thicken/close tiny gaps (very important in practice)
    # Dilation makes black barriers thicker -> seals cracks due to imperfect boxes
    if dilate and dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate * 2 + 1, dilate * 2 + 1))
        mask = cv2.dilate(mask, k, iterations=1)

    return mask


# ----------------------------
# Rooms = connected components in free-space
# ----------------------------

def colorize_rooms_from_wall_mask(original_bgr: np.ndarray, wall_mask_255: np.ndarray, min_area: int = 5000, alpha: float = 0.45):
    """
    wall_mask_255: 255 where walls/closed openings exist, 0 elsewhere.
    rooms are connected components in free space => invert mask.
    """
    H, W = wall_mask_255.shape[:2]
    free = cv2.bitwise_not(wall_mask_255)  # free space becomes 255

    # Optional: remove outside area by flooding from border (keeps only interior rooms)
    # This step helps avoid coloring the "outside" of the plan as a room.
    flood = free.copy()
    ff_mask = np.zeros((H + 2, W + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 0)  # remove outside connected to (0,0)
    free_interior = flood  # 255 only in enclosed spaces

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(free_interior, connectivity=8)

    out = original_bgr.copy()
    room_count = 0

    # generate colors
    colors = []
    for i in range(max(0, num_labels - 1)):
        hue = int(180 * i / max(1, (num_labels - 1)))
        col = cv2.cvtColor(np.uint8([[[hue, 255, 200]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(x) for x in col))

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        color = colors[room_count % len(colors)] if colors else (0, 255, 0)
        overlay = np.zeros_like(out)
        overlay[labels == label] = color

        mask3 = np.stack([labels == label] * 3, axis=-1)
        out = np.where(mask3, cv2.addWeighted(out, 1 - alpha, overlay, alpha, 0), out)
        room_count += 1

    return out, room_count


# ----------------------------
# Main: tiled YOLO -> closed mask -> rooms
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to best.pt")
    ap.add_argument("--image", required=True, help="Path to large floorplan image")

    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=int, default=160)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--iou_merge", type=float, default=0.5)
    ap.add_argument("--device", default="cpu", help="cpu, mps, 0, 0,1 ...")

    # mask/room params
    ap.add_argument("--dilate", type=int, default=3, help="Dilate wall mask (seal small gaps). 0 disables.")
    ap.add_argument("--min_room_area", type=int, default=5000, help="Ignore tiny components.")
    ap.add_argument("--alpha", type=float, default=0.45, help="Room overlay transparency.")

    ap.add_argument("--out_base", default="result")
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
            r = model.predict(tile, imgsz=args.tile, conf=args.conf, iou=0.7, device=args.device, verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue

            b = r.boxes.xyxy.cpu().numpy()
            s = r.boxes.conf.cpu().numpy()
            c = r.boxes.cls.cpu().numpy().astype(int)

            b[:, [0, 2]] += tx
            b[:, [1, 3]] += ty

            all_boxes.append(b)
            all_scores.append(s)
            all_classes.append(c)

    if not all_boxes:
        print("Detections: 0 (no boxes) -> cannot build room mask.")
        return

    all_boxes = np.concatenate(all_boxes, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    all_classes = np.concatenate(all_classes, axis=0)

    # --- global merge
    keep = nms_per_class(all_boxes, all_scores, all_classes, iou_thr=args.iou_merge)
    all_boxes = all_boxes[keep]
    all_scores = all_scores[keep]
    all_classes = all_classes[keep]

    print(f"Detections after merge: {len(all_boxes)}")

    # --- Build closed wall mask from YOLO detections
    wall_mask = build_closed_wall_mask(
        W=W, H=H,
        boxes_xyxy=all_boxes,
        classes=all_classes,
        names=model.names,
        close_door_window=True,
        dilate=args.dilate
    )

    mask_path = f"{args.out_base}_mask.png"
    cv2.imwrite(mask_path, wall_mask)
    print(f"Saved closed-wall mask: {mask_path}")

    # --- Rooms from wall mask
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    rooms_bgr, room_count = colorize_rooms_from_wall_mask(
        original_bgr=img_bgr,
        wall_mask_255=wall_mask,
        min_area=args.min_room_area,
        alpha=args.alpha
    )

    rooms_path = f"{args.out_base}_rooms.png"
    cv2.imwrite(rooms_path, rooms_bgr)
    print(f"Rooms detected: {room_count}")
    print(f"Saved rooms overlay: {rooms_path}")

    # Also save rooms overlay on original (same as rooms_path here)
    # If you want a separate copy, uncomment:
    # rooms_on_original = f"{args.out_base}_rooms_on_original.png"
    # cv2.imwrite(rooms_on_original, rooms_bgr)

if __name__ == "__main__":
    main()

