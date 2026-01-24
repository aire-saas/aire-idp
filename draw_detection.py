import cv2
import os

IMG_PATH = "/Users/albouchiayoub/PycharmProjects/aire-idp/Documentation/Week10/4/Screenshot 2026-01-15 at 14.57.39.png"
TXT_PATH = "/Users/albouchiayoub/PycharmProjects/aire-idp/runs/detect/predict39/labels/Screenshot 2026-01-15 at 14.57.39.txt"
OUT_PATH = "annotated.png"

# Optional: filter out low-confidence predictions (only applies if confidence exists in txt)
CONF_TH = 0.0  # set e.g. 0.5 to only keep boxes with conf >= 0.5


def parse_line_to_box(line, img_w, img_h):
    """
    Returns (x1, y1, x2, y2, cls, conf) in pixel coords.
    Supports:
      - YOLO prediction:   cls xc yc w h conf
      - YOLO label:        cls xc yc w h
      - Pixel coords:      x1 y1 x2 y2
      - Pixel w/ class:    cls x1 y1 x2 y2
    """
    parts = line.strip().split()
    if not parts:
        return None

    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None

    # YOLO prediction with confidence: cls xc yc w h conf
    if len(nums) == 6:
        cls = int(nums[0])
        xc, yc, bw, bh, conf = nums[1], nums[2], nums[3], nums[4], nums[5]

        # normalized YOLO check
        if all(0.0 <= v <= 1.0 for v in (xc, yc, bw, bh)):
            x1 = (xc - bw / 2) * img_w
            y1 = (yc - bh / 2) * img_h
            x2 = (xc + bw / 2) * img_w
            y2 = (yc + bh / 2) * img_h
            return int(x1), int(y1), int(x2), int(y2), cls, float(conf)
        return None

    # YOLO label (no confidence): cls xc yc w h  OR pixel: cls x1 y1 x2 y2
    if len(nums) == 5:
        cls = int(nums[0])
        a, b, c, d = nums[1], nums[2], nums[3], nums[4]

        # if normalized -> YOLO
        if all(0.0 <= v <= 1.0 for v in (a, b, c, d)):
            xc, yc, bw, bh = a, b, c, d
            x1 = (xc - bw / 2) * img_w
            y1 = (yc - bh / 2) * img_h
            x2 = (xc + bw / 2) * img_w
            y2 = (yc + bh / 2) * img_h
            return int(x1), int(y1), int(x2), int(y2), cls, None

        # otherwise treat as pixel coords: cls x1 y1 x2 y2
        x1, y1, x2, y2 = a, b, c, d
        return int(x1), int(y1), int(x2), int(y2), cls, None

    # pixel coords only: x1 y1 x2 y2
    if len(nums) == 4:
        x1, y1, x2, y2 = nums
        return int(x1), int(y1), int(x2), int(y2), None, None

    return None


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return x1, y1, x2, y2


def color_for_class(cls):
    """
    Deterministic BGR color per class id.
    (OpenCV uses BGR, not RGB.)
    """
    if cls is None:
        return (255, 255, 255)  # white fallback

    # A small palette that looks distinct; cycles if you have more classes.
    palette = [
        (0, 255, 0),     # green
        (0, 0, 255),     # red
        (255, 0, 0),     # blue
        (0, 255, 255),   # yellow
        (255, 0, 255),   # magenta
        (255, 255, 0),   # cyan
        (0, 128, 255),   # orange-ish
        (128, 0, 255),   # purple-ish
        (255, 128, 0),   # light blue-ish
    ]
    return palette[cls % len(palette)]


def main():
    if not os.path.exists(IMG_PATH):
        raise FileNotFoundError(f"Image not found: {IMG_PATH}")
    if not os.path.exists(TXT_PATH):
        raise FileNotFoundError(f"Text file not found: {TXT_PATH}")

    img = cv2.imread(IMG_PATH)
    if img is None:
        raise RuntimeError(f"Failed to load image: {IMG_PATH}")

    h, w = img.shape[:2]

    with open(TXT_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    kept = 0
    skipped = 0

    for line in lines:
        parsed = parse_line_to_box(line, w, h)
        if parsed is None:
            skipped += 1
            continue

        x1, y1, x2, y2, cls, conf = parsed

        # confidence filter (only when conf exists)
        if conf is not None and conf < CONF_TH:
            continue

        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)

        # Draw rectangle with per-class color, no text
        color = color_for_class(cls)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        kept += 1

    cv2.imwrite(OUT_PATH, img)
    print(f"Saved: {OUT_PATH}")
    print(f"Boxes drawn: {kept}, lines skipped (unparsed): {skipped}")


if __name__ == "__main__":
    main()
