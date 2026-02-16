# wall_types_from_legend_ocr.py
# Usage example:
# python3 wall_types_from_legend_ocr.py \
#   --plan_image "/path/to/floorplan.png" \
#   --legend_image "/path/to/legend_crop.png" \
#   --txt "/path/to/inference.txt" \
#   --out_base "test1" \
#   --min_conf 0.10 \
#   --dist_threshold 0.35 \
#   --keep_classes "1" \
#   --save_debug_crops
#
# Supports TXT formats:
#   A) cls conf x1 y1 x2 y2    (pixel xyxy)   <-- your current file
#   B) cls cx cy w h conf      (YOLO normalized)

import argparse
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image

# ----------------------------
# Optional OCR backends
# ----------------------------
_OCR_BACKEND = "none"
_easyocr_reader = None
try:
    import easyocr  # type: ignore
    _easyocr_reader = easyocr.Reader(["de", "en"], gpu=False)
    _OCR_BACKEND = "easyocr"
except Exception:
    try:
        import pytesseract  # type: ignore
        _OCR_BACKEND = "tesseract"
    except Exception:
        _OCR_BACKEND = "none"


def ocr_text(bgr: np.ndarray) -> str:
    """OCR a BGR crop. EasyOCR if available, else pytesseract, else ''."""
    if bgr is None or bgr.size == 0:
        return ""

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if _OCR_BACKEND == "easyocr":
        rgb = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
        try:
            res = _easyocr_reader.readtext(rgb, detail=0, paragraph=True)  # type: ignore
            txt = " ".join(res).strip() if res else ""
        except Exception:
            txt = ""
    elif _OCR_BACKEND == "tesseract":
        try:
            txt = pytesseract.image_to_string(bw, lang="deu+eng", config="--psm 6")  # type: ignore
            txt = txt.strip()
        except Exception:
            txt = ""
    else:
        txt = ""

    txt = txt.replace("\n", " ").replace("\t", " ").strip()
    while "  " in txt:
        txt = txt.replace("  ", " ")
    return txt


# ============================================================
# 1) TXT loader (autodetects your pixel-xyxy format)
# ============================================================

def load_detections_txt(
    path: str,
    W: int,
    H: int,
    min_conf: float = 0.0,
    classes_keep: Optional[set] = None
) -> List[Dict[str, Any]]:
    """
    Supports:
      A) cls conf x1 y1 x2 y2    (pixel xyxy)   <-- your current file
      B) cls cx cy w h conf      (YOLO normalized)
    """
    dets: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 6:
                continue

            cls = int(float(parts[0]))
            v1 = float(parts[1])
            v2 = float(parts[2])

            # Heuristic: if "conf" is in [0..1.5] and next values are > 1.5, it's pixel xyxy.
            # Your format: cls conf x1 y1 x2 y2
            if v1 <= 1.5 and v2 > 1.5:
                conf = v1
                x1 = float(parts[2]); y1 = float(parts[3]); x2 = float(parts[4]); y2 = float(parts[5])
                x1 = int(round(x1)); y1 = int(round(y1)); x2 = int(round(x2)); y2 = int(round(y2))
            else:
                # YOLO normalized: cls cx cy w h conf
                cx = float(parts[1]); cy = float(parts[2]); w = float(parts[3]); h = float(parts[4]); conf = float(parts[5])
                bw = w * W
                bh = h * H
                x1 = int(round(cx * W - bw / 2))
                y1 = int(round(cy * H - bh / 2))
                x2 = int(round(cx * W + bw / 2))
                y2 = int(round(cy * H + bh / 2))

            if conf < min_conf:
                continue
            if classes_keep is not None and cls not in classes_keep:
                continue

            x1 = max(0, min(W - 1, x1))
            y1 = max(0, min(H - 1, y1))
            x2 = max(0, min(W - 1, x2))
            y2 = max(0, min(H - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            dets.append({"cls": cls, "conf": float(conf), "xyxy": [x1, y1, x2, y2]})
    return dets


# ============================================================
# 2) Texture descriptor (LBP + HOG + stats + orientation hist)
#    + IMPORTANT: use inner-crop to remove borders/frames
# ============================================================

def inner_crop(gray: np.ndarray, margin_frac: float = 0.12) -> np.ndarray:
    h, w = gray.shape[:2]
    mx = int(w * margin_frac)
    my = int(h * margin_frac)
    x1, y1 = mx, my
    x2, y2 = w - mx, h - my
    if x2 <= x1 + 5 or y2 <= y1 + 5:
        return gray
    return gray[y1:y2, x1:x2]

def preprocess(gray: np.ndarray, size: int = 128) -> np.ndarray:
    patch = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    patch = cv2.equalizeHist(patch)
    _, bw = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw

def lbp_hist_8(bw: np.ndarray) -> np.ndarray:
    img = bw.astype(np.uint8)
    h, w = img.shape
    if h < 3 or w < 3:
        return np.zeros((256,), dtype=np.float32)
    center = img[1:-1, 1:-1]
    codes = np.zeros_like(center, dtype=np.uint8)
    offsets = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]
    for bit, (dy, dx) in enumerate(offsets):
        neigh = img[1+dy:h-1+dy, 1+dx:w-1+dx]
        codes |= ((neigh >= center) << bit).astype(np.uint8)
    hist = np.bincount(codes.ravel(), minlength=256).astype(np.float32)
    hist /= (hist.sum() + 1e-9)
    return hist

def hog_feat(bw: np.ndarray) -> np.ndarray:
    hog = cv2.HOGDescriptor(
        _winSize=(128, 128),
        _blockSize=(32, 32),
        _blockStride=(16, 16),
        _cellSize=(16, 16),
        _nbins=9
    )
    f = hog.compute(bw).reshape(-1).astype(np.float32)
    f /= (np.linalg.norm(f) + 1e-9)
    return f

def edge_density(gray_128: np.ndarray) -> float:
    e = cv2.Canny(gray_128, 50, 150)
    return float((e > 0).mean())

def ink_ratio(bw: np.ndarray) -> float:
    return float((bw < 128).mean())

def orientation_hist(gray: np.ndarray, bins: int = 12) -> np.ndarray:
    g = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.arctan2(gy, gx) + np.pi) * (bins / (2 * np.pi))
    ang = np.clip(ang.astype(np.int32), 0, bins - 1)

    hist = np.zeros((bins,), dtype=np.float32)
    for b in range(bins):
        hist[b] = float(mag[ang == b].sum())
    hist /= (hist.sum() + 1e-9)
    return hist

def descriptor(gray_patch: np.ndarray) -> np.ndarray:
    # remove borders (very important for legend swatches + wall strips)
    gray_patch = inner_crop(gray_patch, 0.10)

    bw = preprocess(gray_patch, size=128)
    gray_128 = cv2.resize(gray_patch, (128, 128), interpolation=cv2.INTER_AREA)

    f1 = lbp_hist_8(bw)
    f2 = hog_feat(bw)
    f3 = np.array([ink_ratio(bw), edge_density(gray_128)], dtype=np.float32)
    f4 = orientation_hist(gray_patch, bins=12)

    feat = np.concatenate([f1, f2, f3, f4]).astype(np.float32)
    feat /= (np.linalg.norm(feat) + 1e-9)
    return feat

def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.clip(np.dot(a, b), -1.0, 1.0))


# ============================================================
# 3) Legend swatches detection + OCR labels
# ============================================================

@dataclass
class SwatchRef:
    idx: int
    swatch_xyxy: Tuple[int, int, int, int]   # in legend coords
    label_xyxy: Tuple[int, int, int, int]    # in legend coords
    label_text: str
    feat: np.ndarray

def auto_extract_swatch_boxes(legend_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(legend_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]

    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 7
    )
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=2)

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < 500 or area > (W * H * 0.40):
            continue
        if w < 28 or h < 18:
            continue

        ar = w / float(h + 1e-9)
        if ar < 1.2 or ar > 6.0:
            continue

        cnt_area = cv2.contourArea(c)
        rectangularity = cnt_area / float(area + 1e-9)
        if rectangularity < 0.45:
            continue

        # left-side bias (labels on the right)
        if x > int(W * 0.58):
            continue

        # must have texture
        roi = gray[y:y+h, x:x+w]
        roi_small = cv2.resize(roi, (96, 96), interpolation=cv2.INTER_AREA)
        edges = cv2.Canny(roi_small, 50, 150)
        ed = float((edges > 0).mean())
        if ed < 0.015:
            continue

        candidates.append((x, y, x + w, y + h, w, h))

    if not candidates:
        return []

    ws = np.array([c[4] for c in candidates], dtype=np.float32)
    hs = np.array([c[5] for c in candidates], dtype=np.float32)
    w_med = float(np.median(ws))
    h_med = float(np.median(hs))

    filtered = []
    for (x1, y1, x2, y2, w, h) in candidates:
        if not (0.70 * w_med <= w <= 1.45 * w_med):
            continue
        if not (0.70 * h_med <= h <= 1.45 * h_med):
            continue
        filtered.append((x1, y1, x2, y2))

    # de-duplicate
    filtered = sorted(filtered, key=lambda b: (b[1], b[0]))
    out = []
    for b in filtered:
        x1, y1, x2, y2 = b
        area_b = (x2 - x1) * (y2 - y1)
        keep = True
        for bb in out:
            X1, Y1, X2, Y2 = bb
            ix1, iy1 = max(x1, X1), max(y1, Y1)
            ix2, iy2 = min(x2, X2), min(y2, Y2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if area_b > 0 and inter / area_b > 0.80:
                keep = False
                break
        if keep:
            out.append(b)

    return sorted(out, key=lambda b: (b[1], b[0]))

def label_crop_for_swatch(
    legend_bgr: np.ndarray,
    swatch_xyxy: Tuple[int, int, int, int],
    gap: int = 8,
    right_frac: float = 0.50,
    y_pad: int = 10,
) -> Tuple[int, int, int, int]:
    H, W = legend_bgr.shape[:2]
    x1, y1, x2, y2 = swatch_xyxy

    lx1 = min(W - 1, x2 + gap)
    lx2 = min(W, x2 + int(W * right_frac))
    ly1 = max(0, y1 - y_pad)
    ly2 = min(H, y2 + y_pad)

    lx2 = max(lx2, lx1 + 1)
    ly2 = max(ly2, ly1 + 1)
    return lx1, ly1, lx2, ly2

def build_swatch_refs_with_ocr(
    legend_bgr: np.ndarray,
    label_gap: int,
    label_right_frac: float,
    label_y_pad: int,
) -> List[SwatchRef]:
    boxes = auto_extract_swatch_boxes(legend_bgr)
    gray = cv2.cvtColor(legend_bgr, cv2.COLOR_BGR2GRAY)

    refs: List[SwatchRef] = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        sw_patch = gray[y1:y2, x1:x2]
        # important: descriptor() does inner crop to ignore borders
        feat = descriptor(sw_patch)

        lx1, ly1, lx2, ly2 = label_crop_for_swatch(
            legend_bgr, (x1, y1, x2, y2),
            gap=label_gap, right_frac=label_right_frac, y_pad=label_y_pad
        )
        lab_crop = legend_bgr[ly1:ly2, lx1:lx2]
        text = ocr_text(lab_crop)

        refs.append(SwatchRef(
            idx=i,
            swatch_xyxy=(x1, y1, x2, y2),
            label_xyxy=(lx1, ly1, lx2, ly2),
            label_text=text,
            feat=feat
        ))
    return refs


# ============================================================
# 4) Robust wall->swatch matching:
#    - sample multiple patches inside wall box
#    - compute median distance (robust to junctions / white space)
# ============================================================

def match_wall_to_swatch_robust(
    gray_full: np.ndarray,
    wall_xyxy: Tuple[int, int, int, int],
    swatches: List[SwatchRef],
    n_samples: int = 9
) -> Tuple[int, float]:
    x1, y1, x2, y2 = wall_xyxy
    w, h = x2 - x1, y2 - y1
    if w < 20 or h < 20 or not swatches:
        return -1, 1e9

    grid = int(np.sqrt(n_samples))
    if grid < 2:
        grid = 2

    xs = np.linspace(x1 + 0.2 * w, x1 + 0.8 * w, grid)
    ys = np.linspace(y1 + 0.2 * h, y1 + 0.8 * h, grid)

    base = max(24, int(min(w, h) * 0.9))
    pw = min(base, w)
    ph = min(base, h)

    feats = []
    for cx in xs:
        for cy in ys:
            cx = int(cx); cy = int(cy)
            sx1 = max(x1, cx - pw // 2)
            sy1 = max(y1, cy - ph // 2)
            sx2 = min(x2, sx1 + pw)
            sy2 = min(y2, sy1 + ph)
            if sx2 <= sx1 or sy2 <= sy1:
                continue
            patch = gray_full[sy1:sy2, sx1:sx2]
            feats.append(descriptor(patch))

    if not feats:
        return -1, 1e9

    best_idx, best_d = -1, 1e9
    for s in swatches:
        ds = [cosine_dist(f, s.feat) for f in feats]
        d_med = float(np.median(ds))
        if d_med < best_d:
            best_d = d_med
            best_idx = s.idx

    return best_idx, best_d


# ============================================================
# 5) Visualization / outputs
# ============================================================

def color_for_idx(i: int) -> Tuple[int, int, int]:
    hue = int((i * 37) % 180)
    bgr = cv2.cvtColor(np.uint8([[[hue, 255, 220]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

def draw_legend_debug(legend_bgr: np.ndarray, swatches: List[SwatchRef]) -> np.ndarray:
    out = legend_bgr.copy()
    for s in swatches:
        col = color_for_idx(s.idx)
        x1, y1, x2, y2 = s.swatch_xyxy
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.putText(out, f"type {s.idx}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)

        lx1, ly1, lx2, ly2 = s.label_xyxy
        cv2.rectangle(out, (lx1, ly1), (lx2, ly2), col, 1)

        txt = s.label_text.strip()
        if len(txt) > 28:
            txt = txt[:28] + "…"
        if txt:
            cv2.putText(out, txt, (lx1, max(0, ly1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
    return out

def draw_plan_overlay(plan_bgr: np.ndarray, dets: List[Dict[str, Any]], assigned: List[Dict[str, Any]]) -> np.ndarray:
    out = plan_bgr.copy()
    for det, a in zip(dets, assigned):
        x1, y1, x2, y2 = det["xyxy"]
        sw = a["swatch_idx"]
        dist = a["distance"]
        name = a["wall_type"]

        if sw >= 0:
            col = color_for_idx(sw)
            label = f"{name}  d={dist:.2f}".strip() if name else f"type {sw}  d={dist:.2f}"
        else:
            col = (0, 0, 255)
            label = f"unknown d={dist:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.putText(out, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out

def save_debug_crops(legend_bgr: np.ndarray, swatches: List[SwatchRef], out_base: str):
    for s in swatches:
        x1, y1, x2, y2 = s.swatch_xyxy
        lx1, ly1, lx2, ly2 = s.label_xyxy
        cv2.imwrite(f"{out_base}_swatch_{s.idx}.png", legend_bgr[y1:y2, x1:x2])
        cv2.imwrite(f"{out_base}_label_{s.idx}.png", legend_bgr[ly1:ly2, lx1:lx2])


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan_image", required=True)
    ap.add_argument("--legend_image", required=True)
    ap.add_argument("--txt", required=True, help="Detections TXT (pixel xyxy or YOLO normalized)")

    ap.add_argument("--out_base", default="walltypes")
    ap.add_argument("--min_conf", type=float, default=0.10)
    ap.add_argument("--dist_threshold", type=float, default=0.35)
    ap.add_argument("--keep_classes", default="", help="Comma-separated class ids to keep (e.g. '1' or '1,2')")

    ap.add_argument("--label_gap", type=int, default=8)
    ap.add_argument("--label_right_frac", type=float, default=0.50)
    ap.add_argument("--label_y_pad", type=int, default=10)

    ap.add_argument("--save_debug_crops", action="store_true")
    args = ap.parse_args()

    plan_pil = Image.open(args.plan_image).convert("RGB")
    plan_bgr = cv2.cvtColor(np.array(plan_pil), cv2.COLOR_RGB2BGR)
    H, W = plan_bgr.shape[:2]

    legend_pil = Image.open(args.legend_image).convert("RGB")
    legend_bgr = cv2.cvtColor(np.array(legend_pil), cv2.COLOR_RGB2BGR)

    classes_keep = None
    if args.keep_classes.strip():
        classes_keep = set(int(x.strip()) for x in args.keep_classes.split(",") if x.strip())

    print(f"OCR backend: {_OCR_BACKEND}")

    # Swatches + OCR
    swatches = build_swatch_refs_with_ocr(
        legend_bgr,
        label_gap=args.label_gap,
        label_right_frac=args.label_right_frac,
        label_y_pad=args.label_y_pad
    )
    print(f"Swatches detected: {len(swatches)}")
    if not swatches:
        print("ERROR: No swatches detected. Use a tighter legend crop or adjust detection thresholds.")
        return

    # Save legend debug + mapping
    legend_debug = draw_legend_debug(legend_bgr, swatches)
    leg_path = f"{args.out_base}_legend_swatches.png"
    cv2.imwrite(leg_path, legend_debug)
    print(f"Saved: {leg_path}")

    mapping = {str(s.idx): s.label_text for s in swatches}
    map_path = f"{args.out_base}_legend_mapping.json"
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"Saved: {map_path}")

    if args.save_debug_crops:
        save_debug_crops(legend_bgr, swatches, args.out_base)
        print(f"Saved: {args.out_base}_swatch_*.png and {args.out_base}_label_*.png")

    # Load detections
    dets = load_detections_txt(args.txt, W=W, H=H, min_conf=args.min_conf, classes_keep=classes_keep)
    print(f"Detections loaded: {len(dets)}")
    if not dets:
        print("ERROR: No detections after filtering.")
        return

    gray_full = cv2.cvtColor(plan_bgr, cv2.COLOR_BGR2GRAY)

    # Match walls -> swatches (robust median matching)
    assigned = []
    for det in dets:
        box = tuple(det["xyxy"])
        sw_idx, dist = match_wall_to_swatch_robust(gray_full, box, swatches, n_samples=9)
        if dist > args.dist_threshold:
            sw_idx = -1

        wall_type = ""
        if sw_idx >= 0 and sw_idx < len(swatches):
            wall_type = swatches[sw_idx].label_text.strip()

        assigned.append({
            "swatch_idx": int(sw_idx),
            "distance": float(dist),
            "wall_type": wall_type
        })

    # Save assignments
    out_json = f"{args.out_base}_assignments.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "ocr_backend": _OCR_BACKEND,
            "swatches": [
                {
                    "idx": s.idx,
                    "swatch_xyxy_in_legend": list(s.swatch_xyxy),
                    "label_xyxy_in_legend": list(s.label_xyxy),
                    "label_text": s.label_text
                } for s in swatches
            ],
            "detections": [
                {"cls": d["cls"], "conf": d["conf"], "xyxy": d["xyxy"], **a}
                for d, a in zip(dets, assigned)
            ]
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_json}")

    # Save overlay
    overlay = draw_plan_overlay(plan_bgr, dets, assigned)
    out_img = f"{args.out_base}_overlay.png"
    cv2.imwrite(out_img, overlay)
    print(f"Saved: {out_img}")


if __name__ == "__main__":
    main()
