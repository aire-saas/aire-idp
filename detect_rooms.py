import numpy as np
import cv2
from skimage.morphology import skeletonize
from scipy.ndimage import convolve, label as ndi_label
from scipy.spatial import cKDTree



"""
# ----------------------------
# Detect rooms in a floorplan using object detections of walls/doors/windows.
#
# IMPORTANT:
# This version expects your detections txt lines in this format (as in your file):
#   class  conf  x1  y1  x2  y2
# where x1,y1,x2,y2 are in PIXELS (not normalized).
# ----------------------------


# Loads YOLO detections and converts bounding boxes from normalized coordinates to image pixels.
#
# Builds a wall/obstacle mask by filling detected wall, door, and window regions, extending each box along its main axis to close small gaps.
#
# Repairs tiny wall breaks by skeletonizing the wall mask and connecting nearby, well-aligned skeleton endpoints.
#
# Detects rooms by flood-filling free space from the image borders to remove the outside area, then labeling enclosed regions as rooms.
#
# Generates visualizations, including wall masks, room labels, colored room maps, and an overlay on the original image.
#
# Output: a labeled room segmentation where each connected enclosed area corresponds to one detected room.
# ----------------------------

"""
IMAGE_PATH = "/Users/albouchiayoub/PycharmProjects/aire-idp/Documentation/Week9/4/Screenshot 2026-01-15 at 14.57.39.png"
DET_TXT_PATH = "/Users/albouchiayoub/PycharmProjects/aire-idp/Documentation/Week9/4/Screenshot 2026-01-15 at 14.57.39_detections.txt"

DOOR_CLASS = 0
WALL_CLASS = 1
WINDOW_CLASS = 2

CONF_TH = 0.10

WALL_EXTEND_PX = 15
DOOR_EXTEND_PX = 10
WINDOW_EXTEND_PX = 10

BRIDGE_D_MAX = 12.0
BRIDGE_MIN_COS = 0.45


# ----------------------------
# 1) Parse detections: class conf x1 y1 x2 y2 (pixels)
# ----------------------------
def load_xyxy_pixel_detections(txt_path: str, conf_th: float = 0.15):
    dets = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            cls = int(float(parts[0]))
            conf = float(parts[1])
            x1, y1, x2, y2 = map(float, parts[2:6])
            if conf >= conf_th:
                dets.append((cls, conf, x1, y1, x2, y2))
    return dets


def clamp_and_order_xyxy(x1, y1, x2, y2, W, H):
    x1, y1, x2, y2 = map(int, map(round, (x1, y1, x2, y2)))

    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return x1, y1, x2, y2


# ----------------------------
# 2) Extend boxes (longer, not wider)
# ----------------------------
def extend_box_along_major_axis(x1, y1, x2, y2, W, H, extend_px=8):
    """
    Extend an axis-aligned rectangle ONLY along its major axis (length), not width.
    extend_px is added on both ends (total added length = 2*extend_px).
    """
    w = x2 - x1
    h = y2 - y1

    if w >= h:
        x1 = max(0, x1 - extend_px)
        x2 = min(W - 1, x2 + extend_px)
    else:
        y1 = max(0, y1 - extend_px)
        y2 = min(H - 1, y2 + extend_px)

    return x1, y1, x2, y2


def build_segmentation_obstacle_mask_from_xyxy(
    img_shape,
    detections,
    wall_class=1,
    door_class=0,
    window_class=2,
    wall_extend_px=12,
    door_extend_px=10,
    window_extend_px=10,
):
    """
    Build ONE obstacle mask for segmentation:
    - Treat walls/doors/windows as obstacles
    - Extend each box ONLY along its major axis (length)
    - Tiny morphology close to remove small holes inside predicted regions
    Returns: walls_seg (bool)
    """
    H, W = img_shape[:2]
    seg_u8 = np.zeros((H, W), dtype=np.uint8)

    for (cls, conf, x1, y1, x2, y2) in detections:
        if cls not in (wall_class, door_class, window_class):
            continue

        x1, y1, x2, y2 = clamp_and_order_xyxy(x1, y1, x2, y2, W, H)

        if cls == wall_class:
            e = wall_extend_px
        elif cls == door_class:
            e = door_extend_px
        else:
            e = window_extend_px

        ex1, ey1, ex2, ey2 = extend_box_along_major_axis(x1, y1, x2, y2, W, H, extend_px=e)
        cv2.rectangle(seg_u8, (ex1, ey1), (ex2, ey2), 255, thickness=-1)

    # remove tiny 1px holes inside predictions
    seg_u8 = cv2.morphologyEx(seg_u8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    return seg_u8 > 0


# ----------------------------
# 3) Skeleton endpoint bridging (adds 1px bridges)
# ----------------------------
def _to_bool(mask: np.ndarray) -> np.ndarray:
    return mask if mask.dtype == np.bool_ else (mask > 0)


def _skeleton_endpoints(skel: np.ndarray) -> np.ndarray:
    skel_u8 = skel.astype(np.uint8)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neigh = convolve(skel_u8, kernel, mode="constant", cval=0)
    endpoints = np.logical_and(skel, neigh == 1)
    return np.argwhere(endpoints)  # (N,2) (y,x)


def _endpoint_direction(skel: np.ndarray, y: int, x: int):
    h, w = skel.shape
    neighs = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and skel[yy, xx]:
                neighs.append((yy, xx))
    if len(neighs) != 1:
        return None
    ny, nx = neighs[0]
    v = np.array([y - ny, x - nx], dtype=np.float32)  # neighbor -> endpoint
    n = np.linalg.norm(v) + 1e-8
    return v / n


def _raster_line_points(y0, x0, y1, x1):
    y_min, y_max = sorted((y0, y1))
    x_min, x_max = sorted((x0, x1))
    pad = 2
    y_min = max(y_min - pad, 0)
    x_min = max(x_min - pad, 0)
    y_max = y_max + pad
    x_max = x_max + pad
    h = y_max - y_min + 1
    w = x_max - x_min + 1
    canvas = np.zeros((h, w), np.uint8)
    cv2.line(canvas, (x0 - x_min, y0 - y_min), (x1 - x_min, y1 - y_min), 1, 1)
    ys, xs = np.nonzero(canvas)
    return ys + y_min, xs + x_min


def bridge_micro_gaps_by_skeleton(walls_mask: np.ndarray, d_max=12.0, min_cos=0.45, mutual_nn=True):
    walls = _to_bool(walls_mask)
    skel = skeletonize(walls)

    endpoints = _skeleton_endpoints(skel)
    if len(endpoints) < 2:
        return walls.copy()

    pts = []
    dirs = []
    for (y, x) in endpoints:
        d = _endpoint_direction(skel, int(y), int(x))
        if d is not None:
            pts.append((int(y), int(x)))
            dirs.append(d)

    if len(pts) < 2:
        return walls.copy()

    pts = np.array(pts, dtype=np.float32)   # (N,2) (y,x)
    dirs = np.stack(dirs, axis=0).astype(np.float32)

    tree = cKDTree(pts)
    k = min(6, len(pts))
    dists, idxs = tree.query(pts, k=k)

    def best_candidate(i):
        yi, xi = pts[i]
        di = dirs[i]
        best_j = None
        best_score = -1e9
        for n in range(1, k):
            j = int(idxs[i, n])
            dist = float(dists[i, n])
            if not np.isfinite(dist) or dist > d_max:
                continue

            yj, xj = pts[j]
            dj = dirs[j]
            v = np.array([yj - yi, xj - xi], dtype=np.float32)
            nv = np.linalg.norm(v) + 1e-8
            vhat = v / nv

            cos_i = float(np.dot(di, vhat))
            cos_j = float(np.dot(dj, -vhat))
            if cos_i < min_cos or cos_j < min_cos:
                continue

            score = (cos_i + cos_j) - 0.15 * dist
            if score > best_score:
                best_score = score
                best_j = j
        return best_j

    proposed = {i: best_candidate(i) for i in range(len(pts))}
    proposed = {i: j for i, j in proposed.items() if j is not None}

    links = set()
    if mutual_nn:
        for i, j in proposed.items():
            if proposed.get(j, None) == i:
                a, b = sorted((i, j))
                links.add((a, b))
    else:
        for i, j in proposed.items():
            a, b = sorted((i, j))
            links.add((a, b))

    walls_seg = walls.copy().astype(np.uint8)
    for a, b in links:
        y0, x0 = map(int, pts[a])
        y1, x1 = map(int, pts[b])
        ys, xs = _raster_line_points(y0, x0, y1, x1)
        walls_seg[ys, xs] = 1

    return walls_seg.astype(bool)


# ----------------------------
# 4) Room detection (closed components)
# ----------------------------
def detect_rooms_from_walls(walls_seg: np.ndarray) -> np.ndarray:
    """
    Rooms = connected components of inside free-space.
    Removes outside by flood fill from borders.
    """
    walls_seg = _to_bool(walls_seg)
    free = (~walls_seg).astype(np.uint8)

    h, w = free.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    img = (free * 255).astype(np.uint8)

    def try_seed(y, x):
        if img[y, x] == 255:
            cv2.floodFill(img, flood_mask, (x, y), 128)

    # flood from all borders
    for x in range(w):
        try_seed(0, x)
        try_seed(h - 1, x)
    for y in range(h):
        try_seed(y, 0)
        try_seed(y, w - 1)

    inside = (img == 255)
    labels, _ = ndi_label(inside.astype(np.uint8), structure=np.ones((3, 3), np.uint8))
    return labels.astype(np.int32)


# ----------------------------
# 5) Visualization (optional)
# ----------------------------
def labels_to_random_colors(room_labels: np.ndarray, seed: int = 123) -> np.ndarray:
    n_rooms = int(room_labels.max())
    h, w = room_labels.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    if n_rooms <= 0:
        return out

    rng = np.random.default_rng(seed)
    colors = np.zeros((n_rooms + 1, 3), dtype=np.uint8)
    colors[1:] = rng.integers(40, 256, size=(n_rooms, 3), dtype=np.uint8)

    out = colors[room_labels]
    out[room_labels == 0] = 0
    return out


def overlay_rooms_on_image(img_bgr: np.ndarray, room_color_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay_bgr = cv2.cvtColor(room_color_rgb, cv2.COLOR_RGB2BGR)
    mask = np.any(overlay_bgr != 0, axis=2).astype(np.uint8) * 255
    mask_3 = cv2.merge([mask, mask, mask])
    blended_part = cv2.addWeighted(img_bgr, 1 - alpha, overlay_bgr, alpha, 0)
    return np.where(mask_3 == 255, blended_part, img_bgr)


# ----------------------------
# Main
# ----------------------------
def main():
    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

    dets = load_xyxy_pixel_detections(DET_TXT_PATH, conf_th=CONF_TH)
    print(f"Loaded detections: {len(dets)} (conf >= {CONF_TH})")

    walls_seg = build_segmentation_obstacle_mask_from_xyxy(
        img.shape,
        dets,
        wall_class=WALL_CLASS,
        door_class=DOOR_CLASS,
        window_class=WINDOW_CLASS,
        wall_extend_px=WALL_EXTEND_PX,
        door_extend_px=DOOR_EXTEND_PX,
        window_extend_px=WINDOW_EXTEND_PX,
    )

    # Sanity check: obstacle coverage
    print(f"walls_seg coverage: {walls_seg.mean():.3f} (fraction of pixels marked obstacle)")

    # Bridge remaining micro-gaps (still only affects segmentation)
    walls_seg = bridge_micro_gaps_by_skeleton(
        walls_seg,
        d_max=BRIDGE_D_MAX,
        min_cos=BRIDGE_MIN_COS,
        mutual_nn=True
    )

    room_labels = detect_rooms_from_walls(walls_seg)
    n_rooms = int(room_labels.max())
    print(f"Detected rooms: {n_rooms}")

    # Debug outputs
    cv2.imwrite("debug_walls_seg.png", (walls_seg.astype(np.uint8) * 255))

    # label visualization (grayscale)
    if n_rooms > 0:
        vis = (room_labels.astype(np.float32) / n_rooms * 255).astype(np.uint8)
    else:
        vis = np.zeros(walls_seg.shape, np.uint8)
    cv2.imwrite("debug_rooms_labels.png", vis)

    # optional colored + overlay
    rooms_rgb = labels_to_random_colors(room_labels, seed=123)
    cv2.imwrite("debug_rooms_colored.png", cv2.cvtColor(rooms_rgb, cv2.COLOR_RGB2BGR))
    overlay = overlay_rooms_on_image(img, rooms_rgb, alpha=0.45)
    cv2.imwrite("debug_rooms_overlay.png", overlay)

    print("Saved outputs:")
    print(" - debug_walls_seg.png")
    print(" - debug_rooms_labels.png")
    print(" - debug_rooms_colored.png")
    print(" - debug_rooms_overlay.png")


if __name__ == "__main__":
    main()
