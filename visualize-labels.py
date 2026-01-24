
import os
import yaml
import cv2
import matplotlib.pyplot as plt


def load_class_names_from_yaml(data_yaml_path: str):
    """
    Reads YOLO data.yaml and returns a dict: {class_id: class_name}
    Works with common formats:
      names: ["a", "b", ...]
      names:
        0: a
        1: b
    """
    if not data_yaml_path or not os.path.exists(data_yaml_path):
        return None

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names", None)
    if names is None:
        return None

    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    if isinstance(names, dict):
        # keys might be strings in YAML
        return {int(k): v for k, v in names.items()}

    return None


def yolo_to_xyxy(cls, x_c, y_c, w, h, img_w, img_h):
    """
    YOLO format: class x_center y_center width height (all normalized 0..1)
    Convert to pixel bbox: (x1, y1, x2, y2)
    """
    x_c *= img_w
    y_c *= img_h
    w *= img_w
    h *= img_h

    x1 = int(round(x_c - w / 2))
    y1 = int(round(y_c - h / 2))
    x2 = int(round(x_c + w / 2))
    y2 = int(round(y_c + h / 2))

    # clamp
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))
    return x1, y1, x2, y2


def visualize_yolo_labels(
    image_path: str,
    label_path: str,
    data_yaml_path: str = None,
    show_class_name: bool = True,
    window_title: str = None
):
    """
    Visualize YOLO txt labels on an image.
    - image_path: path to .jpg/.png...
    - label_path: path to corresponding .txt (YOLO format)
    - data_yaml_path: optional, for class name lookup
    - show_class_name: if True and yaml provided, shows "id:name", else shows "id"
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label not found: {label_path}")

    # Load image (BGR)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img_h, img_w = img.shape[:2]

    class_map = load_class_names_from_yaml(data_yaml_path) if data_yaml_path else None

    # Read label lines
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    # Draw each bbox
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 5:
            # Not a YOLO bbox line (could be segmentation etc.)
            # Skip or handle differently if you want
            print(f"Skipping non-bbox line {i}: {line}")
            continue

        cls = int(float(parts[0]))
        x_c, y_c, w, h = map(float, parts[1:5])
        x1, y1, x2, y2 = yolo_to_xyxy(cls, x_c, y_c, w, h, img_w, img_h)

        # Label text: always include class id
        if show_class_name and class_map and cls in class_map:
            text = f"{cls}:{class_map[cls]}"
        else:
            text = f"{cls}"

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw text background for readability
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = max(th + 5, y1)  # keep within image
        cv2.rectangle(img, (x1, y_text - th - 6), (x1 + tw + 6, y_text + baseline), (0, 255, 0), -1)
        cv2.putText(img, text, (x1 + 3, y_text - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Convert BGR -> RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title(window_title or f"Labels: {os.path.basename(label_path)}")
    plt.show()


if __name__ == "__main__":
    # Example usage:
    # visualize_yolo_labels(
    #     image_path="path/to/image.jpg",
    #     label_path="path/to/label.txt",
    #     data_yaml_path="path/to/data.yaml",  # optional
    #     show_class_name=True
    # )

    # Replace these with your paths:
    image_path = "/Users/albouchiayoub/Downloads/Stairs_dataset/train/images/img_000001.png"
    label_path = "/Users/albouchiayoub/Downloads/Stairs_dataset/train/labels/img_000001.txt"
    data_yaml_path = "data.yaml"  # or None if you don't want it

    visualize_yolo_labels(image_path, label_path, data_yaml_path=data_yaml_path, show_class_name=True)
