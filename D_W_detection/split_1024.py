from PIL import Image, ImageEnhance
import os
import argparse
from glob import glob
from tqdm import tqdm


def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def process_image(img_path, out_img_dir, tile_size=1024, overlap=256,
                   sharpen=0.0, start_index=1):
    
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size

    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be less than tile_size")

    # compute tile start positions ensuring right/bottom edges covered
    xs = list(range(0, max(1, orig_w - tile_size + 1), stride))
    ys = list(range(0, max(1, orig_h - tile_size + 1), stride))
    if len(xs) == 0:
        xs = [0]
    if len(ys) == 0:
        ys = [0]
    # ensure last tile reaches the right/bottom edges
    if xs[-1] + tile_size < orig_w:
        xs.append(max(0, orig_w - tile_size))
    if ys[-1] + tile_size < orig_h:
        ys.append(max(0, orig_h - tile_size))

    index = start_index
    tiles_created = 0

    for ty in ys:
        for tx in xs:
            # crop tile (tile_x, tile_y)
            tile = img.crop((tx, ty, tx + tile_size, ty + tile_size))
            # optionally sharpen/enhance
            if sharpen and sharpen != 0.0:
                enhancer = ImageEnhance.Sharpness(tile)
                tile = enhancer.enhance(sharpen)

            base_name = f"img_{index:06d}"
            out_img_path = os.path.join(out_img_dir, base_name + ".png")
            tile.save(out_img_path, format="PNG")
            tiles_created += 1
            index += 1

    return tiles_created, index


def main():
    parser = argparse.ArgumentParser(description="Tile images into square tiles (no labels, no rotation).")
    parser.add_argument("--input_dir", default="test_dataset", help="Input dataset folder (contains images/train)")
    parser.add_argument("--output_dir", default="test_dataset_1024", help="Output folder for tiles")
    parser.add_argument("--tile_size", type=int, default=1024, help="Tile size in pixels (tile will be tile_size x tile_size)")
    parser.add_argument("--overlap", type=int, default=0, help="Overlap in pixels between tiles")
    
    parser.add_argument("--sharpen", type=float, default=0.0, help="Optional sharpening factor >0 (1.0 is strong). 0.0 means no sharpening.")
    parser.add_argument("--extensions", type=str, default="png,jpg,jpeg", help="Image extensions to read (comma separated)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    in_img_dir = os.path.join(args.input_dir)
    out_img_dir = os.path.join(args.output_dir)
    ensure_dirs([out_img_dir])

    exts = tuple(e.strip().lower() for e in args.extensions.split(","))
    img_paths = []
    for ext in exts:
        img_paths.extend(sorted(glob(os.path.join(in_img_dir, f"*.{ext}"))))
    if not img_paths:
        print(f"No images found in {in_img_dir} with extensions {exts}.")
        return

    counter = 1
    total_tiles = 0
    print(f"Found {len(img_paths)} images. Tiling each to size {args.tile_size} with overlap {args.overlap}.")
    for p in tqdm(img_paths):
        tiles_count, next_index = process_image(p, out_img_dir,
                                               tile_size=args.tile_size, overlap=args.overlap,
                                                sharpen=args.sharpen,
                                               start_index=counter)
        total_tiles += tiles_count
        counter = next_index

    print(f"Done. Created {total_tiles} tiles (including rotated if requested) in {out_img_dir}.")


if __name__ == "__main__":
    main()
