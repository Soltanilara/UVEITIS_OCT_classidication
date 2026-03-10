import argparse
import os
import re
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def is_od_image(rel_path: Path) -> bool:
    """
    Detect OD image from path/filename tokens.
    Matches patterns like "..._OD_..." or path segment "/OD/".
    """
    tokenized = rel_path.as_posix().upper()
    return bool(re.search(r"(^|[/_])OD([/_\.]|$)", tokenized))


def process_image(src: Path, dst: Path, flip: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not flip:
        shutil.copy2(src, dst)
        return

    with Image.open(src) as img:
        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        flipped.save(dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a canonicalized dataset by horizontally flipping OD images."
    )
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Path to original dataset root.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Path to write canonicalized dataset root.",
    )
    parser.add_argument(
        "--copy_os",
        action="store_true",
        help="Also copy non-OD images to output_root (recommended).",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"input_root does not exist: {input_root}")

    image_paths = [
        p for p in input_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    if not image_paths:
        raise RuntimeError(f"No images found under: {input_root}")

    flipped_count = 0
    copied_count = 0
    skipped_count = 0

    for src_path in tqdm(image_paths, desc="Processing images"):
        rel = src_path.relative_to(input_root)
        dst_path = output_root / rel
        od = is_od_image(rel)

        if od:
            process_image(src_path, dst_path, flip=True)
            flipped_count += 1
        elif args.copy_os:
            process_image(src_path, dst_path, flip=False)
            copied_count += 1
        else:
            skipped_count += 1

    print("Done.")
    print(f"Input root  : {input_root}")
    print(f"Output root : {output_root}")
    print(f"Flipped OD  : {flipped_count}")
    print(f"Copied non-OD: {copied_count}")
    print(f"Skipped non-OD: {skipped_count}")
    if not args.copy_os:
        print("Note: --copy_os was not set, so only OD images were written.")


if __name__ == "__main__":
    main()
