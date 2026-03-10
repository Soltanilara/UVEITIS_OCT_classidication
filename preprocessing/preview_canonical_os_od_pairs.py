import argparse
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
EYE_TOKEN_RE = re.compile(r"(^|[/_])(OD|OS)([/_.]|$)", flags=re.IGNORECASE)


def detect_eye(rel_path: Path) -> str | None:
    match = EYE_TOKEN_RE.search(rel_path.as_posix())
    if not match:
        return None
    return match.group(2).upper()


def neutralize_eye_token(path_text: str) -> str:
    # Replace /OD/, /OS/, _OD_, _OS_, etc. with eye-neutral token.
    return re.sub(
        r"(?i)(^|[/_])(OD|OS)([/_.]|$)",
        lambda m: f"{m.group(1)}OX{m.group(3)}",
        path_text,
    )


def build_pairs(dataset_root: Path) -> list[tuple[Path, Path]]:
    groups: dict[str, dict[str, list[Path]]] = {}

    for p in dataset_root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue

        rel = p.relative_to(dataset_root)
        eye = detect_eye(rel)
        if eye not in {"OD", "OS"}:
            continue

        key = neutralize_eye_token(rel.as_posix())
        groups.setdefault(key, {"OD": [], "OS": []})[eye].append(p)

    pairs: list[tuple[Path, Path]] = []
    for bucket in groups.values():
        od_list = sorted(bucket["OD"])
        os_list = sorted(bucket["OS"])
        n = min(len(od_list), len(os_list))
        for i in range(n):
            pairs.append((os_list[i], od_list[i]))

    return pairs


def save_preview(
    sampled_pairs: list[tuple[Path, Path]],
    out_png: Path,
    dataset_root: Path,
) -> None:
    n = len(sampled_pairs)
    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(13, 4 * n))
    if n == 1:
        axes = [axes]  # type: ignore[assignment]

    for i, (os_path, od_path) in enumerate(sampled_pairs):
        os_img = Image.open(os_path).convert("RGB")
        od_img = Image.open(od_path).convert("RGB")
        od_unflipped = od_img.transpose(Image.FLIP_LEFT_RIGHT)

        row_axes = axes[i]
        row_axes[0].imshow(os_img)
        row_axes[0].set_title("OS (canonical)")
        row_axes[0].axis("off")

        row_axes[1].imshow(od_img)
        row_axes[1].set_title("OD (canonical, already flipped)")
        row_axes[1].axis("off")

        row_axes[2].imshow(od_unflipped)
        row_axes[2].set_title("OD flipped again (reference)")
        row_axes[2].axis("off")

        os_rel = os_path.relative_to(dataset_root).as_posix()
        od_rel = od_path.relative_to(dataset_root).as_posix()
        row_axes[0].set_ylabel(f"Set {i + 1}\n{os_rel}\n{od_rel}", fontsize=8)

    fig.suptitle("OS/OD Sanity Check from Canonical Dataset", fontsize=14)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample OS/OD pairs from canonical dataset and save a sanity-check panel."
    )
    parser.add_argument("--dataset_root", type=str, required=True, help="Canonical dataset root path.")
    parser.add_argument("--num_sets", type=int, default=10, help="Number of OS/OD sets to visualize.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--out_png",
        type=str,
        default="sanity_check_os_od_pairs.png",
        help="Output PNG path for the preview panel.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {dataset_root}")

    pairs = build_pairs(dataset_root)
    if not pairs:
        raise RuntimeError("No OS/OD pairs were found in dataset_root.")

    random.seed(args.seed)
    sample_n = min(args.num_sets, len(pairs))
    sampled_pairs = random.sample(pairs, sample_n)

    out_png = Path(args.out_png).expanduser().resolve()
    save_preview(sampled_pairs, out_png, dataset_root)

    print(f"Total matched OS/OD pairs found: {len(pairs)}")
    print(f"Saved preview panel: {out_png}")
    print("Sampled pairs:")
    for i, (os_path, od_path) in enumerate(sampled_pairs, start=1):
        print(f"[{i}] OS: {os_path.relative_to(dataset_root)}")
        print(f"    OD: {od_path.relative_to(dataset_root)}")


if __name__ == "__main__":
    main()
