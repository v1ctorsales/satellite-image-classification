"""
preprocessing.py — Remap + Class-balance Augmentation
======================================================
Steps:
    1. remap_splits()     -> maps original classes to 4 superclasses,
                             writes {split}_remapped.csv and {split}_to_infer.csv
    2. augment_train()    -> analyses class distribution in train split only,
                             generates rotated copies (90/180/270) for minority
                             classes until they reach the majority class count
                             or until each image has produced 3 rotations,
                             appends new rows to train_remapped.csv
    3. main()             -> runs both steps in order
"""

import pandas as pd
from pathlib import Path
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR   = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "preprocessing"
AUG_DIR    = DATA_DIR / "augmented"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUG_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────

CLASS_REMAP = {
    "AnnualCrop":           "Agriculture",
    "PermanentCrop":        "Agriculture",
    "Pasture":              "Agriculture",
    "Forest":               "Vegetation",
    "HerbaceousVegetation": "Vegetation",
    "Industrial":           "Urban",
    "Residential":          "Urban",
    "SeaLake":              "Water",
}

SUPER_CLASSES  = ["Agriculture", "Vegetation", "Urban", "Water"]
LABEL_MAP      = {cls: idx for idx, cls in enumerate(SUPER_CLASSES)}
ROTATIONS      = [90, 180, 270]   # degrees; 4th rotation = original, so we stop at 3
MAX_ROTATIONS  = len(ROTATIONS)

CSV_FILES = {
    "train":      DATA_DIR / "train.csv",
    "validation": DATA_DIR / "validation.csv",
    "test":       DATA_DIR / "test.csv",
}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — REMAP
# ─────────────────────────────────────────────────────────────────────────────

def remap_splits():
    """
    Reads each raw CSV, maps ClassName to a SuperClass, and writes:
        {split}_remapped.csv   — rows with a known superclass + numeric label
        {split}_to_infer.csv   — rows whose class has no mapping (excluded)
    """
    print("=" * 55)
    print("STEP 1 — Remapping classes")
    print("=" * 55)

    for split, path in CSV_FILES.items():
        df = pd.read_csv(path)
        df["SuperClass"] = df["ClassName"].map(CLASS_REMAP)
        df["ImagePath"]  = df["Filename"].apply(lambda f: str(DATA_DIR / f))

        excluded = df[df["SuperClass"].isna()].copy()
        df       = df[df["SuperClass"].notna()].copy()
        df["Label"] = df["SuperClass"].map(LABEL_MAP)

        df.to_csv(OUTPUT_DIR / f"{split}_remapped.csv", index=False)
        excluded.to_csv(OUTPUT_DIR / f"{split}_to_infer.csv", index=False)

        print(f"\n[{split.upper()}]")
        print(f"  kept     : {len(df):>5} → {df['SuperClass'].value_counts().to_dict()}")
        print(f"  excluded : {len(excluded):>5} → {excluded['ClassName'].value_counts().to_dict()}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — AUGMENTATION (train split only)
# ─────────────────────────────────────────────────────────────────────────────

def _rotate_and_save(image_path: str, degrees: int, aug_dir: Path) -> str:
    """
    Rotates an image by `degrees` and saves it to aug_dir.
    Returns the path of the saved file.
    Raises if the source image cannot be opened.
    """
    src = Path(image_path)
    dest = aug_dir / f"{src.stem}_rot{degrees}{src.suffix}"

    if not dest.exists():
        img = Image.open(src)
        img.rotate(degrees).save(dest)

    return str(dest)


def augment_train():
    """
    Analyses class distribution in the train split.
    For each class below the majority count, generates rotated copies
    (90°, 180°, 270°) until the gap is closed or each image has produced
    at most 3 rotations — whichever limit is hit first.
    Appends new rows to train_remapped.csv.
    """
    print("\n" + "=" * 55)
    print("STEP 2 — Augmentation (train only)")
    print("=" * 55)

    csv_path = OUTPUT_DIR / "train_remapped.csv"
    df = pd.read_csv(csv_path)

    counts       = df["SuperClass"].value_counts()
    majority_count = int(counts.max())
    majority_class = counts.idxmax()

    print(f"\n  Class distribution before augmentation:")
    for cls, count in counts.items():
        marker = " ← majority" if cls == majority_class else ""
        print(f"    {cls:<15}: {count:>5}{marker}")

    # Check whether augmentation is actually needed
    minority_classes = counts[counts < majority_count].index.tolist()
    if not minority_classes:
        print("\n  ✅ Classes are already balanced — no augmentation needed.")
        return

    new_rows = []

    for cls in minority_classes:
        gap        = majority_count - int(counts[cls])
        cls_rows   = df[df["SuperClass"] == cls].copy()
        generated  = 0

        print(f"\n  [{cls}] needs {gap} more images (has {int(counts[cls])}, target {majority_count})")

        # Cycle through source images, applying rotations until gap is closed
        # or every image has been rotated MAX_ROTATIONS times
        rotation_idx = 0
        while generated < gap and rotation_idx < MAX_ROTATIONS:
            degrees = ROTATIONS[rotation_idx]
            for _, row in cls_rows.iterrows():
                if generated >= gap:
                    break
                try:
                    aug_path = _rotate_and_save(row["ImagePath"], degrees, AUG_DIR)
                    new_rows.append({
                        "Filename":   Path(aug_path).name,
                        "ClassName":  row["ClassName"],
                        "SuperClass": cls,
                        "Label":      row["Label"],
                        "ImagePath":  aug_path,
                    })
                    generated += 1
                except Exception as e:
                    print(f"    [WARNING] Could not augment {row['ImagePath']}: {e}")

            rotation_idx += 1

        print(f"    → generated {generated} images "
              f"({'gap closed' if generated == gap else f'limit of {MAX_ROTATIONS} rotations reached'})")

    if not new_rows:
        print("\n  No new rows generated.")
        return

    augmented_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    augmented_df.to_csv(csv_path, index=False)

    print(f"\n  Class distribution after augmentation:")
    for cls, count in augmented_df["SuperClass"].value_counts().items():
        print(f"    {cls:<15}: {count:>5}")

    print(f"\n  train_remapped.csv updated — {len(new_rows)} rows added.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    remap_splits()
    augment_train()
    print("\n✅ Preprocessing complete.")


if __name__ == "__main__":
    main()