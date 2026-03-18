import pandas as pd
from pathlib import Path

DATA_DIR   = Path("../data")
OUTPUT_DIR = Path("preprocessing_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

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

SUPER_CLASSES = ["Agriculture", "Vegetation", "Urban", "Water"]
LABEL_MAP     = {cls: idx for idx, cls in enumerate(SUPER_CLASSES)}

CSV_FILES = {
    "Train":      DATA_DIR / "train.csv",
    "Validation": DATA_DIR / "validation.csv",
    "Test":       DATA_DIR / "test.csv",
}


def remap():
    for split, path in CSV_FILES.items():
        df = pd.read_csv(path)
        df["SuperClass"] = df["ClassName"].map(CLASS_REMAP)
        df["ImagePath"]  = df.apply(
            lambda r: str(DATA_DIR / r["Filename"]), axis=1)

        excluded = df[df["SuperClass"].isna()].copy()
        df       = df[df["SuperClass"].notna()].copy()
        df["Label"] = df["SuperClass"].map(LABEL_MAP)

        df.to_csv(OUTPUT_DIR / f"{split.lower()}_remapped.csv", index=False)
        excluded.to_csv(OUTPUT_DIR / f"{split.lower()}_to_infer.csv", index=False)

        print(f"[{split}]")
        print(f"  kept     : {len(df)} → {df['SuperClass'].value_counts().to_dict()}")
        print(f"  excluded : {len(excluded)} → {excluded['ClassName'].value_counts().to_dict()}")


if __name__ == "__main__":
    remap()