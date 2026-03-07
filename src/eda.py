import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

CSV_FILES = {
    "Train":      "C:/Users/Victo/PycharmProjects/satellite-image-classification/data/train.csv",
    "Validation": "C:/Users/Victo/PycharmProjects/satellite-image-classification/data/validation.csv",
    "Test":       'C:/Users/Victo/PycharmProjects/satellite-image-classification/data/test.csv',
}

CLASS_ORDER = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake",
]

COLORS = {
    "Train":      "#4C72B0",
    "Validation": "#55A868",
    "Test":       "#C44E52",
}

OUTPUT_DIR = "eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

dfs = {}
for split, path in CSV_FILES.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    dfs[split] = pd.read_csv(path)
    print(f"[{split}] {len(dfs[split])} rows loaded")

print("\n" + "="*60)
print("CLASS DISTRIBUTION SUMMARY")
print("="*60)

summary = {}
for split, df in dfs.items():
    counts = df["ClassName"].value_counts().reindex(CLASS_ORDER, fill_value=0)
    summary[split] = counts

summary_df = pd.DataFrame(summary)
summary_df["Total"] = summary_df.sum(axis=1)
for split in CSV_FILES:
    summary_df[f"{split} %"] = (summary_df[split] / summary_df[split].sum() * 100).round(2)

print(summary_df.to_string())
print(f"\nDataset total: {summary_df['Total'].sum()} images")

print("\n" + "="*60)
print("BALANCE CHECK (Train split)")
print("="*60)
train_counts = summary_df["Train"]
min_c, max_c = train_counts.min(), train_counts.max()
imbalance_ratio = max_c / min_c
print(f"  Min samples: {min_c}  |  Max samples: {max_c}  |  Ratio: {imbalance_ratio:.2f}x")
if imbalance_ratio < 1.5:
    print("  ✅ Dataset is well balanced.")
elif imbalance_ratio < 3:
    print("  ⚠️  Mild imbalance detected.")
else:
    print("  ❌ Significant imbalance — consider resampling strategies.")

fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(CLASS_ORDER))
bar_w = 0.25

for i, (split, color) in enumerate(COLORS.items()):
    vals = summary_df[split].values
    bars = ax.bar(x + i * bar_w, vals, bar_w, label=split, color=color, alpha=0.88)
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 8,
                    str(int(h)), ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x + bar_w)
ax.set_xticklabels(CLASS_ORDER, rotation=35, ha="right", fontsize=10)
ax.set_ylabel("Number of Images")
ax.set_title("Class Distribution per Split", fontsize=14, fontweight="bold")
ax.legend()
ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
out1 = os.path.join(OUTPUT_DIR, "class_distribution_grouped.png")
plt.savefig(out1, dpi=150)
plt.close()
print(f"\n[Saved] {out1}")

cmap = plt.get_cmap("tab10")
class_colors = {cls: cmap(i) for i, cls in enumerate(CLASS_ORDER)}

fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=False)

for ax, (split, df) in zip(axes, dfs.items()):
    counts = df["ClassName"].value_counts().reindex(CLASS_ORDER, fill_value=0)
    pct    = counts / counts.sum() * 100
    bars   = ax.bar(CLASS_ORDER, pct.values,
                    color=[class_colors[c] for c in CLASS_ORDER], alpha=0.88)
    for bar, v in zip(bars, pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=7.5)
    ax.set_title(f"{split}", fontsize=12, fontweight="bold")
    ax.set_xticklabels(CLASS_ORDER, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% of Split")
    ax.set_ylim(0, pct.max() * 1.2)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.suptitle("Class Proportion within Each Split", fontsize=14, fontweight="bold")
plt.tight_layout()
out2 = os.path.join(OUTPUT_DIR, "class_proportion_per_split.png")
plt.savefig(out2, dpi=150)
plt.close()
print(f"[Saved] {out2}")

total_counts = summary_df["Total"]
fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(
    total_counts.values,
    labels=CLASS_ORDER,
    autopct="%1.1f%%",
    colors=[class_colors[c] for c in CLASS_ORDER],
    startangle=140,
    pctdistance=0.82,
)
for t in autotexts:
    t.set_fontsize(9)
ax.set_title("Overall Class Distribution (All Splits)", fontsize=14, fontweight="bold")
plt.tight_layout()
out3 = os.path.join(OUTPUT_DIR, "class_distribution_pie.png")
plt.savefig(out3, dpi=150)
plt.close()
print(f"[Saved] {out3}")

csv_out = os.path.join(OUTPUT_DIR, "class_distribution_summary.csv")
summary_df.to_csv(csv_out)
print(f"[Saved] {csv_out}")

print("\n✅ EDA class distribution complete. Check the 'eda_outputs/' folder.")