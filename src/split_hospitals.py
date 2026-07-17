"""
Phase 3: Split dataset into 3 simulated hospitals (non-IID)
This creates realistic, UNEVEN data distribution across hospitals,
mimicking how real hospitals see different patient mixes.

Hospital_A -> mostly Stone + Normal
Hospital_B -> mostly Cyst + Tumor
Hospital_C -> balanced mix of everything
"""

import os
import random
import shutil

random.seed(42)  # keeps the split consistent every time you run this

# ---------------------------
# 1. Setup paths
# ---------------------------
SOURCE_DIR = "../data/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
OUTPUT_DIR = "../data/hospitals"

CLASSES = ["Cyst", "Normal", "Stone", "Tumor"]
HOSPITALS = ["Hospital_A", "Hospital_B", "Hospital_C"]

# Percentage of each class going to each hospital (must sum to 1.0 per class)
# This is what makes it "non-IID" -- each hospital sees a different mix
SPLIT_RATIOS = {
    "Cyst":   {"Hospital_A": 0.15, "Hospital_B": 0.55, "Hospital_C": 0.30},
    "Normal": {"Hospital_A": 0.45, "Hospital_B": 0.20, "Hospital_C": 0.35},
    "Stone":  {"Hospital_A": 0.55, "Hospital_B": 0.15, "Hospital_C": 0.30},
    "Tumor":  {"Hospital_A": 0.15, "Hospital_B": 0.55, "Hospital_C": 0.30},
}

# ---------------------------
# 2. Create output folders
# ---------------------------
for hospital in HOSPITALS:
    for cls in CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, hospital, cls), exist_ok=True)

# ---------------------------
# 3. Split and copy files
# ---------------------------
summary = {h: {c: 0 for c in CLASSES} for h in HOSPITALS}

for cls in CLASSES:
    class_folder = os.path.join(SOURCE_DIR, cls)
    if not os.path.isdir(class_folder):
        print(f"WARNING: folder not found -> {class_folder}")
        continue

    files = os.listdir(class_folder)
    random.shuffle(files)

    total = len(files)
    ratios = SPLIT_RATIOS[cls]

    start = 0
    for hospital in HOSPITALS:
        count = int(total * ratios[hospital])
        chunk = files[start:start + count]
        start += count

        for f in chunk:
            src_path = os.path.join(class_folder, f)
            dst_path = os.path.join(OUTPUT_DIR, hospital, cls, f)
            shutil.copyfile(src_path, dst_path)

        summary[hospital][cls] = len(chunk)

# ---------------------------
# 4. Print summary
# ---------------------------
print("\n=== Hospital Data Split Summary ===")
for hospital in HOSPITALS:
    total_h = sum(summary[hospital].values())
    print(f"\n{hospital} (total: {total_h} images)")
    for cls in CLASSES:
        print(f"   {cls}: {summary[hospital][cls]} images")

print("\nDone! Check the '../data/hospitals' folder.")
