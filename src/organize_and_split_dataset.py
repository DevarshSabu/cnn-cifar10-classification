import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_BASE_DIR = "data_raw"
RAW_TRAIN_DIR = os.path.join(RAW_BASE_DIR, "train")
CSV_PATH = os.path.join(RAW_BASE_DIR, "train.csv")

IMAGE_COL = "image_ID"     # column name in CSV for image filenames
LABEL_COL = "label"     # column name for class/sport label
VAL_RATIO = 0.2          # 80% train, 20% validation

OUTPUT_BASE = "data"
TRAIN_OUT = os.path.join(OUTPUT_BASE, "train")
VAL_OUT = os.path.join(OUTPUT_BASE, "val")

def main():
    df = pd.read_csv(CSV_PATH)

    os.makedirs(TRAIN_OUT, exist_ok=True)
    os.makedirs(VAL_OUT, exist_ok=True)

    classes = sorted(df[LABEL_COL].unique())
    print("Classes found in dataset:", classes)

    for cls in classes:
        cls_rows = df[df[LABEL_COL] == cls]

        train_df, val_df = train_test_split(
            cls_rows,
            test_size=VAL_RATIO,
            random_state=42,
            shuffle=True,
        )

        os.makedirs(os.path.join(TRAIN_OUT, cls), exist_ok=True)
        os.makedirs(os.path.join(VAL_OUT, cls), exist_ok=True)

        for _, row in train_df.iterrows():
            src = os.path.join(RAW_TRAIN_DIR, row[IMAGE_COL])
            dst = os.path.join(TRAIN_OUT, cls, row[IMAGE_COL])
            if os.path.exists(src):
                shutil.move(src, dst)

        for _, row in val_df.iterrows():
            src = os.path.join(RAW_TRAIN_DIR, row[IMAGE_COL])
            dst = os.path.join(VAL_OUT, cls, row[IMAGE_COL])
            if os.path.exists(src):
                shutil.move(src, dst)

        print(f"{cls}: {len(train_df)} train, {len(val_df)} val images")

    print("\n✔ Dataset successfully organized into /data/train & /data/val")

if __name__ == "__main__":
    main()
