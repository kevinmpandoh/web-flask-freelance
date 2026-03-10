import os
import shutil
import random
from collections import Counter

# ========== KONFIGURASI ==========
SOURCE_DIR = "dataset_asli"  # folder sumber
OUTPUT_DIR = "data"          # folder tujuan (train/val/test)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BALANCE_DATA = True  # set True untuk menyeimbangkan jumlah foto per kelas

# =================================

def buat_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def salin_data(files, src_class_dir, dst_dir):
    for f in files:
        shutil.copy(os.path.join(src_class_dir, f), os.path.join(dst_dir, f))

def split_dataset():
    random.seed(42)
    classes = os.listdir(SOURCE_DIR)

    for cls in classes:
        class_dir = os.path.join(SOURCE_DIR, cls)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

        if BALANCE_DATA:
            # Ambil jumlah minimum gambar dari semua kelas
            min_count = min([len(os.listdir(os.path.join(SOURCE_DIR, c))) for c in classes])
            images = random.sample(images, min_count)

        random.shuffle(images)
        total = len(images)
        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)

        train_files = images[:train_end]
        val_files = images[train_end:val_end]
        test_files = images[val_end:]

        # Buat folder tujuan
        buat_folder(os.path.join(OUTPUT_DIR, "train", cls))
        buat_folder(os.path.join(OUTPUT_DIR, "val", cls))
        buat_folder(os.path.join(OUTPUT_DIR, "test", cls))

        # Salin file
        salin_data(train_files, class_dir, os.path.join(OUTPUT_DIR, "train", cls))
        salin_data(val_files, class_dir, os.path.join(OUTPUT_DIR, "val", cls))
        salin_data(test_files, class_dir, os.path.join(OUTPUT_DIR, "test", cls))

    print("✅ Dataset berhasil dibagi dan disimpan di folder:", OUTPUT_DIR)

if __name__ == "__main__":
    split_dataset()
