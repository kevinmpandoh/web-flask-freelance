import os
import numpy as np
import shutil
import random
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from keras.optimizers import Adam

# ========== KONFIGURASI ==========
SOURCE_DIR = "dataset_raw"   # folder sumber data asli
OUTPUT_DIR = "dataset"           # folder tujuan dataset terstruktur
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 10
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BALANCE_DATA = True
# =================================


# ===== Fungsi bantu untuk bagi dataset =====
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
            min_count = min([len(os.listdir(os.path.join(SOURCE_DIR, c))) for c in classes])
            images = random.sample(images, min_count)

        random.shuffle(images)
        total = len(images)
        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)

        train_files = images[:train_end]
        val_files = images[train_end:val_end]
        test_files = images[val_end:]

        buat_folder(os.path.join(OUTPUT_DIR, "train", cls))
        buat_folder(os.path.join(OUTPUT_DIR, "val", cls))
        buat_folder(os.path.join(OUTPUT_DIR, "test", cls))

        salin_data(train_files, class_dir, os.path.join(OUTPUT_DIR, "train", cls))
        salin_data(val_files, class_dir, os.path.join(OUTPUT_DIR, "val", cls))
        salin_data(test_files, class_dir, os.path.join(OUTPUT_DIR, "test", cls))

    print("✅ Dataset berhasil dibagi:", OUTPUT_DIR)


# ===== Training CNN =====
def train_model():
    # Data augmentation
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(OUTPUT_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    val_generator = val_datagen.flow_from_directory(
        os.path.join(OUTPUT_DIR, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    test_generator = test_datagen.flow_from_directory(
        os.path.join(OUTPUT_DIR, "test"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    # Model CNN
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Training
    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator
    )

    # ===== Evaluasi =====
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"\n📊 Akurasi di test set: {test_acc:.2f}")

    # Prediksi & Analisis
    Y_pred = model.predict(test_generator)
    y_pred = (Y_pred > 0.5).astype(int).flatten()
    y_true = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    print("\n🧾 Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\n🧾 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

    for i, label in enumerate(class_labels):
        class_correct = np.sum((y_true == i) & (y_pred == i))
        class_total = np.sum(y_true == i)
        class_acc = class_correct / class_total if class_total > 0 else 0
        print(f"🎯 Akurasi kelas '{label}': {class_acc:.2f}")
    
    for layer in model.layers:
        layer.trainable = False

    # Simpan model
    model.save("model_moler.h5")
    print("💾 Model disimpan sebagai model_moler.h5")


if __name__ == "__main__":
    # Bersihkan folder output jika ada
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    # 1. Bagi dataset
    split_dataset()

    # 2. Train dan evaluasi
    train_model()
