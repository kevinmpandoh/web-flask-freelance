import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator

# ==== Load Model ====
model = tf.keras.models.load_model("saved_model/daun_bawang_cnn.h5")

# ==== Test Data ====
test_dir = "dataset/test"
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

# ==== Evaluate ====
loss, acc = model.evaluate(test_generator)
print(f"\n📊 Test Accuracy: {acc:.2f}")

# ==== Predictions ====
Y_pred = model.predict(test_generator)
y_pred = (Y_pred > 0.5).astype(int).flatten()
y_true = test_generator.classes

# ==== Metrics ====
class_labels = list(test_generator.class_indices.keys())

print("\n🧾 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\n🧾 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Akurasi per kelas
for i, label in enumerate(class_labels):
    class_correct = np.sum((y_true == i) & (y_pred == i))
    class_total = np.sum(y_true == i)
    class_acc = class_correct / class_total if class_total > 0 else 0
    print(f"🎯 Akurasi kelas '{label}': {class_acc:.2f}")
