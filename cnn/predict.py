import numpy as np
import os
from keras.preprocessing import image
from keras.models import load_model

class Predictor:
    def __init__(self, model_path, train_dir, img_size=(128, 128)):
        self.model = load_model(model_path)
        self.img_size = img_size
        self.class_labels = sorted(os.listdir(train_dir))

    def predict(self, img_path):
        img = image.load_img(img_path, target_size=self.img_size)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = self.model.predict(img_array)
        predicted_class = 1 if prediction[0][0] >= 0.5 else 0
        confidence = prediction[0][0] if predicted_class == 1 else 1 - prediction[0][0]

        print(f"📷 Image: {img_path}")
        print(f"🔍 Predicted: {self.class_labels[predicted_class]} ({confidence*100:.2f}%)")
        return self.class_labels[predicted_class], confidence

# Contoh pemakaian
if __name__ == "__main__":
    predictor = Predictor("model_moler.h5", "dataset/train")
    predictor.predict("testing_moler.jpg")
