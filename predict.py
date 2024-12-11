import tensorflow as tf
import cv2
import numpy as np
import sys
# from data_loader import image_size, classes
import os

image_size = 122
classes = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

model_path = "models/emotion_detection_model_2.h5"

model = tf.keras.models.load_model(model_path)

# Hàm dự đoán cảm xúc từ hình ảnh
def predict_emotion(image_path):
    # Đọc và xử lý ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image '{image_path}'.")
        return
    img_resized = cv2.resize(img, (image_size, image_size))
    img_array = np.array(img_resized).reshape(-1, image_size, image_size, 3) / 255.0
    
    # Dự đoán
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    print(predicted_class)
    emotion_label = classes[predicted_class]
    
   
    print(f"Predicted Emotion: {emotion_label}")

if __name__ == "__main__":
        predict_emotion(os.path.join("3.jpg"))
