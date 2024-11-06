import tensorflow as tf
import cv2
import numpy as np
import sys
from main import image_size, classes

# Đường dẫn lưu mô hình
model_path = "models/emotion_detection_model.h5"

# Load mô hình
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
    emotion_label = classes[predicted_class]
    
    # Kết quả dự đoán
    print(f"Predicted Emotion: {emotion_label}")

# Kiểm tra và lấy đường dẫn từ tham số dòng lệnh
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        image_path = sys.argv[1]
        predict_emotion(image_path)
