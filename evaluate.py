import tensorflow as tf
import numpy as np
import os
from main import image_size, classes
from sklearn.metrics import classification_report, confusion_matrix

# Đường dẫn lưu mô hình
model_path = "models/emotion_detection_model.h5"
# Đường dẫn dữ liệu kiểm tra
test_data_directory = 'data/test/'

# Load mô hình
model = tf.keras.models.load_model(model_path)

# Tạo dữ liệu kiểm tra
test_data = []

def load_test_data():
    for category in classes:
        path = os.path.join(test_data_directory, category)
        if not os.path.exists(path):
            print(f"Warning: Directory '{path}' does not exist.")
            continue
        class_num = classes.index(category)
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                img_array = cv2.imread(img_path)
                if img_array is None:
                    print(f"Warning: Unable to read image '{img_path}'. Skipping.")
                    continue
                new_array = cv2.resize(img_array, (image_size, image_size))
                test_data.append([new_array, class_num])
            except Exception as e:
                print(f"Error: {e} in image '{img_path}'")

# Load test data
load_test_data()

# Chuẩn bị dữ liệu và nhãn cho kiểm tra
X_test = []
Y_test = []

for features, label in test_data:
    X_test.append(features)
    Y_test.append(label)

X_test = np.array(X_test).reshape(-1, image_size, image_size, 3) / 255.0
Y_test = np.array(Y_test)

# Đánh giá mô hình trên tập kiểm tra
test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Dự đoán và đánh giá chi tiết
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)

# Báo cáo chi tiết và ma trận nhầm lẫn
print("\nClassification Report:")
print(classification_report(Y_test, Y_pred_classes, target_names=classes))

print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, Y_pred_classes))
