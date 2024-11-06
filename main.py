import cv2
import numpy as np
import os
import tensorflow as tf
import random

# Kiểm tra GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print(f"gpu: {gpu}")
    tf.config.experimental.set_memory_growth(gpu, True)

print("gpus:", gpus)

# Đường dẫn dữ liệu và các nhãn lớp
data_directory = 'data/train/'
classes = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Giảm kích thước ảnh để tiết kiệm bộ nhớ
image_size = 112  # Giảm từ 224 xuống 112

# Tạo danh sách dữ liệu huấn luyện
training_data = []

def create_training_data():
    for category in classes:
        path = os.path.join(data_directory, category)
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
                # Resize ảnh và chuẩn hóa để tiết kiệm bộ nhớ
                new_array = cv2.resize(img_array, (image_size, image_size)).astype('float32') / 255.0
                training_data.append([new_array, class_num])
            except Exception as e:
                print(f"Error: {e} in image '{img_path}'")

# Gọi hàm tạo dữ liệu huấn luyện
create_training_data()

# Hiển thị số lượng dữ liệu huấn luyện
print(f"Number of training samples: {len(training_data)}")
random.shuffle(training_data)

# Tách dữ liệu và nhãn
X = []
Y = []

for features, label in training_data:
    X.append(features)
    Y.append(label)

# Đổi X thành kiểu float32 để tiết kiệm bộ nhớ
X = np.array(X, dtype="float32").reshape(-1, image_size, image_size, 3)
Y = np.array(Y, dtype="int32")

print(f"X shape: {X.shape}, Y shape: {Y.shape}")
