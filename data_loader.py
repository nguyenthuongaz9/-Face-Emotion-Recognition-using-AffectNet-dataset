import cv2
import numpy as np
import os
import tensorflow as tf
import random
import torch



if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
    gpu_name = torch.cuda.get_device_name(0) 
    print(f"CUDA is available. Using GPU: {gpu_name}")
    device = torch.device("cuda") 
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device("cpu")  


data_directory = 'data/train/'
classes = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

image_size = 122  

training_data = []

# Tạo dữ liệu huấn luyện
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
                # Resize ảnh và chuẩn hóa
                new_array = cv2.resize(img_array, (image_size, image_size)).astype('float32') / 255.0
                training_data.append([new_array, class_num])
            except Exception as e:
                print(f"Error processing image '{img_path}': {e}")

create_training_data()

print(f"Total number of samples: {len(training_data)}")

random.shuffle(training_data)

train_ratio = 0.8  
train_size = int(len(training_data) * train_ratio)

train_data = training_data[:train_size]
validation_data = training_data[train_size:]

def split_features_and_labels(data):
    X = []
    Y = []
    for features, label in data:
        X.append(features)
        Y.append(label)
    return np.array(X, dtype="float32").reshape(-1, image_size, image_size, 3), np.array(Y, dtype="int32")

X_train, Y_train = split_features_and_labels(train_data)
X_val, Y_val = split_features_and_labels(validation_data)

print(f"Training data: X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"Validation data: X_val shape: {X_val.shape}, Y_val shape: {Y_val.shape}")
