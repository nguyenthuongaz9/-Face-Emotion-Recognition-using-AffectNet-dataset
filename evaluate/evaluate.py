import tensorflow as tf
import numpy as np
import os
import sys
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader.data_loader import image_size, classes
from sklearn.metrics import classification_report, confusion_matrix

model_path = "models/emotion_detection_model_4.h5"
test_data_directory = 'data/test/'

model = tf.keras.models.load_model(model_path)

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

load_test_data()

X_test = []
Y_test = []

for features, label in test_data:
    X_test.append(features)
    Y_test.append(label)

X_test = np.array(X_test).reshape(-1, image_size, image_size, 3) / 255.0
Y_test = np.array(Y_test)

test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)

print("\nClassification Report:")
print(classification_report(Y_test, Y_pred_classes, target_names=classes))

print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, Y_pred_classes))
