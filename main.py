import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print(f"gpu: {gpu}")
    tf.config.experimental.set_memory_growth(gpu, True)



data_directory = 'train/'

classes = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", 'surprise']

# for category in classes:
#     path = os.path.join(data_directory, category)
#     for img in os.listdir(path):
#         img_array = cv2.imread(os.path.join(path, img))
#         plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
#         plt.show()
#         break
#     break




image_size = 224
image_array = cv2.imread(os.path.join(data_directory,"anger", "image0000006.jpg"))
new_array = cv2.resize(image_array, (image_size, image_size))

plt.imshow(new_array)
plt.show()

training_data = []

def create_training_data():
    for category in classes:
        path = os.path.join(data_directory, category)
        class_num = classes.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (image_size, image_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass