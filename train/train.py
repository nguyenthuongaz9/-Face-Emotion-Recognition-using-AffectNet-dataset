import tensorflow as tf
from tensorflow.keras import layers, Model
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader.data_loader import X_train, Y_train, image_size, X_val, Y_val
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# if torch.cuda.is_available():
#     print("CUDA is available. Using GPU.")
#     gpu_name = torch.cuda.get_device_name(0) 
#     print(f"CUDA is available. Using GPU: {gpu_name}")
#     device = torch.device("cuda") 
# else:
#     print("CUDA is not available. Using CPU.")
#     device = torch.device("cpu")  

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

base_model = tf.keras.applications.MobileNetV2(
    include_top=False, 
    weights="imagenet", 
    input_shape=(image_size, image_size, 3)
)

for layer in base_model.layers:
    layer.trainable = False

base_input = base_model.input
base_output = base_model.output

x = layers.GlobalAveragePooling2D()(base_output)
x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation='relu')(x)
final_output = layers.Dense(8, activation='softmax')(x)  

model = Model(inputs=base_input, outputs=final_output)


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],

)


model.summary()

batch_size = 128
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
val_dataset = val_dataset.batch(batch_size)


labels = np.concatenate([y.numpy() for _, y in train_dataset])

class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

history = model.fit(
    train_dataset,
    validation_data=val_dataset,  
    epochs=30,
    class_weights = class_weight_dict
)

model.save("models/emotion_detection_model_4.h5")


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()


unique, counts = np.unique(labels, return_counts=True)
plt.bar(unique, counts)
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.title("Class Distribution in AffectNet Dataset")
plt.show()
