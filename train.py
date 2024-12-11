import tensorflow as tf
from tensorflow.keras import layers, Model
from data_loader import X_train, Y_train, image_size, X_val, Y_val
import torch
import matplotlib.pyplot as plt

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
    gpu_name = torch.cuda.get_device_name(0) 
    print(f"CUDA is available. Using GPU: {gpu_name}")
    device = torch.device("cuda") 
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device("cpu")  

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
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(64, activation='relu')(x)
final_output = layers.Dense(8, activation='softmax')(x)  

model = Model(inputs=base_input, outputs=final_output)


model.compile(
    loss="sparse_categorical_crossentropy", 
    optimizer="adam", 
    metrics=["accuracy"]
)


model.summary()

batch_size = 16
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
val_dataset = val_dataset.batch(batch_size)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,  
    epochs=20
)

model.save("models/emotion_detection_model_2.h5")


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