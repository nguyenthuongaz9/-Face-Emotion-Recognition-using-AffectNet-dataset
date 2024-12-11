import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from data_loader import X, Y, image_size

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

base_model = tf.keras.applications.MobileNetV2(
    include_top=False, 
    weights="imagenet", 
    input_shape=(image_size, image_size, 3)
)

for layer in base_model.layers[:-30]:  
    layer.trainable = False

base_input = base_model.input
base_output = base_model.output

x = layers.GlobalAveragePooling2D()(base_output)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
final_output = layers.Dense(8, activation='softmax')(x) 

model = Model(inputs=base_input, outputs=final_output)

optimizer = Adam(learning_rate=0.001, decay=1e-5)
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator()

train_dataset = train_datagen.flow(X_train, Y_train, batch_size=16)
val_dataset = val_datagen.flow(X_val, Y_val, batch_size=16)

early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


lr_scheduler = LearningRateScheduler(scheduler)

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=25,
    callbacks=[early_stopping, lr_scheduler]
)


model.save("models/emotion_detection_model_optimized.h5")
