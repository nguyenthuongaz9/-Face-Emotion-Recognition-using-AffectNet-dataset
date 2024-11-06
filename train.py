import tensorflow as tf
from tensorflow.keras import layers, Model
from main import X, Y, image_size  # Giả sử X, Y và image_size được định nghĩa trong main.py

# Cấu hình TensorFlow để cho phép tăng dần bộ nhớ GPU
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Tải mô hình MobileNetV2 với trọng số đã được huấn luyện
base_model = tf.keras.applications.MobileNetV2(
    include_top=False, 
    weights="imagenet", 
    input_shape=(image_size, image_size, 3)
)

# Đóng băng các lớp của mô hình gốc để tránh huấn luyện lại
for layer in base_model.layers:
    layer.trainable = False

# Định nghĩa đầu vào và đầu ra của mô hình
base_input = base_model.input
base_output = base_model.output

# Thêm các lớp pooling và dense tùy chỉnh
x = layers.GlobalAveragePooling2D()(base_output)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(64, activation='relu')(x)
final_output = layers.Dense(8, activation='softmax')(x)  # Lớp đầu ra với 8 lớp cảm xúc

# Tạo mô hình hoàn chỉnh
model = Model(inputs=base_input, outputs=final_output)

# Biên dịch mô hình với các tham số
model.compile(
    loss="sparse_categorical_crossentropy", 
    optimizer="adam", 
    metrics=["accuracy"]
)

# In ra tóm tắt mô hình
model.summary()

# Sử dụng tf.data.Dataset để load dữ liệu theo batch
batch_size = 16
train_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Huấn luyện mô hình
model.fit(train_dataset, epochs=25)

# Lưu mô hình đã huấn luyện
model.save("models/emotion_detection_model.h5")
