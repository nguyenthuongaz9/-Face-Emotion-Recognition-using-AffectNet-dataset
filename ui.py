import cv2
import numpy as np
import tensorflow as tf

# Tải mô hình đã huấn luyện
model = tf.keras.models.load_model("models/emotion_detection_model.h5")

# Đặt nhãn cảm xúc tương ứng với các lớp
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral", "Contempt"]

# Khởi động webcam
cap = cv2.VideoCapture(0)

# Kích thước ảnh mô hình đầu vào
image_size = 224  # Kích thước phải khớp với mô hình đã huấn luyện

def preprocess_image(frame):
    # Chuyển ảnh về kích thước cần thiết và chuẩn hóa
    frame_resized = cv2.resize(frame, (image_size, image_size))
    frame_normalized = frame_resized / 255.0
    frame_reshaped = np.reshape(frame_normalized, (1, image_size, image_size, 3))
    return frame_reshaped

while True:
    # Đọc khung hình từ camera
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển sang ảnh màu xám để phát hiện khuôn mặt
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Cắt khuôn mặt từ khung hình
        face = frame[y:y+h, x:x+w]
        face_input = preprocess_image(face)

        # Dự đoán cảm xúc
        predictions = model.predict(face_input)
        emotion_index = np.argmax(predictions)
        emotion_label = emotion_labels[emotion_index]
        confidence = np.max(predictions)

        # Vẽ hình chữ nhật quanh khuôn mặt và nhãn cảm xúc
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"{emotion_label} ({confidence*100:.2f}%)", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Hiển thị khung hình
    cv2.imshow("Emotion Detection", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
