import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

# ================================
# 1. Load Haarcascade và SVM
# ================================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
svm_loaded = joblib.load("face_svm_finetune.pkl")  # SVM đã train với 256-dim feature

# Mapping ID -> tên
id_to_name = {
    1: "Minh Anh",
    2: "Linh",
    3: "Vu",
    4: "Son"
}

# ================================
# 2. Tạo feature_model giống lúc train SVM
# ================================
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
feature_model = Model(inputs=base_model.input, outputs=x)

# ================================
# 3. Hàm detect + crop face
# ================================
def detect_and_crop_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        margin = 20
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img.shape[1] - x, w + 2*margin)
        h = min(img.shape[0] - y, h + 2*margin)
        face = img[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (224, 224))
        return face_resized, (x, y, x+w, y+h)
    return None, None

# ================================
# 4. Hàm extract feature
# ================================
def extract_features(img):
    img_preprocessed = preprocess_input(img.astype(np.float32))
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    features = feature_model.predict(img_batch, verbose=0)
    return features.flatten()

# ================================
# 5. Hàm nhận diện + vẽ
# ================================
def recognize_and_show(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    face_crop, bbox = detect_and_crop_face(img_rgb)

    if face_crop is None:
        cv2.putText(img_rgb, "Khong tim thay khuon mat", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    else:
        features = extract_features(face_crop).reshape(1, -1)
        pred_label = svm_loaded.predict(features)[0]
        pred_prob = svm_loaded.predict_proba(features).max()
        name = id_to_name.get(pred_label, f"ID {pred_label}")

        x1, y1, x2, y2 = bbox
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_rgb, f"Ten : {name} ({pred_prob:.2f})",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
    return img_rgb

# ================================
# 6. Tkinter GUI
# ================================
root = tk.Tk()
root.title("Face Recognition Demo")
root.geometry("900x700")

cap = None
panel = tk.Label(root, bg="gray")
panel.pack(pady=20)

def open_file():
    global panel, cap
    if cap is not None:
        cap.release()
        cap = None

    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
    if not file_path:
        return

    # Đọc ảnh hỗ trợ Unicode
    with open(file_path, 'rb') as f:
        file_bytes = np.frombuffer(f.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    img_rgb = recognize_and_show(img_bgr)
    img_rgb = cv2.resize(img_rgb, (640, 480))
    img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
    panel.config(image=img_tk)
    panel.image = img_tk

def open_camera():
    global cap
    cap = cv2.VideoCapture(0)
    show_frame()

def show_frame():
    global cap, panel
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img_rgb = recognize_and_show(frame)
            img_rgb = cv2.resize(img_rgb, (640, 480))
            img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
            panel.config(image=img_tk)
            panel.image = img_tk
        panel.after(10, show_frame)

# Nút chức năng
btn_frame = tk.Frame(root)
btn_frame.pack()

btn_file = tk.Button(btn_frame, text="Chọn file", command=open_file, width=15, height=2, bg="#FF9800", fg="white")
btn_file.pack(side="left", padx=10)

btn_cam = tk.Button(btn_frame, text="Mở camera", command=open_camera, width=15, height=2, bg="#4CAF50", fg="white")
btn_cam.pack(side="left", padx=10)

root.mainloop()
