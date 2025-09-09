import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import joblib
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from mtcnn import MTCNN
from sklearn.preprocessing import normalize
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ================================
# 1. Load MTCNN Detector
# ================================
detector = MTCNN()

# ================================
# 2. Load SVM
# ================================
svm_loaded = joblib.load("face_recognition_resnet50.pkl")

# Mapping ID -> tên (cập nhật theo dataset)
id_to_name = {
    "1": "Minh Anh",
    "2": "Linh",
    "3": "Vu",
    "4": "Son"
}

# ================================
# 3. ResNet50 feature extractor
# ================================
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224,224,3))
for layer in resnet_model.layers:
    layer.trainable = False

def extract_features(imgs):
    features = []
    for img in imgs:
        x_input = preprocess_input(img.astype(np.float32))
        x_input = np.expand_dims(x_input, axis=0)
        feat = resnet_model.predict(x_input, verbose=0)
        features.append(feat.flatten())
    return np.array(features)

# ================================
# 4. Detect + crop face
# ================================
def detect_and_crop_faces(img):
    results = detector.detect_faces(img)
    face_list = []
    bboxes = []

    for result in results:
        x, y, w, h = result['box']
        x, y = max(0, x), max(0, y)
        x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
        if x2 - x <= 0 or y2 - y <= 0:
            continue
        face_img = img[y:y2, x:x2]
        face_img = cv2.resize(face_img, (224, 224))
        face_list.append(face_img)
        bboxes.append((x, y, x2, y2))

    # fallback resize toàn bộ frame nếu không detect
    if len(face_list) == 0:
        face_list.append(cv2.resize(img, (224, 224)))
        bboxes.append((0,0,img.shape[1], img.shape[0]))

    return face_list, bboxes

# ================================
# 5. Nhận diện + vẽ
# ================================
def recognize_and_show(frame_bgr):
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    faces, bboxes = detect_and_crop_faces(img_rgb)

    for face, bbox in zip(faces, bboxes):
        feat = extract_features([face])
        feat_norm = normalize(feat, norm='l2')
        pred_label = svm_loaded.predict(feat_norm)[0]
        pred_prob = svm_loaded.predict_proba(feat_norm).max()
        name = id_to_name.get(str(pred_label), f"ID {pred_label}")

        x1, y1, x2, y2 = bbox
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_rgb, f"{name} ({pred_prob*100:.2f}%)",
                    (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    return img_rgb

# ================================
# 6. Tkinter GUI
# ================================
root = tk.Tk()
root.title("Face Recognition Demo")
root.geometry("900x700")

panel = tk.Label(root, bg="gray")
panel.pack(pady=20)
cap = None

def open_file():
    global panel, cap
    if cap is not None:
        cap.release()
        cap = None

    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
    if not file_path:
        return

    with open(file_path, 'rb') as f:
        file_bytes = np.frombuffer(f.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    img_rgb = recognize_and_show(img_bgr)
    img_rgb = cv2.resize(img_rgb, (640,480))
    img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
    panel.config(image=img_tk)
    panel.image = img_tk

def open_camera():
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)
    show_frame()

def show_frame():
    global cap, panel
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img_rgb = recognize_and_show(frame)
            img_rgb = cv2.resize(img_rgb, (640,480))
            img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
            panel.config(image=img_tk)
            panel.image = img_tk
        panel.after(30, show_frame)

btn_frame = tk.Frame(root)
btn_frame.pack()

btn_file = tk.Button(btn_frame, text="Chọn file", command=open_file, width=15, height=2, bg="#FF9800", fg="white")
btn_file.pack(side="left", padx=10)

btn_cam = tk.Button(btn_frame, text="Mở camera", command=open_camera, width=15, height=2, bg="#4CAF50", fg="white")
btn_cam.pack(side="left", padx=10)

root.mainloop()