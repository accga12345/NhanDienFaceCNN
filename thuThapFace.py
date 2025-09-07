import cv2
import os

# Tạo thư mục lưu ảnh nếu chưa có
dataset_path = "dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Mở camera
cap = cv2.VideoCapture(0)

# Load bộ phát hiện khuôn mặt Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

count = 0
user_id = input("Nhap ID hoac ten nguoi can thu thap: ")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Cắt và lưu ảnh khuôn mặt
        face = gray[y:y + h, x:x + w]
        count += 1
        filename = f"{dataset_path}/{user_id}_{count}.jpg"
        cv2.imwrite(filename, face)


    cv2.imshow("Face Collection", frame)

    # Nhấn q để thoát hoặc thu thập 100 ảnh
    if cv2.waitKey(1) & 0xFF == ord("q") or count >= 100:
        break

cap.release()
cv2.destroyAllWindows()

print(f"Đã lưu {count} ảnh khuôn mặt vào thư mục '{dataset_path}'")
