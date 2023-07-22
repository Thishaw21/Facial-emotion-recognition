import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

# Load model từ tệp JSON
json_file = open('top_models\\fer.json', 'r')  # Mở tệp JSON chứa kiến trúc mô hình
loaded_model_json = json_file.read()  # Đọc nội dung của tệp JSON
json_file.close()  # Đóng tệp JSON
model = model_from_json(loaded_model_json)  # Tạo đối tượng mô hình dựa trên kiến trúc JSON

# Tải trọng số và gán cho mô hình
model.load_weights('top_models\\fer.h5')  # Tải trọng số mô hình từ tệp H5

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Tải bộ phân loại Haar cascade

cap = cv2.VideoCapture(0)  # Tạo đối tượng video capture để truy cập camera mặc định (chỉ mục 0)

while True:
    ret, img = cap.read()  # Chụp một khung hình từ camera
    if not ret:  # Nếu chụp không thành công, thoát khỏi vòng lặp
        break

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Chuyển đổi khung hình sang ảnh xám

    # Phát hiện khuôn mặt trong ảnh xám
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.1, 6, minSize=(150, 150))

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)  # Vẽ hình chữ nhật xung quanh khuôn mặt

        roi_gray = gray_img[y:y + w, x:x + h]  # Cắt vùng khuôn mặt từ ảnh xám
        roi_gray = cv2.resize(roi_gray, (48, 48))  # Thay đổi kích thước vùng khuôn mặt thành 48x48 điểm ảnh

        img_pixels = image.img_to_array(roi_gray)  # Chuyển đổi ảnh khuôn mặt thành mảng
        img_pixels = np.expand_dims(img_pixels, axis=0)  # Mở rộng kích thước của mảng
        img_pixels /= 255.0  # Chuẩn hóa giá trị điểm ảnh trong khoảng từ 0 đến 1

        predictions = model.predict(img_pixels)  # Dự đoán cảm xúc trên ảnh khuôn mặt
        max_index = int(np.argmax(predictions))  # Lấy chỉ số của cảm xúc dự đoán có xác suất cao nhất

        emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
        predicted_emotion = emotions[max_index]  # Ánh xạ chỉ số với nhãn cảm xúc tương ứng

        cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        # Hiển thị cảm xúc dự đoán dưới dạng văn bản trên ảnh

    resized_img = cv2.resize(img, (1000, 700))  # Thay đổi kích thước ảnh để hiển thị
    cv2.imshow('Facial Emotion Recognition', resized_img)  # Hiển thị ảnh trong cửa sổ có tiêu đề "Facial Emotion Recognition"

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Nếu nhấn phím 'q', thoát khỏi vòng lặp
        break

cap.release()  # Giải phóng video capture
cv2.destroyAllWindows()  # Đóng tất cả các cửa sổ
