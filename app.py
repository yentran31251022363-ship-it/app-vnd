import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Tiêu đề của ứng dụng
st.title("Dự đoán Mệnh giá Tiền Việt Nam (VND Classifier)")
st.write("Tải lên một bức ảnh tiền VNĐ để mô hình phân loại.")

# Hàm tải mô hình (sử dụng cache để không phải tải lại nhiều lần)
@st.cache_resource
def load_model():
    # Tải mô hình đã được huấn luyện
    model = tf.keras.models.load_model('vnd_classifier.h5')
    return model

model = load_model()

# Khai báo các nhãn (Mệnh giá tiền) - Bạn hãy điều chỉnh lại thứ tự cho khớp với lúc training
class_names = ['1000', '2000', '5000', '10000', '20000', '50000', '100000', '200000', '500000']

# Widget tải ảnh lên
uploaded_file = st.file_uploader("Chọn một bức ảnh...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Hiển thị ảnh
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh đã tải lên', use_container_width=True)
    
    st.write("Đang phân loại...")
    
    # Tiền xử lý ảnh (Lưu ý: Thay đổi (224, 224) thành kích thước đầu vào mà mô hình của bạn yêu cầu)
    img_resized = image.resize((224, 224)) 
    img_array = np.array(img_resized)
    
    # Chuẩn hóa ảnh nếu lúc train bạn có chia cho 255
    # img_array = img_array / 255.0 
    
    img_array = np.expand_dims(img_array, axis=0)
    
    # Chạy dự đoán
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    # Hiển thị kết quả
    st.success(f"Dự đoán: {class_names[predicted_index]} VNĐ")
    st.info(f"Độ tin cậy: {confidence * 100:.2f}%")
