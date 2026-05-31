import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. CẤU HÌNH GIAO DIỆN APP
st.set_page_config(page_title="Nhận Diện Tiền VN", page_icon="💵", layout="centered")

st.title("💸 Ứng Dụng Nhận Diện Tiền Việt Nam")
st.write("Sử dụng Camera hoặc tải ảnh lên để hệ thống AI dự đoán mệnh giá tờ tiền.")

# 2. TẢI MÔ HÌNH AI
@st.cache_resource
def load_model():
    # Tải mô hình đã train từ Colab
    return tf.keras.models.load_model('model_tien_vn.h5')

model = load_model()

# Danh sách các mệnh giá (Bạn cần điều chỉnh cho khớp với class_names trên Colab)
class_names = ['1000', '2000', '10000', '50000', '500000'] 

# 3. CHỨC NĂNG XỬ LÝ ẢNH & DỰ ĐOÁN
def predict_image(image):
    # Thay đổi kích thước ảnh cho khớp với input của mô hình (128x128)
    img = image.resize((128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Tạo batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    return predicted_class, confidence

# 4. TÍNH NĂNG CAMERA VÀ TẢI ẢNH (UI)
tab1, tab2 = st.tabs(["📸 Dùng Camera", "📂 Tải ảnh lên"])

with tab1:
    camera_image = st.camera_input("Chụp một bức ảnh tờ tiền")
    if camera_image is not None:
        image = Image.open(camera_image)
        st.image(image, caption="Ảnh vừa chụp", use_column_width=True)
        
        with st.spinner('AI đang phân tích...'):
            label, conf = predict_image(image)
        st.success(f"**Kết quả:** Mệnh giá {label} VNĐ (Độ tin cậy: {conf:.2f}%)")

with tab2:
    uploaded_file = st.file_uploader("Chọn một bức ảnh...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh đã tải lên", use_column_width=True)
        
        with st.spinner('AI đang phân tích...'):
            label, conf = predict_image(image)
        st.success(f"**Kết quả:** Mệnh giá {label} VNĐ (Độ tin cậy: {conf:.2f}%)")