import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Cấu hình trang Web
st.set_page_config(page_title="Nhận Diện Món Ăn VN", page_icon="🍲", layout="centered")

st.title("🍲 AI Nhận Diện Món Ăn Việt Nam")
st.markdown("---")

# 2. Tải bộ não AI (Phiên bản TFLite siêu nhẹ)
@st.cache_resource
def load_tflite_model():
    # Đọc file .tflite
    interpreter = tf.lite.Interpreter(model_path="Model_MonAn_VN.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 3. Khai báo danh sách món ăn
danh_sach_thu_muc = sorted([
    'Banh_mi', 'Banh_xeo', 'Bun_bo_Hue', 'Bun_cha', 
    'Bun_dau_mam_tom', 'Cha_gio', 'Com_tam', 
    'Goi_cuon', 'Mi_Quang', 'Pho_bo'
])

tu_dien_mon = {
    'Banh_mi': 'Bánh Mì Thịt', 'Banh_xeo': 'Bánh Xèo', 
    'Bun_bo_Hue': 'Bún Bò Huế', 'Bun_cha': 'Bún Chả Hà Nội', 
    'Bun_dau_mam_tom': 'Bún Đậu Mắm Tôm', 'Cha_gio': 'Chả Giò (Nem Rán)', 
    'Com_tam': 'Cơm Tấm Sườn Bì Chả', 'Goi_cuon': 'Gỏi Cuốn', 
    'Mi_Quang': 'Mì Quảng', 'Pho_bo': 'Phở Bò'
}

# 4. Hàm dự đoán bằng TFLite
def predict_tflite(image):
    # Resize và chuẩn hóa ảnh
    img_resized = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)
    
    # Bơm ảnh vào mô hình TFLite
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke() # Chạy dự đoán
    
    # Lấy kết quả
    predictions = interpreter.get_tensor(output_details[0]['index'])
    score = tf.nn.softmax(predictions[0])
    
    thu_muc_du_doan = danh_sach_thu_muc[np.argmax(score)]
    ten_mon = tu_dien_mon[thu_muc_du_doan]
    do_tu_tin = 100 * np.max(score)
    
    return ten_mon, do_tu_tin

# 5. Thiết kế Giao diện Người dùng
tab1, tab2 = st.tabs(["📁 Tải ảnh lên", "📸 Chụp Camera"])

with tab1:
    uploaded_file = st.file_uploader("Chọn một bức ảnh món ăn...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Ảnh bạn vừa tải lên', use_container_width=True)
        
        if st.button("Phân tích ảnh này 🚀"):
            with st.spinner('AI đang nếm thử...'):
                ten_mon, do_tu_tin = predict_tflite(image)
                st.success(f"🎯 KẾT QUẢ: Đây là món **{ten_mon.upper()}**")
                st.info(f"Độ tự tin: **{do_tu_tin:.2f}%**")

with tab2:
    camera_file = st.camera_input("Đưa món ăn ra trước Camera")
    if camera_file is not None:
        image = Image.open(camera_file)
        with st.spinner('AI đang soi camera...'):
            ten_mon, do_tu_tin = predict_tflite(image)
            st.success(f"🎯 KẾT QUẢ: Đây là món **{ten_mon.upper()}**")
            st.info(f"Độ tự tin: **{do_tu_tin:.2f}%**")
