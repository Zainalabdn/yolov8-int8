import streamlit as st
import cv2, numpy as np, tempfile, time, os
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Deteksi INT8", layout="wide")

@st.cache_resource
def load_model():
    # Karena foldernya sudah benar, kita tinggal panggil namanya
    return YOLO('best_int8_openvino_model', task='detect')

model = load_model()

st.sidebar.title("⚙️ Pengaturan")
conf_threshold = st.sidebar.slider("Confidence", 0.05, 0.95, 0.35, 0.05)
iou_threshold = st.sidebar.slider("IoU", 0.10, 0.90, 0.70, 0.05)
st.sidebar.markdown("**Kelas Kerusakan:**\n- 🟩 Crack\n- 🟧 Pothole\n- 🟦 Other Corruption")

st.title("🛣️ Deteksi Kerusakan Jalan INT8 (OpenVINO)")
tab1, tab2 = st.tabs(["📷 Gambar", "🎥 Video"])

with tab1:
    uploaded_img = st.file_uploader("Unggah Gambar", type=['jpg', 'jpeg', 'png'])
    if uploaded_img and model:
        img_array = np.array(Image.open(uploaded_img).convert('RGB'))
        start_time = time.time()
        results = model.predict(img_array, conf=conf_threshold, iou=iou_threshold, imgsz=960)
        inf_time = (time.time() - start_time) * 1000
        res_rgb = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
        st.image(res_rgb, caption=f"Waktu Inferensi: {inf_time:.2f} ms", use_container_width=True)

with tab2:
    uploaded_vid = st.file_uploader("Unggah Video", type=['mp4', 'mov'])
    if uploaded_vid and model:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_vid.read())
        cap = cv2.VideoCapture(tfile.name)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        out = cv2.VideoWriter(out_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(cap.get(3)), int(cap.get(4))))
        
        start_time = time.time()
        frames = 0
        status = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            out.write(model.predict(frame, conf=conf_threshold, iou=iou_threshold, imgsz=960, verbose=False)[0].plot())
            frames += 1
            status.text(f"Memproses frame {frames}...")
            
        cap.release()
        out.release()
        status.empty()
        st.video(out_file.name)
        st.success(f"⚡ Kecepatan Rata-rata: {frames / (time.time() - start_time):.2f} FPS")