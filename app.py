import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Camera + Image Processing + Graph", layout="wide")
st.title("Streamlit Camera Demo (Processing + Graph)")

with st.sidebar:
    st.header("Image Processing")
    proc = st.selectbox(
        "เลือกวิธีประมวลผล",
        ["None", "Grayscale", "Blur (Gaussian)", "Threshold (Binary)"],
        index=0,
    )

    # ค่าพารามิเตอร์ตามวิธี
    ksize = sigma = t1 = t2 = thr = 0
    if proc == "Blur (Gaussian)":
        ksize = st.slider("Kernel size (odd)", min_value=3, max_value=31, step=2, value=7)
        sigma = st.slider("Sigma", min_value=0, max_value=10, step=1, value=1)
    elif proc == "Threshold (Binary)":
        thr = st.slider("Threshold", 0, 255, 128)

    st.header("Graph")
    graph_type = st.radio(
        "เลือกกราฟคุณสมบัติภาพ",
        ["Histogram (Grayscale)", "RGB Channel Means"],
        index=0
    )

def apply_processing(img_bgr):
    """รับภาพ BGR -> คืนภาพ BGR หลังประมวลผลตามที่เลือก"""
    if proc == "None":
        return img_bgr

    if proc == "Grayscale":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if proc == "Blur (Gaussian)":
        return cv2.GaussianBlur(img_bgr, (ksize, ksize), sigma)

    if proc == "Threshold (Binary)":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, bin_img = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)

    return img_bgr

def plot_graph(img_bgr):
    """วาดกราฟตาม graph_type แล้วคืน st.pyplot()"""
    fig = plt.figure(figsize=(4.5, 3))

    if graph_type == "Histogram (Grayscale)":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        hist, bins = np.histogram(gray.flatten(), 256, [0, 256])
        plt.plot(hist)            
        plt.title("Grayscale Histogram")
        plt.xlabel("Intensity (0–255)")
        plt.ylabel("Count")

    elif graph_type == "RGB Channel Means":
        means = img_bgr.reshape(-1, 3).mean(axis=0)  # B, G, R
        labels = ["Blue", "Green", "Red"]
        plt.bar(labels, means)      
        plt.title("RGB Channel Means")
        plt.ylabel("Mean Value (0–255)")

    st.pyplot(fig)

mode = st.selectbox(
    "เลือกแหล่งวิดีโอ",
    ["Webcam (snapshot ง่ายสุด)", "Webcam (live stream)"],
)

if mode == "Webcam (snapshot ง่ายสุด)":
    st.write("กดปุ่มเพื่อถ่ายภาพจากกล้องโน้ตบุ๊ก (ภาพนิ่งทีละรูป)")
    img_file = st.camera_input("ถ่ายภาพด้วย webcam")

    col1, col2 = st.columns(2)
    if img_file is not None:
        # bytes -> np.ndarray (BGR)
        file_bytes = np.frombuffer(img_file.getvalue(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # ประมวลผล
        out_bgr = apply_processing(img_bgr)

        # แสดงต้นฉบับ/ผลลัพธ์
        with col1:
            st.subheader("ต้นฉบับ")
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
        with col2:
            st.subheader("หลังประมวลผล")
            st.image(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

        st.subheader("กราฟคุณสมบัติภาพ (จากรูปที่ประมวลผลแล้ว)")
        plot_graph(out_bgr)

elif mode == "Webcam (live stream)":
    try:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
        import av
        from collections import deque
        import threading

        class VideoProcessor(VideoProcessorBase):
            def __init__(self):
                # พารามิเตอร์ (อัปเดตจาก UI)
                self.proc = "None"
                self.ksize = 7
                self.sigma = 1
                self.t1 = 100
                self.t2 = 200
                self.thr = 128

                # เก็บเฟรมล่าสุดเพื่อวิเคราะห์/ทำกราฟ
                self.last_frame = None
                self.lock = threading.Lock()

            def update_params(self, proc, ksize, sigma, t1, t2, thr):
                self.proc = proc
                self.ksize = ksize
                self.sigma = sigma
                self.t1 = t1
                self.t2 = t2
                self.thr = thr

            def _apply(self, img_bgr):
                if self.proc == "None":
                    return img_bgr
                if self.proc == "Grayscale":
                    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
                if self.proc == "Blur (Gaussian)":
                    k = max(3, int(self.ksize) | 1)  
                    return cv2.GaussianBlur(img_bgr, (k, k), int(self.sigma))
                if self.proc == "Threshold (Binary)":
                    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    _, b = cv2.threshold(g, int(self.thr), 255, cv2.THRESH_BINARY)
                    return cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
                return img_bgr

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                out = self._apply(img)
                # เก็บเฟรมล่าสุด 
                with self.lock:
                    self.last_frame = out.copy()
                return av.VideoFrame.from_ndarray(out, format="bgr24")

            def get_latest_processed(self):
                with self.lock:
                    if self.last_frame is None:
                        return None
                    return self.last_frame.copy()

        st.write("อนุญาตกล้องในเบราว์เซอร์เพื่อสตรีมแบบเรียลไทม์ ปรับพารามิเตอร์ด้านซ้ายได้เลย")
        ctx = webrtc_streamer(
            key="webcam",
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
        )

        if ctx.video_processor:
            ctx.video_processor.update_params(proc, ksize, sigma, t1, t2, thr)

            st.divider()
            st.subheader("วิเคราะห์เฟรมล่าสุด (หลังประมวลผล)")
            if st.button("จับภาพเฟรมล่าสุดเพื่อแสดงผล + วาดกราฟ"):
                latest = ctx.video_processor.get_latest_processed()
                if latest is None:
                    st.info("ยังไม่มีเฟรม กรุณารอสตรีมเริ่มทำงานสักครู่")
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(cv2.cvtColor(latest, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
                        st.caption("เฟรมล่าสุด (หลังประมวลผล)")
                    with c2:
                        plot_graph(latest)

    except Exception as e:
        st.error("ต้องติดตั้ง streamlit-webrtc และ av ก่อน (`pip install streamlit-webrtc av`).")
        st.exception(e)
