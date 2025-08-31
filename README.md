# 📷 Streamlit Camera + Image Processing Project  

โปรเจกต์นี้เป็นเว็บแอปที่สร้างด้วย **Streamlit** เพื่อทำงานดังนี้:  
1. เปิดกล้องจาก **Webcam** ได้ 2 โหมด (Snapshot, Live Stream)  
2. ทำ **Image Processing อย่างง่าย** (Grayscale, Blur, Canny Edge, Threshold)  
3. สามารถ **ปรับค่าพารามิเตอร์** ของการประมวลผลได้จาก Sidebar  
4. แสดง **ผลลัพธ์หลังประมวลผล**  
5. แสดง **กราฟคุณสมบัติภาพ** เช่น Histogram และค่าเฉลี่ยสี RGB  

---

## ⚙️ การติดตั้ง

1. Clone หรือดาวน์โหลดโปรเจกต์  
   ```bash
   git clone https://github.com/sakkarin31/camera_image-process
2. ติดตั้งตาม requirements
   ```bash
   pip install -r requirements.txt

## การใช้งาน

รันคำสั่งนี้ใน terminal:
    ```bash
    streamlit run app.py
จากนั้นเปิดเบราว์เซอร์ไปที่ลิงก์ที่ Streamlit แสดง

## ฟีเจอร์
- โหมด Snapshot → กดถ่ายภาพทีละรูป แล้วแสดงผลลัพธ์พร้อมกราฟ

- โหมด Live Stream → เปิดกล้องแบบเรียลไทม์ สามารถประมวลผลทีละเฟรม และจับภาพล่าสุดเพื่อวิเคราะห์

### Image Processing ที่รองรับ

- Grayscale

- Gaussian Blur

- Binary Threshold

### Graph ที่รองรับ

- Grayscale Histogram

- RGB Channel Means

## ตัวอย่างผลลัพธ์
