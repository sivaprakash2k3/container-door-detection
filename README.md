# 🚛 Container Door Open/Close Detection using YOLOv8

This project detects whether a container door is **Open** or **Closed** from video footage using a **YOLOv8 object detection model** along with custom **image processing and dual intensity-based logic**.

It is designed for **real-time monitoring of container trucks** in logistics yards, ports, and warehouses.

---

## 📌 Features
- ✅ Real-time container detection and tracking using YOLOv8
- ✅ Dual-logic door state detection:
  - Left vs Right **intensity comparison**
  - **Current frame vs reference frame** difference
- ✅ Works on recorded video input
- ✅ Displays:
  - Bounding boxes
  - Object ID
  - Open / Closed door state
- ✅ Edge detection & contour analysis using OpenCV

---

## 🛠️ Tech Stack
- Python  
- OpenCV  
- NumPy  
- Ultralytics YOLOv8  

---

## 📂 Project Structure
container-door-detection/
│── door_state_detection.py
│── requirements.txt
│── README.md
│── .gitignore

---

## ▶️ How to Run the Project

### 1️⃣ Install Requirements

---

### 2️⃣ Update Paths in Code

Edit these lines in `door_state_detection.py`:

```python
model = YOLO(r"D:\Container-truck images\best.pt")
video_path = r"C:\Users\siva-intern\Downloads\open7.mp4"

python door_state_detection.py
