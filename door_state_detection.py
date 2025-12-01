import cv2
import numpy as np
from ultralytics import YOLO

# === Load YOLOv8 model ===
model = YOLO(r"D:\Container-truck images\best.pt")

# === Preprocessing with contour extraction ===
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((9, 9), np.uint8)
    eroded = cv2.erode(thresh, kernel, iterations=1)
    _, thresh1 = cv2.threshold(eroded, 127, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh1, np.ones((7, 7), np.uint8), iterations=1)
    edged = cv2.Canny(dilated, 100, 150)
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, edged, gray

# === Intensity difference between left and right halves ===
def detect_door_state_by_intensity(gray_img):
    h, w = gray_img.shape
    left_half = gray_img[:, :w // 2]
    right_half = gray_img[:, w // 2:]
    mean_left = np.mean(left_half)
    mean_right = np.mean(right_half)
    intensity_diff = abs(mean_left - mean_right)
    state = "Open" if intensity_diff > 30 else "Closed"
    return state, intensity_diff

# === Difference between current frame and reference ===
def detect_door_state_vs_reference(current_gray, reference_gray, threshold=10):
    current_mean = np.mean(current_gray)
    reference_mean = np.mean(reference_gray)
    diff = abs(current_mean - reference_mean)
    state = "Open" if diff > threshold else "Closed"
    return state, diff

# === Setup video ===
video_path = r"C:\Users\siva-intern\Downloads\open7.mp4"
cap = cv2.VideoCapture(video_path)

reference_gray_dict = {}  # {object_id: reference_gray}

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.track(frame, persist=True, classes=[0])
    result = results[0]

    if result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)

        for box, obj_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2].copy()

            contours, edges, gray = preprocess(crop)

            # === Store reference frame for each container ===
            if obj_id not in reference_gray_dict:
                reference_gray_dict[obj_id] = gray.copy()

            # === Logic 1: Left vs Right intensity ===
            door_state1, intensity_diff = detect_door_state_by_intensity(gray)

            # === Logic 2: Current vs Reference ===q
            reference_gray = reference_gray_dict[obj_id]
            door_state2, ref_diff = detect_door_state_vs_reference(gray, reference_gray)

            # === Final Decision ===
            if door_state1 == "Open" or door_state2 == "Open":
                final_state = "Open"
            else:
                final_state = "Closed"

            label_color = (0, 255, 0) if final_state == "Open" else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), label_color, 2)
            cv2.putText(frame, f"ID:{obj_id} {final_state}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, label_color, 2)

            # Draw shifted contours
            for cnt in contours:
                cnt_shifted = cnt + [x1, y1]
                cv2.drawContours(frame, [cnt_shifted], -1, (255, 255, 0), 1)

            # 🔍 Debug Info
            print(f"[ID: {obj_id}] IntensityDiff: {intensity_diff:.2f}, RefDiff: {ref_diff:.2f} => Final: {final_state}")

    cv2.imshow("YOLO + Dual Logic Open/Close", frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()