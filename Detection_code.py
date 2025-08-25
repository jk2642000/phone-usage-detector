import os
import cv2
import json
import time
from datetime import datetime
import numpy as np
from ultralytics import YOLO

INPUT_VIDEO = r"C:\Users\yashc\Downloads\Test taks video-20250819T055608Z-1-001\Yash\AVI_video_example.avi"


VIDEO_FOLDER = os.path.splitext(os.path.basename(INPUT_VIDEO))[0]
os.makedirs(VIDEO_FOLDER, exist_ok=True)

OUTPUT_VIDEO = os.path.join(VIDEO_FOLDER, f"{VIDEO_FOLDER}.mp4")
MODEL_PATH = "yolo11m.pt"
SUMMARY_JSON = os.path.join(VIDEO_FOLDER, "summary.json")
CONF_THRESH = 0.50
IOU_THRESH = 0.45
PHONE_INSIDE_RATIO = 0.25
MOTION_BASE = 6.0
MOTION_SCALE = 1.0

def phone_overlap_ratio(phone_box: np.ndarray, person_box: np.ndarray) -> float:
    x1 = max(phone_box[0], person_box[0])
    y1 = max(phone_box[1], person_box[1])
    x2 = min(phone_box[2], person_box[2])
    y2 = min(phone_box[3], person_box[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    phone_area = (phone_box[2] - phone_box[0]) * (phone_box[3] - phone_box[1])
    if phone_area <= 0:
        return 0.0
    return inter_area / phone_area

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    inter = w * h
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / (union + 1e-9))

prev_gray = None
motion_buf = []

def crop_roi(img, box, width, height):
    x1, y1, x2, y2 = box.astype(int)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(width - 1, x2); y2 = min(height - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return img[0:1,0:1]
    return img[y1:y2, x1:x2]

def get_motion_magnitude(frame, box, width, height):
    global prev_gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mag = 0.0
    if prev_gray is not None:
        roi_now = crop_roi(gray, box, width, height)
        roi_prev = crop_roi(prev_gray, box, width, height)
        h = min(roi_now.shape[0], roi_prev.shape[0])
        w = min(roi_now.shape[1], roi_prev.shape[1])
        if h > 2 and w > 2:
            a = roi_now[:h,:w]
            b = roi_prev[:h,:w]
            diff = cv2.absdiff(a, b)
            diff = cv2.GaussianBlur(diff, (3,3), 0)
            mag = float(np.mean(diff))
    prev_gray = gray
    return mag

def update_motion_threshold(val):
    global motion_buf
    motion_buf.append(val)
    if len(motion_buf) > 200:
        motion_buf.pop(0)

def get_motion_threshold():
    if len(motion_buf) < 30:
        return MOTION_BASE
    arr = np.array(motion_buf, dtype=np.float32)
    p = float(np.percentile(arr, 60.0))
    return max(MOTION_BASE, p)

def run_inference(model, frame):
    results = model.predict(frame, conf=CONF_THRESH, iou=IOU_THRESH, verbose=False, classes=[0,67])[0]
    people = []
    phones = []
    if results.boxes is None:
        return people, phones
    names = model.model.names if hasattr(model.model, 'names') else results.names
    for b in results.boxes:
        xyxy = b.xyxy[0].cpu().numpy()
        score = float(b.conf[0].cpu().numpy())
        cls_idx = int(b.cls[0].cpu().numpy())
        cls_name = names[cls_idx] if isinstance(names, (list, dict)) else str(cls_idx)
        if cls_name.lower() == 'person':
            people.append((xyxy, score))
        elif cls_name.lower() in ('cell phone', 'cellphone', 'mobile phone', 'phone'):
            phones.append((xyxy, score))
    return people, phones

def draw_box(img: np.ndarray, box: np.ndarray, label: str, color=(0,0,255)):
    x1,y1,x2,y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
    cv2.putText(img, label, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open input video: {INPUT_VIDEO}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
if not out_writer.isOpened():
    raise RuntimeError(f"Cannot open output writer: {OUTPUT_VIDEO}")

model = YOLO(MODEL_PATH)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
used_frames = 0
frames_with_usage = 0
phone_usage_count = 0
timestamps = []

idx = 0
frame_idx = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    people, phones = run_inference(model, frame)
    
    frame_has_usage = False
    active_phone_indices = []

    for i, (phone_box, phone_conf) in enumerate(phones):
        best_person = None
        best_iou = 0.0
        for person_box, person_conf in people:
            iou = iou_xyxy(phone_box, person_box)
            if iou > best_iou:
                best_iou = iou
                best_person = person_box
        
        if best_person is None:
            continue

        inside_ratio = phone_overlap_ratio(phone_box, best_person)
        if inside_ratio < PHONE_INSIDE_RATIO:
            continue

        mag = get_motion_magnitude(frame, phone_box, width, height)
        update_motion_threshold(mag)
        thr = get_motion_threshold()
        is_active = mag >= (thr * MOTION_SCALE)

        if is_active:
            frame_has_usage = True
            active_phone_indices.append(i)
            x1, y1, x2, y2 = map(int, phone_box)
            cv2.putText(frame, "PHONE USAGE", (x1, y1 - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            phone_usage_count += 1
            timestamps.append(frame_idx / fps)

    for i, (phone_box, phone_conf) in enumerate(phones):
        x1, y1, x2, y2 = map(int, phone_box)
        if i in active_phone_indices:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Phone: {phone_conf*100:.0f}%", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Phone: {phone_conf*100:.0f}%", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame, f"Phone Activity Count: {phone_usage_count}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if frame_has_usage:
        frames_with_usage += 1

    out_writer.write(frame)
    cv2.imshow("Phone Usage Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
    idx += 1
    used_frames += 1

cap.release()
out_writer.release()
cv2.destroyAllWindows()

try:
    import subprocess
    
    check_audio = subprocess.run([
        'ffprobe', '-v', 'quiet', '-select_streams', 'a:0', 
        '-show_entries', 'stream=codec_name', '-of', 'csv=p=0', INPUT_VIDEO
    ], capture_output=True, text=True)
    
    if check_audio.stdout.strip():
        temp_output = OUTPUT_VIDEO.replace('.mp4', '_temp.mp4')
        subprocess.run([
            'ffmpeg', '-i', OUTPUT_VIDEO, '-i', INPUT_VIDEO, 
            '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
            '-shortest', '-y', temp_output
        ], check=True)
        
        os.replace(temp_output, OUTPUT_VIDEO)
        print("Audio added successfully (fast method)")
    else:
        print("No audio track found in input video")
        
except Exception as e:
    print(f"FFmpeg not available, skipping audio: {e}")
    print("Install FFmpeg for audio support: https://ffmpeg.org/download.html")

processing_time = time.time() - start_time
phone_usage_percent = (frames_with_usage / used_frames * 100) if used_frames > 0 else 0

summary = {
    "input": INPUT_VIDEO,
    "output": OUTPUT_VIDEO,
    "models": {
        "segmentation_model": MODEL_PATH
    },
    "fps": fps,
    "resolution": {
        "width": width,
        "height": height
    },
    "total_frames": used_frames,
    "frames_with_phone_usage": frames_with_usage,
    "phone_usage_frame_percent": round(phone_usage_percent, 2),
    "processing_seconds": processing_time,
    "phone_usage_timestamps": timestamps
}

with open(SUMMARY_JSON, 'w') as f:
    json.dump(summary, f, indent=2)

print("Summary saved:", SUMMARY_JSON)
print(json.dumps(summary, indent=2))