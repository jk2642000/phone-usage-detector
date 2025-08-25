import cv2
import time
import json
import os
from ultralytics import YOLO

INPUT_VIDEO = r"C:\Users\yashc\Downloads\Test taks video-20250819T055608Z-1-001\Test taks video\20250718_150650_075a44fc.mp4"

VIDEO_FOLDER = os.path.splitext(os.path.basename(INPUT_VIDEO))[0]
os.makedirs(VIDEO_FOLDER, exist_ok=True)

OUTPUT_VIDEO = os.path.join(VIDEO_FOLDER, f"{VIDEO_FOLDER}.mp4")
PHONE_MODEL_PATH = "yolo11m.pt"
POSE_MODEL_PATH = "yolo11m-pose.pt"
SUMMARY_JSON = os.path.join(VIDEO_FOLDER, "summary.json")
CONF_THRESH = 0.5
IOU_THRESH = 0.45
PADDING = 0.5

phone_model = YOLO(PHONE_MODEL_PATH)
pose_model = YOLO(POSE_MODEL_PATH)
cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

PHONE_ID = 67
LEFT_WRIST_ID = 9
RIGHT_WRIST_ID = 10

def pad_box(x1, y1, x2, y2, pad=0.15):
    w, h = x2 - x1, y2 - y1
    dw, dh = w * pad, h * pad
    return int(x1 - dw), int(y1 - dh), int(x2 + dw), int(y2 + dh)

def point_in_box(px, py, box):
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2
phone_usage_count = 0
frames_with_phone_usage = 0
timestamps = []

start_time = time.time()
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    phone_results = phone_model(frame, classes=[PHONE_ID],
                                conf=CONF_THRESH, iou=IOU_THRESH, imgsz=800)
    phone_data = []
    for r in phone_results:
        for box in r.boxes:
            if int(box.cls[0]) == PHONE_ID:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = pad_box(x1, y1, x2, y2, PADDING)
                phone_data.append((x1, y1, x2, y2, conf))

    pose_results = pose_model(frame, conf=CONF_THRESH, iou=IOU_THRESH, imgsz=800)
    phone_detected_in_frame = False
    active_phones = []

    for r in pose_results:
        if r.keypoints is not None:
            for kp in r.keypoints:  # each person
                kps = kp.xy[0].cpu().numpy()  # (17,2) array
                if len(kps) > max(LEFT_WRIST_ID, RIGHT_WRIST_ID):
                    lx, ly = kps[LEFT_WRIST_ID]
                    rx, ry = kps[RIGHT_WRIST_ID]

                    using_phone = False
                    for (px1, py1, px2, py2, conf) in phone_data:
                        if point_in_box(lx, ly, (px1, py1, px2, py2)) or point_in_box(rx, ry, (px1, py1, px2, py2)):
                            using_phone = True
                            active_phones.append((px1, py1, px2, py2, conf))
                            break

                    if using_phone:
                        cv2.circle(frame, (int(lx), int(ly)), 6, (0, 0, 255), -1)
                        cv2.circle(frame, (int(rx), int(ry)), 6, (0, 0, 255), -1)
                        cv2.putText(frame, "Person USING PHONE", (int(lx), int(ly) - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        phone_detected_in_frame = True
                    else:
                        cv2.circle(frame, (int(lx), int(ly)), 6, (0, 255, 0), -1)
                        cv2.circle(frame, (int(rx), int(ry)), 6, (0, 255, 0), -1)

    for (x1, y1, x2, y2, conf) in active_phones:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"Phone: {conf*100:.0f}%", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    unused_phones = [phone for phone in phone_data if phone not in active_phones]
    for (x1, y1, x2, y2, conf) in unused_phones:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Phone: {conf*100:.0f}%", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if phone_detected_in_frame:
        frames_with_phone_usage += 1
        phone_usage_count += 1
        timestamps.append(frame_idx / fps)

    cv2.putText(frame, f"Phone Activity Count: {phone_usage_count}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    out.write(frame)
    cv2.imshow("Phone Usage Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
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
phone_usage_percent = (frames_with_phone_usage / total_frames * 100) if total_frames > 0 else 0
summary = {
    "input": INPUT_VIDEO,
    "output": OUTPUT_VIDEO,
    "models": {
        "detection_model": PHONE_MODEL_PATH,
        "pose_model": POSE_MODEL_PATH
    },
    "fps": fps,
    "resolution": {
        "width": width,
        "height": height
    },
    "total_frames": total_frames,
    "frames_with_phone_usage": frames_with_phone_usage,
    "phone_usage_frame_percent": round(phone_usage_percent, 2),
    "processing_seconds": processing_time,
    "phone_usage_timestamps": timestamps
}

with open(SUMMARY_JSON, "w") as f:
    json.dump(summary, f, indent=2)

print("Summary saved:", SUMMARY_JSON)
print(json.dumps(summary, indent=2))
