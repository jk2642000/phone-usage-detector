import cv2
import time
import json
import os
from ultralytics import YOLO

INPUT_VIDEO = r"C:\Users\yashc\Downloads\Test taks video-20250819T055608Z-1-001\Yash\yash.webm"

VIDEO_FOLDER = os.path.splitext(os.path.basename(INPUT_VIDEO))[0]
os.makedirs(VIDEO_FOLDER, exist_ok=True)

OUTPUT_VIDEO = os.path.join(VIDEO_FOLDER, f"{VIDEO_FOLDER}.mp4")
SEG_MODEL_PATH = "yolo11m-seg.pt"
SUMMARY_JSON = os.path.join(VIDEO_FOLDER, "summary.json")
CONF_THRESH = 0.5
IOU_THRESH = 0.45

seg_model = YOLO(SEG_MODEL_PATH)

cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

PERSON_ID = 0
PHONE_ID = 67

def masks_overlap(mask1, mask2):
    import numpy as np
    return np.any(mask1 & mask2)

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

    results = seg_model(frame, classes=[PERSON_ID, PHONE_ID],
                       conf=CONF_THRESH, iou=IOU_THRESH)
    
    person_masks = []
    phone_data = []
    active_phones = []
    phone_detected_in_frame = False
    
    for r in results:
        if r.masks is not None:
            for i, mask in enumerate(r.masks):
                cls = int(r.boxes[i].cls[0])
                conf = float(r.boxes[i].conf[0])
                x1, y1, x2, y2 = map(int, r.boxes[i].xyxy[0])
                mask_array = mask.data[0].cpu().numpy().astype(bool)
                
                if cls == PERSON_ID:
                    person_masks.append((mask_array, x1, y1, x2, y2, conf))
                elif cls == PHONE_ID:
                    phone_data.append((mask_array, x1, y1, x2, y2, conf))
    
    active_phone_indices = []
    for i, (phone_mask, px1, py1, px2, py2, pconf) in enumerate(phone_data):
        is_active = False
        for person_mask, x1, y1, x2, y2, conf in person_masks:
            if masks_overlap(person_mask, phone_mask):
                is_active = True
                active_phones.append((phone_mask, px1, py1, px2, py2, pconf))
                active_phone_indices.append(i)
                cv2.putText(frame, "PHONE USAGE", (px1, py1 - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                phone_detected_in_frame = True
                break

    for (mask, x1, y1, x2, y2, conf) in active_phones:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"Phone: {conf*100:.0f}%", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    for i, (mask, x1, y1, x2, y2, conf) in enumerate(phone_data):
        if i not in active_phone_indices:
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
    import os
    
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
        "segmentation_model": SEG_MODEL_PATH
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