# Student Attendance (YUNet + FaceNet)

Ye project aapki requirement ke hisab se banaya gaya hai:
- Face detection: **YUNet**
- Face recognition embeddings: **FaceNet (facenet-pytorch)**
- Multi-angle enrollment: front/left/right/up/down
- Live camera matching: known face -> `Present: Name | ID`, unknown -> `Unknown`

## 1) Setup

```bash
cd "student_attendance"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) YUNet model download

YUNet ONNX model download karo:
- `face_detection_yunet_2023mar.onnx`
- Source (OpenCV Zoo): [https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)

Model ko example ke liye yahan rakho:
`student_attendance/data/face_detection_yunet_2023mar.onnx`

## 3) Enrollment (different angles embeddings save)

```bash
python src/enroll.py \
  --yunet "data/face_detection_yunet_2023mar.onnx" \
  --student-id "S101" \
  --name "Rahul Kumar"
```

Flow:
- Har angle prompt aayega
- Camera me face align karo
- `c` dabao capture ke liye
- `q` se exit
- Embeddings SQLite DB me save hoti hain: `data/attendance.db`

## 4) Live recognition and attendance

```bash
python src/recognize.py --yunet "data/face_detection_yunet_2023mar.onnx"
```

Output:
- Registered face detect hua -> rectangle green + `Present: <Name> | ID: <ID>`
- Unknown/unregistered face -> rectangle red + `Unknown`
- Quit key: `q`
- Attendance CSV auto-log: `data/attendance_log.csv`
- Right-side professional dashboard panel with counters:
  - Total Faces
  - Known Faces
  - Unknown Faces
  - Present Marked

## 5) Threshold tuning

Default cosine threshold: `0.65`

Agar false positives aa rahe ho:
```bash
python src/recognize.py --yunet "data/face_detection_yunet_2023mar.onnx" --threshold 0.70
```

Agar registered face miss ho raha ho:
```bash
python src/recognize.py --yunet "data/face_detection_yunet_2023mar.onnx" --threshold 0.60
```

## 6) Duplicate present रोकने ka smart rule

System cooldown use karta hai, default `60s`.
Is duration me same student repeatedly detect ho to duplicate CSV entry nahi banti.

Custom cooldown example:
```bash
python src/recognize.py \
  --yunet "data/face_detection_yunet_2023mar.onnx" \
  --cooldown-seconds 120
```
