# Face Recognition Attendance System

A webcam app that recognises registered people, greets them with
**"Welcome, Mr. &lt;Lastname&gt;"** and marks their attendance in a daily CSV file.

Built with OpenCV's YuNet (detection) and SFace (recognition) models, so it
only needs `opencv-python`, `numpy` and `pillow`. No GPU or dlib needed.

## Setup

```
py -m pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt
```
(macOS/Linux: use `python3 -m pip ...`)

## Register a person

1. Make a folder named `Firstname Lastname`, e.g. `Vivaan Kumar`.
2. Put about 5 clear photos of the face in it (.jpg/.png).
3. Copy it into `known_faces/`, **or** run the app and click **Add Person Folder...**

The greeting uses the **second word** of the folder name: `Vivaan Kumar` → "Welcome, Mr. Kumar".

## Run

```
py attendance_system.py
```

Click **Start Camera**. Recognised faces get a green box and the welcome message;
unknown faces get a red "Unknown" box. Each person is marked present once per day in
`attendance/attendance_YYYY-MM-DD.csv`.

If it confuses people, move the **Match strictness** slider up; if it doesn't recognise
you, move it down a little or add more photos.

## Files

| File | Purpose |
|---|---|
| `attendance_system.py` | The attendance app (detection + recognition + CSV log) |
| `face_detection_gui.py` | Detection-only demo (finds faces, doesn't identify them) |
| `face_detection_yunet_2023mar.onnx` | YuNet face detector model (0.2 MB) |
| `face_recognition_sface_2021dec_int8.onnx` | SFace face recognition model (10 MB) |
| `requirements.txt` | Dependencies + install command |
| `known_faces/` | Your registered people (photos are not uploaded to GitHub) |
