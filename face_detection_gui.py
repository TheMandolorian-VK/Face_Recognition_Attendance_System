"""
Face Detection GUI
==================

Real-time face detection with a Tkinter GUI, using OpenCV.

Detectors:
  * YuNet (deep-learning, accurate, fast) - the model file (~230 KB) is
    downloaded automatically on first run and saved next to this script.
  * Haar cascade (classic, bundled with OpenCV) - used automatically while
    YuNet is loading, or if the YuNet model cannot be downloaded
    (e.g. no internet), so the program always works.

Features:
  * Live webcam detection (choose camera index)
  * Open an image file and detect faces in it
  * Confidence slider, landmark toggle, mirror toggle
  * Save a snapshot of the current (annotated) frame
  * Face count + FPS display

Install dependencies (see requirements.txt), then run:
    python face_detection_gui.py
"""

import os
import ssl
import sys
import threading
import time
import tkinter as tk
import urllib.request
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_PATH = os.path.join(SCRIPT_DIR, YUNET_FILENAME)
YUNET_URLS = [
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://huggingface.co/opencv/face_detection_yunet/resolve/main/"
    "face_detection_yunet_2023mar.onnx",
]
MIN_MODEL_BYTES = 100_000  # the real model is ~230 KB; anything smaller is an error page

DISPLAY_W, DISPLAY_H = 800, 600
MAX_IMAGE_SIDE = 1280  # large photos are scaled down to this before detection

BOX_COLOR = (0, 220, 0)        # BGR green
TEXT_COLOR = (0, 255, 255)     # BGR yellow
LANDMARK_COLORS = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255), (0, 255, 255)]


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #
class HaarDetector:
    """Classic OpenCV Haar cascade (ships with opencv-python 4.x)."""

    name = "Haar cascade (basic)"

    def __init__(self):
        path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.cascade = cv2.CascadeClassifier(path)
        if self.cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade from {path}")

    def detect(self, frame, confidence):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        # Map the 0.30-0.95 confidence slider onto minNeighbors 3-12
        min_neighbors = int(round(3 + (confidence - 0.3) / 0.65 * 9))
        boxes = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=max(3, min_neighbors), minSize=(40, 40)
        )
        return [(int(x), int(y), int(w), int(h), None, None) for (x, y, w, h) in boxes]


class YuNetDetector:
    """OpenCV YuNet deep-learning face detector (accurate, with 5 landmarks)."""

    name = "YuNet (deep learning)"

    def __init__(self, model_path):
        create = getattr(cv2, "FaceDetectorYN", None)
        create = create.create if create is not None else cv2.FaceDetectorYN_create
        self.net = create(model_path, "", (320, 320), 0.6, 0.3, 5000)
        self.input_size = (320, 320)

    def detect(self, frame, confidence):
        h, w = frame.shape[:2]
        if (w, h) != self.input_size:
            self.net.setInputSize((w, h))
            self.input_size = (w, h)
        self.net.setScoreThreshold(float(confidence))
        _, faces = self.net.detect(frame)
        results = []
        if faces is not None:
            for f in faces:
                x, y, bw, bh = (int(v) for v in f[:4])
                landmarks = [(int(f[4 + 2 * i]), int(f[5 + 2 * i])) for i in range(5)]
                results.append((x, y, bw, bh, float(f[14]), landmarks))
        return results


def _download(url, dest, verify_ssl=True):
    context = None if verify_ssl else ssl._create_unverified_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20, context=context) as resp:
        data = resp.read()
    if len(data) < MIN_MODEL_BYTES:
        raise ValueError(f"downloaded file too small ({len(data)} bytes)")
    tmp = dest + ".part"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, dest)


def load_yunet():
    """Return a YuNetDetector, downloading the model if needed. Raises on failure."""
    if not hasattr(cv2, "FaceDetectorYN") and not hasattr(cv2, "FaceDetectorYN_create"):
        raise RuntimeError("this OpenCV version has no FaceDetectorYN (need 4.8+)")

    if os.path.exists(YUNET_PATH) and os.path.getsize(YUNET_PATH) >= MIN_MODEL_BYTES:
        return YuNetDetector(YUNET_PATH)

    errors = []
    # Try normal (verified) HTTPS first; school/office networks that need
    # pip --trusted-host often intercept SSL, so retry without verification.
    for verify in (True, False):
        for url in YUNET_URLS:
            try:
                print(f"[INFO] Downloading YuNet model from {url} ...")
                _download(url, YUNET_PATH, verify_ssl=verify)
                detector = YuNetDetector(YUNET_PATH)
                print("[INFO] YuNet model ready.")
                return detector
            except Exception as exc:  # noqa: BLE001 - try the next source
                errors.append(f"{url}: {exc}")
                if os.path.exists(YUNET_PATH):
                    try:
                        os.remove(YUNET_PATH)
                    except OSError:
                        pass
    raise RuntimeError("could not download YuNet model:\n  " + "\n  ".join(errors))


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.detector = HaarDetector()  # always available instantly
        self.yunet_status = "loading..."
        self.cap = None
        self.running = False
        self.static_image = None   # BGR image loaded from file
        self.last_annotated = None  # last frame drawn on screen (BGR)
        self.photo = None           # keep a reference so Tk doesn't garbage-collect it
        self.fps = 0.0
        self._last_time = time.time()

        self.confidence = tk.DoubleVar(value=0.70)
        self.show_landmarks = tk.BooleanVar(value=True)
        self.show_scores = tk.BooleanVar(value=True)
        self.mirror = tk.BooleanVar(value=True)
        self.camera_index = tk.IntVar(value=0)

        self._build_ui()
        self._show_placeholder("Click 'Start Camera' or 'Open Image'")

        threading.Thread(target=self._load_yunet_background, daemon=True).start()
        self._refresh_status()

    # ---------------- UI layout ---------------- #
    def _build_ui(self):
        main = ttk.Frame(self.root, padding=8)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.video_label = tk.Label(main, bg="black")
        self.video_label.grid(row=0, column=0, sticky="nsew")

        panel = ttk.Frame(main, padding=(12, 0, 0, 0))
        panel.grid(row=0, column=1, sticky="ns")

        ttk.Label(panel, text="Face Detection", font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 10))

        cam_row = ttk.Frame(panel)
        cam_row.pack(fill="x", pady=2)
        ttk.Label(cam_row, text="Camera index:").pack(side="left")
        ttk.Spinbox(cam_row, from_=0, to=9, width=4, textvariable=self.camera_index).pack(side="left", padx=5)

        self.start_btn = ttk.Button(panel, text="Start Camera", command=self.start_camera)
        self.start_btn.pack(fill="x", pady=2)
        self.stop_btn = ttk.Button(panel, text="Stop Camera", command=self.stop_camera, state="disabled")
        self.stop_btn.pack(fill="x", pady=2)
        ttk.Button(panel, text="Open Image...", command=self.open_image).pack(fill="x", pady=2)
        ttk.Button(panel, text="Save Snapshot...", command=self.save_snapshot).pack(fill="x", pady=2)

        ttk.Separator(panel).pack(fill="x", pady=10)

        ttk.Label(panel, text="Confidence threshold:").pack(anchor="w")
        self.conf_value_label = ttk.Label(panel, text="0.70")
        self.conf_value_label.pack(anchor="e")
        ttk.Scale(panel, from_=0.30, to=0.95, variable=self.confidence,
                  command=self._on_conf_change).pack(fill="x")

        ttk.Checkbutton(panel, text="Show landmarks (eyes/nose/mouth)", variable=self.show_landmarks,
                        command=self._redraw_static).pack(anchor="w", pady=(8, 0))
        ttk.Checkbutton(panel, text="Show confidence scores", variable=self.show_scores,
                        command=self._redraw_static).pack(anchor="w")
        ttk.Checkbutton(panel, text="Mirror camera", variable=self.mirror).pack(anchor="w")

        ttk.Separator(panel).pack(fill="x", pady=10)

        self.detector_label = ttk.Label(panel, text="", wraplength=220, justify="left")
        self.detector_label.pack(anchor="w")
        self.faces_label = ttk.Label(panel, text="Faces: 0", font=("Segoe UI", 12, "bold"))
        self.faces_label.pack(anchor="w", pady=(6, 0))
        self.fps_label = ttk.Label(panel, text="FPS: -")
        self.fps_label.pack(anchor="w")

        ttk.Button(panel, text="Quit", command=self.on_close).pack(side="bottom", fill="x", pady=(10, 0))

    # ---------------- detector loading ---------------- #
    def _load_yunet_background(self):
        try:
            self.detector = load_yunet()
            self.yunet_status = "ready"
        except Exception as exc:  # noqa: BLE001 - fall back to Haar
            print(f"[WARN] {exc}\n[WARN] Using Haar cascade instead.")
            self.yunet_status = "unavailable (using Haar)"

    def _refresh_status(self):
        text = f"Detector: {self.detector.name}"
        if self.yunet_status == "loading...":
            text += "\n(YuNet model loading...)"
        elif self.yunet_status != "ready":
            text += f"\nYuNet {self.yunet_status}"
        self.detector_label.config(text=text)
        self.root.after(500, self._refresh_status)

    # ---------------- camera ---------------- #
    def _open_camera(self, index):
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]
        for backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    return cap
            cap.release()
        return None

    def start_camera(self):
        if self.running:
            return
        try:
            index = int(self.camera_index.get())
        except (tk.TclError, ValueError):
            index = 0
        self.cap = self._open_camera(index)
        if self.cap is None:
            messagebox.showerror(
                "Camera error",
                f"Could not open camera {index}.\n\n"
                "- Make sure a webcam is connected and not used by another app.\n"
                "- Try a different camera index (0, 1, 2...).\n"
                "- On macOS, allow camera access for Terminal/your IDE in System Settings.",
            )
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.static_image = None
        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self._last_time = time.time()
        self._update_frame()

    def stop_camera(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.fps_label.config(text="FPS: -")

    def _update_frame(self):
        if not self.running or self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.stop_camera()
            messagebox.showerror("Camera error", "Lost connection to the camera.")
            return
        if self.mirror.get():
            frame = cv2.flip(frame, 1)

        annotated, count = self._detect_and_draw(frame)
        self._display(annotated)
        self.faces_label.config(text=f"Faces: {count}")

        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps else 1.0 / dt
            self.fps_label.config(text=f"FPS: {self.fps:.1f}")

        self.root.after(10, self._update_frame)

    # ---------------- images ---------------- #
    def open_image(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"), ("All files", "*.*")],
        )
        if not path:
            return
        # np.fromfile + imdecode works with non-ASCII paths on Windows (cv2.imread does not)
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        if image is None:
            messagebox.showerror("Error", f"Could not read image:\n{path}")
            return
        h, w = image.shape[:2]
        scale = MAX_IMAGE_SIDE / max(h, w)
        if scale < 1:
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        self.stop_camera()
        self.static_image = image
        self._redraw_static()

    def _redraw_static(self):
        if self.static_image is None or self.running:
            return
        annotated, count = self._detect_and_draw(self.static_image)
        self._display(annotated)
        self.faces_label.config(text=f"Faces: {count}")

    def save_snapshot(self):
        if self.last_annotated is None:
            messagebox.showinfo("Nothing to save", "Start the camera or open an image first.")
            return
        frame = self.last_annotated.copy()
        path = filedialog.asksaveasfilename(
            title="Save snapshot",
            defaultextension=".png",
            initialfile=time.strftime("faces_%Y%m%d_%H%M%S.png"),
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
        )
        if not path:
            return
        ext = os.path.splitext(path)[1] or ".png"
        ok, buf = cv2.imencode(ext, frame)
        if ok:
            buf.tofile(path)
            messagebox.showinfo("Saved", f"Snapshot saved to:\n{path}")
        else:
            messagebox.showerror("Error", "Could not save the snapshot.")

    # ---------------- drawing ---------------- #
    def _on_conf_change(self, _value=None):
        self.conf_value_label.config(text=f"{self.confidence.get():.2f}")
        self._redraw_static()

    def _detect_and_draw(self, frame):
        detector = self.detector  # read once; the background thread may swap it
        faces = detector.detect(frame, self.confidence.get())
        out = frame.copy()
        thickness = max(2, int(round(max(out.shape[:2]) / 400)))
        font_scale = max(0.5, max(out.shape[:2]) / 1400)
        for (x, y, w, h, score, landmarks) in faces:
            cv2.rectangle(out, (x, y), (x + w, y + h), BOX_COLOR, thickness)
            if self.show_scores.get() and score is not None:
                cv2.putText(out, f"{score * 100:.0f}%", (x, max(15, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, TEXT_COLOR, thickness)
            if self.show_landmarks.get() and landmarks:
                for (lx, ly), color in zip(landmarks, LANDMARK_COLORS):
                    cv2.circle(out, (lx, ly), thickness + 1, color, -1)
        cv2.putText(out, f"Faces: {len(faces)}", (10, int(30 * font_scale / 0.5)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, BOX_COLOR, thickness)
        return out, len(faces)

    def _display(self, frame_bgr):
        self.last_annotated = frame_bgr
        h, w = frame_bgr.shape[:2]
        scale = min(DISPLAY_W / w, DISPLAY_H / h)
        resized = cv2.resize(frame_bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                             interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.config(image=self.photo, width=DISPLAY_W, height=DISPLAY_H)

    def _show_placeholder(self, message):
        canvas = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
        (tw, th), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(canvas, message, ((DISPLAY_W - tw) // 2, (DISPLAY_H + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.config(image=self.photo, width=DISPLAY_W, height=DISPLAY_H)

    def on_close(self):
        self.stop_camera()
        self.root.destroy()


def main():
    root = tk.Tk()
    FaceDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
