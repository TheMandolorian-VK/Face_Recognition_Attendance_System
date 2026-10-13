import csv
import os
import shutil
import ssl
import sys
import time
import tkinter as tk
import urllib.request
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(SCRIPT_DIR, "known_faces")
ATTENDANCE_DIR = os.path.join(SCRIPT_DIR, "attendance")

MODEL_BASE_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
DETECTOR_FILE = "face_detection_yunet_2023mar.onnx"
RECOGNIZER_FILE = "face_recognition_sface_2021dec_int8.onnx"
MODELS = {
    DETECTOR_FILE: MODEL_BASE_URL + "face_detection_yunet/" + DETECTOR_FILE,
    RECOGNIZER_FILE: MODEL_BASE_URL + "face_recognition_sface/" + RECOGNIZER_FILE,
}

CAMERA_W, CAMERA_H = 640, 480
DETECT_SCALE = 0.5              # detect on a half-size frame (much faster)
DETECT_CONFIDENCE = 0.75        # YuNet score needed to count as a face
DEFAULT_MATCH_THRESHOLD = 0.40  # SFace cosine similarity (OpenCV suggests 0.363)
RECHECK_KNOWN_EVERY = 30        # frames between re-recognising a known face
RECHECK_UNKNOWN_EVERY = 5       # frames between re-trying an unknown face
WELCOME_SECONDS = 4             # how long the welcome message stays up
PHOTO_MAX_SIDE = 1024           # registration photos are shrunk to this first
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

GREEN, RED, WHITE, DARK = (0, 200, 0), (0, 0, 230), (255, 255, 255), (30, 30, 30)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def last_name_from_folder(folder_name):
    """'Vivaan Kumar' -> 'Kumar'. Uses the 2nd word (falls back to the 1st)."""
    words = folder_name.replace("_", " ").split()
    if not words:
        return folder_name
    word = words[1] if len(words) >= 2 else words[0]
    return word if any(c.isupper() for c in word) else word.capitalize()


def read_image(path):
    """cv2.imread that also works with non-English characters in the path."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def ensure_model(filename):
    """Return the path of a model file, downloading it if it is missing."""
    path = os.path.join(SCRIPT_DIR, filename)
    if os.path.exists(path) and os.path.getsize(path) > 100_000:
        return path
    url = MODELS[filename]
    print(f"[INFO] Downloading {filename} ...")
    last_error = None
    # Normal HTTPS first; school networks that need pip --trusted-host often
    # intercept SSL, so retry once without certificate verification.
    for context in (None, ssl._create_unverified_context()):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60, context=context) as resp:
                data = resp.read()
            if len(data) < 100_000:
                raise ValueError(f"file too small ({len(data)} bytes)")
            with open(path + ".part", "wb") as fh:
                fh.write(data)
            os.replace(path + ".part", path)
            return path
        except Exception as exc:  # noqa: BLE001 - try again / report below
            last_error = exc
    raise RuntimeError(
        f"Could not download {filename}: {last_error}\n\n"
        f"Download it manually from:\n{url}\nand put it next to attendance_system.py"
    )


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------- #
# Face engine: detection + recognition
# --------------------------------------------------------------------------- #
class FaceEngine:
    def __init__(self):
        create_det = getattr(cv2, "FaceDetectorYN", None)
        create_det = create_det.create if create_det else cv2.FaceDetectorYN_create
        create_rec = getattr(cv2, "FaceRecognizerSF", None)
        create_rec = create_rec.create if create_rec else cv2.FaceRecognizerSF_create

        self.detector = create_det(ensure_model(DETECTOR_FILE), "", (320, 320),
                                   DETECT_CONFIDENCE, 0.3, 50)
        self.recognizer = create_rec(ensure_model(RECOGNIZER_FILE), "")
        self._input_size = (320, 320)
        self.people = {}  # folder name -> array of face embeddings (N x 128)

    def detect(self, image, confidence=DETECT_CONFIDENCE):
        """Return YuNet face rows (N x 15) for the image, or an empty array."""
        h, w = image.shape[:2]
        if (w, h) != self._input_size:
            self.detector.setInputSize((w, h))
            self._input_size = (w, h)
        self.detector.setScoreThreshold(confidence)
        _, faces = self.detector.detect(image)
        return faces if faces is not None else np.empty((0, 15), np.float32)

    def embed(self, image, face_row):
        aligned = self.recognizer.alignCrop(image, face_row)
        feat = self.recognizer.feature(aligned).flatten()
        return feat / (np.linalg.norm(feat) + 1e-9)

    def identify(self, image, face_row, threshold):
        """Return (folder_name or None, best similarity)."""
        if not self.people:
            return None, 0.0
        feat = self.embed(image, face_row)
        best_name, best_score = None, -1.0
        for name, feats in self.people.items():
            score = float(np.max(feats @ feat))  # cosine similarity
            if score > best_score:
                best_name, best_score = name, score
        return (best_name if best_score >= threshold else None), best_score

    def load_known_faces(self):
        """Read known_faces/<person>/*.jpg. Returns a list of report lines."""
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        self.people = {}
        report = []
        for person in sorted(os.listdir(KNOWN_FACES_DIR)):
            folder = os.path.join(KNOWN_FACES_DIR, person)
            if not os.path.isdir(folder):
                continue
            feats, skipped = [], []
            for fname in sorted(os.listdir(folder)):
                if not fname.lower().endswith(IMAGE_EXTS):
                    continue
                img = read_image(os.path.join(folder, fname))
                if img is None:
                    skipped.append(fname)
                    continue
                scale = PHOTO_MAX_SIDE / max(img.shape[:2])
                if scale < 1:
                    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                faces = self.detect(img, confidence=0.6)
                if len(faces) == 0:
                    skipped.append(fname)
                    continue
                biggest = faces[np.argmax(faces[:, 2] * faces[:, 3])]
                feats.append(self.embed(img, biggest))
            if feats:
                self.people[person] = np.array(feats)
            line = f"{person}: {len(feats)} photo(s) used"
            if skipped:
                line += f", no face found in: {', '.join(skipped)}"
            report.append(line)
        return report


# --------------------------------------------------------------------------- #
# Attendance log
# --------------------------------------------------------------------------- #
class AttendanceLog:
    def __init__(self):
        os.makedirs(ATTENDANCE_DIR, exist_ok=True)
        self.date = datetime.now().strftime("%Y-%m-%d")
        self.path = os.path.join(ATTENDANCE_DIR, f"attendance_{self.date}.csv")
        self.marked = {}  # name -> time string
        if os.path.exists(self.path):
            with open(self.path, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    self.marked[row["Name"]] = row["Time"]

    def mark(self, name):
        """Mark a person present. Returns True if this is a new entry today."""
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self.date:  # program left running past midnight
            self.__init__()
        if name in self.marked:
            return False
        now = datetime.now().strftime("%H:%M:%S")
        new_file = not os.path.exists(self.path)
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if new_file:
                writer.writerow(["Name", "Last Name", "Date", "Time"])
            writer.writerow([name, last_name_from_folder(name), self.date, now])
        self.marked[name] = now
        return True


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
class AttendanceApp:
    def __init__(self, root, engine):
        self.root = root
        self.engine = engine
        self.log = AttendanceLog()
        self.cap = None
        self.running = False
        self.frame_no = 0
        self.tracks = []          # [{"box", "name", "score", "checked"}]
        self.welcome = {}         # name -> time it was last seen
        self.photo = None
        self.fps = 0.0
        self._last_time = time.time()

        self.threshold = tk.DoubleVar(value=DEFAULT_MATCH_THRESHOLD)
        self.mirror = tk.BooleanVar(value=True)
        self.camera_index = tk.IntVar(value=0)

        root.title("Face Recognition Attendance")
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._build_ui()
        self._show_message("Click 'Start Camera'")
        self.reload_faces(show_popup=False)
        self._refresh_attendance_list()

    # ---------------- layout ---------------- #
    def _build_ui(self):
        main = ttk.Frame(self.root, padding=8)
        main.grid(row=0, column=0, sticky="nsew")

        self.video = tk.Label(main, bg="black")
        self.video.grid(row=0, column=0, rowspan=2, sticky="nsew")

        self.welcome_label = tk.Label(main, text="", font=("Segoe UI", 20, "bold"),
                                      fg="#0a8a0a", anchor="w", justify="left")
        self.welcome_label.grid(row=2, column=0, sticky="ew", pady=(6, 0))

        panel = ttk.Frame(main, padding=(12, 0, 0, 0))
        panel.grid(row=0, column=1, sticky="n")

        ttk.Label(panel, text="Attendance System", font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 8))

        row = ttk.Frame(panel)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Camera index:").pack(side="left")
        ttk.Spinbox(row, from_=0, to=9, width=4, textvariable=self.camera_index).pack(side="left", padx=5)

        self.start_btn = ttk.Button(panel, text="Start Camera", command=self.start_camera)
        self.start_btn.pack(fill="x", pady=2)
        self.stop_btn = ttk.Button(panel, text="Stop Camera", command=self.stop_camera, state="disabled")
        self.stop_btn.pack(fill="x", pady=2)
        ttk.Button(panel, text="Add Person Folder...", command=self.add_person).pack(fill="x", pady=2)
        ttk.Button(panel, text="Reload Faces", command=self.reload_faces).pack(fill="x", pady=2)

        ttk.Separator(panel).pack(fill="x", pady=8)
        ttk.Label(panel, text="Match strictness (higher = stricter):").pack(anchor="w")
        self.thr_label = ttk.Label(panel, text=f"{DEFAULT_MATCH_THRESHOLD:.2f}")
        self.thr_label.pack(anchor="e")
        ttk.Scale(panel, from_=0.30, to=0.70, variable=self.threshold,
                  command=lambda _v: self.thr_label.config(text=f"{self.threshold.get():.2f}")).pack(fill="x")
        ttk.Checkbutton(panel, text="Mirror camera", variable=self.mirror).pack(anchor="w", pady=(6, 0))

        ttk.Separator(panel).pack(fill="x", pady=8)
        self.people_label = ttk.Label(panel, text="", wraplength=240, justify="left")
        self.people_label.pack(anchor="w")
        self.fps_label = ttk.Label(panel, text="FPS: -")
        self.fps_label.pack(anchor="w", pady=(4, 0))

        ttk.Label(panel, text="Present today:", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(8, 2))
        self.present_list = tk.Listbox(panel, height=8, width=34)
        self.present_list.pack(fill="x")
        self.total_label = ttk.Label(panel, text="Total present: 0", font=("Segoe UI", 11, "bold"))
        self.total_label.pack(anchor="w", pady=(4, 0))

        ttk.Button(panel, text="Quit", command=self.on_close).pack(fill="x", pady=(10, 0))

    # ---------------- people ---------------- #
    def reload_faces(self, show_popup=True):
        self.people_label.config(text="Loading faces...")
        self.root.update_idletasks()
        report = self.engine.load_known_faces()
        self.tracks = []
        if self.engine.people:
            names = ", ".join(self.engine.people)
            self.people_label.config(text=f"Registered ({len(self.engine.people)}): {names}")
        else:
            self.people_label.config(text="No one registered yet.\nClick 'Add Person Folder...'")
        print("[INFO] " + ("\n[INFO] ".join(report) if report else "known_faces/ is empty"))
        if show_popup:
            messagebox.showinfo("Faces loaded", "\n".join(report) or
                                "No person folders found in known_faces/.")

    def add_person(self):
        src = filedialog.askdirectory(title="Choose the folder with the person's photos "
                                            "(named 'Firstname Lastname')")
        if not src:
            return
        name = os.path.basename(os.path.normpath(src))
        photos = [f for f in os.listdir(src) if f.lower().endswith(IMAGE_EXTS)]
        if not photos:
            messagebox.showerror("No photos", f"No .jpg/.png photos found in:\n{src}")
            return
        dest = os.path.join(KNOWN_FACES_DIR, name)
        if os.path.normpath(src) != os.path.normpath(dest):
            os.makedirs(dest, exist_ok=True)
            for f in photos:
                shutil.copy2(os.path.join(src, f), os.path.join(dest, f))
        self.reload_faces(show_popup=True)

    # ---------------- camera ---------------- #
    def _open_camera(self, index):
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]
        for backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always show the newest frame
                ok, _ = cap.read()
                if ok:
                    return cap
            cap.release()
        return None

    def start_camera(self):
        if self.running:
            return
        if not self.engine.people:
            if not messagebox.askyesno("No one registered",
                                       "No faces are registered yet, so everyone will show as Unknown.\n"
                                       "Start the camera anyway?"):
                return
        try:
            index = int(self.camera_index.get())
        except (tk.TclError, ValueError):
            index = 0
        self.cap = self._open_camera(index)
        if self.cap is None:
            messagebox.showerror("Camera error",
                                 f"Could not open camera {index}.\n\n"
                                 "- Check the webcam is connected and not used by another app (Zoom, Teams...).\n"
                                 "- Try camera index 1 or 2.\n"
                                 "- On macOS, allow camera access for Terminal/your IDE.")
            return
        self.running = True
        self.tracks = []
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self._last_time = time.time()
        self._loop()

    def stop_camera(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.fps_label.config(text="FPS: -")

    def _loop(self):
        if not self.running:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.stop_camera()
            messagebox.showerror("Camera error", "Lost connection to the camera.")
            return
        if frame.shape[1] > CAMERA_W:  # some cameras ignore the size request
            frame = cv2.resize(frame, (CAMERA_W, int(frame.shape[0] * CAMERA_W / frame.shape[1])))
        if self.mirror.get():
            frame = cv2.flip(frame, 1)

        self.frame_no += 1
        self._process(frame)
        self._draw(frame)
        self._show(frame)
        self._update_welcome()

        now = time.time()
        dt, self._last_time = now - self._last_time, now
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 / dt if self.fps else 1 / dt
            self.fps_label.config(text=f"FPS: {self.fps:.1f}")
        self.root.after(5, self._loop)

    # ---------------- detection + recognition ---------------- #
    def _process(self, frame):
        small = cv2.resize(frame, None, fx=DETECT_SCALE, fy=DETECT_SCALE, interpolation=cv2.INTER_LINEAR)
        faces = self.engine.detect(small)
        new_tracks = []
        for row in faces:
            full_row = row.copy()
            full_row[:14] /= DETECT_SCALE  # box + landmarks back to full-size coordinates
            box = tuple(int(v) for v in full_row[:4])

            # Re-use the identity of the matching face from the previous frame
            prev = max(self.tracks, key=lambda t: iou(t["box"], box), default=None)
            if prev is not None and iou(prev["box"], box) > 0.3:
                track = dict(prev, box=box)
            else:
                track = {"box": box, "name": None, "score": 0.0, "checked": -10_000}

            wait = RECHECK_KNOWN_EVERY if track["name"] else RECHECK_UNKNOWN_EVERY
            if self.frame_no - track["checked"] >= wait:
                track["name"], track["score"] = self.engine.identify(frame, full_row, self.threshold.get())
                track["checked"] = self.frame_no

            if track["name"]:
                self.welcome[track["name"]] = time.time()
                if self.log.mark(track["name"]):
                    self._refresh_attendance_list()
            new_tracks.append(track)
        self.tracks = new_tracks

    def _draw(self, frame):
        for t in self.tracks:
            x, y, w, h = t["box"]
            color = GREEN if t["name"] else RED
            label = f"Mr. {last_name_from_folder(t['name'])}" if t["name"] else "Unknown"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty = y if y - th - 10 >= 0 else y + h + th + 10  # tag below the box if no room above
            cv2.rectangle(frame, (x, ty - th - 10), (x + tw + 8, ty), color, -1)
            cv2.putText(frame, label, (x + 4, ty - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

        names = self._current_welcome_names()
        if names:
            text = "Welcome, " + " & ".join(f"Mr. {last_name_from_folder(n)}" for n in names)
            scale = 0.9 if len(text) < 30 else 0.7
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
            fh, fw = frame.shape[:2]
            cv2.rectangle(frame, (0, fh - th - 24), (fw, fh), DARK, -1)
            cv2.putText(frame, text, ((fw - tw) // 2, fh - 12), cv2.FONT_HERSHEY_SIMPLEX, scale, GREEN, 2)

    def _current_welcome_names(self):
        now = time.time()
        self.welcome = {n: t for n, t in self.welcome.items() if now - t < WELCOME_SECONDS}
        return sorted(self.welcome)

    def _update_welcome(self):
        names = self._current_welcome_names()
        text = "\n".join(f"Welcome, Mr. {last_name_from_folder(n)}" for n in names)
        if self.welcome_label.cget("text") != text:
            self.welcome_label.config(text=text)

    # ---------------- display ---------------- #
    def _show(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video.config(image=self.photo)

    def _show_message(self, text):
        canvas = np.zeros((CAMERA_H, CAMERA_W, 3), np.uint8)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(canvas, text, ((CAMERA_W - tw) // 2, (CAMERA_H + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        self._show(canvas)

    def _refresh_attendance_list(self):
        self.present_list.delete(0, "end")
        for name, t in sorted(self.log.marked.items(), key=lambda kv: kv[1]):
            self.present_list.insert("end", f"{t}  {name}")
        self.total_label.config(text = "%s" % len(self.log.marked))

    def open_attendance_folder(self):
        os.makedirs(ATTENDANCE_DIR, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(ATTENDANCE_DIR)  # noqa: S606 - opens Explorer
            else:
                import subprocess
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", ATTENDANCE_DIR])
        except Exception:  # noqa: BLE001
            messagebox.showinfo("Attendance folder", ATTENDANCE_DIR)

    def on_close(self):
        self.stop_camera()
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        engine = FaceEngine()
    except Exception as exc:  # noqa: BLE001 - show the problem instead of crashing silently
        root.withdraw()
        messagebox.showerror("Could not load face models", str(exc))
        root.destroy()
        return
    AttendanceApp(root, engine)
    root.mainloop()


if __name__ == "__main__":
    main()
