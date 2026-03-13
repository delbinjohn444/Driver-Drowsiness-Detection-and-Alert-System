"""
Drowsiness Detection Alert System — ESP32 via USB Serial
=========================================================
No WiFi needed. The ESP32 is connected directly via USB cable.
Python sends plain-text commands over the serial port:
  "ALERT\\n"  → ESP32 turns RED LED ON + buzzer beeps
  "CLEAR\\n"  → ESP32 turns LED & buzzer OFF
  "PING\\n"   → ESP32 replies "PONG" (connection check)
"""

import cv2
import numpy as np
import threading
import time
import os
import urllib.request
import serial               # pip install pyserial
import serial.tools.list_ports

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision.face_landmarker import Blendshapes
from mediapipe.tasks.python.vision.core import image as mp_image

# ── Configuration ─────────────────────────────────────────────────────────────

BLINK_THRESHOLD    = 0.5   # Blendshape score above this = eyes closed
CONSECUTIVE_FRAMES = 30    # ~2 seconds at 15 FPS
ALERT_COOLDOWN     = 3.0   # Seconds between repeated alerts
CAMERA_INDEX       = 0

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# ── Serial / ESP32 Settings ───────────────────────────────────────────────────
# Set SERIAL_PORT to None to auto-detect the ESP32, or specify it explicitly:
#   Windows  →  "COM3"   (check Device Manager)
#   macOS    →  "/dev/cu.usbserial-XXXX"  or  "/dev/cu.SLAB_USBtoUART"
#   Linux    →  "/dev/ttyUSB0"  or  "/dev/ttyACM0"
SERIAL_PORT    = None       # None = auto-detect
SERIAL_BAUD    = 9600       # must match ESP32 sketch
SERIAL_TIMEOUT = 2          # seconds for readline()
ESP32_ENABLED  = True       # set False to skip hardware entirely


# ── Serial helpers ────────────────────────────────────────────────────────────

def auto_detect_port():
    """Return the first USB-serial port that looks like an ESP32."""
    keywords = ["usbserial", "usbmodem", "slab_usbtouar", "ch340",
                "cp210", "ftdi", "ttyusb", "ttyacm", "wchusb"]
    for port in serial.tools.list_ports.comports():
        desc = (port.description + port.device).lower()
        if any(kw in desc for kw in keywords):
            print(f"[Serial] Auto-detected ESP32 on {port.device}")
            return port.device
    return None


def connect_serial():
    """Open and return a serial.Serial object, or None on failure."""
    if not ESP32_ENABLED:
        return None

    port = SERIAL_PORT or auto_detect_port()
    if port is None:
        print("[Serial] No ESP32 found. Check USB cable and driver.")
        print("[Serial]  → Available ports:")
        for p in serial.tools.list_ports.comports():
            print(f"           {p.device}  —  {p.description}")
        return None

    try:
        ser = serial.Serial(port, SERIAL_BAUD, timeout=SERIAL_TIMEOUT)
        time.sleep(2)           # wait for ESP32 to reset after DTR toggle
        ser.reset_input_buffer()

        # Wait for READY message from ESP32
        deadline = time.time() + 5
        while time.time() < deadline:
            line = ser.readline().decode(errors="ignore").strip()
            if line == "READY":
                print(f"[Serial] ESP32 connected on {port} — READY")
                return ser
            elif line:
                print(f"[Serial] ESP32 says: {line}")

        print("[Serial] Warning: ESP32 did not send READY within 5 s — continuing anyway.")
        return ser

    except serial.SerialException as exc:
        print(f"[Serial] Could not open {port}: {exc}")
        return None


class ESP32Serial:
    """Thread-safe wrapper around a serial.Serial connection."""

    def __init__(self):
        self._ser  = connect_serial()
        self._lock = threading.Lock()

    @property
    def connected(self):
        return self._ser is not None and self._ser.is_open

    def send(self, command: str):
        """Send a command string (newline appended). Non-blocking."""
        if not self.connected:
            return
        def _write():
            with self._lock:
                try:
                    self._ser.write((command + "\n").encode())
                    # Optionally read and log the response
                    response = self._ser.readline().decode(errors="ignore").strip()
                    if response:
                        print(f"[ESP32] {command} → {response}")
                except serial.SerialException as exc:
                    print(f"[Serial] Write error: {exc}")
        threading.Thread(target=_write, daemon=True).start()

    def ping(self):
        """Returns True if ESP32 responds to PING."""
        if not self.connected:
            return False
        with self._lock:
            try:
                self._ser.write(b"PING\n")
                resp = self._ser.readline().decode(errors="ignore").strip()
                return resp == "PONG"
            except serial.SerialException:
                return False

    def alert(self):
        self.send("ALERT")

    def clear(self):
        self.send("CLEAR")

    def close(self):
        if self.connected:
            try:
                self._ser.close()
            except Exception:
                pass


# ── Model helpers ─────────────────────────────────────────────────────────────

def get_model_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "face_landmarker.task")
    if not os.path.exists(model_path):
        print("Downloading face landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("Model downloaded.")
    return model_path


# ── PC audio alert ────────────────────────────────────────────────────────────

def play_alert_sound():
    def _play():
        try:
            if PYGAME_AVAILABLE:
                pygame.mixer.init()
                sr, dur, freq = 22050, 0.5, 440
                n = int(round(dur * sr))
                buf = np.sin(2 * np.pi * freq * np.linspace(0, dur, n))
                buf = (buf * 32767).astype(np.int16)
                sound = pygame.sndarray.make_sound(np.column_stack((buf, buf)))
                sound.play()
                time.sleep(1.0)
            else:
                print("\a", end="", flush=True)
        except Exception as e:
            print(f"Alert: WAKE UP! ({e})")
    threading.Thread(target=_play, daemon=True).start()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    model_path = get_model_path()

    # Connect to ESP32 over USB Serial
    esp32 = ESP32Serial()
    if esp32.connected:
        print(f"[ESP32] Ping test: {'OK' if esp32.ping() else 'No response (continuing anyway)'}")
    else:
        print("[ESP32] Running WITHOUT hardware alerts.")

    options = FaceLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=model_path),
        running_mode=running_mode.VisionTaskRunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    face_landmarker = FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        esp32.close()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    closed_eye_count   = 0
    last_alert_time    = 0
    status_text        = "Awake"
    status_color       = (0, 255, 0)
    frame_timestamp_ms = 0
    was_drowsy         = False

    print("Drowsiness Detection System Started — Press 'q' to quit")
    print("-" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame     = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _   = frame.shape

        mp_img = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=rgb_frame)
        frame_timestamp_ms += 33
        result = face_landmarker.detect_for_video(mp_img, frame_timestamp_ms)

        drowsy_now = False

        if result.face_landmarks and result.face_blendshapes:
            blendshapes = result.face_blendshapes[0]

            def get_score(idx):
                for cat in blendshapes:
                    if cat.index == idx:
                        return cat.score or 0.0
                return 0.0

            avg_blink = (get_score(Blendshapes.EYE_BLINK_LEFT) +
                         get_score(Blendshapes.EYE_BLINK_RIGHT)) / 2.0

            if avg_blink > BLINK_THRESHOLD:
                closed_eye_count += 1
                if closed_eye_count >= CONSECUTIVE_FRAMES:
                    drowsy_now = True
                    now = time.time()
                    if now - last_alert_time > ALERT_COOLDOWN:
                        esp32.alert()           # ← serial command to ESP32
                        play_alert_sound()
                        last_alert_time = now
                    status_text  = "DROWSY! ALERT!"
                    status_color = (0, 0, 255)
            else:
                closed_eye_count = 0
                status_text      = "Awake"
                status_color     = (0, 255, 0)

            cv2.putText(frame, f"Blink: {avg_blink:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Blink: {avg_blink:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        else:
            closed_eye_count = 0
            status_text      = "No face detected"
            status_color     = (128, 128, 128)

        # Send CLEAR when drowsiness ends
        if was_drowsy and not drowsy_now:
            esp32.clear()
        was_drowsy = drowsy_now

        # ── HUD ──────────────────────────────────────────────────────────────
        cv2.rectangle(frame, (0, h - 50), (w, h), (40, 40, 40), -1)
        cv2.putText(frame, status_text, (20, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

        # Serial connection badge (top-right)
        badge_text  = "ESP32: USB" if esp32.connected else "ESP32: --"
        badge_color = (0, 220, 0)  if esp32.connected else (60, 60, 60)
        cv2.putText(frame, badge_text, (w - 170, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, badge_color, 2)

        if "DROWSY" in status_text:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
            cv2.putText(frame, "WAKE UP!", (w // 2 - 120, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)

        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    esp32.clear()
    esp32.close()
    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()
    print("System stopped.")


if __name__ == "__main__":
    main()
