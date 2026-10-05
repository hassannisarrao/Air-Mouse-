import sys
import cv2
import mediapipe as mp
import pyautogui
import pyttsx3
import threading
import time
import numpy as np
import speech_recognition as sr
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QSlider, QTextEdit, 
                             QFrame, QGraphicsDropShadowEffect, QCheckBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QColor, QPainter, QPainterPath

pyautogui.FAILSAFE = False

# --- BACKGROUND VOICE ENGINE (Speaking) ---
class VoiceEngine:
    @staticmethod
    def speak(text):
        def run():
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 160)
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"Voice Error: {e}")
        threading.Thread(target=run, daemon=True).start()

# --- VOICE COMMAND LISTENER (Hearing) ---
class VoiceListenerThread(QThread):
    command_signal = pyqtSignal(str)
    
    def run(self):
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
        
        while True:
            try:
                with mic as source:
                    audio = recognizer.listen(source, timeout=1, phrase_time_limit=3)
                text = recognizer.recognize_google(audio).lower()
                
                if "start engine" in text or "start tracking" in text:
                    self.command_signal.emit("START")
                elif "stop engine" in text or "stop tracking" in text or "pause" in text:
                    self.command_signal.emit("STOP")
                elif "hand mode" in text:
                    self.command_signal.emit("HAND")
                elif "face mode" in text or "eye mode" in text:
                    self.command_signal.emit("FACE")
            except:
                pass # Ignore background noise errors

class CursorSmoother:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.prev_x, self.prev_y = None, None

    def smooth(self, x, y):
        if self.prev_x is None:
            self.prev_x, self.prev_y = x, y
            return x, y
        smooth_x = self.alpha * x + (1 - self.alpha) * self.prev_x
        smooth_y = self.alpha * y + (1 - self.alpha) * self.prev_y
        self.prev_x, self.prev_y = smooth_x, smooth_y
        return int(smooth_x), int(smooth_y)

# --- THE NEURAL TRACKING ENGINE ---
class TrackingThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    log_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str, str, str)

    def __init__(self):
        super().__init__()
        self.is_running = True
        self.is_paused = False
        self.tracking_mode = "FACE"
        self.dwell_click_enabled = True
        
        self.smoother = CursorSmoother(alpha=0.2)
        self.screen_w, self.screen_h = pyautogui.size()
        
        self.is_dragging = False
        self.current_action_text = "SYSTEM STANDBY"
        
        # Dynamic Calibration Bounds (Learns user limits)
        self.is_calibrating = False
        self.calib_end_time = 0
        self.bounds = {'x_min': 0.3, 'x_max': 0.7, 'y_min': 0.3, 'y_max': 0.7}
        
        self.wink_thresh = 0.18
        self.mouth_thresh = 0.35
        self.pinch_thresh = 0.04
        
        self.last_action = time.time()
        self.eyes_closed_start = None
        self.dwell_start = time.time()
        self.dwell_x, self.dwell_y = 0, 0
        self.dwell_radius = 20
        self.dwell_time = 1.5 

    def start_calibration(self):
        """Triggers the 5-second boundary learning mode"""
        self.is_calibrating = True
        self.calib_end_time = time.time() + 5.0
        self.bounds = {'x_min': 1.0, 'x_max': 0.0, 'y_min': 1.0, 'y_max': 0.0}
        VoiceEngine.speak("Calibration started. Move your head or hand in a large circle.")
        self.log_signal.emit("CALIBRATION INITIATED: RECORDING USER LIMITS...")

    def calc_dist(self, p1, p2, w, h):
        return np.linalg.norm(np.array([p1.x * w, p1.y * h]) - np.array([p2.x * w, p2.y * h]))

    def get_ear(self, lm, indices, w, h):
        v1 = self.calc_dist(lm[indices[1]], lm[indices[5]], w, h)
        v2 = self.calc_dist(lm[indices[2]], lm[indices[4]], w, h)
        hd = self.calc_dist(lm[indices[0]], lm[indices[3]], w, h)
        return (v1 + v2) / (2.0 * hd) if hd != 0 else 0
        
    def draw_hud(self, frame, mode_text, action_text):
        h, w, _ = frame.shape
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (10, 15, 20), -1)
        cv2.rectangle(overlay, (0, h-60), (w, h), (10, 15, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, f"[ {mode_text} ]", (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
        
        if self.is_calibrating:
            rem = max(0, int(self.calib_end_time - time.time()))
            cv2.putText(frame, f"CALIBRATING: MOVE IN CIRCLES ({rem}s left)", (20, h-22), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
        else:
            action_color = (0, 150, 255) if "DRAGGING" in action_text else (0, 255, 120)
            cv2.putText(frame, f"CURRENT ACTION: {action_text}", (20, h-22), cv2.FONT_HERSHEY_DUPLEX, 0.7, action_color, 2)

    def run(self):
        cap = cv2.VideoCapture(0)
        mp_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        mp_hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret: continue
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            curr_time = time.time()
            target_x, target_y = None, None
            self.current_action_text = "TRACKING CURSOR"
            
            raw_x, raw_y = 0.5, 0.5 # Default center

            if self.tracking_mode == "FACE":
                results = mp_mesh.process(rgb)
                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    l_ear = self.get_ear(lm, [33, 160, 158, 133, 153, 144], w, h)
                    r_ear = self.get_ear(lm, [362, 385, 387, 263, 373, 380], w, h)
                    
                    l_iris, r_iris = lm[468], lm[473] 
                    cv2.circle(frame, (int(l_iris.x * w), int(l_iris.y * h)), 12, (0, 255, 200), 1)
                    cv2.circle(frame, (int(r_iris.x * w), int(r_iris.y * h)), 12, (0, 255, 200), 1)

                    if l_ear < self.wink_thresh and r_ear < self.wink_thresh:
                        if self.eyes_closed_start is None: self.eyes_closed_start = curr_time
                        elif curr_time - self.eyes_closed_start > 1.5:
                            self.is_paused = not self.is_paused
                            self.eyes_closed_start = None
                            self.last_action = curr_time + 1.0
                    else:
                        self.eyes_closed_start = None

                    if not self.is_paused and not self.is_calibrating:
                        if curr_time - self.last_action > 1.0:
                            if l_ear < self.wink_thresh and r_ear > self.wink_thresh:
                                pyautogui.click()
                                self.current_action_text = "LEFT CLICK (WINK)"
                                self.last_action = curr_time
                            elif r_ear < self.wink_thresh and l_ear > self.wink_thresh:
                                pyautogui.click(button='right')
                                self.current_action_text = "RIGHT CLICK (WINK)"
                                self.last_action = curr_time
                    
                    raw_x, raw_y = lm[4].x, lm[4].y
                
                self.draw_hud(frame, "FACE NEURAL ACTIVE", "PAUSED" if self.is_paused else self.current_action_text)

            elif self.tracking_mode == "HAND":
                results = mp_hands.process(rgb)
                if results.multi_hand_landmarks:
                    hand_lm = results.multi_hand_landmarks[0].landmark
                    thumb, index_tip = hand_lm[4], hand_lm[8]
                    middle_tip, middle_pip = hand_lm[12], hand_lm[10]
                    pinky_tip = hand_lm[20]
                    
                    for i in range(len([0, 5, 6, 7, 8])-1):
                        p1, p2 = hand_lm[[0, 5, 6, 7, 8][i]], hand_lm[[0, 5, 6, 7, 8][i+1]]
                        cv2.line(frame, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (0, 200, 255), 4)

                    raw_x, raw_y = index_tip.x, index_tip.y
                    
                    if not self.is_calibrating and not self.is_paused:
                        pinky_thumb_pinch = self.calc_dist(thumb, pinky_tip, w, h)
                        if pinky_thumb_pinch < (self.pinch_thresh * w):
                            if not self.is_dragging:
                                pyautogui.mouseDown()
                                self.is_dragging = True
                            self.current_action_text = "DRAGGING & COPYING [HOLDING CLICK]"
                        else:
                            if self.is_dragging:
                                pyautogui.mouseUp()
                                self.is_dragging = False
                            
                            index_pinch = self.calc_dist(thumb, index_tip, w, h)
                            if index_pinch < (self.pinch_thresh * w) and (curr_time - self.last_action > 0.8):
                                pyautogui.click()
                                self.current_action_text = "LEFT CLICK (PINCH)"
                                self.last_action = curr_time
                            
                            if middle_tip.y < middle_pip.y - 0.03:
                                self.current_action_text = "SCROLLING ENGINE ACTIVE"
                                ny = np.clip((raw_y - self.bounds['y_min']) / (self.bounds['y_max'] - self.bounds['y_min'] + 1e-6), 0, 1)
                                if ny < 0.35: pyautogui.scroll(60)
                                elif ny > 0.65: pyautogui.scroll(-60)
                                raw_x = None # Disable movement while scrolling

                self.draw_hud(frame, "HAND NEURAL ACTIVE", self.current_action_text)

            # --- CALIBRATION LOGIC ---
            if self.is_calibrating:
                if curr_time < self.calib_end_time:
                    self.bounds['x_min'] = min(self.bounds['x_min'], raw_x)
                    self.bounds['x_max'] = max(self.bounds['x_max'], raw_x)
                    self.bounds['y_min'] = min(self.bounds['y_min'], raw_y)
                    self.bounds['y_max'] = max(self.bounds['y_max'], raw_y)
                else:
                    self.is_calibrating = False
                    VoiceEngine.speak("Calibration complete")
                    self.log_signal.emit(f"LIMITS SAVED. X:[{self.bounds['x_min']:.2f}, {self.bounds['x_max']:.2f}]")

            # --- MOVEMENT EXECUTION ---
            if raw_x is not None and not self.is_paused and not self.is_calibrating:
                nx = np.clip((raw_x - self.bounds['x_min']) / (self.bounds['x_max'] - self.bounds['x_min'] + 1e-6), 0, 1)
                ny = np.clip((raw_y - self.bounds['y_min']) / (self.bounds['y_max'] - self.bounds['y_min'] + 1e-6), 0, 1)
                
                target_x, target_y = nx * self.screen_w, ny * self.screen_h
                sx, sy = self.smoother.smooth(target_x, target_y)
                
                sx = max(1, min(int(sx), self.screen_w - 2))
                sy = max(1, min(int(sy), self.screen_h - 2))
                pyautogui.moveTo(sx, sy)
                
                if self.dwell_click_enabled and not self.is_dragging:
                    if abs(sx - self.dwell_x) < self.dwell_radius and abs(sy - self.dwell_y) < self.dwell_radius:
                        dwell_duration = curr_time - self.dwell_start
                        if dwell_duration > self.dwell_time and (curr_time - self.last_action > 1.5):
                            pyautogui.click()
                            VoiceEngine.speak("Click")
                            self.last_action = curr_time
                            self.dwell_start = curr_time
                        elif dwell_duration > 0.4:
                            cv2.circle(frame, (w//2, h//2), int(dwell_duration*40), (0, 255, 120), 3)
                    else:
                        self.dwell_start = curr_time
                        self.dwell_x, self.dwell_y = sx, sy

            qt_img = QImage(frame.data, w, h, frame.strides[0], QImage.Format.Format_RGB888).rgbSwapped()
            self.change_pixmap_signal.emit(qt_img)

        cap.release()

    def stop(self):
        self.is_running = False
        self.wait()


# --- MODERN HIGH-END USER INTERFACE ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EyeMouse Pro - Accessibility Edition")
        self.setFixedSize(1150, 780)
        
        self.setStyleSheet("""
            QMainWindow { background-color: #0d1117; }
            QLabel { color: #e6edf3; font-family: 'Segoe UI', sans-serif; }
            QFrame#Sidebar { background-color: #161b22; border-radius: 15px; }
            QFrame#Card { background-color: #21262d; border-radius: 12px; padding: 10px; }
            QPushButton {
                background-color: #238636; color: white; border-radius: 8px;
                font-weight: bold; font-size: 14px; padding: 14px; border: 1px solid #2ea043;
            }
            QPushButton:hover { background-color: #2ea043; }
            QPushButton#StopBtn { background-color: #da3633; border: 1px solid #f85149; }
            QPushButton#StopBtn:hover { background-color: #f85149; }
            QPushButton#ModeBtn { background-color: #1f6feb; border: 1px solid #388bfd; }
            QPushButton#CalibBtn { background-color: #9e6a03; border: 1px solid #d29922; }
            QTextEdit {
                background-color: #010409; color: #8b949e; border: 1px solid #30363d;
                border-radius: 10px; font-family: 'Consolas', monospace; padding: 12px;
            }
            QSlider::groove:horizontal { border-radius: 3px; height: 6px; background: #30363d; }
            QSlider::handle:horizontal { background: #58a6ff; width: 16px; height: 16px; border-radius: 8px; margin: -5px 0; }
            QCheckBox { color: #e6edf3; font-weight: bold; font-size: 13px;}
        """)

        self.thread = None
        self.voice_listener = VoiceListenerThread()
        self.voice_listener.command_signal.connect(self.process_voice_command)
        self.voice_listener.start()
        
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        layout.setContentsMargins(20, 20, 20, 20)

        # --- LEFT SIDEBAR ---
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(360)
        side_layout = QVBoxLayout(sidebar)

        title = QLabel("EYEMOUSE PRO")
        title.setStyleSheet("font-size: 26px; font-weight: 900;")
        subtitle = QLabel("Paralysis Assistive Tech v5.0 | Voice Active 🎙️")
        subtitle.setStyleSheet("color: #3fb950; font-size: 12px;")
        side_layout.addWidget(title)
        side_layout.addWidget(subtitle)

        self.btn_toggle = QPushButton("START NEURAL ENGINE")
        self.btn_toggle.clicked.connect(self.toggle_tracking)
        side_layout.addWidget(self.btn_toggle)

        self.btn_mode = QPushButton("CURRENT: FACE & EYES")
        self.btn_mode.setObjectName("ModeBtn")
        self.btn_mode.clicked.connect(self.switch_mode)
        side_layout.addWidget(self.btn_mode)

        self.btn_calib = QPushButton("CALIBRATE USER LIMITS")
        self.btn_calib.setObjectName("CalibBtn")
        self.btn_calib.clicked.connect(self.trigger_calibration)
        side_layout.addWidget(self.btn_calib)

        controls_card = QFrame()
        controls_card.setObjectName("Card")
        controls_layout = QVBoxLayout(controls_card)
        self.check_dwell = QCheckBox(" Enable Auto-Click on Hover (1.5s)")
        self.check_dwell.setChecked(True)
        self.slider_smooth = QSlider(Qt.Orientation.Horizontal)
        self.slider_smooth.setRange(5, 50)
        self.slider_smooth.setValue(20)
        controls_layout.addWidget(self.check_dwell)
        controls_layout.addWidget(QLabel("Movement Smoothness Filter"))
        controls_layout.addWidget(self.slider_smooth)
        side_layout.addWidget(controls_card)

        side_layout.addStretch()

        self.lbl_instructions = QLabel()
        self.lbl_instructions.setStyleSheet("font-size: 14px;")
        self.set_face_instructions()
        side_layout.addWidget(self.lbl_instructions)

        # --- RIGHT MAIN AREA ---
        main_area = QFrame()
        main_layout = QVBoxLayout(main_area)

        self.video_container = QLabel()
        self.video_container.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_container.setFixedSize(700, 525)
        self.video_container.setStyleSheet("background-color: #010409; border-radius: 20px; border: 2px solid #30363d;")
        main_layout.addWidget(self.video_container)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(120)
        self.log_box.append("> SYSTEM INITIALIZED. VOICE LISTENER ONLINE...")
        self.log_box.append("> Say 'Start Engine' to begin.")
        main_layout.addWidget(self.log_box)

        layout.addWidget(sidebar)
        layout.addWidget(main_area)

    def process_voice_command(self, cmd):
        self.update_log(f"🎤 VOICE COMMAND DETECTED: {cmd}")
        if cmd == "START" and (self.thread is None or not self.thread.isRunning()):
            self.toggle_tracking()
        elif cmd == "STOP" and self.thread and self.thread.isRunning():
            self.toggle_tracking()
        elif cmd == "HAND" and "FACE" in self.btn_mode.text():
            self.switch_mode()
        elif cmd == "FACE" and "HAND" in self.btn_mode.text():
            self.switch_mode()

    def trigger_calibration(self):
        if self.thread and self.thread.isRunning():
            self.thread.start_calibration()
        else:
            self.update_log("ERROR: MUST START ENGINE BEFORE CALIBRATING.")

    def set_face_instructions(self):
        self.lbl_instructions.setText(
            "<b style='color:#ffffff; font-size:15px;'>Voice Commands Available:</b><br>"
            "<span style='color:#3fb950;'>🗣️ 'Start Engine'</span> | <span style='color:#f85149;'>'Stop Engine'</span><br>"
            "<span style='color:#3fb950;'>🗣️ 'Hand Mode'</span> | <span style='color:#79c0ff;'>'Face Mode'</span><br><br>"
            "<b style='color:#ffffff; font-size:15px;'>Face Mode Actions:</b><br>"
            "⏱️ Hover ➔ Auto-Click | 😉 Wink ➔ Quick Click<br>"
            "😴 Close Eyes ➔ Pause System"
        )

    def set_hand_instructions(self):
        self.lbl_instructions.setText(
            "<b style='color:#ffffff; font-size:15px;'>Voice Commands Available:</b><br>"
            "<span style='color:#3fb950;'>🗣️ 'Start Engine'</span> | <span style='color:#f85149;'>'Stop Engine'</span><br>"
            "<span style='color:#3fb950;'>🗣️ 'Hand Mode'</span> | <span style='color:#79c0ff;'>'Face Mode'</span><br><br>"
            "<b style='color:#ffffff; font-size:15px;'>Hand Mode Actions:</b><br>"
            "☝️ Index ➔ Move | 🤏 Pinch ➔ Click<br>"
            "🖕 Mid Finger ➔ Scroll | 🤙 Pinky+Thumb ➔ Drag"
        )

    def switch_mode(self):
        if "FACE" in self.btn_mode.text():
            self.btn_mode.setText("CURRENT: HAND & FINGERS")
            self.set_hand_instructions()
            if self.thread: self.thread.tracking_mode = "HAND"
            VoiceEngine.speak("Hand tracking activated")
        else:
            self.btn_mode.setText("CURRENT: FACE & EYES")
            self.set_face_instructions()
            if self.thread: self.thread.tracking_mode = "FACE"
            VoiceEngine.speak("Face tracking activated")

    def toggle_tracking(self):
        if self.thread is None or not self.thread.isRunning():
            self.thread = TrackingThread()
            self.thread.tracking_mode = "FACE" if "FACE" in self.btn_mode.text() else "HAND"
            self.thread.dwell_click_enabled = self.check_dwell.isChecked()
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.log_signal.connect(self.update_log)
            self.thread.start()
            self.btn_toggle.setText("STOP NEURAL ENGINE")
            self.btn_toggle.setObjectName("StopBtn")
            self.btn_toggle.setStyleSheet(self.btn_toggle.styleSheet())
            VoiceEngine.speak("Engine online")
        else:
            self.thread.stop()
            self.btn_toggle.setText("START NEURAL ENGINE")
            self.btn_toggle.setObjectName("")
            self.btn_toggle.setStyleSheet(self.btn_toggle.styleSheet())
            self.video_container.setPixmap(QPixmap())
            VoiceEngine.speak("Engine disengaged")

    def update_image(self, qt_img):
        scaled = qt_img.scaled(700, 525, Qt.AspectRatioMode.KeepAspectRatioByExpanding)
        rounded_pixmap = QPixmap(700, 525)
        rounded_pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(rounded_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(0, 0, 700, 525, 20, 20)
        painter.setClipPath(path)
        x_offset = (700 - scaled.width()) // 2
        y_offset = (525 - scaled.height()) // 2
        painter.drawImage(x_offset, y_offset, scaled)
        painter.end()
        self.video_container.setPixmap(rounded_pixmap)

    def update_log(self, text):
        time_str = time.strftime("%H:%M:%S")
        self.log_box.append(f"<span style='color:#8b949e;'>[{time_str}]</span> <span style='color:#58a6ff;'>></span> {text}")
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())