import sys
import cv2
import os
import numpy as np
import psycopg2
import insightface
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QHBoxLayout, QWidget, QFrame, QPushButton, QInputDialog)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QThread, pyqtSignal, Qt

# ---------------------------------------------------------
# 1. THE BRAIN (Background Processing Thread)
# ---------------------------------------------------------
class SecurityThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_status_signal = pyqtSignal(str, str, str)
    
    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.mode = "SCAN"
        self.register_name = ""
        self.dataset_path = "dataset"
        self.save_request = False
        
        # Database Connection
        try:
            self.conn = psycopg2.connect(
                host="localhost", dbname="security_ai",
                user="postgres", password="0000", port=5432
            )
            self.cur = self.conn.cursor()
            
            # AI Model Loading
            self.app_ai = insightface.app.FaceAnalysis(name="buffalo_l")
            self.app_ai.prepare(ctx_id=0, det_size=(640,640))
            
            # Load initial memory
            self.reload_database()
        except Exception as e:
            print(f"Startup Error: {e}")

    def reload_database(self):
        """Refreshes the known faces list from PostgreSQL"""
        self.cur.execute("""
            SELECT p.id, p.name, f.embedding
            FROM persons p
            JOIN face_embeddings f ON p.id = f.person_id
            WHERE p.enabled = TRUE
        """)
        db_data = self.cur.fetchall()
        self.names = [x[1] for x in db_data]
        self.embeddings = [np.array(x[2]) for x in db_data]
        print(f"🧠 Memory Updated: {len(self.names)} faces loaded.")

    def cosine(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def set_mode_register(self, name):
        self.register_name = name
        self.mode = "REGISTER"
        os.makedirs(os.path.join(self.dataset_path, name), exist_ok=True)

    def set_mode_scan(self):
        self.mode = "SCAN"

    def trigger_save(self):
        self.save_request = True

    def run(self):
        cap = cv2.VideoCapture(0)
        THRESHOLD = 0.5

        while self._run_flag:
            ret, frame = cap.read()
            if not ret: break

            # --- MODE: RECORDING NEW PERSON ---
            if self.mode == "REGISTER":
                if self.save_request:
                    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
                    save_path = os.path.join(self.dataset_path, self.register_name, filename)
                    cv2.imwrite(save_path, frame)
                    self.save_request = False
                    # Visual feedback: Flash
                    cv2.rectangle(frame, (0,0), (640,480), (255,255,255), -1)

                self.update_status_signal.emit(f"New: {self.register_name}", "RECORDING...", "blue")
            
            # --- MODE: SECURITY SCANNING ---
            else:
                faces = self.app_ai.get(frame)
                if not faces:
                    self.update_status_signal.emit("No Face", "Waiting", "gray")
                
                for face in faces:
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    emb = face.embedding
                    best_score, best_name = 0, "Unknown"

                    for i in range(len(self.embeddings)):
                        score = self.cosine(emb, self.embeddings[i])
                        if score > best_score:
                            best_score, best_name = score, self.names[i]

                    if best_score > THRESHOLD:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        self.update_status_signal.emit(best_name, "ACCESS GRANTED", "green")
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        self.update_status_signal.emit("Unknown", "ACCESS DENIED", "red")

            self.change_pixmap_signal.emit(frame)
        
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

# ---------------------------------------------------------
# 2. THE DASHBOARD (User Interface)
# ---------------------------------------------------------
class SecurityDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pro AI Door Security")
        self.setGeometry(100, 100, 1100, 600)
        self.setStyleSheet("background-color: #121212;")

        # Layout
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Video Section
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("border: 3px solid #333; background: black;")
        layout.addWidget(self.video_label)

        # UI Section
        panel = QFrame()
        panel.setStyleSheet("background: #1e1e1e; border-radius: 15px; padding: 10px;")
        panel_layout = QVBoxLayout(panel)

        self.name_display = QLabel("Initializing...")
        self.name_display.setFont(QFont("Segoe UI", 24, QFont.Bold))
        self.name_display.setStyleSheet("color: #00d4ff;")
        self.name_display.setAlignment(Qt.AlignCenter)

        self.status_box = QLabel("READY")
        self.status_box.setFont(QFont("Segoe UI", 18, QFont.Bold))
        self.status_box.setStyleSheet("background: #333; color: white; border-radius: 8px; padding: 15px;")
        self.status_box.setAlignment(Qt.AlignCenter)

        self.add_btn = QPushButton("Register New Person")
        self.add_btn.setStyleSheet("""
            QPushButton { background: #007bff; color: white; padding: 15px; border-radius: 8px; font-size: 16px; font-weight: bold; }
            QPushButton:hover { background: #0056b3; }
        """)
        self.add_btn.clicked.connect(self.start_registration)

        self.info_text = QLabel("Normal Mode")
        self.info_text.setStyleSheet("color: #666; font-size: 12px;")
        self.info_text.setAlignment(Qt.AlignCenter)

        panel_layout.addWidget(self.name_display)
        panel_layout.addWidget(self.status_box)
        panel_layout.addStretch()
        panel_layout.addWidget(self.info_text)
        panel_layout.addWidget(self.add_btn)
        layout.addWidget(panel)

        # Start AI Thread
        self.thread = SecurityThread()
        self.thread.change_pixmap_signal.connect(self.update_video)
        self.thread.update_status_signal.connect(self.update_labels)
        self.thread.start()

    def start_registration(self):
        name, ok = QInputDialog.getText(self, "Registration", "Enter Full Name:")
        if ok and name:
            self.thread.set_mode_register(name)
            self.add_btn.setEnabled(False)
            self.info_text.setText("[SPACE] Capture | [ESC] Save & Exit")

    def keyPressEvent(self, event):
        if self.thread.mode == "REGISTER":
            if event.key() == Qt.Key_Space:
                self.thread.trigger_save()
            elif event.key() == Qt.Key_Escape:
                # Save person and train
                new_person = self.thread.register_name
                self.thread.set_mode_scan()
                self.run_auto_train(new_person)
                self.add_btn.setEnabled(True)
                self.info_text.setText("System Updated ✅")

    def run_auto_train(self, name):
        """The magic part: Turns photos into vectors and reloads DB"""
        self.status_box.setText("TRAINING...")
        self.status_box.setStyleSheet("background: orange; color: black;")
        QApplication.processEvents()

        try:
            # 1. Create person record
            self.thread.cur.execute("INSERT INTO persons (name) VALUES (%s) RETURNING id", (name,))
            pid = self.thread.cur.fetchone()[0]
            
            # 2. Extract features from new photos
            path = os.path.join("dataset", name)
            for img_file in os.listdir(path):
                img = cv2.imread(os.path.join(path, img_file))
                if img is None: continue
                faces = self.thread.app_ai.get(img)
                if faces:
                    emb = faces[0].embedding.tolist()
                    self.thread.cur.execute(
                        "INSERT INTO face_embeddings (person_id, embedding) VALUES (%s, %s)",
                        (pid, emb)
                    )
            self.thread.conn.commit()
            
            # 3. RELOAD the thread's memory immediately
            self.thread.reload_database()
        except Exception as e:
            print(f"Auto-train failed: {e}")

    def update_labels(self, name, status, color):
        self.name_display.setText(name)
        self.status_box.setText(status)
        bg = {"green": "#28a745", "red": "#dc3545", "blue": "#17a2b8", "gray": "#333"}
        self.status_box.setStyleSheet(f"background: {bg.get(color, '#333')}; color: white; border-radius: 8px; padding: 15px;")

    def update_video(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg).scaled(640, 480, Qt.KeepAspectRatio))

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SecurityDashboard()
    window.show()
    sys.exit(app.exec_())