import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import dlib
import numpy as np
import os

class FaceShapeRecognizer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Face Shape Recognition")
        self.root.geometry("800x600")
        
        # Initialize video capture
        self.init_camera()

        # Load dlib's face detector and shape predictor
        try:
            predictor_path = r"tools\shape_predictor_68_face_landmarks.dat"
            if not os.path.exists(predictor_path):
                raise FileNotFoundError(f"Predictor file not found at: {predictor_path}")
            
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(predictor_path)
        except Exception as e:
            messagebox.showerror("Error", f"Error loading dlib models: {e}")
            self.root.destroy()
            return

        # Variables
        self.is_running = False
        self.current_frame = None

        # Create GUI elements
        self.create_widgets()

    def init_camera(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
            print("Camera initialized successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Camera initialization failed: {e}")
            self.root.destroy()

    def create_widgets(self):
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(pady=10, padx=10, fill="both", expand=True)
        
        # Create video frame
        self.video_label = ttk.Label(self.main_frame)
        self.video_label.pack(pady=10)
        
        # Create info label
        self.info_label = ttk.Label(self.main_frame, text="Face Shape: Unknown", font=('Arial', 14))
        self.info_label.pack(pady=10)
        
        # Create control buttons
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(pady=10)
        
        self.start_button = ttk.Button(self.control_frame, text="Start", command=self.start_video)
        self.start_button.pack(side="left", padx=5)
        
        self.stop_button = ttk.Button(self.control_frame, text="Stop", command=self.stop_video)
        self.stop_button.pack(side="left", padx=5)

    def determine_face_shape(self, face_landmarks):
        """
        Determines face shape based on facial landmarks.
        Returns: round, oval, or square
        """
        points = np.array([[face_landmarks.part(i).x, face_landmarks.part(i).y] 
                        for i in range(68)])
        
        # Key measurements
        face_length = points[8][1] - points[19][1]  # Vertical length from chin to forehead
        jaw_width = np.linalg.norm(points[3] - points[13])  # Width at jawline
        cheekbone_width = np.linalg.norm(points[2] - points[14])  # Width at cheekbones
        
        # Calculate ratios
        length_width_ratio = face_length / cheekbone_width
        jaw_cheek_ratio = jaw_width / cheekbone_width

        # Define thresholds
        SQUARE_JAW_CHEEK_THRESHOLD = 0.95  # For square face, jaw width should be similar to cheekbone width
        ROUND_LENGTH_WIDTH_THRESHOLD = 1.1  # For round face, length should be similar to width
        
        # Square face: Similar jaw and cheekbone width
        if jaw_cheek_ratio >= SQUARE_JAW_CHEEK_THRESHOLD:
            return "Square"
        
        # Round face: Similar length and width, curved jawline
        elif length_width_ratio <= ROUND_LENGTH_WIDTH_THRESHOLD:
            return "Round"
        
        # Oval face: Length greater than width, cheekbones wider than jawline
        else:
            return "Oval"

    def process_frame(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.root.after(10, self.process_frame)
            return

        frame = cv2.flip(frame, 1)  # Mirror the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using dlib
        faces = self.detector(gray)
        for face in faces:
            # Get landmarks
            landmarks = self.predictor(gray, face)

            # Calculate face shape
            face_shape = self.determine_face_shape(landmarks)

            # Draw the face bounding box and landmarks
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)
            for i in range(68):
                cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), 1, (0, 255, 0), -1)

            # Display face shape
            cv2.putText(frame, face_shape, (face.left(), face.top() - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Update GUI label
            self.info_label.config(text=f"Face Shape: {face_shape}")

        # Convert frame for display
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.process_frame)

    def start_video(self):
        if not self.is_running:
            self.is_running = True
            self.video_thread = threading.Thread(target=self.process_frame)
            self.video_thread.start()

    def stop_video(self):
        self.is_running = False
        if hasattr(self, 'video_thread'):
            self.video_thread.join()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        self.stop_video()
        if hasattr(self, 'cap'):
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    app = FaceShapeRecognizer()
    app.run()