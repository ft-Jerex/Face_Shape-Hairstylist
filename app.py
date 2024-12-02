import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import dlib
import numpy as np
import os
import time

class FaceShapeRecognizer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Face Shape Recognition")
        self.root.state('zoomed')
        
        # Initialize variables first
        self.is_running = False
        self.current_frame = None

        self.message_shown = False
        
        # Create GUI elements before camera initialization
        self.create_widgets()
        
        # Initialize camera and models after GUI setup
        self.init_camera()
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
        

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        # Start video processing after a short delay
        self.root.after(100, self.start_video)
        self.root.mainloop()

    def init_camera(self):
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
            self.cap.set(cv2.CAP_PROP_FPS, 30)  # Set FPS
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
            print("Camera initialized successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Camera initialization failed: {e}")
            self.root.destroy()

    def create_widgets(self):
        # Create main frame
        # Create main frame with custom background
        self.main_frame = ttk.Frame(self.root, style='Custom.TFrame')
        self.main_frame.pack(pady=10, padx=10, fill="both", expand=True)
        
        # Configure custom style for frame and buttons
        style = ttk.Style()
        style.configure('Custom.TFrame', background='#FFB5C1')
        style.configure('Custom.TButton', 
                        background='#FFB5C1',  
                        foreground='#000000',  
                        font=('Arial', 10, 'bold'))
                        
        # Create video frame
        self.video_label = ttk.Label(self.main_frame)
        self.video_label.pack(pady=10)
        
        # Create info label
        self.info_label = ttk.Label(self.main_frame, 
                                    text="Face Shape: Unknown", 
                                    font=('Arial', 14), 
                                    style='Custom.TFrame')
        self.info_label.pack(pady=10)
        
        # Create control buttons frame
        self.control_frame = ttk.Frame(self.main_frame, style='Custom.TFrame')
        self.control_frame.pack(pady=10)
        
        # Start button with custom style
        self.start_button = ttk.Button(self.control_frame, 
                                    text="Start", 
                                    command=self.start_video,
                                    style='Custom.TButton')
        self.start_button.pack(side="left", padx=5)
        
        # Stop button with custom style
        self.stop_button = ttk.Button(self.control_frame, 
                                    text="Stop", 
                                    command=self.stop_video,
                                    style='Custom.TButton')
        self.stop_button.pack(side="left", padx=5)
        
        # Restart button with custom style
        self.restart_button = ttk.Button(self.control_frame, 
                                        text="Restart Analysis", 
                                        command=self.restart_analysis,
                                        style='Custom.TButton')
        self.restart_button.pack(side="left", padx=5)
        
        # Initially disable restart button
        self.restart_button.config(state=tk.DISABLED)

    def restart_analysis(self):
        # Reset all analysis-related attributes
        if hasattr(self, 'shape_history'):
            del self.shape_history
        if hasattr(self, 'shape_start_time'):
            del self.shape_start_time
        
        # Reset message flag
        self.message_shown = False
        
        # Reset info label
        self.info_label.config(text="Face Shape: Unknown")
        
        # Disable restart button
        self.restart_button.config(state=tk.DISABLED)
        
        # Restart video processing
        self.start_video()

    def determine_face_shape(self, face_landmarks):
        """
        Enhanced face shape detection including Round, Oval, Square, Diamond, and Heart shapes
        """
        points = np.array([[face_landmarks.part(i).x, face_landmarks.part(i).y] 
                        for i in range(68)])
        
        # Key measurements
        face_length = points[8][1] - points[19][1]  # Chin to forehead
        jaw_width = np.linalg.norm(points[3] - points[13])  # Width at jawline
        cheekbone_width = np.linalg.norm(points[2] - points[14])  # Width at cheekbones
        forehead_width = np.linalg.norm(points[17] - points[26])  # Width at forehead
        chin_to_cheekbone = points[8][1] - points[2][1]  # Chin to cheekbone height
        
        # Calculate ratios
        length_width_ratio = face_length / cheekbone_width
        jaw_cheek_ratio = jaw_width / cheekbone_width
        forehead_jaw_ratio = forehead_width / jaw_width
        chin_angle = np.arctan2(points[8][1] - points[7][1], points[8][0] - points[7][0])

        # Add timing verification
        if not hasattr(self, 'shape_history'):
            self.shape_history = []
            self.shape_start_time = time.time()

        # Define thresholds with more precise measurements
        if length_width_ratio <= 1.1 and jaw_cheek_ratio > 0.9:
            shape = "Round"
        elif jaw_cheek_ratio < 0.8 and forehead_jaw_ratio < 0.9:
            shape = "Heart"
        elif jaw_cheek_ratio > 0.9 and length_width_ratio > 1.2:
            shape = "Square"
        elif cheekbone_width > jaw_width and cheekbone_width > forehead_width:
            shape = "Diamond"
        else:
            shape = "Oval"

        # Store shape in history
        self.shape_history.append(shape)
        
        # Check if we have consistent readings for 3 seconds
        elapsed_time = time.time() - self.shape_start_time
        time_threshold = 10.0
        if elapsed_time >= time_threshold:
            # Get most common shape in history
            from collections import Counter
            most_common_shape = Counter(self.shape_history).most_common(1)[0][0]
            
            # Show message box if we haven't shown it yet
            if not getattr(self, 'message_shown', False):
                self.message_shown = True
                message = f"Face shape analysis complete!\nYour face shape is: {most_common_shape}\n\n"
                message += self.get_face_shape_description(most_common_shape)
                messagebox.showinfo("Face Shape Result", message)
                
                # Enable restart button after analysis is complete
                self.restart_button.config(state=tk.NORMAL)
            
            return most_common_shape
        
        return shape
    
    def get_face_shape_description(self, shape):
        """
        Returns detailed description for each face shape
        """
        descriptions = {
            "Round": "Characterized by soft curves and similar face width and length. Best suited for angular hairstyles to add definition.",
            "Oval": "Considered the ideal face shape with balanced proportions. Suits most hairstyles and facial features.",
            "Square": "Strong jaw and angular features. Characterized by a wide hairline and jawline of similar width.",
            "Diamond": "Wide cheekbones with narrow forehead and jawline. Features dramatic angles and defined cheekbones.",
            "Heart": "Wider forehead and cheekbones with a narrow, pointed chin. Often considered a very feminine face shape."
        }
        return descriptions.get(shape, "")

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
            self.info_label.config(text="Initializing camera...")
            self.is_running = True
            self.video_thread = threading.Thread(target=self.process_frame)
            self.video_thread.start()

    def stop_video(self):
        self.is_running = False
        if hasattr(self, 'video_thread'):
            self.video_thread.join()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        # Process first frame immediately
        self.root.after(100, self.process_frame)
        self.root.mainloop()

    def on_closing(self):
        self.stop_video()
        if hasattr(self, 'cap'):
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    app = FaceShapeRecognizer()
    app.run()
    