import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import numpy as np

class FaceShapeRecognizer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Face Shape Recognition")
        self.root.geometry("800x600")
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        
        # Load the face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Create GUI elements
        self.create_widgets()
        
        # Variables
        self.is_running = True
        self.current_frame = None
        
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
        
    def determine_face_shape(self, face_width, face_height):
        ratio = face_height / face_width
        
        if ratio > 1.5:
            return "Oblong"
        elif 1.3 <= ratio <= 1.5:
            return "Oval"
        elif 1.1 <= ratio < 1.3:
            return "Round"
        else:
            return "Square"
    
    def process_frame(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                # Convert frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    # Determine face shape
                    face_shape = self.determine_face_shape(w, h)
                    
                    # Update info label
                    self.info_label.config(text=f"Face Shape: {face_shape}")
                    
                    # Draw face shape text
                    cv2.putText(frame, face_shape, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                
                # Convert frame for display
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
            
            self.root.update()
    
    def start_video(self):
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
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    app = FaceShapeRecognizer()
    app.run()