import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import numpy as np
import os

class FaceShapeRecognizer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Face Shape Recognition")
        self.root.geometry("800x600")
        
        # Initialize video capture with error handling
        self.init_camera()

        # Load the cascade classifier
        try:
            cascade_path = r"C:\Users\Jerard\Documents\GitHub\Face_Shape-Hairstylist\tools\haarcascade_frontalface_default.xml"
            if not os.path.exists(cascade_path):
                raise FileNotFoundError(f"Cascade file not found at: {cascade_path}")
            
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                raise Exception("Error loading cascade classifier")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading cascade classifier: {e}")
            self.root.destroy()
            return

        # Load the classifier model
        try:
            classifier_path = r"C:\Users\Jerard\Documents\GitHub\Face_Shape-Hairstylist\tools\classifier.yml"
            if not os.path.exists(classifier_path):
                raise FileNotFoundError(f"Classifier file not found at: {classifier_path}")
            
            self.clf = cv2.face.LBPHFaceRecognizer_create()
            self.clf.read(classifier_path)
        except Exception as e:
            messagebox.showerror("Error", f"Error loading classifier: {e}")
            self.root.destroy()
            return
        
        # Variables
        self.is_running = False
        self.current_frame = None
        
        # Create GUI elements
        self.create_widgets()
        
    def init_camera(self):
        try:
            self.cap = cv2.VideoCapture(0)  # Use default camera
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
            ret, frame = self.cap.read()
            if not ret:
                raise Exception("Could not read from camera")
            print("Successfully connected to camera")
        except Exception as e:
            messagebox.showerror("Error", f"Camera initialization failed: {str(e)}")
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

    def determine_face_shape(self, face_width, face_height, face_landmarks=None):
        # Calculate basic ratio
        ratio = face_height / face_width
        
        # Add tolerance for measurement variations
        tolerance = 0.1
        
        # Enhanced face shape determination logic
        if ratio > 1.5:
            return "Oblong"
        elif 1.3 - tolerance <= ratio <= 1.5 + tolerance:
            return "Oval"
        elif 1.1 - tolerance <= ratio < 1.3 - tolerance:
            return "Round"
        elif ratio < 1.1:
            return "Square"
        else:
            return "Undefined"

    def process_frame(self):
        if not self.is_running:
            return

        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("Failed to grab frame")
                self.root.after(10, self.process_frame)  # Retry after a short delay
                return

            # Mirror the frame
            frame = cv2.flip(frame, 1)

            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            try:
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, 
                                                        minNeighbors=5,
                                                        minSize=(30, 30))
            except Exception as e:
                print(f"Error in face detection: {e}")
                self.root.after(10, self.process_frame)
                return

            for (x, y, w, h) in faces:
                try:
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                    # Get region of interest
                    roi_gray = gray[y:y+h, x:x+w]

                    # Calculate face shape
                    face_shape = self.determine_face_shape(w, h)

                    # Update info label in a thread-safe way
                    self.info_label.config(text=f"Face Shape: {face_shape}")

                    # Draw face shape text
                    cv2.putText(frame, face_shape, (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                except Exception as e:
                    print(f"Error processing detected face: {e}")
                    continue

            # Convert frame for display
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        except Exception as e:
            print(f"Error in process_frame: {str(e)}")

        # Schedule the next frame update
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