import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import subprocess
import sys
import dlib
import numpy as np
import math
import os
import time

class FaceShapeRecognizer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Face Shape Recognition")
        self.root.state('zoomed')
        
        # Get screen dimensions
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        
        # Initialize variables first
        self.is_running = False
        self.current_frame = None
        self.message_shown = False
        self.current_shape = None
        self.timer_started = False
        self.timer_value = 10
        self.timer_paused = False
        self.elapsed_time = 0
        
        # Calculate dynamic sizes based on screen dimensions
        self.hairstyle_img_size = min(int(self.screen_width * 0.15), int(self.screen_height * 0.2))
        
        # Set video size to maintain 4:3 aspect ratio
        video_height = int(self.screen_height * 0.4)
        video_width = int(video_height * 4/3)
        self.video_size = (video_width, video_height)
        
        # Load hairstyle images with dynamic sizing
        self.male_images = self.load_hairstyle_images("male")
        self.female_images = self.load_hairstyle_images("female")
        
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

    def load_hairstyle_images(self, gender):
        images = {}
        shapes = ["Round", "Oval", "Square", "Diamond", "Heart"]
        for shape in shapes:
            shape_images = []
            path = f"{gender}/{shape.lower()}"
            if os.path.exists(path):
                for img_file in os.listdir(path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        try:
                            img = Image.open(os.path.join(path, img_file))
                            img = img.resize((self.hairstyle_img_size, self.hairstyle_img_size), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            shape_images.append(photo)
                        except Exception as e:
                            print(f"Error loading image {img_file}: {e}")
            images[shape] = shape_images[:5]  # Take up to 5 images
        return images

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
        # Calculate dynamic padding and font sizes
        side_padding = int(self.screen_width * 0.02)
        title_font_size = int(min(self.screen_width, self.screen_height) * 0.03)
        label_font_size = int(min(self.screen_width, self.screen_height) * 0.017)
        button_font_size = int(min(self.screen_width, self.screen_height) * 0.012)
        
        # Create main container
        container = ttk.Frame(self.root, style='Custom.TFrame')
        container.pack(fill="both", expand=True)

        # Create left frame for male images
        left_frame = ttk.Frame(container, style='Custom.TFrame')
        left_frame.pack(side="left", padx=side_padding, fill="y")

        # Create center frame
        center_frame = ttk.Frame(container, style='Custom.TFrame')
        center_frame.pack(side="left", expand=True, fill="both", padx=10)

        # Create right frame for female images
        right_frame = ttk.Frame(container, style='Custom.TFrame')
        right_frame.pack(side="right", padx=side_padding, fill="y")

        # Configure custom style with dynamic sizes
        style = ttk.Style()
        style.configure('Custom.TFrame', 
                        background='#FFB5C1', 
                        foreground='#000000')
        style.configure('Custom.TButton',
                        background='#FFB5C1',
                        foreground='#000000',
                        font=('Cambria', button_font_size, 'bold'))
        style.configure('Custom.TLabel',
                        background='#FFB5C1',
                        foreground='#000000')

        # Add MALE label with dynamic font size
        male_label = tk.Label(left_frame, 
                            text="MALE", 
                            font=("Cambria", title_font_size, "bold"),
                            fg="#8B0000",
                            bg='#FFB5C1',
                            justify="center",
                            anchor="center")
        male_label.pack(pady=int(self.screen_height * 0.03), padx=110)

        # Create image grid for male side
        self.male_image_labels = []
        placeholder_frame_left = ttk.Frame(left_frame, style='Custom.TFrame')
        placeholder_frame_left.pack(side="top", fill="both", expand=True)
        
        # First row - 2 images
        row1_frame = ttk.Frame(placeholder_frame_left, style='Custom.TFrame')
        row1_frame.pack(pady=5)
        for i in range(2):
            label = ttk.Label(row1_frame, style='Custom.TLabel')
            label.pack(side="left", padx=2)
            self.male_image_labels.append(label)
            
        # Second row - 2 images
        row2_frame = ttk.Frame(placeholder_frame_left, style='Custom.TFrame')
        row2_frame.pack(pady=5)
        for i in range(2):
            label = ttk.Label(row2_frame, style='Custom.TLabel')
            label.pack(side="left", padx=2)
            self.male_image_labels.append(label)
            
        # Third row - 1 image
        row3_frame = ttk.Frame(placeholder_frame_left, style='Custom.TFrame')
        row3_frame.pack(pady=2)
        label = ttk.Label(row3_frame, style='Custom.TLabel')
        label.pack()
        self.male_image_labels.append(label)

        # Add FEMALE label with dynamic font size
        female_label = tk.Label(right_frame, 
                            text="FEMALE", 
                            font=("Cambria", title_font_size, "bold"),
                            fg="#8B0000",
                            bg='#FFB5C1',
                            justify="center",
                            anchor="center")
        female_label.pack(pady=int(self.screen_height * 0.03), padx=110)

        # Create image grid for female side
        self.female_image_labels = []
        placeholder_frame_right = ttk.Frame(right_frame, style='Custom.TFrame')
        placeholder_frame_right.pack(side="top", fill="both", expand=True)
        
        # First row - 2 images
        row1_frame = ttk.Frame(placeholder_frame_right, style='Custom.TFrame')
        row1_frame.pack(pady=5)
        for i in range(2):
            label = ttk.Label(row1_frame, style='Custom.TLabel')
            label.pack(side="left", padx=2)
            self.female_image_labels.append(label)
            
        # Second row - 2 images
        row2_frame = ttk.Frame(placeholder_frame_right, style='Custom.TFrame')
        row2_frame.pack(pady=2)
        for i in range(2):
            label = ttk.Label(row2_frame, style='Custom.TLabel')
            label.pack(side="left", padx=2)
            self.female_image_labels.append(label)
            
        # Third row - 1 image
        row3_frame = ttk.Frame(placeholder_frame_right, style='Custom.TFrame')
        row3_frame.pack(pady=5)
        label = ttk.Label(row3_frame, style='Custom.TLabel')
        label.pack()
        self.female_image_labels.append(label)

        # Title Label with dynamic font size
        title_label = tk.Label(
            center_frame,
            text="Face Shape\nHairstylist",
            font=("Cambria", 32, "bold"),
            fg="#8B0000",
            bg="#FFB5C1",
        )
        title_label.pack(pady=(20, 10))

        # Create video frame in center
        self.video_label = ttk.Label(center_frame)
        self.video_label.pack(pady=int(self.screen_height * 0.01))
        
        # Create timer label with dynamic font size
        self.timer_label = ttk.Label(center_frame,
                                   text="",
                                   font=('Cambria', label_font_size),
                                   style='Custom.TLabel')
        self.timer_label.pack(pady=5)
        
        # Create info label with dynamic font size
        self.info_label = ttk.Label(center_frame,
                                    text="Face Shape: Unknown",
                                    font=('Cambria', label_font_size),
                                    style='Custom.TLabel')
        self.info_label.pack(pady=10)
        
        # Create control buttons frame
        self.control_frame = ttk.Frame(center_frame, style="Custom.TFrame")
        self.control_frame.pack(pady=int(self.screen_height * 0.01))
        
        # Control buttons with dynamic padding
        button_padding = int(self.screen_width * 0.005)
        
        # Start button
        self.start_button = tk.Button(self.control_frame,
                                    text="▶",  # Play icon
                                    command=self.start_video,
                                    bg="#90EE90",  # Light green
                                    activebackground="#32CD32",  # Darker green on hover
                                    font=("Arial", 14),  # Larger font for icon
                                    relief="groove"
                                    )
        self.start_button.pack(side="left", padx=button_padding)
        
        # Stop button
        self.stop_button = tk.Button(self.control_frame,
                                    text="⏸",  # Pause icon
                                    command=self.stop_video,
                                    bg="red",  # Light red
                                    activebackground="#DC143C",  # Darker red on hover
                                    font=("Arial", 14),  # Larger font for icon
                                    relief="groove"
                                    )
        self.stop_button.pack(side="left", padx=button_padding)
        
        # Restart button
        self.restart_button = tk.Button(self.control_frame,
                                        text="↻",  # Repeat/restart icon
                                        command=self.restart_analysis,
                                        bg="#FFB347",  # Light orange
                                        activebackground="#FF8C00",  # Darker orange on hover
                                        font=("Arial", 14),  # Larger font for icon
                                        relief="groove"
                                        )
        self.restart_button.pack(side="left", padx=button_padding)

        # Back to Main Menu button
        self.back_button = tk.Button(self.control_frame,
                                    text="⌂",  # Home icon
                                    command=self.back_to_main_menu,
                                    bg="#87CEEB",  # Light blue
                                    activebackground="#4169E1",  # Darker blue on hover
                                    font=("Arial", 14),  # Larger font for icon
                                    relief="groove"
                                    )
        self.back_button.pack(side="left", padx=button_padding)

        # Create result label with dynamic font size and wraplength
        self.result_label = ttk.Label(center_frame,
                                    text="",
                                    font=('Cambria', label_font_size),
                                    wraplength=int(self.screen_width * 0.4),
                                    justify="center",
                                    style='Custom.TLabel')
        self.result_label.pack(pady=10)
        
        # Initially disable restart button
        self.restart_button.config(state=tk.DISABLED)

    def back_to_main_menu(self):
        """Launch the start.py file in a new Python process."""
        try:
            script_path = os.path.abspath(r"start.py")
            if os.path.exists(script_path):
                # Open the script in the same directory
                subprocess.Popen([sys.executable, script_path], cwd=os.path.dirname(script_path))
                self.on_closing()  # Close current window after launching main menu
            else:
                messagebox.showerror("Error", f"File not found: {script_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open start window: {e}")

    def update_hairstyle_images(self, face_shape):
        if face_shape != self.current_shape:
            self.current_shape = face_shape
            # Update male images
            male_images = self.male_images.get(face_shape, [])
            for i, label in enumerate(self.male_image_labels):
                if i < len(male_images):
                    label.configure(image=male_images[i])
                    label._image = male_images[i]  # Keep a reference
                else:
                    label.configure(image='')

            # Update female images
            female_images = self.female_images.get(face_shape, [])
            for i, label in enumerate(self.female_image_labels):
                if i < len(female_images):
                    label.configure(image=female_images[i])
                    label._image = female_images[i]  # Keep a reference
                else:
                    label.configure(image='')

    def restart_analysis(self):
        # Reset all analysis-related attributes
        if hasattr(self, 'shape_history'):
            del self.shape_history
        if hasattr(self, 'shape_start_time'):
            del self.shape_start_time
        
        # Reset message flag and current shape
        self.message_shown = False
        self.current_shape = None
        self.timer_started = False
        self.timer_value = 10
        self.timer_paused = False
        self.elapsed_time = 0
        
        # Clear all image labels
        for label in self.male_image_labels + self.female_image_labels:
            label.configure(image='')
        
        # Reset info label and result label
        self.info_label.config(text="Face Shape: Unknown")
        self.result_label.config(text="")
        self.timer_label.config(text="")
        
        # Disable restart button
        self.restart_button.config(state=tk.DISABLED)
        
        # Restart video processing
        self.start_video()

    def determine_face_shape(self, face_landmarks):
        """
        Advanced face shape detection using comprehensive landmark measurements
        
        Args:
            face_landmarks (dlib.full_object_detection): Facial landmarks from dlib
        
        Returns:
            str: Detected face shape with high confidence
        """
        # Extract precise landmark coordinates
        points = np.array([
            [face_landmarks.part(i).x, face_landmarks.part(i).y] 
            for i in range(68)
        ])
        
        # Advanced measurement points
        # Forehead points
        forehead_left = points[17]
        forehead_right = points[26]
        
        # Cheekbone points
        cheekbone_left = points[2]
        cheekbone_right = points[14]
        
        # Jawline points
        jaw_left = points[5]
        jaw_right = points[11]
        jaw_bottom = points[8]
        
        # Precise measurements
        forehead_width = np.linalg.norm(forehead_left - forehead_right)
        cheekbone_width = np.linalg.norm(cheekbone_left - cheekbone_right)
        jaw_width = np.linalg.norm(jaw_left - jaw_right)
        
        # Face length calculations
        face_length = np.linalg.norm(points[19] - points[8])  # From forehead to chin
        
        # Detailed angle calculations
        def calculate_angle(p1, p2, p3):
            """Calculate angle between three points"""
            ba = p1 - p2
            bc = p3 - p2
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            return np.degrees(angle)
        
        # Jaw angle calculation
        jaw_angle_left = calculate_angle(jaw_left, jaw_bottom, cheekbone_left)
        jaw_angle_right = calculate_angle(jaw_right, jaw_bottom, cheekbone_right)
        avg_jaw_angle = (jaw_angle_left + jaw_angle_right) / 2
        
        # Advanced ratio calculations
        ratios = {
            'length_to_width': face_length / cheekbone_width,
            'forehead_to_jaw': forehead_width / jaw_width,
            'cheekbone_to_jaw': cheekbone_width / jaw_width,
            'face_aspect_ratio': face_length / (forehead_width + jaw_width)
        }
        
        # Enhanced face shape detection logic
        def classify_face_shape():
            # Comprehensive classification criteria
            if (ratios['length_to_width'] <= 1.2 and 
                ratios['cheekbone_to_jaw'] >= 0.9 and 
                avg_jaw_angle < 70):
                return "Round"
            
            elif (ratios['length_to_width'] >= 1.3 and 
                  ratios['cheekbone_to_jaw'] >= 0.8 and 
                  ratios['forehead_to_jaw'] > 1.1):
                return "Oval"
            
            elif (forehead_width > cheekbone_width and 
                  cheekbone_width > jaw_width and 
                  avg_jaw_angle > 80):
                return "Heart"
            
            elif (math.isclose(forehead_width, jaw_width, rel_tol=0.1) and 
                  avg_jaw_angle > 75):
                return "Square"
            
            elif (cheekbone_width > forehead_width and 
                  cheekbone_width > jaw_width and 
                  ratios['length_to_width'] > 1.2):
                return "Diamond"
            
            else:
                return "Cannot determine"
        
        # Get initial shape classification
        shape = classify_face_shape()
        
        # Initialize shape history if not exists
        if not hasattr(self, 'shape_history'):
            self.shape_history = []
            self.shape_start_time = time.time()
            self.timer_started = True

        # Store shape in history
        self.shape_history.append(shape)
        
        # Check if we have consistent readings
        if not self.timer_paused:
            self.elapsed_time = time.time() - self.shape_start_time
        time_threshold = 10.0
        
        # Update timer display
        if self.timer_started and self.elapsed_time < time_threshold:
            remaining_time = max(0, int(time_threshold - self.elapsed_time))
            self.timer_label.config(text=f"Analysis in progress... {remaining_time}s")
        
        if self.elapsed_time >= time_threshold:
            # Get most common shape in history with higher threshold
            from collections import Counter
            shape_counts = Counter(self.shape_history)
            most_common = shape_counts.most_common(1)[0]
            most_common_shape = most_common[0]
            confidence = most_common[1] / len(self.shape_history)
            
            # Only show result if confidence is high enough
            if confidence > 0.6 and not getattr(self, 'message_shown', False):
                self.message_shown = True
                self.timer_label.config(text="Analysis complete!")
                result_text = f"Face Shape Analysis Complete!\nYour face shape is: {most_common_shape}\n\n"
                result_text += self.get_face_shape_description(most_common_shape)
                self.result_label.config(text=result_text)
                self.restart_button.config(state=tk.NORMAL)
                self.update_hairstyle_images(most_common_shape)
            
            return most_common_shape
        
        return shape
    def get_face_shape_description(self, face_shape):
        """
        Provides a detailed description for each face shape
        
        Args:
            face_shape (str): Detected face shape
        
        Returns:
            str: Detailed description of the face shape
        """
        descriptions = {
            "Round": "A face shape where the width and length are nearly equal, characterized by soft, rounded contours and gentle curves. The cheeks are typically full, and the jawline appears less defined, creating a harmonious, youthful appearance.",
            "Oval": "Considered the most versatile and balanced face shape, featuring a forehead slightly wider than the chin, with gently rounded edges. The face appears longer than it is wide, creating an elegant profile that complements most hairstyles.",
            "Heart": "Defined by a broader forehead and cheekbones that taper down to a delicate, pointed chin, mimicking the shape of a heart. This distinctive silhouette is marked by prominent cheekbones and a narrow jawline.",
            "Square": "A bold face shape characterized by a strong, angular jawline and nearly uniform width across the forehead, cheekbones, and jaw. Sharp, defined lines create a sense of structure and symmetry, giving the face a powerful appearance.",
            "Diamond": "A unique face shape featuring narrow, more angular forehead and jawline with dramatically prominent, wide cheekbones. The cheekbones are the widest part of the face, creating a refined look that draws attention to the facial features.",
            "Cannot determine": "Your facial structure presents a unique combination of characteristics that makes a standard face shape classification challenging. This complexity reflects the individual beauty of your specific facial anatomy."
        }
        
        return descriptions.get(face_shape, "No description available.")

    def process_frame(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.root.after(10, self.process_frame)
            return

        frame = cv2.flip(frame, 1)  # Mirror the frame
        
        # Resize frame to fit screen while maintaining aspect ratio
        frame = cv2.resize(frame, self.video_size)
        
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
            self.timer_paused = False
            if hasattr(self, 'shape_start_time'):
                # Resume timer from where it was paused
                self.shape_start_time = time.time() - self.elapsed_time
            self.video_thread = threading.Thread(target=self.process_frame)
            self.video_thread.start()

    def stop_video(self):
        self.is_running = False
        self.timer_paused = True
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