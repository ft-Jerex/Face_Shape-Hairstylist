import os
import sys
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Shape Hairstylist")
        self.root.geometry("500x700")
        self.root.configure(bg="#FFB5C1")  # Light pink gradient-like background
        self.center_window(self.root, 400, 600)
        self.root.resizable(False, False)

        # Title Label
        title_label = tk.Label(
            self.root,
            text="Face Shape\nHairstylist",
            font=("Lucida Calligraphy", 24, "bold"),
            fg="#8B0000",  # Dark red color
            bg="#FFB5C1",
        )
        title_label.pack(pady=50)

        # Main Menu Button
        main_menu_frame = tk.Frame(self.root, bg="#FFB5C1")
        main_menu_frame.pack(pady=10)

        # Main Menu Label
        main_menu_label = tk.Label(
            main_menu_frame,
            text="Main Menu",
            font=("Lucida Calligraphy", 18, "bold"),
            fg="#FFFFFF",
            bg="#68102C",
            padx=40,
            pady=10,
        )
        main_menu_label.pack()

        # Buttons
        button_frame = tk.Frame(self.root, bg="#FFB5C1")
        button_frame.pack(pady=10)

        # Start Button
        start_button = tk.Button(
            button_frame,
            text="| Start |",
            font=("Lucida Calligraphy", 14),
            bg="#FFB5C1",
            fg="#8B0000",
            relief="flat",
            command=self.open_start_window,
        )
        start_button.pack(pady=10)

        # Help Button
        help_button = tk.Button(
            button_frame,
            text="| Help |",
            font=("Lucida Calligraphy", 14),
            bg="#FFB5C1",
            fg="#8B0000",
            relief="flat",
            command=self.open_help_window,
        )
        help_button.pack(pady=10)

        # Exit Button
        exit_button = tk.Button(
            button_frame,
            text="| Exit |",
            font=("Lucida Calligraphy", 14),
            bg="#FFB5C1",
            fg="#8B0000",
            relief="flat",
            command=self.exit_app,
        )
        exit_button.pack(pady=10)

    def center_window(self, window, width, height):
        """Centers a window on the screen."""
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        window.geometry(f'{width}x{height}+{x}+{y}')

    def open_start_window(self):
        """Launch the app.py file in a new Python process and close current window."""
        try:
            script_path = os.path.abspath(r"app.py")
            if os.path.exists(script_path):
                # Open the script in the same directory
                subprocess.Popen([sys.executable, script_path], cwd=os.path.dirname(script_path))
                self.root.destroy()  # Close the current window
            else:
                tk.messagebox.showerror("Error", f"File not found: {script_path}")
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to open start window: {e}")

    def open_help_window(self):
        """Opens the Help Window."""
        self.root.withdraw()  # Hide the main window
        help_window = tk.Toplevel(self.root)
        help_window.title("Help Window")
        help_window.geometry("300x200")
        self.center_window(help_window, 300, 200)
        help_window.configure(bg="#FFC3C3")
        help_window.resizable(False, False)

        tk.Label(
            help_window,
            text="Help: Use 'Start' to begin\nand 'Exit' to close the app.",
            font=("Arial", 14),
            bg="#FFC3C3",
            fg="#8B0000",
            justify="center",
        ).pack(pady=50)

        # Add a close button for the help window
        close_button = tk.Button(
            help_window,
            text="Close Help",
            font=("Arial", 12),
            bg="#FFC3C3",
            fg="#8B0000",
            relief="flat",
            command=lambda: self.close_window(help_window),
        )
        close_button.pack(pady=10)

        help_window.protocol("WM_DELETE_WINDOW", lambda: self.close_window(help_window))

    def close_window(self, window):
        """Closes the given window and restores the main window."""
        window.destroy()
        self.root.deiconify()

    def exit_app(self):
        """Closes the application."""
        self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
