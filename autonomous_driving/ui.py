import customtkinter as ctk
from tkinter import messagebox

# Window setup
app = ctk.CTk()
app.title("Autonomous Driving System")
app.geometry("900x600")

# Title
title = ctk.CTkLabel(app, text="Autonomous Driving Dashboard", font=("Arial", 24))
title.pack(pady=20)

# Functions
def start_system():
    messagebox.showinfo("Start", "System Started")

def stop_system():
    messagebox.showinfo("Stop", "System Stopped")

def lane_detection():
    messagebox.showinfo("Lane Detection", "Lane Detection Activated")

def sign_recognition():
    messagebox.showinfo("Sign Recognition", "Sign Recognition Activated")

def obstacle_detection():
    messagebox.showinfo("Obstacle Detection", "YOLO Obstacle Detection Running")

# Buttons
btn1 = ctk.CTkButton(app, text="Start System", command=start_system)
btn1.pack(pady=10)

btn2 = ctk.CTkButton(app, text="Stop System", command=stop_system)
btn2.pack(pady=10)

btn3 = ctk.CTkButton(app, text="Lane Detection", command=lane_detection)
btn3.pack(pady=10)

btn4 = ctk.CTkButton(app, text="Sign Recognition", command=sign_recognition)
btn4.pack(pady=10)

btn5 = ctk.CTkButton(app, text="Obstacle Detection (YOLO)", command=obstacle_detection)
btn5.pack(pady=10)

# Run app
app.mainloop()