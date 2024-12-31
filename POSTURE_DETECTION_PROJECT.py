import cv2
import mediapipe as mp
import math as m
import tkinter as tk
from tkinter import messagebox, ttk
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Initialize Mediapipe Pose class and drawing utilities
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# History file
HISTORY_FILE = "posture_history.json"

# Colors
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
yellow = (0, 255, 255)

# Set angle and distance calculation functions
def findDistance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def findAngle(x1, y1, x2, y2):
    """Calculate angle in degrees between two points and the vertical axis."""
    theta = m.atan2(y2 - y1, x2 - x1)
    return abs(theta * (180 / m.pi))

# Function to send alert for bad posture
def sendWarning(frame):
    cv2.putText(frame, "Warning: Bad posture detected!", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, yellow, 2)
    print("Warning: Bad posture detected! Please adjust your posture.")

# Function to save session data
def save_session_data(duration, good_percentage):
    try:
        # Load existing history
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    except FileNotFoundError:
        history = []

    # Add new session
    history.append({"session_number": len(history) + 1, "duration": duration, "good_percentage": good_percentage})

    # Save updated history
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

# Function to load session history
def load_session_history():
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# Function to clear all session history
def clear_history():
    if messagebox.askyesno("Clear All History", "Are you sure you want to clear all session history?"):
        try:
            # Overwrite the history file with an empty list
            with open(HISTORY_FILE, "w") as f:
                json.dump([], f)
            messagebox.showinfo("Clear All History", "All session history has been cleared!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while clearing history: {e}")

# Function to display history
def display_history():
    history = load_session_history()
    if not history:
        messagebox.showinfo("History", "No session history available!")
        return

    # Create a new window for the history
    history_window = tk.Toplevel(root)
    history_window.title("Session History")
    history_window.geometry("400x400")

    # Create treeview to display history
    columns = ("Session", "Duration (s)", "Good Posture (%)")
    tree = ttk.Treeview(history_window, columns=columns, show="headings")

    tree.column("Session", width=80, minwidth=50, stretch=tk.NO)  
    tree.column("Duration (s)", width=120, minwidth=100, stretch=tk.NO)
    tree.column("Good Posture (%)", width=150, minwidth=120, stretch=tk.YES)

    tree.heading("Session", text="Session")
    tree.heading("Duration (s)", text="Duration (s)")
    tree.heading("Good Posture (%)", text="Good Posture (%)")
    tree.pack(fill=tk.BOTH, expand=True)
    
    # Insert history data into treeview
    for session in history:
        round_up = f"{session['good_percentage']:.2f}"
        tree.insert("", tk.END, values=(session["session_number"], session["duration"], round_up))

    # Button to plot the graph
    plot_button = tk.Button(history_window, text="Plot Graph", command=plot_graph)
    plot_button.pack(pady=10)

    # Button to clear all history
    clear_button = tk.Button(history_window, text="Clear All History", command=lambda: [clear_history(), history_window.destroy()])
    clear_button.pack(pady=10)

# Function to plot history graph
def plot_graph():
    history = load_session_history()
    if not history:
        messagebox.showinfo("History", "No session history available!")
        return

    session_numbers = [session["session_number"] for session in history]
    good_percentages = [session["good_percentage"] for session in history]

    # Plot graph using Matplotlib
    plt.figure(figsize=(8, 5))
    plt.plot(session_numbers, good_percentages, marker="o", linestyle="-", color="blue")
    plt.title("Good Posture Percentage Across Sessions")
    plt.xlabel("Session Number")
    plt.ylabel("Good Posture Percentage (%)")
    plt.grid()
    plt.show()    

# Function to calculate session summary
def calculate_summary(good_frames, total_frames):
    """Calculate percentages of good and bad posture."""
    good_percentage = (good_frames / total_frames) * 100 if total_frames > 0 else 0
    bad_percentage = 100 - good_percentage if total_frames > 0 else 0
    return good_percentage, bad_percentage

# Function to recognize posture
def start_posture_recognition():
    # Initialize counters
    good_frames = 0
    bad_frames = 0
    total_frames = 0
    score = 0
    fps_threshold = 180  # Bad posture threshold in frames
    session_start_time = datetime.now()

    # Start video capture
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with file path

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Get fps and frame dimensions
        fps = cap.get(cv2.CAP_PROP_FPS)
        h, w = frame.shape[:2]

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # Check if landmarks were detected
        if results.pose_landmarks:
            # Get landmarks
            landmarks = results.pose_landmarks.landmark
            l_shldr = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_shldr = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
            l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]

            # Calculate screen coordinates for points
            l_shldr_x, l_shldr_y = int(l_shldr.x * w), int(l_shldr.y * h)
            r_shldr_x, r_shldr_y = int(r_shldr.x * w), int(r_shldr.y * h)
            l_ear_x, l_ear_y = int(l_ear.x * w), int(l_ear.y * h)
            l_hip_x, l_hip_y = int(l_hip.x * w), int(l_hip.y * h)

            # Calculate neck and torso inclination
            neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
            torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

            # Measure shoulder distance (relative to image width)
            shoulder_distance = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
            relative_distance = shoulder_distance / w

            # Distance warning if too close 
            if relative_distance > 0.65:
                cv2.putText(frame, "Too Close to Camera!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, red, 2)

            # Update counters and feedback
            if neck_inclination < 121 and neck_inclination > 112 and torso_inclination < 85 and torso_inclination > 79:
                good_frames += 1
                score += 1
                bad_frames = 0
                cv2.putText(frame, f"Good Posture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, green, 2)
            else:
        
                bad_frames += 1
                cv2.putText(frame, f"Bad Posture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, red, 2)
            # Display angles and score
            cv2.putText(frame, f"Neck Angle: {int(neck_inclination)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, yellow, 2)
            cv2.putText(frame, f"Torso Angle: {int(torso_inclination)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, yellow, 2)
            cv2.putText(frame, f"Score: {score}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, green, 2)

            # Trigger warning if bad posture persists
            if bad_frames > fps_threshold:
                sendWarning(frame)

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        total_frames += 1

        print(f"Good Frames: {good_frames}, Total Frames: {total_frames}")

        cv2.putText(frame, "Press E to Exit", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, red, 2)
        cv2.imshow("Posture Detection (Beta)", frame)

        # Exit on 'e' key
        if cv2.waitKey(5) & 0xFF == ord('e'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Calculate session summary
    good_percentage, bad_percentage = calculate_summary(good_frames, total_frames)
    session_duration = (datetime.now() - session_start_time).seconds
    
    # Save session data
    save_session_data(session_duration, good_percentage)
    messagebox.showinfo(
        "Session Summary",
        f"Session Duration: {session_duration} seconds\n"
        f"Good Posture: {good_percentage:.2f}%\n"
        f"Bad Posture: {bad_percentage:.2f}%\n"
        f"Final Score: {score}"
    )

# Function to display instructions
def display_instructions():
    messagebox.showinfo(
        "Instructions",
        "1. Position yourself in front of the camera.\n"
        "2. Maintain good posture to score points.\n"
        "3. Stay within the frame and avoid being too close to the camera.\n"
        "4. Press 'E' to exit the session."
    )

# Function to quit the program
def quit_program():
    root.destroy()

# Tkinter UI menu
root = tk.Tk()
root.title("Posture Recognition")
root.geometry("400x400")

title = tk.Label(root, text="Posture Recognition Program", font=("Helvetica", 16, "bold"))
title.pack(pady=20)

start_button = tk.Button(root, text="Start", font=("Helvetica", 14), command=start_posture_recognition)
start_button.pack(pady=10)

instruction_button = tk.Button(root, text="Instructions", font=("Helvetica", 14), command=display_instructions)
instruction_button.pack(pady=10)

history_button = tk.Button(root, text="History", font=("Helvetica", 14), command=display_history)
history_button.pack(pady=10)

quit_button = tk.Button(root, text="Quit", font=("Helvetica", 14), command=quit_program)
quit_button.pack(pady=10)

root.mainloop()
