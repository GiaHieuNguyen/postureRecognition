import cv2
import mediapipe as mp
import math as m

# Initialize Mediapipe Pose class and drawing utilities
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Set angle calculation functions
def findDistance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def findAngle(x1, y1, x2, y2):
    """Calculate angle in degrees between two points and the vertical axis."""
    theta = m.atan2(y2 - y1, x2 - x1)
    return abs(theta * (180 / m.pi))

# Function to send alert for bad posture
def sendWarning():
    cv2.putText(frame, f"Warning: Bad posture detected! Please adjust your posture", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, red, 2)
    print("Warning: Bad posture detected! Please adjust your posture.")

# Initialize frame countersq
good_frames = 0
bad_frames = 0
fps_threshold = 180  # Set threshold for warning in seconds

# Colors
green = (127, 255, 0)
red = (50, 50, 255)

# Start video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with file path

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Get image dimensions
    h, w = frame.shape[:2]

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Check if landmarks were detected
    if results.pose_landmarks:
        # Get necessary landmarks
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
        
        cv2.putText(frame, f"Press Q to quit", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, red, 2)
        # Set posture feedback based on thresholds
        if neck_inclination < 118 and neck_inclination > 106 and torso_inclination < 85 and torso_inclination >80:
            good_frames += 1
            bad_frames = 0
            cv2.putText(frame, f"Good Posture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, green, 2)
        else:
            good_frames = 0
            bad_frames += 1
            cv2.putText(frame, f"Bad Posture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, red, 2)

        # Display neck and torso angles
        cv2.putText(frame, f"Neck Angle: {int(neck_inclination)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, green, 2)
        cv2.putText(frame, f"Torso Angle: {int(torso_inclination)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, green, 2)

        # Check time spent in bad posture
        if bad_frames > fps_threshold:
            sendWarning()  # Trigger warning if bad posture persists

        # Draw landmarks and connections
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the output
    cv2.imshow("Posture Detection(Beta)", frame)

    # Exit on 'q' key
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
