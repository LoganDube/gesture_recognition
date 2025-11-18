# Import libraries
from typing import Tuple, Union
import cv2
import mediapipe as mp
import numpy as np
import math



# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2
)

def define_box(midpoint, frame):
    # Get frame dimensions
    h, w = frame.shape[:2]
    
    # Convert normalized coordinates to pixel coordinates
    mid_x = int(midpoint[0] * w)
    mid_y = int(midpoint[1] * h)
    
    # Define box corners (pixel offsets)
    offset = 150  # pixels
    start_point = (mid_x - offset, mid_y + offset)  # top-left corner
    end_point = (mid_x + offset, mid_y - offset)    # bottom-right corner
    

    return start_point, end_point
    

def detect_hand(frame):
    global frame_width, frame_height
    
    
    

    # Convert BGR image to RGB
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the RGB image
    results = hands.process(frameRGB)

    # If hands are detected
    if results.multi_hand_landmarks:
        # Extract x coordinate from first landmark of first hand
        midpoint = results.multi_hand_landmarks[0].landmark[0].x, results.multi_hand_landmarks[0].landmark[0].y - 0.2 # gives x and y coordinates of midpoint
        start_point, end_point = define_box(midpoint, frame)
        return start_point, end_point
        
