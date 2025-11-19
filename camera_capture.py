import cv2
from matplotlib import pyplot as plt
import detect_hand as dh
import image_classification_testing as ict
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

TEXT_COLOR = (0, 255, 0)  # green

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # send frame pixels to model for prediction here
    detect_hand = dh.detect_hand(frame)
    if (detect_hand != None):
        start_point, end_point = detect_hand
        rectangle = cv2.rectangle(frame, start_point, end_point, TEXT_COLOR, 2)
        # Extract coordinates ensuring correct ordering for array slicing
        if (start_point[0] > 0 and end_point[0] > 0 and start_point[1] > 0 and end_point[1] > 0): # if all values on the frame
            x_min, x_max = min(start_point[0], end_point[0]), max(start_point[0], end_point[0])
            y_min, y_max = min(start_point[1], end_point[1]), max(start_point[1], end_point[1])

            # Clamp coordinates to frame bounds
            h, w = frame.shape[:2]
            x_min = max(0, min(x_min, w - 1))
            x_max = max(0, min(x_max, w))
            y_min = max(0, min(y_min, h - 1))
            y_max = max(0, min(y_max, h))

            # Make sure we have a valid region
            if x_max > x_min and y_max > y_min:
                hand_frame = frame[y_min:y_max, x_min:x_max]

                # Create a copy and resize for display (keep aspect ratio)
                display_w = 300
                aspect = hand_frame.shape[1] / float(hand_frame.shape[0]) if hand_frame.shape[0] != 0 else 1
                display_h = int(display_w / aspect) if aspect != 0 else display_w
                hand_display = cv2.resize(hand_frame, (display_w, display_h))

                # Show the extracted hand frame in a separate window
                cv2.imshow('Hand View', hand_display)
                
            
                # handle preprocessing of the smaller frame s.t it can be fed into the model
                hand_frame = cv2.resize(hand_frame, (28, 28))
                hand_frame = cv2.cvtColor(hand_frame, cv2.COLOR_BGR2GRAY)
                hand_frame = np.array(hand_frame)
                hand_frame = hand_frame.reshape(-1, 28, 28, 1).astype('float32') / 255.0
                print(hand_frame.shape)
    

                ict.predict_input(hand_frame)
                
                # Showing the processes hand frame image in grayscale (used in testing)
                # plt.imshow(hand_frame.reshape(28, 28)) #plt.imshow() requires a 2D array, so we reshape the input
                # plt.show()
        
    cv2.imshow('Camera Capture', frame)
    
    
    # press c to close the camera window and exit the program
    if cv2.waitKey(1) & 0xFF == ord('c'):
        break

cap.release()
cv2.destroyAllWindows()