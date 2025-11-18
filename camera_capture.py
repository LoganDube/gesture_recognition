import cv2
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
    
    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # send frame pixels to model for prediction here
    detect_hand = dh.detect_hand(frame)
    if (detect_hand != None):
        start_point, end_point = detect_hand
        rectangle = cv2.rectangle(frame, start_point, end_point, TEXT_COLOR, 2)
        # Extract coordinates ensuring correct ordering for array slicing
        x_min, x_max = min(start_point[0], end_point[0]), max(start_point[0], end_point[0])
        y_min, y_max = min(start_point[1], end_point[1]), max(start_point[1], end_point[1])
        hand_frame = frame[y_min:y_max, x_min:x_max]
        
        # handle preprocessing of the smaller frame s.t it can be fed into the model
        hand_frame = cv2.resize(hand_frame, (28, 28))
        hand_frame = cv2.cvtColor(hand_frame, cv2.COLOR_BGR2GRAY)
        hand_frame = hand_frame.astype('float32') / 255.0
        hand_frame = np.expand_dims(hand_frame, axis=-1)  # add channel dimension
        hand_frame = np.expand_dims(hand_frame, axis=0)   # add batch dimension
        
        ict.predict_input(hand_frame)
        

    cv2.imshow("HP Webcam", frame)
    

    
    # press c to close the camera window and exit the program
    if cv2.waitKey(1) & 0xFF == ord('c'):
        break

cap.release()
cv2.destroyAllWindows()