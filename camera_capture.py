import cv2
import detect_hand as dh
import image_classification_testing as ict
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
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
        # Preprocessing image to match model input requirements
        img = cv2.resize(rectangle, (28, 28))

        # Convert to grayscale (model expects 1 channel)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Normalize
        img = img.astype("float32") / 255.0

        # Add channel dim (H, W, 1)
        img = np.expand_dims(img, axis=-1)

        # Add batch dim (1, H, W, 1)
        img = np.expand_dims(img, axis=0)
        

        ict.predict_input(img)
        

    cv2.imshow("HP Webcam", frame)
    

    
    # press c to close the camera window and exit the program
    if cv2.waitKey(1) & 0xFF == ord('c'):
        break

cap.release()
cv2.destroyAllWindows()