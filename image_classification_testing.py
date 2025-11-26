# -------- IMPORTS --------
from matplotlib import pyplot as plt
from keras.preprocessing.image import load_img
import numpy as np
import pandas as pd
from keras.models import load_model



# Mapping for ASL MNIST (A–Y, excluding J and Z)
asl_map = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: 'SPACE', 27: 'DEL', 28: 'NOTHING'
}

# -------- SETTINGS --------
model_path = "visual_recognition_model.h5"
test_csv_path = "asl_alphabet_test"
img_width, img_height = 28, 28  # MNIST image size

# -------- LOAD MODEL --------
model = load_model(model_path)
print("Model loaded successfully!")

# One-hot encode labels
num_classes = 29

image = load_img(
    f'{test_csv_path}/A/A_test.jpg',
    target_size=(28, 28),
    color_mode='grayscale'
)

img = np.array(image).astype("float32") / 255.0
img = img.reshape(1, 28, 28, 1)

predicted_probs = model.predict(img, verbose=0)
predicted_label = np.argmax(predicted_probs)


    
# Showing the processes hand frame image in grayscale (used in testing)
# plt.imshow(sample_image.reshape(28, 28)) #plt.imshow() requires a 2D array, so we reshape the input
# plt.show()


def predict_input(frame):
    # making prediction
    prediction = model.predict(frame, verbose=0)
    label = np.argmax(prediction)
    predicted_probability = prediction[0][label] 
    # Make prediction
    if predicted_probability > 0.99:
        return "predicted letter:", asl_map[label]
    else:
        return "Nothing"
    
# -------- LOAD AND PREPROCESS TEST DATA --------