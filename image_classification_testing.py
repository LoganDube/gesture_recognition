# -------- IMPORTS --------
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.utils import to_categorical


# Mapping for ASL MNIST (A–Y, excluding J and Z)
asl_map = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
}

# -------- SETTINGS --------
model_path = "visual_recognition_model.h5"
test_csv_path = "sign_mnist_dataset/sign_mnist_test.csv"
img_width, img_height = 28, 28  # MNIST image size

# -------- LOAD MODEL --------
model = load_model(model_path)
print("Model loaded successfully!")

# -------- LOAD AND PREPROCESS TEST DATA --------
test_df = pd.read_csv(test_csv_path)
y_test = test_df['label'].values
X_test = test_df.drop('label', axis=1).values

# Reshape and normalize
X_test = X_test.reshape(-1, img_width, img_height, 1).astype('float32') / 255.0

# One-hot encode labels
num_classes = 25
y_test_categorical = to_categorical(y_test, num_classes)


# -------- EVALUATE MODEL --------
loss, accuracy = model.evaluate(X_test, y_test_categorical, verbose=1)


# -------- OPTIONAL: PREDICT SAMPLE --------


sample_index = 91 # Change to test different samples
sample_image = X_test[sample_index].reshape(1, img_width, img_height, 1)
predicted_class = np.argmax(model.predict(sample_image), axis=1)[0]
true_class = y_test[sample_index]


  
if predicted_class == true_class:
    result = "Correct"
else:
    result = "Incorrect"
    
print(result)
    
# Showing the processes hand frame image in grayscale (used in testing)
plt.imshow(sample_image.reshape(28, 28)) #plt.imshow() requires a 2D array, so we reshape the input
plt.show()
        



def predict_input(frame):
    # making prediction
    label = round(np.argmax((model.predict(frame))))

    # Make prediction
    print("Predicted probabilities:", label)
    print("predicted letter:", asl_map[label])
    
# -------- LOAD AND PREPROCESS TEST DATA --------