# -------- IMPORTS --------
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.utils import to_categorical

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

# Training settings
batch_size = 128
num_classes = 24
epochs = 10

train_df = pd.read_csv('sign_mnist_dataset/sign_mnist_train.csv')
test_df  = pd.read_csv('sign_mnist_dataset/sign_mnist_test.csv')

# -------------------------
# SPLIT INTO FEATURES AND LABELS
# -------------------------
y_train = train_df['label'].values
X_train = train_df.drop('label', axis=1).values

y_test = test_df['label'].values
X_test = test_df.drop('label', axis=1).values

# -------------------------
# RESHAPE + NORMALIZE
# -------------------------
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test  = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# -------------------------
# ONE-HOT ENCODE LABELS
# -------------------------
num_classes = 26
y_train = to_categorical(y_train, num_classes)
y_test  = to_categorical(y_test, num_classes)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
# -------- BUILD MODEL (same architecture as before) --------

model = Sequential([
    
    Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.20),

    Dense(num_classes, activation='softmax')
]) 

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# -------- EARLY STOPPING CALLBACK -------- occurs when val_loss stops decreasing
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# -------- TRAIN MODEL --------
history = model.fit(X_train, y_train,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=(X_test, y_test),
          callbacks=[early_stop]
          )



plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend()
plt.title('Accuracy')
plt.show()

# Save entire model (better than save_weights)
model.save("visual_recognition_model.h5") # .h5 has hdf5 formatting, used for storing large data and models

