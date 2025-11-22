# -------- IMPORTS --------
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.utils import image_dataset_from_directory
from keras import backend as K

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

# Training settings
batch_size = 16
epochs = 8
num_classes = 29

train_data_dir = 'asl_alphabet_train/asl_alphabet_train'
validation_data_dir = 'asl_alphabet_test'

img_width, img_height = 28, 28   # must match your model input
input_shape = (img_width, img_height, 1)  # grayscale


# -------- LOAD DATASETS USING IMAGE_DATASET_FROM_DIRECTORY --------
train_ds = image_dataset_from_directory(
    train_data_dir,
    labels="inferred",
    label_mode="categorical",
    color_mode="grayscale",       # important: model expects 1 channel
    image_size=(img_width, img_height),
    batch_size=batch_size,
    shuffle=True
)

val_ds = keras.utils.image_dataset_from_directory(
    validation_data_dir,
    labels="inferred",
    label_mode="categorical",
    color_mode="grayscale",
    image_size=(img_width, img_height),
    batch_size=batch_size,
    shuffle=False
)

# Normalize images (0–255 → 0–1)
train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
val_ds = val_ds.map(lambda x, y: (x / 255.0, y))


# -------- BUILD MODEL --------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# -------- EARLY STOPPING --------
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)


# -------- TRAIN MODEL --------
history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    callbacks=[early_stop]
)


# -------- PLOT --------
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend()
plt.title('Accuracy')
plt.show()


# -------- SAVE MODEL --------
model.save("visual_recognition_model.h5")
