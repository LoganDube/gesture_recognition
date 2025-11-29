FROM tensorflow/tensorflow:2.17.0

WORKDIR /gesture_recognition_mnist

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy your app
COPY camera_capture.py .

ENV MODEL=/gesture_recognition_mnist/visual_recognition_model.hdf5

CMD ["python", "camera_capture.py"]