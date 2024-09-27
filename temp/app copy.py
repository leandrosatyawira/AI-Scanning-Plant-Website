from flask import Flask, render_template, request, Response, jsonify
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import os
import base64
import re
from PIL import Image
import io

# Define a flask app
app = Flask(__name__)

# Load your trained model
model = load_model('plant.h5')

def predict_leaf_disease(img, model):
    x = cv2.resize(img, (256, 256))
    x = x / 255.0  # Normalize the image
    x = np.expand_dims(x, axis=0)

    # Make predictions
    predictions = model.predict(x)
    class_indices = {0: "Pepper bell | Disease: Bacterial spot",
                     1: "Pepper bell | No Disease: Healthy",
                     2: "Potato | Disease: Early blight",
                     3: "Potato | Disease: Late blight",
                     4: "Potato | No Disease: Healthy",
                     5: "Tomato | Disease: Bacterial spot",
                     6: "Tomato | Disease: Early blight",
                     7: "Tomato | Disease: Late blight",
                     8: "Tomato | Disease: Leaf Mold",
                     9: "Tomato | Disease: Septoria leaf spot",
                     10: "Tomato | Disease: Spider mites Two spotted spider mite",
                     11: "Tomato | Disease: Target Spot",
                     12: "Tomato | Disease: Yellow Leaf Curl Virus",
                     13: "Tomato | Disease: Mosaic Virus",
                     14: "Tomato | No Disease: Healthy"}

    predicted_class_index = np.argmax(predictions)
    predicted_class = class_indices[predicted_class_index]

    return predicted_class

def gen():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            result = predict_leaf_disease(frame, model)
            yield (b'--frame\r\n'
                   b'Content-Type: text/plain\r\n\r\n' + result.encode() + b'\r\n\r\n')

        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/register')
def login():
    return render_template('register.html')
@app.route('/scan')
def scan(): 
    return render_template('scan.html')
@app.route('/aboutus')
def scan():
    return render_template('aboutus.html')
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Convert the data URL to a numpy array
    img_data = re.sub('^data:image/.+;base64,', '', data['image'])
    img_data = base64.b64decode(img_data)
    img_data = Image.open(io.BytesIO(img_data))
    img_data = np.array(img_data)

    # Make a prediction using your model
    prediction = predict_leaf_disease(img_data, model)

    return jsonify({ 'prediction': prediction })

if __name__ == '__main__':
    app.run(port=5001,debug=True)
