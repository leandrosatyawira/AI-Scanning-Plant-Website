
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as nps
#import tensorflow as tf
import tensorflow as tf

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*2)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Keras
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'plant.h5' 

# Load your trained model
model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(256, 256))

    # Preprocessing the image
    x = image.img_to_array(img) # [25*256]
    ## x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = nps.expand_dims(x, axis=0)  # np 
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x)

    preds = model.predict(x)
    preds=nps.argmax(preds, axis=1) # np
    if preds==0:
        preds="Plant Name : Pepper bell | Disease : Bacterial spot"
    elif preds==1:
        preds="Plant Name : Pepper bell| No Disease : healthy"
    elif preds==2:
        preds="Plant Name : Potato | Disease : Early blight"
    elif preds==3:
        preds="Plant Name : Potato | Disease : Late blight"
    elif preds==4:
        preds="Plant Name : Potato | No Disease : healthy"
    elif preds==5:
        preds="Plant Name : Tomato | Disease : Bacterial spot"
    elif preds==6:
        preds="Plant Name : Tomato | Disease : Early blight"
    elif preds==7:
        preds="Plant Name : Tomato | Disease : Late blight"
    elif preds==8:
        preds="Plant Name : Tomato | Disease : Leaf Mold"
    elif preds==9:
        preds="Plant Name : Tomato | Disease : Septoria leaf spot"
    elif preds==10:
        preds="Plant Name : Tomato | Disease : Spider mites Two spotted spider mite"
    elif preds==12:
        preds="Plant Name : Tomato | Disease : Target Spot"
    elif preds==13:
        preds="Plant Name : Tomato | Disease : Yellow Leaf Curl Virus"
    elif preds==14:
        preds="Plant Name : Tomato | Disease : Mosaic Virus"
    else:
        preds ="Plant Name : Tomato | No Disease : healthy"
    
    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the file from the form
        f = request.files['file']

        if f:
            # Save the file to a temporary location
            upload_folder = 'uploads'  # Directory to save uploaded files
            os.makedirs(upload_folder, exist_ok=True)  # Ensure the directory exists
            temp_path = os.path.join(upload_folder, secure_filename(f.filename))
            f.save(temp_path)

            # Make a prediction using your model
            prediction = model_predict(temp_path, model)

            # Remove the temporary file
            os.remove(temp_path)

            return render_template('index.html', prediction=prediction)
    
    # If no file was uploaded or an error occurred, return to the homepage
    return redirect('/')
if __name__ == '__main__':
    app.run(port=5001,debug=True)
