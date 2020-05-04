from flask import Flask, render_template, request, flash, redirect
from flask_uploads import UploadSet, configure_uploads,IMAGES
from flask_session import Session

from tempfile import mkdtemp

from keras.preprocessing import image
from tensorflow.python.keras.backend import set_session
#from scipy.isc import imsave, imread, imresize
import numpy as np
import keras.models
import re
import sys
import os
import base64
import cv2
sys.path.append(os.path.abspath("./model_№1"))
from load import *

global graph, model, sess

model, graph = init()

app = Flask(__name__)
photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = '.'
configure_uploads(app, photos)

# Configure session to use filesystem (instead of signed cookies)
# app.config["SESSION_FILE_DIR"] = mkdtemp()
# app.config["SESSION_PERMANENT"] = False
# app.config["SESSION_TYPE"] = "filesystem"
# Session(app)

#depends on CNN
ROWS = 224
COLS = 224

# for vgg16
def read_and_prep_data(image_path):
    data = []
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (ROWS,COLS), interpolation=cv2.INTER_CUBIC)
    data.append(np.array(image))
    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST' and request.files['photo'].filename:  #'photo' in request.files:
        filename = photos.save(request.files['photo'])
        img_path = './'+filename
        img = image.load_img(img_path, target_size=(ROWS, COLS), color_mode="grayscale")#remove grayscale for color
        x = image.img_to_array(img)/255.0 #with normalization
        x = np.expand_dims(x, axis=0)
        #print(x.shape)
        with graph.as_default():
            #use global session
            set_session(sess)
            prediction = model.predict(x)
            #print(prediction)
            # for binary classification
            if prediction > 0.5:
                probability = prediction[0][0] * 100 # '{0:.5%}'.format(prediction[0][0])
                diagnosis = 'Pneumonia'
            else:
                probability = (1-prediction[0][0]) * 100
                diagnosis = 'Normal'
            # for multi classification
            # if prediction[0] >= prediction[1]:
            #     probability = prediction[0] * 100
            #     diagnosis = 'Normal'
            # else:
            #     probability = prediction[1] * 100
            #     diagnosis = 'Pneumonia'
            return render_template("results.html", probability=probability, diagnosis=diagnosis)
    if not request.files['photo'].filename:
        flash("Select Image First!")
        return redirect("/")

if __name__ == "__main__":
	#decide what port to run the app in
	#port = int(os.environ.get('PORT', 8000))
	#run the app locally on the givn port
	#app.run(host='0.0.0.0', port=port)
	#optional if we want to run in debugging mode
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'

    #app.debug = True
    app.run(port=8000)
	#app.run(debug=True, port=8000)