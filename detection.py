from keras.models import load_model
from flask_debugtoolbar import DebugToolbarExtension
from flask import Flask, render_template, request,jsonify
from flask_ngrok import run_with_ngrok

from PIL import Image
import io
import re
import cv2
import numpy as np
import base64


img_size=100

app = Flask(__name__, template_folder='/content/flask_webapp/templates') 
run_with_ngrok(app)
app.debug = True

model=load_model('/content/drive/MyDrive/model-016.model')

labelss={0:'Covid-19 Negative', 1:'Covid-19 Positive'}

def preprocess(img):

	img=np.array(img)

	if(img.ndim==3):
		grayscale=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	else:
		grayscale=img

	grayscale=grayscale/255
	resiziee=cv2.resize(grayscale,(img_size,img_size))
	reshaped=resiziee.reshape(1,img_size,img_size)
	return reshaped

@app.route("/")
def index():
	return(render_template("index.html"))
 
@app.route("/predict", methods=["POST"])
def predict():
	print('HERE')
	msg = request.get_json(force=True)
	encod = msg['image']
	decod = base64.b64decode(encod)
	dataBytesIO=io.BytesIO(decod)
	dataBytesIO.seek(0)
	image = Image.open(dataBytesIO)

	test_image=preprocess(image)

	predictions = model.predict(test_image)
	results=np.argmax(predictions,axis=1)[0]
	acc=float(np.max(predictions,axis=1)[0])

	label=labelss[results]

	print(predictions,results,acc)

	response = {'predictions': {'results': label,'accuracy': acc}}

	return jsonify(response)
 
app.run()