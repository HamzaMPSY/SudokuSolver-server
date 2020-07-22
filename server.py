from backtracing import backtracingSolver
from preprocessing import preprocesse
import numpy as np
import tensorflow as tf
from keras.models import model_from_json
from keras.backend import set_session
import os
import cv2
import flask
import werkzeug
import warnings
warnings.filterwarnings("ignore")

def loadModel():
	global graph,sess
	with graph.as_default():
		set_session(sess)
		json_file = open('leNet5.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		loaded_model.load_weights("leNet5.h5")
		print("Loaded model from disk")
		return loaded_model

sess = tf.Session()
graph = tf.get_default_graph()
model = None
test = "test"
app = flask.Flask(__name__)

@app.route('/solve', methods = ['POST'])
def solve():
	global graph,sess
	imagefile = flask.request.files['img']
	filename = werkzeug.utils.secure_filename(imagefile.filename)
	print("\n[!]Received image File name : " + imagefile.filename)
	imagefile.save(test +'/' +filename)

	img = cv2.imread(test +'/' +filename)
	with graph.as_default():
		set_session(sess)
		grid_digits,sudoku_clr,rois = preprocesse(img,model)
		print(grid_digits)
		grid_digits = backtracingSolver(grid_digits)
		

@app.before_first_request
def initialize():
	global model  
	# Here the server begin
	print("[*] Please wait until model is loaded")
	model = loadModel()
	print("[+] DL model is loaded")

app.run(host="0.0.0.0", port=8000, debug=True)