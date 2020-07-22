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
		grid_digits,sudoku_clr,rois,sud_coords,full_coords,width,height,cn_img = preprocesse(img,model)
		print(grid_digits)
		grid_digits1 = grid_digits.copy()
		solution = backtracingSolver(grid_digits)
	grid_digits  = grid_digits1.reshape((81,))
	solution  = solution.reshape((81,))
	for e in range(81):
		if grid_digits[e]!=0:
			continue
		sudoku_clr=cv2.putText(sudoku_clr, str(solution[e]), ((rois[e][2]+rois[e][3])//2, rois[e][1]),cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), thickness=2)
	# Now define the homography map and apply warp perspective
	# to fit the top-down sudoku back on our frame.
	h, mask = cv2.findHomography(sud_coords, full_coords)
	im_out = cv2.warpPerspective(sudoku_clr, h, (width, height))

	final_im = im_out + cn_img
	final_im = cv2.resize(final_im,(720,720))
	cv2.imshow('solution',final_im)
	cv2.waitKey(0)
	return "OK lol!"
		

@app.before_first_request
def initialize():
	global model  
	# Here the server begin
	print("[*] Please wait until model is loaded")
	model = loadModel()
	print("[+] DL model is loaded")

app.run(host="0.0.0.0", port=8000, debug=True)