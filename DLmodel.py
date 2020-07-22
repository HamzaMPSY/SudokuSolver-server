import tensorflow as tf
import numpy as np
import os 
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout,MaxPooling2D,AveragePooling2D
from keras.models import Model,Sequential


image_width = 28
image_height = 28

def build(width, height, depth, classes):
	model = Sequential()
	input_shape = (height, width, depth)

	model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
	model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(units=128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(units=classes, activation = 'softmax'))

	return model

batch_size = 64
datagen = ImageDataGenerator(rescale=1. / 255,validation_split=0.2)
train_data = datagen.flow_from_directory('dataset',target_size=(image_width,image_height),color_mode='grayscale',batch_size=batch_size,class_mode='categorical',subset='training')
val_data  = datagen.flow_from_directory('dataset',target_size=(image_width,image_height),color_mode='grayscale',batch_size=batch_size,class_mode='categorical',subset='validation')


model = build(image_width,image_height,1,9)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit_generator(train_data,epochs=15,validation_data =val_data)

# serialize model to JSON
model_json = model.to_json()
with open("leNet5.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("leNet5.h5")
print("Saved model to disk")