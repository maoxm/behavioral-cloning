from keras.layers import Input, Flatten, Dense, Dropout, Activation, ELU, Lambda
from keras.models import Model, Sequential
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.datasets import mnist


import matplotlib.image as mpimg
import csv
import numpy as np
import os
import json

LIMIT_TRAIN_NUM = 80000
RESIZE_FILE = "resize.json"
IN_FOLDER = "data"
IN_FILE = os.path.join(IN_FOLDER, "driving_log.csv")
USE_LEFT_RIGHT = 0 # Use 3 camera instead of 1
USE_GENERATOR = 0 # Flag to use generator 

G_COUNT = 0


# Preprocessing image, for generator
def pre_image(image_path, flip=False):
    global G_COUNT
    ex = mpimg.imread(image_path)
    G_COUNT += 1;
    # Done
    x = []
    x.append(ex)
    x = np.array(x)
    return x


def model3(input_shape):
  model = Sequential()
  model.add(Lambda(lambda x: x/255.0 - 0.5,
    input_shape=input_shape))
  model.add(Cropping2D(((70, 25), (0,0))))
  model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
  model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
  model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
  model.add(Convolution2D(64,3,3,activation="relu"))
  model.add(Convolution2D(64,3,3,activation="relu"))
  model.add(Flatten())
  model.add(Dense(100))
  model.add(Dense(50))
  model.add(Dense(10))
  model.add(Dense(1))
  return model


# model
def model1(input_shape):
  model = Sequential()
  model.add(Lambda(lambda x: x/255.0 - 0.5,
    input_shape=input_shape))
  print("After normalized", model.output_shape)
  model.add(Cropping2D(((70, 25), (0,0))))
  print("After crop", model.output_shape)
#  model.add(Convolution2D(3, 5, 5))
#  model.add(ELU())
#  model.add(Convolution2D(5, 5,5))
#  model.add(ELU())
  model.add(Convolution2D(16, 8, 8))
  model.add(ELU())
  model.add(Convolution2D(32, 5,5))
  model.add(ELU())
  model.add(Convolution2D(64, 5,5))
  model.add(Flatten())
  model.add(Dropout(0.2))
  model.add(ELU())
  #model.add(Activation("relu"))
  model.add(Dense(256))
#  model.add(Activation("relu"))
  model.add(Dropout(0.5))
  model.add(Dense(1))
  return model

def model2(input_shape):
  model = Sequential()
  model.add(Convolution2D(32, 3, 3, border_mode="same", input_shape=input_shape))
  model.add(MaxPooling2D())
  model.add(Dropout(0.5))
  model.add(Activation("relu"))
  model.add(Flatten())
  model.add(Dense(128))
  model.add(Activation("relu"))
  model.add(Dense(1))
  return model

def get_image_path(path):
  return os.path.join(IN_FOLDER, "IMG", get_file_name(path))

# Support both format in linux and windows
def get_file_name(path):
  words = path.split("/")
  if len(words) < 5: words = path.split("\\")
  if len(words) < 5: Exception("wrong path %s" % path)
  return words[-1]


class Auto(object):
  
  def __init__(self):
    self.gather()
    self.train()

  # Gather all the image path and label, but not creating image yet as image
  # takes a lot memory
  def gather(self):
    data = []
    with open(IN_FILE) as fin:
        csv_data = csv.reader(fin, delimiter=',')
        for i, row in enumerate(csv_data):
          if i == 0: continue # Remove header
          steering_adjust = [0, 0.08, -0.08]
          steering = float(row[3])
          if USE_LEFT_RIGHT: total = 3
          else: total = 1
          for k in range(total):
            path = row[k]
            source = get_image_path(path)
            if not os.path.exists(source):
                Exception("source not exist %s" % source)
            data.append((source, steering + steering_adjust[k]))
          if i > LIMIT_TRAIN_NUM: break
##    data= shuffle(data)
    if USE_GENERATOR:
      self.data_t, self.data_v = train_test_split(
	    data, test_size=0.33, random_state=0)
      print("Data train valid:", len(data), len(self.data_t), len(self.data_v))
    else:
      self.data_t = data


  # For generator
  def gen_data(self, data, msg):
    i = 0
    l = len(data)
    while True:
        y = []
        y.append(data[i][1])
        y = np.array(y)
        yield pre_image(data[i][0]), y
        yield pre_image(data[i][0], True), y
        i += 1
        i %= l
        

  # For whole training image and label generation
  def create_train_data(self):
      Xt = []
      yt = []
      for each in self.data_t:
          image_data = mpimg.imread(each[0])
          Xt.append(image_data)
          yt.append(each[1])
          Xt.append(np.fliplr(image_data))
          yt.append(-each[1])
      return np.array(Xt), np.array(yt)
      

  def train(self):
    input_shape = (160, 320, 3)
    print("input", input_shape)
    model = model3(input_shape)
    model.compile('adam', 'mean_squared_error')
    if USE_GENERATOR:
      model.fit_generator(
	self.gen_data(self.data_t, "train"), 
	samples_per_epoch=3,
	nb_epoch=1,
        #validation_data=self.gen_data(self.data_v, "valid"),
        #nb_worker=16,
	nb_val_samples=1
      )
    else:
      # Train the whole data on big machine
      Xt, yt = self.create_train_data()
      history = model.fit(Xt, yt, batch_size=512, nb_epoch=1, validation_split=0.1, shuffle=True)
    model.save("model.h5")
  
  

def main():
  auto = Auto()

if __name__ == "__main__":
    main()
