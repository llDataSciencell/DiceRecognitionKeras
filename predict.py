#!/usr/bin/env python
#TODO このプログラムを本番は使う。
import math

from keras.models import Model, Sequential,load_model
from keras.layers import Conv2D, Dense, Dropout, Flatten
import data_generator
import os
import cv2

NUM_EPOCHS = 50
BATCH_SIZE = 16
INPUT_SIZE=(128,128)

def train():

  model = load_model('../models/model_cnn_dice.h5')

  data_directory=os.environ['HOME']+"/DATA/NicoTechDice/DiceDataset/"
  dgf = data_generator.DataGeneratorFactory(data_dir=data_directory,
                                            input_shape=INPUT_SIZE,
                                            batch_size=BATCH_SIZE,
                                            num_channels=1
                                            )
  print("train_size:" + str(dgf.train_size))
  print("valid_size:"+str(dgf.valid_size))

  #https://keras.io/models/model/


  while True:
    x = cv2.imread(os.environ['HOME']+"/DATA/NicoTechDice/DiceDataset/train/1/00239.png",cv2.IMREAD_GRAYSCALE)
    result = model.predict_generator(dgf.pred_gen(x), steps=1)
    print("result:" + str(result))
    #TODO 返却値はこんな感じ　[[8.7736458e-01 7.4880212e-05 1.5932435e-09 4.3771642e-10 7.2609147e-15 1.4786409e-14]]

if __name__ == '__main__':
  train()