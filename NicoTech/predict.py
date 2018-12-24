#!/usr/bin/env python
#TODO このプログラムを本番は使う。
import math

from keras.models import Model, Sequential,load_model
from keras.layers import Conv2D, Dense, Dropout, Flatten
import data_generator
import os
import cv2
import numpy as np

NUM_EPOCHS = 50
BATCH_SIZE = 16
INPUT_SIZE=(128,128)

def train():

  model = load_model('/Users/user/DATA/NicoTech/model_cnn_temp.h5')
  dgf = data_generator.DataGeneratorFactory(input_shape=INPUT_SIZE,
                                            batch_size=BATCH_SIZE,
                                            num_channels=1
                                            )

  while True:
    x = cv2.imread(os.environ['HOME']+"/DATA/NicoTech/DiceDataset/1/00135.bmp",cv2.IMREAD_GRAYSCALE)
    result = model.predict_generator(dgf.pred_gen(x), steps=1)
    print("result:" + str(result))

    print(np.argmax(result[0]))
    #[[3.1851310e-01 5.8983585e-10 1.9802104e-18 6.2612086e-23 4.6781291e-26,2.5079543e-31]]
    #TODO 返却値はこんな感じ　[[8.7736458e-01 7.4880212e-05 1.5932435e-09 4.3771642e-10 7.2609147e-15 1.4786409e-14]]

if __name__ == '__main__':
  train()