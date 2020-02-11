# 参考資料: https://github.com/keras-team/keras/tree/master/examples

import math
import pandas as pd
from keras.models import Model, Sequential,load_model
from keras.optimizers import Adam
from keras.layers import Conv2D, Dense, Dropout, Flatten, Convolution1D
import data_generator # DataGeneratorFactory クラスが記述されているファイルをimport
import os
from keras.callbacks import EarlyStopping,ModelCheckpoint
NUM_EPOCHS = 20 #TODO エポック数を必要に応じて増やす
BATCH_SIZE = 16
INPUT_SHAPE=(64,64)

def get_model():
  model = Sequential()
  model.add(Conv2D(12, (3, 3), activation='relu',
                   input_shape=(64, 64, 1)))
  model.add(Conv2D(24, (3, 3), activation='relu'))
  model.add(Conv2D(48, (3, 3), activation='relu'))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dense(6, activation='sigmoid'))
  return model

def train():
  model = get_model()
  model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999),
                metrics=['accuracy'])

  data_directory=os.environ['HOME']+"/DATA/NicoTechDice/DiceDataset/"
  dgf = data_generator.DataGeneratorFactory(data_dir=data_directory,
                                            input_shape=INPUT_SHAPE,
                                            batch_size=BATCH_SIZE,
                                            num_channels=1
                                            )
  print("train_size:" + str(dgf.train_size))
  print("valid_size:"+str(dgf.valid_size))

  early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
  checkpoint_callback = ModelCheckpoint('../models/model_cnn_temp_20200211.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                        mode='min')
  
  history = model.fit_generator(
      dgf.train_generator,
      steps_per_epoch=math.ceil(dgf.train_size / BATCH_SIZE),
      validation_data=dgf.valid_generator,
      validation_steps=math.ceil(dgf.valid_size / BATCH_SIZE),
      epochs=NUM_EPOCHS, verbose=1,callbacks=[early_stopping_callback])#,checkpoint_callback])
  
  model.save('../models/model_cnn_dice_20200211.h5')
  #model = load_model('../models/model_cnn_dice.h5')
  print(history)

  metrics = model.evaluate_generator(dgf.test_generator, steps=dgf.test_size)
  print("Evaluate Accuracy for test dataset...")
  for i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[i]
    metric_value = metrics[i]
    print('%s: %s' % (metric_name, metric_value))
  print("=================================")

  #RGB画像を直接モデルに与えて推論を行い、評価する  (https://keras.io/models/model/)
  """
  for i in range(100):
    result=model.predict_generator(dgf.pred_gen_using_image(image),steps=1)
    print("result:"+str(result))
  """
if __name__ == '__main__':
  train()
