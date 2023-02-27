# %%
# in order to install the required packages for the snn conversion, 
# you need the following packages
# furthemore you must have installed miniconda or anaconda
# and activated a virtual environment to execute the following commands

# %conda install akida
# %conda install cnn2snn
# %conda install akida-models

# %%
import tensorflow_datasets as tfds
import tensorflow as tf
import pandas as pd
from tensorflow.image import resize
import numpy as np
import utils

# %%

input_shapes = [
(224,224,3),
(112,112,3),
(56,56,3),
(28,28,3),
(14,14,3),
(7,7,3),
(4,)]

  

# %%
ds_train, ds_train_info = utils.getDataset('train')
ds_test, ds_test_info = utils.getDataset('test')
ds_eval, ds_eval_info = utils.getDataset('validation')

# %%
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_file
from tensorflow.keras.layers import Dense, Flatten, ReLU, Conv2D, BatchNormalization
from tensorflow.keras import Sequential
import tensorflow as tf

model = Sequential()

for input_shape in input_shapes:
  
  model.add(Conv2D(2,7, input_shape=input_shape)) 
  model.add(BatchNormalization(input_shape=input_shape)) 
  model.add(ReLU(input_shape=input_shape))

model.add(Flatten())
model.add(Dense(4))

# %%
model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error"), 
              metrics=['accuracy'])
model.get_layer(index=-1)


cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="./imagenet_models",
                                                 save_weights_only=True,
                                                 verbose=1)

# %%
history = model.fit(
    ds_train,
    epochs=100,
    validation_data=ds_test,
)

# %%
model.evaluate(
    ds_eval
)