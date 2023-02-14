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
ds_train, ds_train_info = tfds.load('wider_face', split='train', shuffle_files=True, with_info=True)
ds_train = ds_train.filter(lambda x: len(x["faces"]["bbox"]) <= 1) 
ds_test, ds_test_info = tfds.load('wider_face', split='test', shuffle_files=True, with_info=True)
ds_test = ds_test.filter(lambda x: len(x["faces"]["bbox"]) <= 1)


# %%
from tensorflow.image import resize
import numpy as np

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""

  print("\n\n\n\n",label,"\n\n",len(label),"\n\n")

  return tf.cast(resize(image, [224,224]),tf.float32) / 255., label

# %%
ds_train = ds_train.map(lambda x : normalize_img(image=x["image"], label=x["faces"]["bbox"]), num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_train_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

# %%
ds_test = ds_test.map(lambda x : normalize_img(image=x["image"], label=x["faces"]["bbox"]), num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.cache()
ds_test = ds_test.shuffle(ds_test_info.splits['test'].num_examples)
ds_test = ds_test.batch(128)
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

# %%
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_file
from tensorflow.keras.layers import Dense, Flatten, ReLU, Conv2D, BatchNormalization
from tensorflow.keras import Sequential
import tensorflow as tf

model = Sequential()

for input_shape in input_shapes:
  
  print(input_shape)
  
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
model.fit(
    ds_train,
    epochs=2,
    validation_data=ds_test,
)