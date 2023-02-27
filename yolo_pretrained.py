# %%
# in order to install the required packages for the snn conversion, 
# you need the following packages
# furthemore you must have installed miniconda or anaconda
# and activated a virtual environment to execute the following commands

# %conda install akida
# %conda install cnn2snn
# %conda install akida-models

# %%
import tensorflow as tf
import utils
from tensorflow.keras.layers import Reshape
from tensorflow.keras import Model

# %%
num_anchors = 5
grid_size = (7, 7)
classes = 1

ds_train, ds_train_info = utils.getDataset('train')
ds_test, ds_test_info = utils.getDataset('test')
ds_eval, ds_eval_info = utils.getDataset('validation')

# %%
from akida_models import yolo_widerface_pretrained

model, anchor = yolo_widerface_pretrained()

output = Reshape((grid_size[1], grid_size[0], num_anchors, 4 + 1 + classes),
                 name="YOLO_output")(model.output)

# %%
model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error"), 
              metrics=['accuracy'])

# Build the complete model
model = Model(model.input, output)
model.output

# %%
model.evaluate(
    ds_eval
)