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

# %%
ds_train, ds_train_info = utils.getDataset('train')
ds_test, ds_test_info = utils.getDataset('test')
ds_eval, ds_eval_info = utils.getDataset('validation')

# %%
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_file
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential

# Retrieve the float model with pretrained weights and load it
model_file = get_file(
    "akidanet_imagenet_224_alpha_50.h5",
    "http://data.brainchip.com/models/akidanet/akidanet_imagenet_224_alpha_50.h5",
    cache_subdir='models/akidanet_imagenet')

model = Sequential()
model.add(load_model(model_file)) 
model.add(Dense(4, activation='relu'))


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
    epochs=5,
    validation_data=ds_test,
)

# %%

results = model.evaluate(
  ds_eval
)

print("test loss, test acc:", results)

model.save("models/imagenet_model")

# %% 
import cv2
import numpy as np

model = tf.keras.models.load_model("models/imagenet_model")

# %%
bbox = None
image = None

for ele in ds_eval.take(1):
  
  image = np.array(ele[0])
  bbox = np.array(ele[1])
  
  cv2.imshow("bounding_box", image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
  print(bbox)
  break



prediction = model.predict(
  image
)

# %%

for box in [bbox, prediction]:

  x = box[0]
  y = box[1]
  xx = box[2]
  yy = box[3]

  print("x ", x, "y ", y, "xx ", xx, "yy", yy)

  cv2.rectangle(image, (x, y), (xx, yy), (0, 0, 255), 2)
  print("x,y,w,h:",x,y,xx,yy)
  
# save resulting image
cv2.imwrite('example.jpg', image)      

# show thresh and result    
cv2.imshow("bounding_box", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
