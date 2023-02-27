import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.image import resize

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""

  print("\n\n\n\n",label,"\n\n",len(label),"\n\n")

  return tf.cast(resize(image, [224,224]),tf.float32) / 255., label

def dataFrameMap(x):
  
  x["faces"]["bbox"] = tf.convert_to_tensor([.0,.0,.0,.0], tf.float32) if len(x["faces"]["bbox"]) == 0 else x["faces"]["bbox"]
  
  return x

def getDataset(name : str) :
  ds, ds_info = tfds.load('wider_face', split=name, shuffle_files=True, with_info=True)
  ds = ds.filter(lambda x: len(x["faces"]["bbox"]) == 1) 
  #ds = ds.map(dataFrameMap) 
  
  ds = ds.map(lambda x : normalize_img(image=x["image"], label=x["faces"]["bbox"]), num_parallel_calls=tf.data.AUTOTUNE)
  ds = ds.cache()
  ds = ds.shuffle(ds_info.splits[name].num_examples)
  ds = ds.batch(128)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  
  return ds, ds_info