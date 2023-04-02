import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.image import resize
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""

  return tf.cast(resize(image, [224,224]),tf.float32) / 255., label[0] #- [0.5 for _ in range(4)]

def dataFrameMap(x):
  
  x["faces"]["bbox"] = tf.convert_to_tensor([.0,.0,.0,.0], tf.float32) if len(x["faces"]["bbox"]) == 0 else x["faces"]["bbox"]
  
  return x

def displayBoundingBox(image, bbox: list, title: str):
  
  bbox = np.array(bbox)
  
  y = int(bbox[0]*image.shape[0])
  x = int(bbox[1]*image.shape[1])
  h = int(bbox[2]*image.shape[0])
  b = int(bbox[3]*image.shape[1])
  
  plt.figure()
  plt.title(title)
  plt.xlabel(f"shape: {image.shape}\nbbox: {bbox}")
  #plt.axis('off')
  plt.imshow(cv2.rectangle((np.array(image)*255).astype(np.uint8), (x,y), (b,h),(255,0,0), 1))
    

def getDataset(name : str, only_one: bool) -> tf.data.Dataset:
  
  ds, ds_info = tfds.load('wider_face', split=name, shuffle_files=True, with_info=True)
  
  if(name == "test"):
    display(tfds.as_dataframe(ds.take(10), ds_info))
  
  if only_one:
    ds = ds.filter(lambda x: len(x["faces"]["bbox"]) == 1) 
  #ds = ds.map(dataFrameMap) 
  
  if(name == "test"):
    display(tfds.as_dataframe(ds.take(10), ds_info))

  for ele in ds.take(1):
    
    displayBoundingBox(ele["image"]/255, ele["faces"]["bbox"][0], f"{name} VOR NORMALIZATION")
     
    x = normalize_img(image=ele["image"], label=ele["faces"]["bbox"])
    
    displayBoundingBox(x[0], x[1], f"{name} NACH NORMALIZATION") 
     
    break
  
  ds = ds.map(lambda x : normalize_img(image=x["image"], label=x["faces"]["bbox"]), num_parallel_calls=tf.data.AUTOTUNE)
  
  if name == "train":
    ds = ds.cache()
    #ds = ds.shuffle(ds_info.splits[name].num_examples)
    ds = ds.batch(128)
  else:
    ds = ds.batch(128)
    ds = ds.cache()
        
  ds = ds.prefetch(tf.data.AUTOTUNE)
  
  return ds, ds_info

def display_imgs(images, folder : str, n=5):

  image = None

  fig, axs = plt.subplots(1,n)
  
  fig.set_size_inches(30,20)

  for i in range(n):
    
    image = images[i]

    image = (np.array(image)*255).astype(np.uint8)
    
    im = Image.fromarray(image)
    im.save(f"images/{folder}/image_{i}.jpeg")

    axs[i].imshow(image)        
    axs[i].get_xaxis().set_visible(False)
    axs[i].get_yaxis().set_visible(False)
    
def predict(model, model_name, ds):
  
  for ele in ds.take(1):

    predictions = tf.convert_to_tensor([np.array([bbox]) for bbox in model.predict(ele[0])])

    print(f"\n\nGroundtruth: {ele[1]}\n\n")
    print(f"\n\nPredicetd: {predictions}\n\n")
    
    images = tf.image.draw_bounding_boxes(
      ele[0], ele[1], [(0, 0, 1, 1) for _ in range(len(ele[0]))], name=None
      )
    
    images = tf.image.draw_bounding_boxes(
      images, predictions, [(0, 1, 0, 1) for _ in range(len(images))], name=None
      )
    
    display_imgs(images, model_name) 