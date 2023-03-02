import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.image import resize
import matplotlib.pyplot as plt

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

def display_imgs(images, n=6):

    image = None

    fig, axs = plt.subplots(1,n)
    
    fig.set_size_inches(30,20)

    for i, image in enumerate(images):

        print(i)

        image = (np.array(image)*255).astype(np.uint8)

        from PIL import Image
        im = Image.fromarray(image)
        im.save("tensor.jpeg")

        #axs[i].imshow(image)        
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)

    plt.imshow(image)