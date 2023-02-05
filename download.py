import tensorflow_datasets as tfds
import tensorflow as tf

ds = tfds.load('wider_face', split='train', shuffle_files=True)