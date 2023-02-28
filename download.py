import tensorflow_datasets as tfds
import tensorflow as tf

dl_config = tfds.download.DownloadConfig(verify_ssl = False)

tfds.load('wider_face', split="train", shuffle_files=True, with_info=True, download_and_prepare_kwargs={
      'download_config': dl_config,
})