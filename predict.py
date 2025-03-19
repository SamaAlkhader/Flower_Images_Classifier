#!/usr/bin/env python
# coding: utf-8

# In[7]:


with open('predict.py', 'w') as f:
  f.write("""\
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import logging
logger= tf.get_logger()
logger.setLevel(logging.ERROR)
from PIL import Image



def process_image(image):
    image= np.array(image)
    image= tf.image.resize(image, (224,224))
    image/=255.0
    return image.numpy()


def predict(image_path, model, top_k, class_names):
    image= Image.open(image_path)
    image= process_image(image)
    image= np.expand_dims(image, axis=0)
    prediction= model.predict(image)[0]
    sorted_indices= np.argsort(prediction)[-top_k:][::-1]
    probabilities= prediction[sorted_indices]
    names=[class_names[str(i)] for i in sorted_indices]
    return probabilities, names


def main():
  parser= argparse.ArgumentParser(description="Predict flower name from an image")
  parser.add_argument('image_path', type=str, help='Path to image')
  parser.add_argument('model_path', type=str, help='Path to model')
  parser.add_argument('--top_k', type=int, default=5,help='Return the top K most likely classes')
  parser.add_argument('--category_names', type=str, help='Path to JSON file mapping labels to flower names')
  args= parser.parse_args()

  class_names={}
  if args.category_names:
    with open(args.category_names, 'r') as f:
      class_names= json.load(f)

  model= tf.keras.models.load_model(args.model_path, custom_objects={'KerasLayer':hub.KerasLayer})
  probabilities, classes= predict(args.image_path, model, args.top_k, class_names)


  print("Flower names and their probabilities:")
  for i in range(len(classes)):
    print(f"{classes[i]}: {probabilities[i]:.5f}")






if __name__ == '__main__':
  main()

""")


# In[8]:


get_ipython().system('python predict.py ./test_images/wild_pansy.jpg models/best_model.h5 --top_k 3 --category_names label_map.json')


# In[ ]:




