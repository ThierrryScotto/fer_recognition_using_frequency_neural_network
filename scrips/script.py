import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

HEIGHT = 48
WIDTH = 48

def convert_1d_vector_to_image(dataset):
  images = []
  images_label = []

  for index, row in dataset.iterrows():
    one_dimensional_str = row['pixels'].split()
    one_dimensional_vec = np.array(list(map(int, one_dimensional_str)))
    
    if one_dimensional_vec.size == HEIGHT * WIDTH:
      image_2d = one_dimensional_vec.reshape((HEIGHT, WIDTH))

      images.append(image_2d)
      images_label.append(row['emotion'])
    else:
      print()
      raise ValueError(f"The size of the vector {one_dimensional_vec.size} does not correspond to the product of the dimensions ({HEIGHT}, {WIDTH})")
  
  return images, images_label

# 0 : Anger (45 samples)
# 1 : Disgust (59 samples)
# 2 : Fear (25 samples)
# 3 : Happiness (69 samples)
# 4 : Sadness (28 samples)
# 5 : Surprise (83 samples)
# 6 : Neutral (593 samples)
# 7 : Contempt (18 samples)