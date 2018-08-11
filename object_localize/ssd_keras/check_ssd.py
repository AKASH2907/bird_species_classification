from imageio import imread
import numpy as np
from matplotlib import pyplot as plt

img_height = 512
img_width = 512
orig_images = [] # Store the images here.
input_images = [] # Store resized versions of the images here.

# We'll only load one image in this example.
img_path = '../train/blasti/'

orig_images.append(imread(img_path))
img = image.load_img(img_path, target_size=(img_height, img_width))
print(img.shape)
img = image.img_to_array(img)
print(img.shape)
input_images.append(img)
input_images = np.array(input_images)

print(input_images)