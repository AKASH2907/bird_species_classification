import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
from os.path import isfile, join
from os import rename, listdir, rename, makedirs
# Root directory of the project
ROOT_DIR = os.path.abspath("./")

species = ["blasti", "bonegl", "brhkyt", "cbrtsh", "cmnmyn", "gretit", "hilpig", "himbul", "himgri", "hsparo", "indvul"
    , "jglowl", "lbicrw", "mgprob", "rebimg", "wcrsrt"]

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "coco/"))  # To find local version
import coco

# %matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()



# Create model object in inference mode.# Creat 
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
image_path = '../train_data/'

for i in species:
	specie = join(image_path, i)

	files = listdir(specie)
	files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

	birds = 1

	for file in files:

		img_path = join(specie, file)

		print(img_path)
		image = cv2.imread(img_path, 1)

		result = model.detect([image], verbose=1)

		r = result[0]

		l = len(r['rois'])
		print(l)
		print(r['class_ids'])
		imgs = 1
		for j in range(l):
			if(r['class_ids'][j]==15):

				y1, x1, y2, x2 = r['rois'][j]

				crop = image[y1:y2, x1:x2]

				cv2.imwrite('../mask_rcnn_cropped/' + i + '/' + str(birds) + str(imgs) + '.jpg', crop)

				imgs+=1

		birds+=1






