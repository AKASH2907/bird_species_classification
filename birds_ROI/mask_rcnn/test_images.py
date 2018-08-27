import sys
import random
import math
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Dropout, Flatten, Input, AveragePooling2D
from keras.models import Model
from keras.utils import plot_model, np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, LearningRateScheduler
from time import time
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras import backend as K
from os.path import isfile, join
from os import rename, listdir, rename, makedirs
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.utils.generic_utils import get_custom_objects


BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1
N_CLASSES = 16
EPOCHS = 7

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

species = ["blasti", "bonegl", "brhkyt", "cbrtsh", "cmnmyn", "gretit", "hilpig", "himbul", "himgri", "hsparo", "indvul"
	, "jglowl", "lbicrw", "mgprob", "rebimg", "wcrsrt"]
species_check = ["wcrsrt"]
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

class InferenceConfig(coco.CocoConfig):
	# Set batch size to 1 since we'll be running inference on
	# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()



# Create model object in inference mode.# Creat 
model_coco = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model_coco.load_weights(COCO_MODEL_PATH, by_name=True)


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



def swish(x):
	return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': Activation(swish)})


def build_inception_resnet_V2(img_shape=(416, 416, 3), n_classes=16, l2_reg=0.,
                load_pretrained=True, freeze_layers_from='base_model'):
    # Decide if load pretrained weights from imagenet
    if load_pretrained:
        weights = 'imagenet'
    else:
        weights = None

    # Get base model
    base_model = InceptionResNetV2(include_top=False, weights=weights,
                             input_tensor=None, input_shape=img_shape)

    # Add final layers
    x = base_model.output
    x = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(512, activation='swish', name='dense_1', kernel_initializer='he_uniform')(x)
    x = Dropout(0.25)(x)
    predictions = Dense(n_classes, activation='softmax', name='predictions', kernel_initializer='he_uniform')(x)

    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze some layers
    if freeze_layers_from is not None:
        if freeze_layers_from == 'base_model':
            print ('   Freezing base model layers')
            for layer in base_model.layers:
                layer.trainable = False
        else:
            for i, layer in enumerate(model.layers):
                print(i, layer.name)
            print ('   Freezing from layer 0 to ' + str(freeze_layers_from))
            for layer in model.layers[:freeze_layers_from]:
               layer.trainable = False
            for layer in model.layers[freeze_layers_from:]:
               layer.trainable = True

    adam = Adam(0.0001) 
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[precision, recall, f1])  
    return model 



# model_incept_res1 = build_inception_resnet_V2()
# model_incept_res1.load_weights('../inception_resnet_crops.h5')
# model_incept_res2 = build_inception_resnet_V2()
# model_incept_res2.load_weights('../inception_resnet_crops+11.h5')
image_path = '../test/'

y_pred = []

for i in species:
	specie = join(image_path, i)

	files = listdir(specie)
	files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

	for file in files:

		img_path = join(specie, file)

		print(img_path)
		image = cv2.imread(img_path, 1)

		result = model_coco.detect([image], verbose=1)

		r = result[0]

		l = len(r['rois'])
		print(l)
		print(r['class_ids'])
		print(r['scores'])
		batches = []
		for j in range(l):
			if(r['class_ids'][j]==15):

				y1, x1, y2, x2 = r['rois'][j]

				crop = image[y1:y2, x1:x2]
				crop = cv2.resize(crop, (416, 416))


				batches.append(crop)

				# cv2.imwrite('../mask_rcnn_cropped/' + i + '/' + str(birds) + str(imgs) + '.jpg', crop)

		batches = np.asarray(batches).astype('float32')
		batches/=255

		if(batches.shape[0]>0):
			p = model_incept1.predict(batches)

			flipped = []
			for i in range(batches.shape[0]):
				flip = np.flip(np.argsort(p)[i], axis=0)
				flipped.append(flip[0])

			print(flipped)
			x = np.bincount(flipped)
			print(x)

			maxi = np.argmax(x)
			print(maxi)

			y_pred+=[maxi]


		else:
			im = cv2.resize(image, (416, 416))
			im = np.reshape(im, (1, 416, 416, 3))

			p = model_incept2.predict(im)
			print(p)
			maxi = np.argmax(p)
			y_pred+=[maxi]
			# predict result directly on image


	# 	break

	# break


	

print(y_pred)

Y_pred = np_utils.to_categorical(y_pred, 16)
print(Y_pred)
np.save('../Y_pred_inceptionv3.npy', Y_pred)

print(len(y_pred))