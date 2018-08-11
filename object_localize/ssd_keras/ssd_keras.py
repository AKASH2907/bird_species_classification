from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt
from os.path import isfile, join
from os import rename, listdir, rename, makedirs
from keras_ssd512 import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
# from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
# from keras_layers.keras_layer_DecodeDetections import DecodeDetections
# from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
# from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

# from data_generator.object_detection_2d_data_generator import DataGenerator
# from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
# from data_generator.object_detection_2d_geometric_ops import Resize
# from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

img_height = 512
img_width = 512

species = ["blasti", "bonegl", "brhkyt", "cbrtsh", "cmnmyn", "gretit", "hilpig", "himbul", "himgri", "hsparo", "indvul"
    , "jglowl", "lbicrw", "mgprob", "rebimg", "wcrsrt"]
datapath = '../train_data/'
model = ssd_512(image_size=(img_height, img_width, 3),
                n_classes=20,
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05], # The scales for MS COCO are [0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06]
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
               two_boxes_for_ar1=True,
               steps=[8, 16, 32, 64, 128, 256, 512],
               offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
               clip_boxes=False,
               variances=[0.1, 0.1, 0.2, 0.2],
               normalize_coords=True,
               subtract_mean=[123, 117, 104],
               swap_channels=[2, 1, 0],
               confidence_thresh=0.3,
               iou_threshold=0.45,
               top_k=200,
               nms_max_output_size=400
               )


weights_path = '../VGG_VOC0712Plus_SSD_512x512_ft_iter_160000.h5'

model.load_weights(weights_path, by_name=True)

# 3: Compile the model so that Keras won't complain the next time you load it.

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

# model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
#                                                'L2Normalization': L2Normalization,
#                                                'DecodeDetections': DecodeDetections,
#                                                'compute_loss': ssd_loss.compute_loss})





# We'll only load one image in this example.
for i in species:

	path = join(datapath, i)
	files = listdir(path)
	files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
	for file in files:
		orig_images = [] # Store the images here.
		input_images = [] # Store resized versions of the images here.
		img_path= join(path, file)
		print(img_path)
		orig_images.append(imread(img_path))
		img = image.load_img(img_path, target_size=(img_height, img_width))
		# print(img)
		img = image.img_to_array(img)
		# print(img)
		input_images.append(img)
		input_images = np.array(input_images)

		# print(input_images)

		y_pred = model.predict(input_images)
		confidence_threshold = 0.3

		y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

		np.set_printoptions(precision=2, suppress=True, linewidth=90)
		print("Predicted boxes:\n")
		print('   class   conf xmin   ymin   xmax   ymax')
		print(y_pred_thresh[0])