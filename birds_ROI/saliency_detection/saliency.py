import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from os.path import isfile, join
from os import rename, listdir, rename, makedirs

# img = "../final_data/train/blasti/100101.jpg"
# image = cv2.imread(img)

species = ["blasti", "bonegl", "brhkyt", "cbrtsh", "cmnmyn", "gretit", "hilpig", "himbul", "himgri", "hsparo", "indvul"
    , "jglowl", "lbicrw", "mgprob", "rebimg", "wcrsrt"]
destination = '../train/'
datapath = '../saliency_cropped_txt/'

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to BING objectness saliency model")	
ap.add_argument("-n", "--max-detections", type=int, default=10,
	help="maximum # of detections to examine")
args = vars(ap.parse_args())

# initialize OpenCV's objectness saliency detector and set the path
# to the input model files
saliency = cv2.saliency.ObjectnessBING_create()
saliency.setTrainingPath(args["model"])


for i in species:

	path = join(destination, i)
	files = listdir(path)
	files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
	txt_path = join(datapath, i)

	for image in files:
		im = join(path, image)

		read = cv2.imread(im, 1)

		# compute the bounding box predictions used to indicate saliency
		(success, saliencyMap) = saliency.computeSaliency(read)
		numDetections = saliencyMap.shape[0]

		# loop over the detections
		# points = []
		for i in range(0, min(numDetections, args["max_detections"])):
			# extract the bounding box coordinates
			
			(startX, startY, endX, endY) = saliencyMap[i].flatten()
			
			# randomly generate a color for the object and draw it on the image
			output = read.copy()
			color = np.random.randint(0, 255, size=(3,))
			color = [int(c) for c in color]
			cv2.rectangle(output, (startX, startY), (endX, endY), color, 2)

			with open(txt_path+ '/'+ str(image[0:6]) + '.txt', 'a+') as f:
				f.write(str(startX) + " " + str(endX) + " " + str(startY) + " " + str(endY) +"\n")