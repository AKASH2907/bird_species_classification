import cv2
from matplotlib import pyplot as plt
import numpy as np
from keras.utils import *
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from os.path import isfile, join
from os import rename, listdir, rename, makedirs
import argparse
from shutil import copyfile, move
import imgaug as ia
from imgaug import augmenters as iaa
# import random
from random import randint, shuffle, sample

# datapath = '../train_data/'
species = ["blasti", "bonegl", "brhkyt", "cbrtsh", "cmnmyn", "gretit", "hilpig", "himbul", "himgri", "hsparo", "indvul"
	, "jglowl", "lbicrw", "mgprob", "rebimg", "wcrsrt"]
datapath = './final_data/train/'

# path = join(datapath)
# files = listdir(datapath)

# new_species = species
# print(new_species)
# del new_species[0]
# print(new_species)


def create_pairs():


	pairs = []
	labels = []
	# print(new_species)
	# c = 1
	count = 0
	for i in species:
		# print(i)
		positive = join(datapath, i)


		# new_species = species
		# del new_species[count]
		new_species = [x for x in species if x!=i]

		files = listdir(positive)
		files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
		# print(len(files))
		
		# print(neg)
		# negative = join(datapath, neg)

		for j in files:
			im_j = join(positive, j)
			img_j = cv2.imread(im_j, 1)
			img_j = cv2.resize(img_j, (416, 416))
			for k in files:
				if(j==k):
					continue
				else:
					im_k = join(positive, k)
					img_k = cv2.imread(im_k, 1)
					img_k = cv2.resize(img_k, (416, 416))
					# print(im_j, im_k)
					pairs+=[[img_j, img_k]]
					labels+=[1]
			

			neg = sample(new_species, len(files)-1)
			# print(len(neg))
			# print(neg)
			for negs in neg:
				negative = join(datapath, negs)

				neg_files = listdir(negative)
				shuffle(neg_files)
				for l in neg_files:
					im_l = join(negative, l)
					img_l = cv2.imread(im_l, 1)
					img_l = cv2.resize(img_l, (416, 416))
					# print(im_j, im_l)
					pairs+=[[img_j, img_l]]
					labels+=[0]
					break

			# break
		# break
		print(len(pairs))
		# print(labels)
		# break
	count+=1
	# c+=1 


	combined = list(zip(pairs, labels))
	shuffle(combined)

	pairs[:], labels[:] = zip(*combined)
	# print(pairs)
	# print(len(pairs))
	# pr = np.array(pairs)
	# print(pr.shape)
	# print(labels)
	# print(pr[:,0])

	return np.array(pairs), np.array (labels)







# if __name__== "__main__":
# 	tr_pairs, tr_labels = create_pairs()
# 	print(tr_labels)






# X_train = np.array([])
# X_train = []
# for i in files:
# 	im = join(destination, i)
# 	img = cv2.imread(im,1)
# 	# print(img)
# 	X_train.append(img)
# 	# break

# # print(X_train)
# # print(X_train.shape)
# # print(X_train_batch)
# # print(len(X_train))

# X_train = np.asarray(X_train)
# # print(arr)
# # print(arr.shape)

# X_train = X_train.reshape(150, 416*416*3)