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
import random
from random import randint, shuffle
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

# datapath = '../train_data/'
species = [  "blasti","bonegl",  "brhkyt", "cbrtsh", "cmnmyn", "gretit", "hilpig", "himbul","himgri", "hsparo", "indvul"
	, "jglowl", "lbicrw", "mgprob", "rebimg", "wcrsrt"]
# new_species = 
datapath = './'

# files = listdir(datapath)

def gen_random_batch(in_groups, batch_halfsize = 8):
	out_img_a, out_img_b, out_score = [], [], []
	all_groups = list(range(len(in_groups)))
	for match_group in [True, False]:
		group_idx = np.random.choice(all_groups, size = batch_halfsize)
		out_img_a += [in_groups[c_idx][np.random.choice(range(in_groups[c_idx].shape[0]))] for c_idx in group_idx]
		if match_group:
			b_group_idx = group_idx
			out_score += [1]*batch_halfsize
		else:
			# anything but the same group
			non_group_idx = [np.random.choice([i for i in all_groups if i!=c_idx]) for c_idx in group_idx] 
			b_group_idx = non_group_idx
			out_score += [0]*batch_halfsize
			
		out_img_b += [in_groups[c_idx][np.random.choice(range(in_groups[c_idx].shape[0]))] for c_idx in b_group_idx]
			
	return np.stack(out_img_a,0), np.stack(out_img_b,0), np.stack(out_score,0)



def create_pairs():
	X_train = []
	Y_train = []
	X_test = []
	Y_test = []
	count=0
	for i in species:

		train_samples = join(datapath, 'train/'+i)
		test_samples = join(datapath, 'test/'+i)
		train_files = listdir(train_samples)
		test_files = listdir(test_samples)
		train_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
		test_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

		for j in train_files:
			# im = join(train_samples, j)
			# img = cv2.imread(im,1)
			# img = cv2.resize(img, (416, 416))
			# # print(img)
			# X_train.append(img)
			Y_train+=[count]
			# break
		# print(count)

		for k in test_files:
			# im = join(test_samples, k)
			# img = cv2.imread(im,1)
			# img = cv2.resize(img, (416, 416))
			# # print(img)
			# X_test.append(img)
			Y_test+=[count]
		

		count+=1

	# print(X_train)
	# print(X_train.shape)
	# print(X_train_batch)
	# print(len(X_train))

	# X_train = np.asarray(X_train)
	# # print(X_train)
	# X_train = X_train.astype('float32')
	# X_train/= 255
	Y_train = np.asarray(Y_train)
	# Y_train = to_categorical(Y_train, len(species)-1)
	# print(X_train)
	# print(y_train)
	# print(arr)
	# print(arr.shape)
	# print(X_train.shape)
	# print(y_train.shape)
	# print(y_train)
	# X_train = X_train.reshape(150, 416*416*3)

	# X_test = np.asarray(X_test)
	# X_test = X_test.astype('float32')
	# X_test /= 255
	Y_test = np.asarray(Y_test)
	# Y_test = to_categorical(Y_test, len(species)-1)
	return X_train, Y_train, X_test, Y_test


x_train, y_train, x_test, y_test = create_pairs()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# print(y_test)
# print(y_train)



train_groups = [x_train[np.where(y_train==i)[0]] for i in np.unique(y_train)]
test_groups = [x_test[np.where(y_test==i)[0]] for i in np.unique(y_test)]

# print(train_groups)
# test_groups = [x_test[np.where(y_test==i)[0]] for i in np.unique(y_train)]
# print('train groups:', [x.shape[0] for x in train_groups])
# print('test groups:', [x.shape[0] for x in test_groups])

y_train = to_categorical(y_train, len(species))
y_test = to_categorical(y_test, len(species))
pv_a, pv_b, pv_sim = gen_random_batch(train_groups, 4)
# print(pv_a.shape)
# print(pv_b.shape)
batch_size=4
def siam_gen(in_groups, batch_size = 32):
    while True:
        pv_a, pv_b, pv_sim = gen_random_batch(train_groups, batch_size//2)
        yield [pv_a, pv_b], pv_sim

valid_a, valid_b, valid_sim = gen_random_batch(test_groups, batch_size)

# print(pv_a.shape)

# print(valid_sim)
# print(valid_sim.shape)
# fig, m_axs = plt.subplots(2, pv_a.shape[0], figsize = (12, 6))
# for c_a, c_b, c_d, (ax1, ax2) in zip(pv_a, pv_b, pv_sim, m_axs.T):
#     ax1.imshow(c_a[:,:,0])
#     ax1.set_title('Image A')
#     ax1.axis('off')
#     ax2.imshow(c_b[:,:,0])
#     ax2.set_title('Image B\n Similarity: %3.0f%%' % (100*c_d))
#     ax2.axis('off')