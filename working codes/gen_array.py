import cv2
from matplotlib import pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Dropout, Flatten, Input
from keras.models import Model
from keras.utils import plot_model, np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, LearningRateScheduler
from time import time
import math
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras import backend as K
from os.path import isfile, join
from os import rename, listdir, rename, makedirs
from sklearn.model_selection import train_test_split, StratifiedKFold


species = [  "blasti","bonegl",  "brhkyt", "cbrtsh", "cmnmyn", "gretit", "hilpig", "himbul","himgri", "hsparo", "indvul"
    , "jglowl", "lbicrw", "mgprob", "rebimg", "wcrsrt"]
# new_species = 
datapath = './'


def gen_data():
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
            im = join(train_samples, j)
            img = cv2.imread(im,1)
            img = cv2.resize(img, (1000, 1000))
            # print(img)
            X_train.append(img)
            Y_train+=[count]
            # break
        # print(count)

        for k in test_files:
            im = join(test_samples, k)
            img = cv2.imread(im,1)
            img = cv2.resize(img, (1000, 1000))
            # print(img)
            X_test.append(img)
            Y_test+=[count]
        

        count+=1

    # print(X_train)
    # print(X_train.shape)
    # print(X_train_batch)
    # print(len(X_train))

    X_train = np.asarray(X_train)
    # print(X_train)
    X_train = X_train.astype('float32')
    X_train/= 255
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

    X_test = np.asarray(X_test)
    X_test = X_test.astype('float32')
    X_test /= 255
    Y_test = np.asarray(Y_test)
    # Y_test = to_categorical(Y_test, len(species)-1)
    return X_train, Y_train, X_test, Y_test


x_train, y_train, x_test, y_test = gen_data()
y_train = np_utils.to_categorical(y_train, N_CLASSES)
y_test = np_utils.to_categorical(y_test, N_CLASSES)
np.save('X_tr1k.npy', x_train)
np.save('Y_tr1k.npy', y_train)
np.save('X_te1k.npy', x_test)
np.save('Y_te1k.npy', y_test) 
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# print(y_train, y_test)