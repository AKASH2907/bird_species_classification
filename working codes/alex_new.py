import cv2
from matplotlib import pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Dropout, Flatten, Input, BatchNormalization
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

BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1
N_CLASSES = 16
EPOCHS = 7


def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.1
    epochs_drop = 4.0
    lrate = initial_lrate * math.pow(drop, math.floor((epoch)/epochs_drop))
    return lrate

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    print(precision, recall)
    return (2*((precision*recall)/(precision+recall+K.epsilon())))



def alex():
    v1 = Input(shape=(416, 416, 3))

    # 1st Layer
    conv_1 = Conv2D(96, (11, 11), strides=(4,4), padding='valid')(v1)
    conv_1 = Activation('relu')(conv_1)
    conv_1 = BatchNormalization()(conv_1)  
    conv_1 = MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid')(conv_1)
 
    # 2nd Layer
    conv_2 = Conv2D(256, (5, 5))(conv_1)
    conv_2 = Activation('relu')(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(conv_2)

    # 3rd Layer
    conv_3 = Conv2D(384, (3, 3))(conv_2)
    # conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation('relu')(conv_3)
    # conv_3 = MaxPooling2D((3, 3))(conv_3)

    # 4th Layer
    conv_4 = Conv2D(384, (3, 3))(conv_3)
    # conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation('relu')(conv_4)
    # conv_4 = MaxPooling2D((3, 3))(conv_4)

    # 5th Layer
    conv_5 = Conv2D(256, (3, 3))(conv_4)
    conv_5 = Activation('relu')(conv_5)
    conv_5 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(conv_5)
    
    # Flatten Layer
    f = Flatten()(conv_5)

    # 6th Layer
    dense_1 = Dense(4096)(f)
    # dense_1 = BatchNormalization()(dense_1)
    dense_1 = Activation('relu')(dense_1)
    drop_1  = Dropout(0.25)(dense_1)

    # 7th Layer
    dense_2 = Dense(4096)(drop_1)
    # dense_2 = BatchNormalization()(dense_2)
    dense_2 = Activation('relu')(dense_2)
    drop_2  = Dropout(0.25)(dense_2)

    dense_3 = Dense(16)(drop_2)

    predictions = Activation('softmax')(dense_3)

    model = Model(inputs=v1, outputs=predictions)
    
    adam = Adam(lr=0.00006)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[precision, recall, f1])

    return model


model = alex()

x_train = np.load('X_train.npy')
y_train = np.load('Y_train.npy')
x_test = np.load('X_test.npy')
y_test = np.load('Y_test.npy')
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# print(y_train)

# model = build_vgg()

filepath = "wts/alex/alex-1-{epoch:02d}-{val_f1:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_f1', verbose =1, save_best_only=True, mode='max', save_weights_only=True)
early_stopper = EarlyStopping(monitor='val_f1', verbose=1, patience=4)
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
# lrate = LearningRateScheduler(step_decay)
callback_list = [checkpoint, early_stopper, tensorboard]

history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose= 1, 
    # steps_per_epoch=x_train.shape[0]//BATCH_SIZE,
                        validation_split=VALIDATION_SPLIT)

model.save_weights('alex_1_7ep.h5')
score = model.evaluate(x_test, y_test, verbose=1, batch_size= BATCH_SIZE)

print(score)

print(history.history.keys())