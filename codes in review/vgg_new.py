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

BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1
N_CLASSES = 16



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


def build_vgg(img_shape=(416, 416, 3), n_classes=16, n_layers=16, l2_reg=0.,
                load_pretrained=False, freeze_layers_from='base_model'):
    # Decide if load pretrained weights from imagenet
    if load_pretrained:
        weights = 'imagenet'
    else:
        weights = None

    # Get base model
    if n_layers==16:
        base_model = VGG16(include_top=False, weights=weights,
                           input_tensor=None, input_shape=img_shape)
    elif n_layers==19:
        base_model = VGG19(include_top=False, weights=weights,
                           input_tensor=None, input_shape=img_shape)
    else:
        raise ValueError('Number of layers should be 16 or 19')

    # Add final layers
    x = base_model.output
    x = Flatten(name="flatten")(x)
    x = Dense(1024, activation='relu', name='dense_1')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', name='dense_2')(x)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, name='dense_3_{}'.format(n_classes))(x)
    predictions = Activation("softmax", name="softmax")(x)

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
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[precision, recall, f1])

    return model 

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
            img = cv2.resize(img, (416, 416))
            # print(img)
            X_train.append(img)
            Y_train+=[count]
            # break
        # print(count)

        for k in test_files:
            im = join(test_samples, k)
            img = cv2.imread(im,1)
            img = cv2.resize(img, (416, 416))
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


# x_train, y_train, x_test, y_test = gen_data()
# y_train = np_utils.to_categorical(y_train, N_CLASSES)
# y_test = np_utils.to_categorical(y_test, N_CLASSES)
# np.save('X_train.npy', x_train)
# np.save('Y_train.npy', y_train)
# np.save('X_test.npy', x_test)
# np.save('Y_test.npy', y_test) 
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# print(y_train, y_test)

x_train = np.load('X_train.npy')
y_train = np.load('Y_train.npy')
x_test = np.load('X_test.npy')
y_test = np.load('Y_test.npy')
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# print(y_train)

model = build_vgg()

history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=20, verbose= 1, 
    # steps_per_epoch=x_train.shape[0]//BATCH_SIZE,
                        validation_split=VALIDATION_SPLIT)

model.save_weights('vgg_1.h5')
score = model.evaluate(x_test, y_test, verbose=1, batch_size= BATCH_SIZE)

print(score)

print(history.history.keys())
