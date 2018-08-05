from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import Adam, SGD
from keras.layers import Dense, Activation, Dropout, Flatten, Input, BatchNormalization, Lambda, merge
from keras.models import Model, Sequential
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, LearningRateScheduler
from time import time
import math
from keras import backend as K
from keras.utils.np_utils import to_categorical
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
from keras.losses import categorical_crossentropy
import numpy.random as rnd
from sklearn.utils import shuffle
import os
from create_pairs import test_groups, train_groups
from keras.optimizers import Adam, RMSprop
import numpy.random as rng
import numpy as np
from keras.utils.generic_utils import get_custom_objects

BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1
N_CLASSES = 16
EPOCHS = 10

def swish(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': Activation(swish)})


def W_init(shape,name=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,name=name)

#//TODO: figure out how to initialize layer biases in keras.
def b_init(shape,name=None):
    """Initialize bias as in paper"""
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)


def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.1
    epochs_drop = 4.0
    lrate = initial_lrate * math.pow(drop, math.floor((epoch)/epochs_drop))
    return lrate


def euclidean_distance(vects):
    # assert len(vects)==2
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    # return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
    return 0.5*K.mean((1-y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0)))


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()

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
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    print(precision, recall)
    return (2*((precision*recall)/(precision+recall+K.epsilon())))

input_size = (416, 416, 3)

left_input = Input(shape=input_size)
right_input = Input(shape=input_size)

def model(inputs):
    get_custom_objects().update({'swish': Activation(swish )})
    conv_1 = Conv2D(64, (10, 10), activation='swish', kernel_initializer=W_init, kernel_regularizer=l2(2e-4))(inputs)
    # conv_1 = Conv2D(64, (10, 10), activation='relu')(inputs)
    pool_1 = MaxPooling2D()(conv_1)

    conv_2 = Conv2D(128,(7,7),activation='swish', kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init)(pool_1)
    # conv_2 = Conv2D(128,(7,7),activation='relu')(pool_1)
    pool_2 = MaxPooling2D()(conv_2)

    conv_3 = Conv2D(128,(4,4),activation='swish',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init)(pool_2)
    pool_3 = MaxPooling2D()(conv_3)

    conv_4 = Conv2D(256,(4,4),activation='swish',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init)(pool_3)


    f = Flatten()(conv_4)

    dense_1 = Dense(1024, activation="swish",kernel_regularizer=l2(1e-3),kernel_initializer=W_init,bias_initializer=b_init)
    return dense_1

convnet = Sequential()
convnet.add(Conv2D(64,(10,10),activation='relu',input_shape=input_size,
                   kernel_initializer=W_init,kernel_regularizer=l2(2e-4)))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(128,(7,7),activation='relu',
                   kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(128,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(256,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))
convnet.add(Flatten())
convnet.add(Dense(512,activation="sigmoid",kernel_regularizer=l2(1e-3),kernel_initializer=W_init,bias_initializer=b_init))


encoded_l = convnet(left_input)
encoded_r = convnet(right_input)

L1_distance = lambda x: K.abs(x[0]-x[1])
L2_distance = lambda x: K.sqrt(K.sum(K.square(x[0] - x[1]), axis=1, keepdims=True))
both = merge([encoded_l,encoded_r], mode = L1_distance, output_shape=lambda x: x[0])
prediction = Dense(16,activation='softmax',bias_initializer=b_init)(both)
siamese_model = Model(inputs=[left_input,right_input],outputs=prediction)
#optimizer = SGD(0.0004,momentum=0.6,nesterov=True,decay=0.0003)
optimizer = Adam(0.001)
#//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
siamese_model.compile(loss=contrastive_loss,optimizer=optimizer, metrics=[precision, recall, f1])
# siamese_net.count_params()



print("Creating Pairs......")
# tr_pairs, tr_y = create_pairs()

# distance = Lambda(euclidean_distance, output_shape=((1, )))([encoded_l, encoded_r])

# siamese_model = Model(input=[left_input, right_input], output=distance)

# train
rms = RMSprop()
adam = Adam()
# siamese_model.compile(loss=contrastive_loss, optimizer=rms)
def siam_gen(in_groups, batch_size =32):
    while True:
        pv_a, pv_b, pv_sim = gen_random_batch(train_groups, batch_size//2)
        yield [pv_a, pv_b], pv_sim

def valid_gen(in_groups, batch_size =16):
    while True:
        pv_a, pv_b, pv_sim = gen_random_batch(test_groups, batch_size//2)
        yield [pv_a, pv_b], pv_sim

valid_a, valid_b, valid_sim = gen_random_batch(test_groups, BATCH_SIZE)
print("Pairs loaded.......")

# filepath = "sia-2-{epoch:02d}-{val_f1:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_f1', verbose =1, save_best_only=True, mode='max', save_weights_only=True)
# early_stopper = EarlyStopping(monitor='val_f1', verbose=1, patience=3)
# tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
# lrate = LearningRateScheduler(step_decay)
# callback_list = [checkpoint, early_stopper, tensorboard]


loss_history = siamese_model.fit_generator(siam_gen(train_groups), 
    # validation_data=([valid_a, valid_b], valid_sim),
    steps_per_epoch=50,epochs=10, verbose=True)
# siamese_model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          # validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
          # batch_size=32,
          # epochs=10)

siamese_model.save_weights('sia_1swish.h5')

score = siamese_model.evaluate_generator(valid_gen(test_groups))

print(score)
# compute final accuracy on training and test sets
# pred = siamese_model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
# tr_acc = compute_accuracy(pred, tr_y)
# pred = siamese_model.predict([te_pairs[:, 0], te_pairs[:, 1]])
# te_acc = compute_accuracy(pred, te_y)

# print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
# print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

























































# # this is the augmentation configuration we will use for training
# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)

# # this is the augmentation configuration we will use for testing:
# # only rescaling
# test_datagen = ImageDataGenerator(rescale=1./255)


# train_generator = train_datagen.flow_from_directory(
#         'data/train',  # this is the target directory
#         target_size=(416, 416),  # images resized to 1000*1000
#         batch_size=batch_size,
#         class_mode='categorical')

# validation_generator = test_datagen.flow_from_directory(
#         'data/test',  
#         target_size=(416, 416),  # images  resized to 1000*1000
#         batch_size=batch_size,
#         class_mode='categorical')

# filepath = "alex/alex-1{epoch:02d}-{val_acc:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose =1, save_best_only=True, mode='max', save_weights_only=True)
# early_stopper = EarlyStopping(monitor='val_acc', verbose=1, patience=3)
# tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
# lrate = LearningRateScheduler(step_decay)
# callback_list = [checkpoint, early_stopper, tensorboard]
# training = alex()
# training.fit_generator(
#         train_generator,
#         steps_per_epoch=1600 // batch_size,
#         epochs=15,
#         validation_data=validation_generator,
#         # validation_steps=800 // batch_size
#         callbacks= callback_list
#         )
# training.save_weights('alex.h5')

# def alex():
#     v1 = Input(shape=(416, 416, 3))
#     conv_1 = Conv2D(64, (11, 11))(v1)
#     conv_1 = BatchNormalization()(conv_1)
#     conv_1 = Activation('relu')(conv_1)
#     conv_1 = MaxPooling2D((3, 3))(conv_1)

#     conv_2 = Conv2D(128, (7, 7))(conv_1)
#     conv_2 = BatchNormalization()(conv_2)
#     conv_2 = Activation('relu')(conv_2)
#     conv_2 = MaxPooling2D((3, 3))(conv_2)

#     conv_3 = Conv2D(192, (3, 3))(conv_2)
#     conv_3 = BatchNormalization()(conv_3)
#     conv_3 = Activation('relu')(conv_3)
#     conv_3 = MaxPooling2D((3, 3))(conv_3)

#     conv_4 = Conv2D(256, (3, 3))(conv_3)
#     conv_4 = BatchNormalization()(conv_4)
#     conv_4 = Activation('relu')(conv_4)
#     conv_4 = MaxPooling2D((3, 3))(conv_4)

#     f = Flatten()(conv_4)

#     dense_1 = Dense(4096)(f)
#     dense_1 = BatchNormalization()(dense_1)
#     dense_1 = Activation('relu')(dense_1)

#     dense_2 = Dense(1024)(dense_1)
#     dense_2 = BatchNormalization()(dense_2)
#     dense_2 = Activation('relu')(dense_2)

#     dense_3 = Dense(16)(dense_2)

#     predictions = Activation('softmax')(dense_3)

#     model = Model(inputs=v1, outputs=predictions)
    
#     adam = Adam(lr=0.0001)
#     model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[f1, 'accuracy'])

#     return mode





# def make_pairs():
#     pairs = []
#     labels = []
#     # print(new_species)
#     c = 1
#     count = 0
#     for i in species:
#         positive = join(datapath, i)
#         new_species = species
#         del new_species[count]

#         files = listdir(positive)
#         files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
#         # print(len(files))
        

#         neg = random.sample(new_species, len(files))
#         print(neg)
#         # negative = join(datapath, neg)


#         for j in files:
#             im_j = join(positive, j)
#             img_j = cv2.imread(im_j, 1)
#             for k in files:
#                 if(j==k):
#                     continue
#                 else:
#                     im_k = join(positive, k)
#                     img_k = cv2.imread(im_k, 1)
#                     # print(im_j, im_k)
#                     pairs+=[[img_j, img_k]]
#                     labels+=[1]
#             for negs in neg:
#                 negative = join(datapath, negs)

#                 neg_files = listdir(negative)

#                 for l in neg_files:
#                     im_l = join(negative, l)
#                     img_l = cv2.imread(im_l, 1)
#                     print(im_j, im_l)
#                     pairs+=[[img_j, img_l]]
#                     labels+=[0]
#                     break

#             break
#         break
#         # c+=1 

#     # print(pairs)
#     # print(len(pairs))
#     pr = np.asarray(pairs)
#     print(pr.shape)
#     print(labels)

#     return