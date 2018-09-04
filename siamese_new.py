from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import random
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Lambda, Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Dropout, Flatten, Input, AveragePooling2D, BatchNormalization
from keras.optimizers import RMSprop, Adam
from keras import backend as K
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.utils.generic_utils import get_custom_objects
from keras.utils import plot_model, np_utils

num_classes = 16
epochs = 10

def swish(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': Activation(swish)})


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    print(precision, recall)
    return (2*((precision*recall)/(precision+recall+K.epsilon())))



def build_inception_resnet_V2(img_shape=(416, 416, 3), n_classes=16, l2_reg=0.,
                load_pretrained=True, freeze_layers_from=0):
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

    model = Model(inputs=base_model.input, outputs=x)
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
 
    return Model(inputs=base_model.input, outputs=x)


def alex():
    v1 = Input(shape=(416, 416, 3))

    # 1st Layer
    conv_1 = Conv2D(96, (11, 11), strides=(4,4), padding='valid')(v1)
    conv_1 = Activation('swish')(conv_1)
    conv_1 = BatchNormalization()(conv_1)  
    conv_1 = MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid')(conv_1)
 
    # 2nd Layer
    conv_2 = Conv2D(256, (5, 5))(conv_1)
    conv_2 = Activation('swish')(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(conv_2)

    # 3rd Layer
    conv_3 = Conv2D(384, (3, 3))(conv_2)
    # conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation('swish')(conv_3)
    # conv_3 = MaxPooling2D((3, 3))(conv_3)

    # 4th Layer
    conv_4 = Conv2D(384, (3, 3))(conv_3)
    # conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation('swish')(conv_4)
    # conv_4 = MaxPooling2D((3, 3))(conv_4)

    # 5th Layer
    conv_5 = Conv2D(256, (3, 3))(conv_4)
    conv_5 = Activation('swish')(conv_5)
    conv_5 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(conv_5)
    
    # Flatten Layer
    f = Flatten()(conv_5)

    # 6th Layer
    dense_1 = Dense(1024)(f)
    # dense_1 = BatchNormalization()(dense_1)
    dense_1 = Activation('swish')(dense_1)
    drop_1  = Dropout(0.25)(dense_1)

    # 7th Layer
    dense_2 = Dense(512)(drop_1)
    # dense_2 = BatchNormalization()(dense_2)
    dense_2 = Activation('swish')(dense_2)
    # drop_2  = Dropout(0.25)(dense_2)

    # dense_3 = Dense(16)(drop_2)

    # predictions = Activation('softmax')(dense_3)

    return Model(inputs=v1, outputs=dense_2)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    # print([len(digit_indices[d]) for d in range(num_classes)])
    # print(n)
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            # print(z1, z2)
            pairs += [[x[z1], x[z2]]]
            # print(pairs)
            inc = random.randrange(1, num_classes)
            # print(inc)
            dn = (d + inc) % num_classes
            # print(dn)
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            # print(z1, z2)
            pairs += [[x[z1], x[z2]]]
            # print(pairs)
            labels += [1, 0]
            # print(labels)
    return np.array(pairs), np.array(labels)

x_train = np.load('X_train_crop.npy')
y_train = np.load('Y_train_crop.npy')
# x_train = np.load('X_train.npy')
# x_train = np.load('X_train_siamese_testing.npy')
# y_train = np.load('Y_train_siamese_testing.npy')
y_test = np.load('Y_test_not_categorical.npy')
# y_train = np.load('Y_train_not_categorical.npy')
x_test = np.load('X_test.npy')
# y_train = np_utils.to_categorical(y_train, num_classes)
# y_test = np_utils.to_categorical(y_test, N_CLASSES)

# x_train = np.load('X_trains_features.npy') 
# y_train = np.load('Y_trains_features.npy')
# x_test = np.load('X_test_features.npy')
# y_test = np.load('Y_test_features.npy')


# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
tr_pairs, tr_y = create_pairs(x_train, digit_indices)

# print(digit_indices)
# print(tr_pairs.shape)
# np.save('training_pairs.npy', tr_pairs)
# print(tr_pairs[:, 0].shape)
# print(tr_pairs[:, 1].shape)
print(tr_y)
digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
te_pairs, te_y = create_pairs(x_test, digit_indices)
# print(digit_indices)
# print(te_pairs.shape)
# print(te_y)



# network definition
# base_network = build_inception_resnet_V2()
base_network = alex()


input_shape = (416, 416, 3)
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])


# print(distance)
model = Model([input_a, input_b], distance)
# model = Model([processed_a, processed_b], distance)
# model.load_weights('siamese_train.h5')
# # train
rms = RMSprop()
adam = Adam(lr=0.0001)
model.compile(loss=contrastive_loss, optimizer=adam, metrics=[accuracy, precision, recall, f1])
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=16,
          epochs=epochs,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y)
          )


model.save_weights('siamese_crops+1.h5')
# compute final accuracy on training and test sets








y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
print(y_pred)
tr_acc = compute_accuracy(tr_y, y_pred)

y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
print(y_pred)
te_acc = compute_accuracy(te_y, y_pred)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))