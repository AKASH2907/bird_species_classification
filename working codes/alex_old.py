from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Dropout, Flatten, Input, BatchNormalization
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, LearningRateScheduler
from time import time
import math
from keras import backend as K

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.1
    epochs_drop = 4.0
    lrate = initial_lrate * math.pow(drop, math.floor((epoch)/epochs_drop))
    return lrate

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

batch_size = 32
# input_size = (416, 416, 3)
# nb_classes = 16
# mean_flag = False
# model = get_alexnet(input_size,nb_classes)

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


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1./255)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        'train/',  # this is the target directory
        target_size=(416, 416),  # images resized to 1000*1000
        batch_size=batch_size,
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
        'test/',  
        target_size=(416, 416),  # images  resized to 1000*1000
        batch_size=batch_size,
        class_mode='categorical')

filepath = "wts/alex/alex-1-{epoch:02d}-{val_f1:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_f1', verbose =1, save_best_only=True, mode='max', save_weights_only=True)
early_stopper = EarlyStopping(monitor='val_f1', verbose=1, patience=4)
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
# lrate = LearningRateScheduler(step_decay)
callback_list = [checkpoint, early_stopper, tensorboard]
training = alex()
# print("Loading weights......")
# training.load_weights('alex_3.h5')

print("weights loaded .......")
history = training.fit_generator(
        train_generator,
        steps_per_epoch=1600//batch_size,
        epochs=20,
        validation_data=test_generator,
        # validation_steps=800 // batch_size
        callbacks= [checkpoint, early_stopper, tensorboard]
        )
training.save_weights('wts/alex_1.h5')



# class LRN(Layer):

#     def __init__(self, alpha=0.0001,k=1,beta=0.75,n=5, **kwargs):
#         self.alpha = alpha
#         self.k = k
#         self.beta = beta
#         self.n = n
#         super(LRN, self).__init__(**kwargs)
    
#     def call(self, x, mask=None):
#         b, ch, r, c = x.shape
#         half_n = self.n // 2 # half the local region
#         # orig keras code
#         #input_sqr = T.sqr(x)  # square the input
#         input_sqr = K.square(x) # square the input
#         # orig keras code
#         #extra_channels = T.alloc(0., b, ch + 2 * half_n, r,c)  # make an empty tensor with zero pads along channel dimension
#         #input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n+ch, :, :],input_sqr) # set the center to be the squared input

#         extra_channels = K.zeros((b, int(ch) + 2 * half_n, r, c))
#         input_sqr = K.concatenate([extra_channels[:, :half_n, :, :],input_sqr, extra_channels[:, half_n + int(ch):, :, :]],axis = 1)

#         scale = self.k # offset for the scale
#         norm_alpha = self.alpha / self.n # normalized alpha
#         for i in range(self.n):
#             scale += norm_alpha * input_sqr[:, i:i+int(ch), :, :]
#         scale = scale ** self.beta
#         x = x / scale
#         return x

#     def get_config(self):
#         config = {"alpha": self.alpha,
#                   "k": self.k,
#                   "beta": self.beta,
#                   "n": self.n}
#         base_config = super(LRN, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))