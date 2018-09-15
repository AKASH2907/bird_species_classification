import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Dropout, Flatten, Input, AveragePooling2D
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import plot_model, np_utils
from keras.callbacks import LearningRateScheduler
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects


BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1
N_CLASSES = 16
EPOCHS = 7


# Swish Activation Function
def swish(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': Activation(swish)})


# Learning Step Decay by 10e-1 after every 4 epochs
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.1
    epochs_drop = 4.0
    lrate = initial_lrate * math.pow(drop, math.floor((epoch)/epochs_drop))
    return lrate


# Calculates Precision Accuracy
def precision(y_true, y_pred):
    """Precision metric.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


# Calculates Recall Accuracy
def recall(y_true, y_pred):
    """Recall metric.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


# Calculates F1 score
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
    return (2*((precision*recall)/(precision+recall+K.epsilon())))



# Inception_ResNet_V2 model define
def build_inception_resnet_V2(img_shape=(416, 416, 3), n_classes=16, l2_reg=0.,
                load_pretrained=True, freeze_layers_from='base_model'):
    
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
    x = Dropout(0.25)(x)
    predictions = Dense(n_classes, activation='softmax', name='predictions', kernel_initializer='he_uniform')(x)

    # This is the model that will  be trained
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


    # Compiling Model with Adam Optimizer
    adam = Adam(0.0001) 
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[precision, recall, f1])  
    return model 


if __name__ == '__main__':
    
    # Loading Cropped Images for Training resized to 416x416
    # x_train_crop = np.load('X_train_crop.npy')
    # y_train_crop = np.load('Y_train_crop.npy')
    # y_train_crop = np_utils.to_categorical(y_train, N_CLASSES)

    # Loading Original Images for training resized to 416x416
    # x_train_original = np.load('X_train.npy')
    # y_train_original = np.load('Y_train.npy')
    # x_valid          = np.load('X_valid.npy')
    # y_valid          = np.load('Y_valid.npy')

    # Loading Original Images for Testing rsized to 416x416
    x_test = np.load('X_test.npy')
    y_test = np.load('Y_test_categorical.npy')

    # print(x_train.shape, y_train.shape)

    # Learning Rate Schedule
    lrate = LearningRateScheduler(step_decay)

    # Loading Model
    model = build_inception_resnet_V2()

    # Loading Trained weights
    model.load_weights('inception_resnet_v2_images+crops.h5')

    # Model Fitting with 10% of the images used for Validation purpose
    # history = model.fit(x_train_original, y_train_original, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose= 1, 
    #     # steps_per_epoch=x_train.shape[0]//BATCH_SIZE,
    #     callbacks = [lrate],
    #     validation_split=VALIDATION_SPLIT
    #     )

    # Save Model Weights
    # model.save_weights('inception_resnet_crops.h5')

    # Calculate score over test data
    score = model.evaluate(x_test, y_test, verbose=1, batch_size= BATCH_SIZE)

    # Prints Precision, Recall, and F-1 score
    print(score)