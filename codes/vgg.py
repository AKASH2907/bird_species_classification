from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Dropout, Flatten, Input
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, LearningRateScheduler
from time import time
import math
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19

batch_size = 16

def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.1
    epochs_drop = 4.0
    lrate = initial_lrate * math.pow(drop, math.floor((epoch)/epochs_drop))
    return lrate



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
    x = Dense(512, activation='relu', name='dense_1')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', name='dense_2')(x)
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

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model 


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        'data/train',  # this is the target directory
        target_size=(416, 416),  # images resized to 1000*1000
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        'data/test',  
        target_size=(416, 416),  # images  resized to 1000*1000
        batch_size=batch_size,
        class_mode='categorical')


training = build_vgg()

filepath = "vgg_wts/vgg16-1-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose =1, save_best_only=True, mode='max', save_weights_only=True)
early_stopper = EarlyStopping(monitor='val_acc', verbose=1, patience=3)
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
lrate = LearningRateScheduler(step_decay)
callback_list = [checkpoint, early_stopper, tensorboard, lrate]

training.fit_generator(
        train_generator,
        steps_per_epoch=1600 // batch_size,
        epochs=15,
        validation_data=validation_generator,
        # validation_steps=800 // batch_size
        callbacks= callback_list
        )
model.save_weights('vgg16_1.h5')