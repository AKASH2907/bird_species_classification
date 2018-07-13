from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Dropout, Flatten, Input
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, LearningRateScheduler
from time import time
import math

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.1
    epochs_drop = 4.0
    lrate = initial_lrate * math.pow(drop, math.floor((epoch)/epochs_drop))
    return lrate


# model
v1 = Input(shape=(416, 416, 3))
l1 = Conv2D(32, (3, 3), activation='relu')(v1)
p1 = MaxPooling2D((2, 2))(l1)
l2 = Conv2D(32, (3, 3), activation='relu')(p1)
p2 = MaxPooling2D((2, 2))(l2)
dr1 = Dropout(0.25)(p2)
l3 = Conv2D(64, (3, 3), activation='relu')(dr1)
p3 = MaxPooling2D((2, 2))(l3)
l4 = Conv2D(128, (3, 3), activation='relu')(p3)
p4 = MaxPooling2D((2, 2))(l4)
# l5 = Conv2D(128, (3, 3), activation='relu')(p4)
# p5 = MaxPooling2D((2, 2))(l5)
f = Flatten()(p4)

d1 = Dense(64, activation='relu')(f)
drop = Dropout(0.25)(d1)
output = Dense(16, activation='softmax')(drop)

model = Model(inputs=v1, outputs=output)

# print(model.summary())
# plot_model(model, to_file='multiple_inputs.png')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
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

filepath = "aug2/drop-0.25-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose =1, save_best_only=True, mode='max', save_weights_only=True)
early_stopper = EarlyStopping(monitor='val_acc', verbose=1, patience=3)
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
lrate = LearningRateScheduler(step_decay)
callback_list = [checkpoint, early_stopper, tensorboard, lrate]
model.fit_generator(
        train_generator,
        steps_per_epoch=1600 // batch_size,
        epochs=15,
        validation_data=validation_generator,
        # validation_steps=800 // batch_size
        callbacks= callback_list
        )
model.save_weights('aug2.h5')