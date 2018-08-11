import cv2
from matplotlib import pyplot as plt
import numpy as np
from keras.utils import *
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from os.path import isfile, join
import os
import argparse
from shutil import copyfile
import imgaug as ia
from imgaug import augmenters as iaa
import glob
from scipy import misc
# datapath = '../modify_data/hsparo/'
# destination = '../modify_data/images/'
dest = './train_data/'
dest1 = './train/'
# aug = '../modify_data/test_aug/'
# 
species = ["blasti","bonegl", "brhkyt", "cbrtsh", "cmnmyn", "gretit", "hilpig", "himbul", "himgri", "hsparo", 
"indvul", "jglowl", "lbicrw", "mgprob", "rebimg", "wcrsrt"]
datagen = ImageDataGenerator()
# new_files = os.listdir(destination)
''' Type of Augmentation:
10 - Normal Image
20 - Gaussian Noise - 0.1* 255
30 - Gaussian Blur - sigma - 3.0
40 - Flip - Horizaontal
50 - Contrast Normalization - (0.5, 1.5)
60 - Hue
70 - Crop and Pad

Flipped
11 - Add - 2,3,4,5,6,12,13,14      7, 15, 16
12 - Multiply - 2,3,4,5,6,12,13,14 7, 15, 16
13 - Sharpen 
14 - Gaussian Noise - 0.2*255
15 - Gaussian Blur - sigma - 0.0-2.0
16 - Affine Translation 50px x, y
17 - Hue Value
'''


def save_images(augmentated_image, destination, k, classes, types):

    # im = cv2.resize(augmentated_image, (416, 416))
    l = str(k)
    k = int(k)
    if classes<10:
        if k<10:
            print(join(destination, str(types) + str(0) + str(classes) + l + '.jpg'))
            cv2.imwrite(join(destination, str(types)+ str(0) + str(classes) + l + '.jpg'), augmentated_image)
        elif k>=10:
            print(join(destination, str(types) + str(0) + str(classes) + l + '.jpg'))
            cv2.imwrite(join(destination, str(types) + str(0) + str(classes) + l + '.jpg'), augmentated_image)

    elif classes>=10:
        if k<10:
            print(join(destination, str(types) + str(classes) + l + '.jpg'))
            cv2.imwrite(join(destination, str(types) + str(classes) + l + '.jpg'), augmentated_image)

        elif k>=10:
            print(join(destination, str(types) + str(classes) + l + '.jpg'))
            cv2.imwrite(join(destination, str(types) + str(classes) + l + '.jpg'), augmentated_image)

# print(batches.shape)
# images = np.array([cv2.imread(join(path, image), 1) for image in files], dtype=np.uint8)    

# Done
gauss = iaa.Sequential([iaa.AdditiveGaussianNoise(scale=0.2*255)])
# blur = iaa.GaussianBlur(sigma=(3.0))
# flip = iaa.Fliplr(1.0)
# contrast = iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)
sharp = iaa.Sharpen(alpha=(0, 0.3), lightness=(0.7, 1.3))
affine = iaa.Affine(translate_px={"x": (-50, 50), "y": (-50, 50)})
# add = iaa.Add((-20, 20), per_channel=0.5)
# multiply  = iaa.Multiply((0.8, 1.2), per_channel=0.5)
hue = iaa.Sequential([
    iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
    iaa.WithChannels(0, iaa.Add((50, 100))),
    iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")
])

aug = iaa.Sequential([
    iaa.Fliplr(1.0),
    iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
    iaa.WithChannels(0, iaa.Add((50, 100))),
    iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")
    # iaa.Sharpen(alpha=(0, 0.3), lightness=(0.7, 1.3))
    # iaa.AdditiveGaussianNoise(scale=0.1*255)
])


# edge = iaa.EdgeDetect(alpha=0.8)
# matrix = np.array([[0, -1, 0],
#                    [-1, 4, -1],
#                    [0, -1, 0]])
# conv = iaa.Convolve(matrix=matrix)

# print(images_aug.shape)
# aug = aug+'flip'
# c =1
for i in species:
    path = join(dest1, i)
    files = os.listdir(path)

    
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    print(len(files))
    # batches = []
    # value = []
    # c = files[0]
    # c = int(c[2:4])
    # # print(c)
    # for image in files:

    #     if int(image[0])==4:
    #         value.append(image[4:6])
    #         # print(value)
    #         im = join(path, image)
    #         # print(im)

    #         img = cv2.imread(im, 1)
    #         # print(i, img.shape)
    #         # resized = cv2.resize(img, (416, 416))
    #         # i += 1
    #         # print(resized.shape)
    #         batches.append(img)
    # # print(value)
    # # print(len(batches))
    # j = 0
    # path1 = join(dest1, i)
    # if len(batches)<9:
    #     for images_aug in aug.augment_images(batches):
    #         # print(value[j])
    #         save_images(images_aug, path1, value[j], c, 17)
    #         j += 1
    #     misc.imshow(images_aug)
    #     resized = cv2.resize(images_aug, (416, 416))
    #     cv2.imshow('i', resized)
    #     key = cv2.waitKey(0)
    #     if key==27:
    #         break
    # c +=1

    # break

# cv2.destroyAllWindows()
# # cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(len([name for name in os.listdir('.') if os.path.isfile(datapath)]))
# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# # path joining version for other paths
# DIR = '/tmp'
# print len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])


# def aug:
#     seq = iaa.Sequential([
#     # iaa.Fliplr(0.5)  # horizontal flips
#     # iaa.Crop(percent=(0, 0.1)),  # random crops

#     # 2) Gaussian Blur
#     # Small gaussian blur with random sigma between 0 and 0.5.
#     # But we only blur about 50% of all images.
#     # iaa.Sometimes(0.5,
#     #               iaa.GaussianBlur(sigma=(0, 0.5))
#     #               )

#     # ,
    
#     # 3) Contrast Normalization
#     # Strengthen or weaken the contrast in each image.
#     # iaa.ContrastNormalization((0.75, 1.5)),
    
#     # 4) gAUSSIAN NOISES    
#     # Add gaussian noise.
#     # For 50% of all images, we sample the noise once per pixel.
#     # For the other 50% of all images, we sample the noise per pixel AND
#     # channel. This can change the color (not only brightness) of the
#     # pixels.
#     iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1 * 255), per_channel=0.5)
#     # ,
    
#     # 5) Brightness
#     # Make some images brighter and some darker.
#     # In 20% of all cases, we sample the multiplier once per channel,
#     # which can end up changing the color of the images.
#     # iaa.Multiply((0.8, 1.2), per_channel=0.2)
#     # ,
    
#     # 6) Affine transformations
#     # Apply affine transformations to each image.
#     # Scale/zoom them, translate/move them, rotate them and shear them.
#     # iaa.Affine(
#     #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
#     #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
#     #     rotate=(-25, 25),
#     #     shear=(-8, 8)
#     # )
#     ], random_order=True)  # apply augmenters in random order