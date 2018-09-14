import cv2
import numpy as np
from os.path import isfile, join
import os
from shutil import copyfile
from imgaug import augmenters as iaa

initial = './train_data/'
final = './train/'
 
species = ["blasti","bonegl", "brhkyt", "cbrtsh", "cmnmyn", "gretit", "hilpig", "himbul", "himgri", "hsparo", 
"indvul", "jglowl", "lbicrw", "mgprob", "rebimg", "wcrsrt"]

datagen = ImageDataGenerator()


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


def save_images(augmentated_image, destination, number_of_images, classes, types):


    image_number = str(number_of_images)
    number_of_images = int(number_of_images)

    if classes<10:

        if number_of_images<10:
            cv2.imwrite(join(destination, str(types)+ str(0) + str(classes) + image_number + '.jpg'), augmentated_image)
        
        elif number_of_images>=10:
            cv2.imwrite(join(destination, str(types) + str(0) + str(classes) + image_number + '.jpg'), augmentated_image)

    elif classes>=10:
        
        if number_of_images<10:
            cv2.imwrite(join(destination, str(types) + str(classes) + image_number + '.jpg'), augmentated_image)

        elif number_of_images>=10:
            cv2.imwrite(join(destination, str(types) + str(classes) + image_number + '.jpg'), augmentated_image)




# Dataset Augmentation

gauss = iaa.AdditiveGaussianNoise(scale=0.2*255)

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


def main():

    for i in species:
        path = join(initial, i)

        files = os.listdir(path)
        files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))


        batches = []
        value = []
        c = files[0]
        c = int(c[2:4])
        # print(c)
        for image in files:

            if int(image[0])==4:
                
                value.append(image[4:6])
                im = join(path, image)

                img = cv2.imread(im, 1)
                batches.append(img)


        j = 0
        if len(batches)<9:
            for images_aug in aug.augment_images(batches):
                save_images(images_aug, path, value[j], c, 17)
                j += 1
        c +=1



if __name__ == '__main__':
    main()