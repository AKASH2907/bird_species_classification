import cv2
from os.path import join
import os
from imgaug import augmenters as iaa

augmented_image_dir = "./train/"

species = [
    "blasti",
    "bonegl",
    "brhkyt",
    "cbrtsh",
    "cmnmyn",
    "gretit",
    "hilpig",
    "himbul",
    "himgri",
    "hsparo",
    "indvul",
    "jglowl",
    "lbicrw",
    "mgprob",
    "rebimg",
    "wcrsrt",
]


""" Naming conventions can be different. This is
what I've used at my time. I just followed the table
present to generate that much number of images.

Type of Augmentation:
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
"""


def save_images(
    augmentated_image,
    destination,
    number_of_images,
    bird_specie_counter,
    types
):

    image_number = str(number_of_images)
    number_of_images = int(number_of_images)

    if bird_specie_counter < 10:

        if number_of_images < 10:
            cv2.imwrite(
                join(
                    destination,
                    str(types)
                    + str(0)
                    + str(bird_specie_counter)
                    + image_number
                    + ".jpg",
                ),
                augmentated_image
            )

        elif number_of_images >= 10:
            cv2.imwrite(
                join(
                    destination,
                    str(types)
                    + str(0)
                    + str(bird_specie_counter)
                    + image_number
                    + ".jpg",
                ),
                augmentated_image
            )

    elif bird_specie_counter >= 10:

        if number_of_images < 10:
            cv2.imwrite(
                join(
                    destination,
                    str(types)
                    + str(bird_specie_counter)
                    + image_number
                    + ".jpg",
                ),
                augmentated_image
            )

        elif number_of_images >= 10:
            cv2.imwrite(
                join(
                    destination,
                    str(types)
                    + str(bird_specie_counter)
                    + image_number
                    + ".jpg",
                ),
                augmentated_image
            )


# Dataset Augmentation

gauss = iaa.AdditiveGaussianNoise(scale=0.2 * 255)
# blur = iaa.GaussianBlur(sigma=(3.0))
# flip = iaa.Fliplr(1.0)
# contrast = iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)
sharp = iaa.Sharpen(alpha=(0, 0.3), lightness=(0.7, 1.3))
affine = iaa.Affine(translate_px={"x": (-50, 50), "y": (-50, 50)})
# add = iaa.Add((-20, 20), per_channel=0.5)
# multiply  = iaa.Multiply((0.8, 1.2), per_channel=0.5)

hue = iaa.Sequential(
    [
        iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
        iaa.WithChannels(0, iaa.Add((50, 100))),
        iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB"),
    ]
)

aug = iaa.Sequential(
    [
        iaa.Fliplr(1.0),
        iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
        iaa.WithChannels(0, iaa.Add((50, 100))),
        iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB"),
    ]
)


def main():
    """Read images, apply augmentation and save images.
    Two types of image augmentation is applied. One is on normal
    image whose image name starts with 1 nad another is one flipped
    image which starts with 4. Bird classes are mentioned above which
    type of augmentation is applied on which type of image and which
    type of specie. We check the first value of image path
    and compare it 1/4 to apply the data augmentation accordingly.
    """
    for bird_specie in species:
        augmented_image_folder = join(augmented_image_dir, bird_specie)
        source_images = os.listdir(augmented_image_folder)
        print(source_images)
        source_images.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

        augmented_images_arr = []
        img_number = []
        bird_specie_number = source_images[0]
        bird_specie_number = int(bird_specie_number[2:4])
        for source_image in source_images:

            if int(source_image[0]) == 1:

                img_number.append(source_image[4:6])
                img_path = join(augmented_image_folder, source_image)

                img = cv2.imread(img_path)
                augmented_images_arr.append(img)

        counter = 0
        if len(augmented_images_arr) < 9:
            # Applying Gaussian image augmentation
            for augmented_image in gauss.augment_images(augmented_images_arr):
                save_images(
                    augmented_image,
                    augmented_image_folder,
                    img_number[counter],
                    bird_specie_number,
                    20,
                )
                counter += 1


if __name__ == "__main__":
    main()
