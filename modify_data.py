from os.path import join, exists
from os import listdir, makedirs
from shutil import copyfile

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


source_folder = "./train_data/"
destination_folder = "./train/"


def rename_files():
    """
    Initially the file names are incosistent. This function
    changes the file name to make it more understanding.

    Example - for example, DSC_6272.jpg may be changed to 100101.jpg
    For bird_specie_counter < 10, in this,
    100 -> original image, 1 -> Class Number, 01 -> Image Number

    Similarly, for the case if the species counter is greater than 10.
    """
    bird_specie_counter = 1

    for bird_specie in species:

        #
        source_image_dir = join(source_folder, bird_specie)
        print(source_image_dir)
        source_images = listdir(source_image_dir)
        print(source_images)

        for source_image in source_images:

            destination = join(destination_folder, bird_specie)
            print(destination)
            if bird_specie_counter < 10:

                images = 0
                for source_image in source_images:

                    if images < 10:
                        copyfile(
                            join(source_image_dir, source_image),
                            join(
                                destination,
                                str(100)
                                + str(bird_specie_counter)
                                + str(0)
                                + str(images)
                                + ".jpg",
                            ),
                        )

                    elif images >= 10:
                        copyfile(
                            join(source_image_dir, source_image),
                            join(
                                destination,
                                str(100)
                                + str(bird_specie_counter)
                                + str(images)
                                + ".jpg",
                            ),
                        )

                    images += 1

            elif bird_specie_counter >= 10:

                images = 0

                for source_image in source_images:

                    if images < 10:
                        copyfile(
                            join(source_image_dir, source_image),
                            join(
                                destination,
                                str(10)
                                + str(bird_specie_counter)
                                + str(0)
                                + str(images)
                                + ".jpg",
                            ),
                        )

                    elif images >= 10:
                        copyfile(
                            join(source_image_dir, source_image),
                            join(
                                destination,
                                str(10)
                                + str(bird_specie_counter)
                                + str(images)
                                + ".jpg",
                            ),
                        )
                    images += 1

        bird_specie_counter += 1


if __name__ == "__main__":
    for bird_specie in species:
        if not exists(join(destination_folder, bird_specie)):
            destination = makedirs(join(destination_folder, bird_specie))
    rename_files()
