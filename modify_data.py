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
destination_folder = "./train_rename/"


def rename_files():
    classes = 1

    for i in species:

        #
        source = join(source_folder, i)
        files = listdir(source)
        files.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

        for file in files:

            if not exists(join(destination_folder, i)):
                destination = makedirs(join(destination_folder, i))

            if classes < 10:

                images = 0
                for file in files:

                    if images < 10:
                        # 00 - Data Augmentation
                        # Classes 01 - Images 100101.jpg
                        copyfile(
                            join(source, file),
                            join(
                                destination,
                                str(100)
                                + str(classes)
                                + str(0)
                                + str(images)
                                + ".jpg",
                            ),
                        )

                    elif images >= 10:
                        copyfile(
                            join(source, file),
                            join(
                                destination,
                                str(100)
                                + str(classes)
                                + str(images)
                                + ".jpg",
                            ),
                        )

                    images += 1

            elif classes >= 10:

                images = 0

                for file in files:

                    if images < 10:
                        copyfile(
                            join(source, file),
                            join(
                                destination,
                                str(10)
                                + str(classes)
                                + str(0)
                                + str(images)
                                + ".jpg",
                            ),
                        )

                    elif images >= 10:
                        copyfile(
                            join(source, file),
                            join(
                                destination,
                                str(10) + str(classes) + str(images) + ".jpg",
                            ),
                        )
                    images += 1

        classes += 1


if __name__ == "__main__":

    rename_files()
