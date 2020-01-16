from os.path import isfile, join
from os import listdir, rename, makedirs
from shutil import copyfile, move
import random

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

initial = "./train/"
final = "./validation/"


def create_validation():

    for i in species:

        source = join(initial, i)
        destination = join(final, i)

        files = listdir(source)
        number = len(files)
        if number < 78:
            samples = random.sample(files, 6)
            for file in samples:
                move(join(source, file), join(destination, file))

        elif 78 <= number <= 81:
            samples = random.sample(files, 8)
            for file in samples:
                move(join(source, file), join(destination, file))

        elif number > 85:
            samples = random.sample(files, 9)
            for file in samples:
                move(join(source, file), join(destination, file))


if __name__ == "__main__":

    makedirs(final)

    for i in species:

        makedirs(join(final, i))

    create_validation()
