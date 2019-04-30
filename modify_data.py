from os.path import isfile, join
from os import rename, listdir, rename, makedirs
from shutil import copyfile, move
import random

species = ["blasti", "bonegl", "brhkyt", "cbrtsh", "cmnmyn", "gretit", "hilpig", "himbul", "himgri", "hsparo", "indvul"
    , "jglowl", "lbicrw", "mgprob", "rebimg", "wcrsrt"]


initial = './train_data/'
final = './train/'


def rename_files():
	classes = 1

	for i in species:

		# 
		source = join(initial, i)
		files = listdir(source)
		files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

		for file in files:

			destination = join(final, i)
			
			if classes<10:
				
				images = 0
				for file in files:

					if images<10:
						# 00 - Data Augmentation 00- Classes 01 - Images 100101.jpg
						copyfile(join(source, file), join(destination,  str(100) + str(classes) + str(0) + str(images) + '.jpg'))
					
					elif images>=10:
						copyfile(join(source, file), join(destination, str(100) + str(classes) + str(images) + '.jpg'))

					images+=1

			elif classes>=10:

				images = 0
				
				for file in files:
				
					if images<10:
						copyfile(join(source, file), join(destination, str(10) + str(classes) + str(0) + str(images) + '.jpg'))
				
					elif images>=10:
						copyfile(join(source, file), join(destination, str(10) + str(classes) + str(images) + '.jpg'))
					images+=1

		classes+=1



if __name__ == '__main__':

	makedirs(final)

	for i in species:
	
		makedirs(join(final, i))

	rename_files()

	

