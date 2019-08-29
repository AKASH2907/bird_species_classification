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

		
		source = join(initial, i)
		files = listdir(source)
		
		
		# 
		
		#
		#
		
		#lambda(f)=int(''.join(filter(str.isdigit,f)))
		#str.join函数 举例，'-v-'.join('a','b','c') == 'a-v-b-v-c'
		#filter(过滤器)参数 filter(function,list or str or tutle.....)
		#filter按照function规则从list中筛选出满足的元素，返回元素的迭代器对象，
		#其中迭代器对象a，可以通过list(a)转换成list从而查看

		#list.sort(key,reverse)
		#这句话就是将files中的每一个元素file，传入key对应的函数，得到一个返回值，，
		#此时，list:files排序等价于对其中的file进行lambda变换后的返回值的排序
		#而此中的lambda的含义为：将传入的字符串（比如说'1234e4567.txt'），筛选出其中的所有数字字符['1','2','3','4','5','6','7','8']，
		#并返回这个列表的迭代器对象，''.join()将收到的迭代器对象的所有元素（字符），把他们连接起来，中间用‘’分开，合并成一个字符串
		
		#
		#
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

	

