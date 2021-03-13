import os
import random
from shutil import copyfile


def split_data(source, training, testing, split_size):
    files = []
    for filename in os.listdir(source):
        file = source + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
    training_length = int(len(files) * split_size)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0: training_length]
    testing_set = shuffled_set[:testing_length]

    for filename in training_set:
        this_file = source + filename
        destination = training + filename
        # print(this_file)
        # print(destination)
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = source + filename
        destination = testing + filename
        copyfile(this_file, destination)


cat_source = 'PetImages/Cat/'
training_cats_dir = 'cats-vs-dogs/training/cats/'
testing_cats_dir = 'cats-vs-dogs/testing/cats/'
dog_source = 'PetImages/Dog/'
training_dog_dir = 'cats-vs-dogs/training/dogs/'
testing_dog_dir = 'cats-vs-dogs/testing/dogs/'

split = 0.7

catsize = len(os.listdir(cat_source))
dogsize = len(os.listdir(dog_source))

split_data(cat_source, training_cats_dir, testing_cats_dir, split)
split_data(dog_source, training_dog_dir, testing_dog_dir, split)
