from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = keras.models.load_model('cat_vs_dog.h5')

model.summary()

datagen = ImageDataGenerator(rescale=1.0 / 255.)

testing_dir = 'cats-vs-dogs/testing'

test_data = datagen.flow_from_directory(testing_dir, batch_size=64, class_mode='binary', target_size=(200, 200))

step_size = test_data.n // test_data.batch_size

test = model.evaluate(test_data, return_dict=True, batch_size=64)

print(test)
