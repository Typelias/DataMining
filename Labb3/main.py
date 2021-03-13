import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

traingin_dir = 'cats-vs-dogs/training'
train_datagen = ImageDataGenerator(rescale=1.0 / 255.)
train_generator = train_datagen.flow_from_directory(traingin_dir, batch_size=64, class_mode='binary',
                                                    target_size=(200, 200))

model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

with tf.device('/gpu:0'):
    model.compile(optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
    start = time.time()
    model.fit(train_generator, epochs=50, verbose=1)
    end = time.time()
    elapsed_time = end - start
    print(elapsed_time)

model.save('cat_vs_dog.h5')
