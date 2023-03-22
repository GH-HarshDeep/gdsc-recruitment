import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from keras.datasets import cifar10
(x_train, y_train),(x_test, y_test) = cifar10.load_data()

classification = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

x_train = x_train / 255
x_test = x_test / 255

model = Sequential()

#first_layer
model.add( Conv2D(32, (5,5), activation='relu', input_shape = (32,32,3)))

#pooling_layer
model.add(MaxPooling2D( pool_size = (2,2)))

#convolution_layer
model.add( Conv2D(32, (5,5), activation='relu'))

#another_pooling_layer
model.add(MaxPooling2D( pool_size = (2,2)))

#flatten_layer
model.add(Flatten())

#layer with 1000 neurons
model.add(Dense(1000, activation='relu'))

#Drop out layer
model.add(Dropout(0.5))

#layer with 500 neurons
model.add(Dense(500, activation='relu'))

#Drop out layer
model.add(Dropout(0.5))

#layer with 250 neurons
model.add(Dense(250, activation='relu'))

#layer with 10 neurons
model.add(Dense(10, activation='softmax'))



model.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

hist = model.fit(x_train,y_train_one_hot,
                 batch_size = 256,
                 epochs = 10,
                 validation_split = 0.2)

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Model Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Val'], loc='upper right')
plt.show()
