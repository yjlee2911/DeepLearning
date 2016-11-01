import os
import scipy.misc
import numpy as np
from PIL import Image
import skimage.io as io
from keras.optimizers import SGD
from keras.models import Sequential
from keras.constraints import maxnorm
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D

model2 = Sequential()
model2.add(Convolution2D(36, 3, 3, border_mode='same', activation='relu', dim_ordering = 'th',
                        W_constraint=maxnorm(3), input_shape=(3, 18, 18)))
print model2.output_shape

model2.add(Dropout(0.2))

model2.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu', dim_ordering = 'th', 
                        W_constraint=maxnorm(3)))
print model2.output_shape

model2.add(MaxPooling2D(pool_size=(2, 2), border_mode='same', dim_ordering = 'th'))
print model2.output_shape

model2.add(Convolution2D(12, 3, 3, border_mode='same', activation='relu', dim_ordering = 'th', 
                        W_constraint=maxnorm(3)))
print model2.output_shape

model2.add(MaxPooling2D(pool_size=(2, 2), border_mode='same', dim_ordering = 'th'))
print model2.output_shape

model2.add(Flatten())
print model2.output_shape

model2.add(Dense(64, activation='relu', W_constraint=maxnorm(3)))
print model2.output_shape

model2.add(Dropout(0.5))

model2.add(Dense(10, activation='softmax'))
print model2.output_shape

epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model2.compile(optimizer = 'sgd', loss='categorical_crossentropy', metrics=['accuracy'])

model2.fit(train_data, train_target, nb_epoch=epochs, verbose=2)