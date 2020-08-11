from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

from DCGAN_image import load_image


import matplotlib.pyplot as plt

import sys

import numpy as np

class DCGAN():
    def __init__(self, rows, cols, channels, z = 10):
        #input Shape
        self.img_rows =rows
        self.img_cols = cols
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = z
        self.noise_shape = self.img_shape

        optimizer = Adam(0.0002, 0.5)

        #Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss = 'binary_crossentropy',
            optimizer = optimizer,
            metrics=['accuracy'])

        #Build the generator
        self.generator = self.build_generator()

        #The generator takes noise as input and generates imgs
        z = Input(shape=(self.noise_shape))
        img = self.generator(z)

        #for the combined model we will only train the generator
        self.discriminator.trainable = False

        #the discriminator takes the generated images as input and determines validity
        valid = self.descriminator(img)

        #The combined model (stacked generator and discriminator)
        #trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7*7, activation='relu', input_dim =self.latent_dim))
        model.add(Reshape((7,7,128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same", input_shape=(self.noise_shape), activation='relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(activation("relu"))
        model.add(Upsampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.noise_shape))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(432, kernel_size = 3, strides=2, input_shape= self.img_shape, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding = "same"))
        model.add(ZeroPadding2D(padding=((0,1), (0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding = "same"))
        model.add(ZeroPadding2D(padding=((0,1), (0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding = "same"))
        model.add(ZeroPadding2D(padding=((0,1), (0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activatioion='sigmooid'))

        model.summary()

        img =Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=256, save_interval=50):

        #load the dataset 
        X_train = load_images('D:/data/Gan/Dog')
        noise = load_image('D:/data/Gan/Human')

        #Rescale -1 to 1
        X_train = X_train / 127.5 -1.
        #X_train = np.expand_dims(X_train, axis = 3)
        noise = noise / 127.5 -1.

        #Adversarial ground truths
        valid = np.ones((batch_size, 1))