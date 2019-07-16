import os
import numpy as np
#from IPython.core.debugger import Tracer
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, UpSampling2D, Conv2D
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam
#import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class GAN(object):
    """ Generative Adversarial Network class """
    def __init__(self, width=64, height=64, channels=3):

        self.width = width
        self.height = height
        self.channels = channels

        self.shape = (self.width, self.height, self.channels)

        self.optimizer = Adam(lr=0.0002, beta_1=0.5, decay=8e-8)
        self.G = self.__generator()
        self.G.compile(loss='binary_crossentropy', optimizer=self.optimizer)

        self.D = self.__discriminator()
        self.D.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        self.stacked_generator_discriminator = self.__stacked_generator_discriminator()

        self.stacked_generator_discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer)


    def __generator(self):
        """ Declare generator """

        generator=Sequential()
        generator.add(Dense(128, input_shape=(100,), activation='linear', use_bias=True))
        generator.add(LeakyReLU(alpha=0.2))
        generator.add(Dense(256, activation='linear', use_bias=True))
        generator.add(BatchNormalization())
        generator.add(LeakyReLU(alpha=0.2))
        generator.add(Dense(512, activation='linear', use_bias=True))
        generator.add(BatchNormalization())
        generator.add(LeakyReLU(alpha=0.2))
        generator.add(Dense(12288, activation='linear', use_bias=True))
        generator.add(Dense(12288, activation='tanh'))
        generator.add(Reshape((64,64,3)))

        return  generator

    def __discriminator(self):
        """ Declare discriminator """

        discriminator=Sequential()
        discriminator.add(Flatten())
        discriminator.add(Dense(512, activation='linear', use_bias=True))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dense(256, activation='linear', use_bias=True))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dense(1, activation='linear', use_bias=True))
        discriminator.add(Dense(1, activation='sigmoid'))

        return discriminator


    def __stacked_generator_discriminator(self):

        self.D.trainable = False

        model = Sequential()
        model.add(self.G)
        model.add(self.D)

        return model

    def train(self, X_train, epochs=50000, batch = 512):

        for cnt in range(epochs):

            ## train discriminator
            for i in range(5):
                self.D.trainable = True
                random_index = np.random.randint(0, len(X_train) - np.int64(batch/2))
                np.random.shuffle(X_train)
                legit_images = X_train[random_index : random_index + np.int64(batch/2)].reshape(np.int64(batch/2), self.width, self.height, self.channels)

                gen_noise = np.random.normal(0, 1, (np.int64(batch/2), 100))
                syntetic_images = self.G.predict(gen_noise)

                x_combined_batch = np.concatenate((legit_images, syntetic_images))
                y_combined_batch = np.concatenate((np.ones((np.int64(batch/2), 1)), np.zeros((np.int64(batch/2), 1))))

                d_loss = self.D.train_on_batch(x_combined_batch, y_combined_batch)

            # train generator
            self.D.trainable = False
            noise = np.random.normal(0, 1, (batch, 100))
            y_mislabled = np.ones((batch, 1))

            g_loss = self.stacked_generator_discriminator.train_on_batch(noise, y_mislabled)

            print ('epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (cnt, d_loss[0], g_loss))
            if (cnt+1)%1000==0:
                self.stacked_generator_discriminator.save('model/dcgan_model_{}.h5'.format(cnt+1))
                self.G.save('model/dcgan_{}.h5'.format(cnt+1))


if __name__ == '__main__':

    IMGs=np.load('real_images.npy')
    X_train = IMGs.astype('float32')/255.0
    X_train = 2*X_train-1

    gan = GAN()
    gan.train(X_train)