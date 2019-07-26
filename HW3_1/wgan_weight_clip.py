import os
import numpy as np
# from IPython.core.debugger import Tracer
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, UpSampling2D, Conv2D, Activation, ZeroPadding2D
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
import sys
import keras.backend as K
from HW3_1.generate_dataset_v2 import load_h5py_to_np, OUTPUT_DATA

# import matplotlib.pyplot as plt

# os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

d_iter = 5


def wloss(y_true, y_pred):
    return K.mean(y_true * y_pred)


class GAN(object):
    """ Generative Adversarial Network class """

    def __init__(self, width=64, height=64, channels=3):

        self.width = width
        self.height = height
        self.channels = channels

        self.shape = (self.width, self.height, self.channels)
        self.LossFunction = wloss

        self.clip_value = 0.01
        self.optimizer = RMSprop(lr=0.00005)
        self.G = self.__generator()
        # self.G.compile(loss=self.LossFunction, optimizer=self.optimizer)

        self.D = self.__discriminator()
        self.D.compile(loss=self.LossFunction, optimizer=self.optimizer, metrics=['accuracy'])

        self.stacked_generator_discriminator = self.__stacked_generator_discriminator()

        self.stacked_generator_discriminator.compile(loss=self.LossFunction, optimizer=self.optimizer)

    def __generator(self):
        """ Declare generator """

        generator = Sequential()
        # generator.add(Dense(128 * 16 * 16, input_shape=(100,), activation='relu'))
        #
        # generator.add(Reshape((16, 16, 128)))
        # generator.add(UpSampling2D())
        # generator.add(Conv2D(128, kernel_size=4, activation='relu'))
        #
        # generator.add(UpSampling2D())
        # generator.add(Conv2D(64, kernel_size=4, activation='relu'))
        #
        # generator.add(Conv2D(3, kernel_size=4, activation='relu'))
        #
        # generator.add(Flatten())
        # generator.add(Dense(64 * 64 * 3, activation='tanh'))
        # generator.add(Reshape((64, 64, 3)))

        generator.add(Dense(128 * 16 * 16, activation="relu", input_dim=100))
        generator.add(Reshape((16, 16, 128)))
        generator.add(UpSampling2D())
        generator.add(Conv2D(128, kernel_size=4, padding="same"))
        generator.add(BatchNormalization(momentum=0.8))
        generator.add(Activation("relu"))
        generator.add(UpSampling2D())
        generator.add(Conv2D(64, kernel_size=4, padding="same"))
        generator.add(BatchNormalization(momentum=0.8))
        generator.add(Activation("relu"))
        generator.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        generator.add(Activation("tanh"))

        return generator

    def __discriminator(self):
        """ Declare discriminator """

        discriminator = Sequential()
        # discriminator.add(Conv2D(32, kernel_size=4, input_shape=(64, 64, 3), activation='relu'))
        # 
        # discriminator.add(Conv2D(64, kernel_size=4, padding='same', activation='relu'))
        # 
        # discriminator.add(Conv2D(128, kernel_size=4, activation='relu'))
        # 
        # discriminator.add(Conv2D(256, kernel_size=4, activation='relu'))
        # 
        # discriminator.add(Flatten())
        # discriminator.add(Dense(1))
        
        discriminator.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.shape, padding="same"))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dropout(0.25))
        discriminator.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        discriminator.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        discriminator.add(BatchNormalization(momentum=0.8))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dropout(0.25))
        discriminator.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        discriminator.add(BatchNormalization(momentum=0.8))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dropout(0.25))
        discriminator.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        discriminator.add(BatchNormalization(momentum=0.8))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dropout(0.25))
        discriminator.add(Flatten())
        discriminator.add(Dense(1))
        
        return discriminator

    def __stacked_generator_discriminator(self):

        self.D.trainable = False
        # self.D.compile(loss=self.LossFunction, optimizer=self.optimizer, metrics=['accuracy'])

        model = Sequential()
        model.add(self.G)
        model.add(self.D)


        return model

    def train(self, X_train, epochs=6000, batch=128):
        # self.stacked_generator_discriminator=load_model('model/gan_tips_model_1700.h5')####
        # self.G=load_model('model/gan_tips_1700.h5')####
        valid = -np.ones((batch, 1))
        fake = np.ones((batch, 1))
        for cnt in range(epochs):  #####

            ## train discriminator
            for i in range(d_iter):
                # random_index = np.random.randint(0, len(X_train) - np.int64(batch / 2))
                # legit_images = X_train[random_index: random_index + np.int64(batch / 2)].reshape(np.int64(batch / 2),
                #                                                                                  self.width,
                #                                                                                  self.height,
                #                                                                                  self.channels)

                idx = np.random.randint(0, X_train.shape[0], batch)
                imgs = X_train[idx]

                # gen_noise = np.random.normal(0, 1, (np.int64(batch / 2), 100))
                noise = np.random.normal(0, 1, (batch, 100))
                syntetic_images = self.G.predict(noise)

                # x_combined_batch = np.concatenate((legit_images, syntetic_images))
                # y_combined_batch = np.concatenate(
                #     (np.ones((np.int64(batch / 2), 1)), np.zeros((np.int64(batch / 2), 1))))
                #
                # d_loss = self.D.train_on_batch(x_combined_batch, y_combined_batch)

                d_loss_real = self.D.train_on_batch(imgs, valid)
                d_loss_fake = self.D.train_on_batch(syntetic_images, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                for l in self.D.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # train generator

            # noise = np.random.normal(0, 1, (batch, 100))
            # y_mislabled = np.ones((batch, 1))

            g_loss = self.stacked_generator_discriminator.train_on_batch(noise, valid)

            print('epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (cnt, 1-d_loss[0], 1-g_loss))
            if cnt % 100 == 0:
                self.stacked_generator_discriminator.save('data/WGAN_model/weight_clip/org/gd/model_{}.h5'.format(cnt))
                self.G.save('data/WGAN_model/weight_clip/org/g/wgan_{}.h5'.format(cnt))


if __name__ == '__main__':
    IMGs,labels = load_h5py_to_np(OUTPUT_DATA)
    X_train = IMGs.astype('float32') / 255.0
    X_train = 2 * X_train - 1

    gan = GAN()
    gan.train(X_train)