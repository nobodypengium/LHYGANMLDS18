import os
import numpy as np
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, UpSampling2D, Conv2D
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model
from keras.models import Sequential
from keras.optimizers import Adam
from keras import regularizers
from HW3_1.generate_dataset_v2 import load_h5py_to_np, OUTPUT_DATA


# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
        generator.add(Dense(256*16*16, input_shape=(100,), activation='relu',kernel_regularizer=regularizers.l2(0.01)))
        generator.add(Reshape((16,16,256)))
        generator.add(UpSampling2D())
        generator.add(Conv2D(128, kernel_size=4, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
        generator.add(UpSampling2D())
        generator.add(Conv2D(64, kernel_size=4, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
        generator.add(Conv2D(3, kernel_size=4, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
        generator.add(Flatten())
        generator.add(Dense(64*64*3, activation='tanh',kernel_regularizer=regularizers.l2(0.01)))
        generator.add(Reshape((64,64,3)))

        return  generator

    def __discriminator(self):
        """ Declare discriminator """

        discriminator=Sequential()
        discriminator.add(Conv2D(32, kernel_size=4, input_shape=(64,64,3), activation='relu',kernel_regularizer=regularizers.l2(0.01)))
        discriminator.add(Conv2D(64, kernel_size=4, padding='same', activation='relu',kernel_regularizer=regularizers.l2(0.01)))
        discriminator.add(Conv2D(128, kernel_size=4, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
        discriminator.add(Conv2D(256, kernel_size=4, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
        discriminator.add(Flatten())
        discriminator.add(Dense(1, activation='sigmoid',kernel_regularizer=regularizers.l2(0.01)))

        return discriminator


    def __stacked_generator_discriminator(self):

        self.D.trainable = False

        model = Sequential()
        model.add(self.G)
        model.add(self.D)

        return model

    def train(self, X_train, epochs=15000, batch = 128):
        #load old model
        self.stacked_generator_discriminator = load_model('data/GAN_model/gd/gan_model_5000.h5')
        # self.G = load_model('data/GAN_model/g/gan_5000.h5')
        base = 5000

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
                self.stacked_generator_discriminator.save('data/GAN_model/gd/gan_model_{}.h5'.format(cnt+1+base))
                self.G.save('data/GAN_model/g/gan_{}.h5'.format(cnt+1+base))


if __name__ == '__main__':
    IMGs,labels = load_h5py_to_np(OUTPUT_DATA)
    X_train = IMGs.astype('float32') / 255.0
    X_train = 2 * X_train - 1

    gan = GAN()
    gan.train(X_train)
