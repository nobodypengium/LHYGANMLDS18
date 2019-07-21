import os
import numpy as np
# from IPython.core.debugger import Tracer
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, UpSampling2D, Conv2D
from keras.layers.merge import _Merge
from keras import Model
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
import sys
import keras.backend as K
from HW3_1.generate_dataset_v2 import load_h5py_to_np, OUTPUT_DATA
from functools import partial

# import matplotlib.pyplot as plt

# os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

d_iter = 10


# def wloss(y_true, y_pred):
#     V = (2 * y_true - 1) * (2 * y_pred - 1)
#     return K.mean(V)

class RandomWeightedAverage(_Merge):
    """在原图和生成图片之间的区域进行采样，用于计算惩罚项"""

    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 64, 64, 3))  # 在生成图片和真实图片中间采样32个图片
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class GAN(object):
    """ Generative Adversarial Network class """

    def __init__(self, width=64, height=64, channels=3):

        self.width = width
        self.height = height
        self.channels = channels

        self.image_shape = (self.width, self.height, self.channels)
        self.latent_dim = 100
        self.gradient_penalty_loss = self.__gradient_penalty_loss
        self.wasserstein_loss = self.__wasserstein_loss

        self.clip_value = 0.01
        self.optimizer = RMSprop(lr=0.0002)

        self.G = self.__generator()
        self.D = self.__discriminator()

        # -------------------------------
        # Construct Computational Graph
        #       for the Critic
        # -------------------------------

        # Freeze generator's layers while training critic
        self.G.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.image_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        # Generate image based of noise (fake sample)
        fake_img = self.G(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.D(fake_img)
        valid = self.D(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.D(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                                  averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Keras requires function names

        self.discriminator_model = Model(inputs=[real_img, z_disc],
                                         outputs=[valid, fake, validity_interpolated])
        self.discriminator_model.compile(loss=[self.wasserstein_loss,
                                               self.wasserstein_loss,
                                               partial_gp_loss],
                                         optimizer=self.optimizer,
                                         loss_weights=[1, 1, 10])
        # self.D.compile(loss=[self.wasserstein_loss,self.wasserstein_loss,partial_gp_loss],optimizer=self.optimizer,loss_weights=[1,1,10])

        # -------------------------------
        # Construct Computational Graph
        #         for Generator
        # -------------------------------

        # For the generator we freeze the critic's layers
        self.D.trainable = False
        self.G.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(100,))
        # Generate images based of noise
        img = self.G(z_gen)
        # Discriminator determines validity
        valid = self.D(img)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=self.optimizer)

    def __gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """Calculates the gradient penalty loss for a batch of "averaged" samples.
        In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the
        loss function that penalizes the network if the gradient norm moves away from 1.
        However, it is impossible to evaluate this function at all points in the input
        space. The compromise used in the paper is to choose random points on the lines
        between real and generated samples, and check the gradients at these points. Note
        that it is the gradient w.r.t. the input averaged samples, not the weights of the
        discriminator, that we're penalizing!
        In order to evaluate the gradients, we must first run samples through the generator
        and evaluate the loss. Then we get the gradients of the discriminator w.r.t. the
        input averaged samples. The l2 norm and penalty can then be calculated for this
        gradient.
        Note that this loss function requires the original averaged samples as input, but
        Keras only supports passing y_true and y_pred to loss functions. To get around this,
        we make a partial() of the function with the averaged_samples argument, and use that
        for model training."""
        # first get the gradients:
        #   assuming: - that y_pred has dimensions (batch_size, 1)
        #             - averaged_samples has dimensions (batch_size, nbr_features)
        # gradients afterwards has dimension (batch_size, nbr_features), basically
        # a list of nbr_features-dimensional gradient vectors
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def __wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def __generator(self):
        """ Declare generator """

        generator = Sequential()
        generator.add(Dense(128 * 16 * 16, input_shape=(100,), activation='relu'))

        generator.add(Reshape((16, 16, 128)))
        # generator.add(UpSampling2D())
        generator.add(Conv2D(128, kernel_size=4, activation='relu'))

        generator.add(UpSampling2D())
        generator.add(Conv2D(64, kernel_size=4, activation='relu'))

        generator.add(Conv2D(3, kernel_size=4, activation='relu'))

        generator.add(Flatten())
        generator.add(Dense(64 * 64 * 3, activation='tanh'))
        generator.add(Reshape((64, 64, 3)))

        noise = Input(shape=(self.latent_dim,))
        img = generator(noise)

        return Model(noise, img)

    def __discriminator(self):
        """ Declare discriminator """

        discriminator = Sequential()
        discriminator.add(Conv2D(32, kernel_size=4, input_shape=(64, 64, 3), activation='relu'))

        discriminator.add(Conv2D(64, kernel_size=4, padding='same', activation='relu'))

        discriminator.add(Conv2D(128, kernel_size=4, activation='relu'))

        discriminator.add(Conv2D(256, kernel_size=4, activation='relu'))

        discriminator.add(Flatten())
        discriminator.add(Dense(1))

        img = Input(shape=self.image_shape)
        validity = discriminator(img)
        return Model(img, validity)

    def __stacked_generator_discriminator(self):

        self.D.trainable = False
        # self.D.compile(loss=self.LossFunction, optimizer=self.optimizer, metrics=['accuracy'])

        model = Sequential()
        model.add(self.G)
        model.add(self.D)

        return model

    def train(self, X_train, epochs=1000, batch=32):
        # self.stacked_generator_discriminator=load_model('model/gan_tips_model_1700.h5')####
        # self.G=load_model('model/gan_tips_1700.h5')####
        # Adversarial ground truths
        valid = -np.ones((batch, 1))
        fake = np.ones((batch, 1))
        dummy = np.zeros((batch, 1))  # Dummy gt for gradient penalty
        for cnt in range(epochs):  #####
            ## train discriminator
            for i in range(d_iter):
                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch)
                imgs = X_train[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch, self.latent_dim))
                # Train the critic
                d_loss = self.discriminator_model.train_on_batch([imgs, noise],
                                                          [valid, fake, dummy])

            # train generator
            noise = np.random.normal(0, 1, (batch, self.latent_dim))
            g_loss = self.generator_model.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f] [G loss: %f]" % (cnt, d_loss[0], g_loss))

            if cnt % 500 == 0:
                self.discriminator_model.save('data/WGAN_model/GP/d/model_{}.h5'.format(cnt))
                self.generator_model.save('data/WGAN_model/GP/g/wgan_{}.h5'.format(cnt))


if __name__ == '__main__':
    IMGs, labels = load_h5py_to_np(OUTPUT_DATA)
    X_train = IMGs.astype('float32') / 255.0
    X_train = 2 * X_train - 1

    gan = GAN()
    gan.train(X_train)
