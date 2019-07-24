import os
import numpy as np
# from IPython.core.debugger import Tracer
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, UpSampling2D, Conv2D, Activation, Conv2DTranspose
from keras.optimizers import Adam
from keras.engine.topology import Layer
from keras.layers.merge import _Merge
from keras import Model
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
import sys
import keras.backend as K
import keras.initializers
from HW3_1.generate_dataset_v2 import load_h5py_to_np, OUTPUT_DATA
from functools import partial


class DiscriminatorLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(DiscriminatorLossLayer, self).__init__(**kwargs)  # python3.x也可以写作super().__init__(**kwargs)

    def lossfun(self, y_real, y_real_rec, y_fake, y_fake_rec):
        """
        真实图片与虚假图片的损失的差，如果损失看不出什么差距，那么虚假图片就可以以假乱真了
        """
        loss_real = K.mean(K.abs(y_real - y_real_rec))
        loss_fake = K.mean(K.abs(y_fake - y_fake_rec))
        loss = loss_real - loss_fake
        return loss

    def call(self, inputs):
        y_real = inputs[0]
        y_real_rec = inputs[1]
        y_fake = inputs[2]
        y_fake_rec = inputs[3]

        loss = self.lossfun(y_real, y_real_rec, y_fake, y_fake_rec)
        self.add_loss(loss, inputs=inputs)

        return y_real


class GeneratorLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(GeneratorLossLayer, self).__init__(**kwargs)

    def lossfun(self, y_fake, y_fake_rec):
        """
        重构图片与原图片的相似性
        """
        loss = K.mean(K.abs(y_fake - y_fake_rec))
        return loss

    def call(self, inputs):
        y_fake = inputs[0]
        y_fake_rec = inputs[1]

        loss = self.lossfun(y_fake, y_fake_rec)
        self.add_loss(loss, inputs=inputs)

        return y_fake


class EBGAN(object):
    def __init__(self, width=64, height=64, channels=3):
        # Define the image shape
        self.width = width
        self.height = height
        self.channels = channels
        self.img_shape = (self.width, self.height, self.channels)

        # Define the noise shape
        self.z_dims = 100

        # Define the components
        self.encoder = self.__encoder()
        self.decoder = self.__decoder()
        self.generator = self.__decoder()
        self.discriminator = self.__autoencoder()

        # Define the loss function ???
        self.zero_loss = self.__zero_loss

        # Define the iteration number of the discriminator
        self.d_iter = 5

        ## Build the model
        # Generate fake and real img
        x_real = Input(shape=self.img_shape)
        z_sample = Input(shape=(self.z_dims,))
        x_fake = self.generator(z_sample)

        # Discriminator
        x_real_rec = self.discriminator(x_real)
        x_fake_rec = self.discriminator(x_fake)

        # TODO: 此处存疑
        d_loss = DiscriminatorLossLayer()([x_real, x_real_rec, x_fake, x_fake_rec])
        g_loss = GeneratorLossLayer()([x_fake, x_fake_rec])

        # Build discriminator trainer
        self.generator.trainable = False
        self.discriminator.trainable = True

        self.dis_trainer = Model(inputs=[x_real, z_sample], outputs=[d_loss])
        self.dis_trainer.compile(loss=[self.zero_loss], optimizer=Adam(lr=2.0e-4, beta_1=0.5))
        self.dis_trainer.summary()

        # Build generator trainer
        self.generator.trainable = True
        self.discriminator.trainable = False

        self.gen_trainer = Model(inputs=[x_real, z_sample],
                                 outputs=[g_loss])
        self.gen_trainer.compile(loss=[self.zero_loss],
                                 optimizer=Adam(lr=2.0e-4, beta_1=0.5))
        self.gen_trainer.summary()

    def __decoder(self):
        # Define the input layer
        noise = Input(shape=(self.z_dims,))
        w = self.img_shape[0] // (2 ** 3)

        # Define the Dense layers
        decoder = Sequential()
        decoder.add(Dense(w * w * 512, input_shape=(100,)))
        decoder.add(BatchNormalization())
        decoder.add(Activation('relu'))

        # Prepare for the convolution layers
        decoder.add(Reshape((w, w, 512)))
        kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        bias_init = keras.initializers.Zeros()

        # Convolution layers Conv2D -> LeakyReLU ->Conv2D->LeakyReLU->Conv2D->LeakyReLU->Conv2DT->LeakyReLU
        decoder.add(
            Conv2D(filters=512, kernel_size=(5, 5), strides=(2, 2), padding='same', kernel_initializer=kernel_init,
                   bias_initializer=bias_init))
        decoder.add(LeakyReLU(0.1))
        decoder.add(
            Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2), padding='same', kernel_initializer=kernel_init,
                   bias_initializer=bias_init))
        decoder.add(LeakyReLU(0.1))
        decoder.add(
            Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2), padding='same', kernel_initializer=kernel_init,
                   bias_initializer=bias_init))
        decoder.add(LeakyReLU(0.1))
        # # 反卷积层 卷积层的前向操作可以表示为和矩阵C相乘，那么 我们很容易得到卷积层的反向传播就是和C的转置相乘。
        # decoder.add(Conv2DTranspose(filters=self.img_shape[2], kernel_size=(5, 5), strides=(1, 1),
        #                             kernel_initializer=kernel_init, bias_initializer=bias_init, padding='same'))
        decoder.add(UpSampling2D())
        decoder.add(
            Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', kernel_initializer=kernel_init,
                   bias_initializer=bias_init))
        decoder.add(LeakyReLU(0.1))
        decoder.add(
            Conv2D(filters=3, kernel_size=(5, 5), strides=(2, 2), padding='same', kernel_initializer=kernel_init,
                   bias_initializer=bias_init))
        decoder.add(LeakyReLU(0.1))

        decoder.add(Flatten())
        decoder.add(Dense(64 * 64 * 3, activation='tanh'))
        decoder.add(Reshape((64, 64, 3)))

        decoder.add(LeakyReLU(0.1))
        decoder.add(Reshape(self.img_shape))

        # Define the output
        # NOTE: Here, we just know z in dim(x,y,z) for img is 3, but we concat decoder and encoder together to give x,y information for decoder's output.
        img = decoder(noise)
        return Model(noise, img)

    def __encoder(self):
        # Define the input layer
        img = Input(shape=self.img_shape)

        # Prepare for the convolution layers
        encoder = Sequential()
        kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        bias_init = keras.initializers.Zeros()

        # Convolution layers
        encoder.add(
            Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2), padding='same', kernel_initializer=kernel_init,
                   bias_initializer=bias_init))
        encoder.add(LeakyReLU(0.1))
        encoder.add(
            Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2), padding='same', kernel_initializer=kernel_init,
                   bias_initializer=bias_init))
        encoder.add(LeakyReLU(0.1))
        encoder.add(
            Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2), padding='same', kernel_initializer=kernel_init,
                   bias_initializer=bias_init))
        encoder.add(LeakyReLU(0.1))
        encoder.add(
            Conv2D(filters=512, kernel_size=(5, 5), strides=(2, 2), padding='same', kernel_initializer=kernel_init,
                   bias_initializer=bias_init))
        encoder.add(LeakyReLU(0.1))

        # Encode as an 1_dim vector
        encoder.add(Flatten())
        encoder.add(Dense(1024, activation='relu'))
        encoder.add(Dense(self.z_dims, activation='linear'))

        # Define output
        code = encoder(img)
        return Model(img, code)

    def __autoencoder(self):
        img_in = Input(self.img_shape)
        code = self.encoder(img_in)
        img_out = self.decoder(code)
        return Model(img_in, img_out)

    def __zero_loss(self, y_true, y_pred):
        return K.zeros_like(y_true)

    # Start training
    def train(self, X_train, epochs=5001, batch=32):
        base_idx = 0
        for epoch in range(epochs):
            # Train discriminator and generator for 5:1 epoch
            for i in range(self.d_iter):
                # Select a random image batch,
                # ---!!!!!NOTE THAT THE IMAGES ARE PRE-SHUFFLED!!!!!---
                # idx = np.random.randint(0, X_train.shape[0], batch)
                idx = (np.array(range(batch)) + base_idx) % X_train.shape[0]
                imgs = X_train[idx]
                base_idx = base_idx + batch
                # Generate noise code
                z_sample = np.random.uniform(-1.0, 1.0, size=(batch, self.z_dims))
                # Train the discriminator
                dummy = np.zeros_like(imgs)  # TODO:这是啥
                # g_loss = self.gen_trainer.train_on_batch([imgs,z_sample],dummy)
                d_loss = self.dis_trainer.train_on_batch([imgs, z_sample], dummy)

            # Train generator
            # Generate images
            idx = (np.array(range(batch)) + base_idx) % X_train.shape[0]
            imgs = X_train[idx]
            base_idx = base_idx + batch
            # Generate noise code
            z_sample = np.random.uniform(-1.0, 1.0, size=(batch, self.z_dims))
            # Train on batch
            dummy = np.zeros_like(imgs)
            g_loss = self.gen_trainer.train_on_batch([imgs, z_sample], dummy)

            # Print training state
            print("Epoch: %d [D loss: %f] [G loss: %f]" % (epoch, d_loss, g_loss))

            if epoch % 100 == 0:
                self.discriminator.save('data/EBGAN_model/d/discriminator_{}.h5'.format(epoch))
                self.generator.save('data/EBGAN_model/g/generator_{}.h5'.format(epoch))

if __name__ == '__main__':
    IMGs, labels = load_h5py_to_np(OUTPUT_DATA)
    X_train = IMGs.astype('float32') / 255.0
    X_train = 2 * X_train - 1

    gan = EBGAN()
    gan.train(X_train,epochs=500)