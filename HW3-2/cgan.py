from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import sys
import os
import numpy as np
from HW3_1.generate_dataset_v2 import load_h5py_to_np
import tensorflow as tf

class CGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.hair_classes = 12
        self.eye_classes = 10
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise1 = Input(shape=(int(self.latent_dim / 2),))
        noise2 = Input(shape=(int(self.latent_dim / 2),))
        label1 = Input(shape=(1,))
        label2 = Input(shape=(1,))
        img = self.generator([noise1, noise2, label1, label2])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label1, label2])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise1, noise2, label1, label2], valid)
        self.combined.compile(loss=['binary_crossentropy'],
                              optimizer=optimizer)

    def build_generator(self):

        generator = Sequential()
        # generator.add(Dense(50 * 16 * 16, input_dim=self.latent_dim, activation='relu'))
        # # generator.add(LeakyReLU())
        # generator.add(BatchNormalization(momentum=0.8))
        # generator.add(Reshape((16, 16, 50)))
        # generator.add(UpSampling2D())
        # generator.add(Conv2D(128, kernel_size=4, activation='relu'))
        # # generator.add(LeakyReLU())
        # generator.add(BatchNormalization(momentum=0.8))
        # generator.add(UpSampling2D())
        # generator.add(Conv2D(64, kernel_size=4, activation='relu'))
        # # generator.add(LeakyReLU())
        # generator.add(BatchNormalization(momentum=0.8))
        # generator.add(Conv2D(3, kernel_size=4, activation='relu'))
        # # generator.add(LeakyReLU())
        # generator.add(BatchNormalization(momentum=0.8))
        # generator.add(Flatten())
        # generator.add(Dense(64 * 64 * 3, activation='tanh'))
        # generator.add(Reshape((64, 64, 3)))

        generator.add(Dense(128 * 16 * 16, input_shape=(100,), activation='relu'))

        generator.add(Reshape((16, 16, 128)))
        generator.add(UpSampling2D())
        generator.add(Conv2D(128, kernel_size=4, activation='relu'))

        generator.add(UpSampling2D())
        generator.add(Conv2D(64, kernel_size=4, activation='relu'))

        generator.add(Conv2D(3, kernel_size=4, activation='relu'))

        generator.add(Flatten())
        generator.add(Dense(64 * 64 * 3, activation='tanh'))
        generator.add(Reshape((64, 64, 3)))

        generator.summary()

        # 为两个输入分别进行 将标签融入噪声 操作
        noise1 = Input(shape=(int(self.latent_dim / 2),))
        noise2 = Input(shape=(int(self.latent_dim / 2),))
        label1 = Input(shape=(1,), dtype='int32')
        label2 = Input(shape=(1,), dtype='int32')
        label_embedding1 = Flatten()(Embedding(self.hair_classes, int(self.latent_dim / 2))(label1))
        label_embedding2 = Flatten()(Embedding(self.eye_classes, int(self.latent_dim / 2))(label2))
        noise = Concatenate(axis=-1)([noise1, noise2])
        label_embedding = Concatenate(axis=-1)([label_embedding1, label_embedding2])

        model_input = multiply([noise, label_embedding])
        img = generator(model_input)

        return Model([noise1, noise2, label1, label2], img)

    def build_discriminator(self):

        discriminator = Sequential()
        discriminator.add(Dense(80 * 16 * 16, input_dim=np.prod(self.img_shape), activation='relu'))
        # discriminator.add(LeakyReLU())
        discriminator.add(Reshape((16, 16, 80)))
        discriminator.add(Conv2D(32, kernel_size=4, activation='relu'))
        # discriminator.add(LeakyReLU())
        discriminator.add(Conv2D(64, kernel_size=4, padding='same', activation='relu'))
        # discriminator.add(LeakyReLU())
        discriminator.add(Conv2D(128, kernel_size=4, activation='relu'))
        # discriminator.add(LeakyReLU())
        discriminator.add(Conv2D(256, kernel_size=4, activation='relu'))
        # discriminator.add(LeakyReLU())
        discriminator.add(Flatten())
        discriminator.add(Dense(1, activation='sigmoid'))
        discriminator.summary()

        img = Input(shape=self.img_shape)
        label1 = Input(shape=(1,), dtype='int32')
        label2 = Input(shape=(1,), dtype='int32')
        label_embedding1 = Flatten()(Embedding(self.hair_classes, int(np.prod(self.img_shape)/2))(label1))
        label_embedding2 = Flatten()(Embedding(self.eye_classes, int(np.prod(self.img_shape)/2))(label2))
        label_embedding = Concatenate(axis=-1)([label_embedding1, label_embedding2])
        flat_img = Flatten()(img)

        model_input = multiply([flat_img, label_embedding])
        # model_input = multiply([img, label_embedding])

        validity = discriminator(model_input)

        return Model([img, label1, label2], validity)

    def train(self, X_train, y_train, epochs=20000, batch_size=4):

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]
            label1 = labels[:, 0]
            label2 = labels[:, 1]

            # Sample noise as generator input
            noise1 = np.random.normal(0, 1, (batch_size, int(100 / 2)))
            noise2 = np.random.normal(0, 1, (batch_size, int(100 / 2)))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise1, noise2, label1, label2])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, label1, label2], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, label1, label2], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            label1 = np.random.randint(0, self.hair_classes, batch_size)
            label2 = np.random.randint(0, self.eye_classes, batch_size)
            # print(sampled_labels.shape)

            # Train the generator
            g_loss = self.combined.train_on_batch([noise1, noise2, label1, label2], valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
            if epoch % 1000 == 0:
                self.generator.save_weights('data/model/CGAN/g/cgan1_relu32_g_{}.h5'.format(epoch))
                self.combined.save_weights('data/model/CGAN/gd/cgan1_relu32_c_{}.h5'.format(epoch))


if __name__ == '__main__':
    IMGs, labels = load_h5py_to_np('data/anime_face_not_onehot_label.h5')
    X_train = IMGs.astype('float32') / 255.0
    X_train = 2 * X_train - 1
    y_train = labels

    cgan = CGAN()
    cgan.train(X_train, y_train)
