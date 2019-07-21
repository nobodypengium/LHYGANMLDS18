from keras.layers import Input, Dense, Reshape, Flatten, Dropout, UpSampling2D, Conv2D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import h5py
import tensorflow as tf
import os
import numpy as np
import sys
import keras.backend as K

# py cuda epoch
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

T = int(sys.argv[2])
np.random.seed(126)

def wloss(y_true, y_pred):
    V = (2*y_true - 1)*(2*y_pred-1)
    return K.mean(V)


def save_imgs(generator):
    import matplotlib.pyplot as plt

    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))

    # gen_imgs should be shape (25, 64, 64, 3)
    gen_imgs = generator.predict(noise)
    gen_imgs = (gen_imgs+1)/2
    gen_imgs.astype(float)

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt,:,:,:])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("output.png")
    plt.close()

generator = load_model('model/wgan_{}.h5'.format(T), custom_objects={'wloss':wloss})

save_imgs(generator)