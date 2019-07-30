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
import glob
from HW3_1.generate_dataset_v3 import load_h5py_to_np
from HW3_2.cgan import CGAN

# py cuda epoch
# os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

# T = int(sys.argv[2])
T = 0
np.random.seed(126)


def save_imgs(generator,hair_tags,eyes_tags):
    import matplotlib.pyplot as plt

    r, c = 5, 5
    noise1 = np.random.normal(0, 1, (r * c, 50))
    noise2 = np.random.normal(0, 1, (r * c, 50))
    label1 = np.random.randint(0, 12, r * c)
    label2 = np.random.randint(0, 10, r * c)

    # gen_imgs should be shape (25, 64, 64, 3)
    gen_imgs = generator.predict([noise1, noise2, label1, label2])
    gen_imgs = (gen_imgs + 1) / 2
    gen_imgs.astype(float)

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, :])
            axs[i, j].axis('off')
            axs[i, j].set_title(hair_tags[label1[cnt]].decode('utf8')+' '+eyes_tags[label2[cnt]].decode('utf8'))
            cnt += 1

    # 保存文件
    # filepath, tmpfilename = os.path.split(generator)
    # shotname, extension = os.path.splitext(tmpfilename)

    fig.savefig("data/outputs/CGAN/epoch_" + str(T) + ".png")
    plt.close()


def create_file_list(root_dir):
    print("开始读取训练模型：" + root_dir)
    # 获取一个子目录中所有的图片文件
    extensions = ['h5']  # 列出所有扩展名, 'jpeg', 'JPG', 'JPEG'
    file_list = []
    # 针对不同的扩展名，将其文件名加入文件列表
    for extension in extensions:
        # INPUT_DATA是存放图片的文件夹
        # file_glob形如"INPUT_DATA/dir_name/*.extension"
        file_glob = os.path.join(root_dir, '*.' + extension)
        # extend()的作用是将glob.glob(file_glob)加入file_list
        # glob.glob()返回所有匹配的文件路径列表,此处返回的是所有在INPUT_DATA/dir_name文件夹中，且扩展名是extension的文件
        file_list.extend(glob.glob(file_glob))
    print("所有训练模型读取完毕")
    # 包含有存有路径的string的list
    return file_list


if __name__ == '__main__':
    file_list = create_file_list('data/models/CGAN/g/')
    file_list.sort(key=lambda x: os.path.getctime(x))
    _,_,hair_tags,eyes_tags = load_h5py_to_np('data/anime_face_not_onehot_label_with_tags.h5')
    # generator = load_model('data/WGAN_model/g/gan_{}.h5'.format(T))
    # save_imgs(generator)
    # file_list=['C:\Study\Python\LHYGANMLDS18\HW3_1\data\WGAN_model\GP\g\wgan_5000.h5']
    for file in file_list:
        print("第{}次epoch的结果已生成".format(T))
        cgan = CGAN()
        generator = cgan.build_generator()
        generator.load_weights(file)
        save_imgs(generator,hair_tags,eyes_tags)
        T = T + 1000
    # for i in (2500,3000,3500,4000,4500):
    #     print("第{}次epoch的结果已生成".format(i))
    #     file = 'C:\Study\Python\LHYGANMLDS18\HW3_1\data\WGAN_model\GP\g\wgan_'+str(i)+'.h5'
    #     model = load_model(file, custom_objects={'__wasserstein_loss': wasserstein_loss,
    #                                              '__gradient_penalty_loss': gradient_penalty_loss,
    #                                              '__asserstein_loss': asserstein_loss})
    #     generator = Sequential()
    #     generator.add(model.get_layer('input_5'))
    #     generator.add(model.get_layer('model_1'))
    #     save_imgs(generator)
    #     T=T+500
    #