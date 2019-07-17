import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from PIL import Image
import h5py
import matplotlib.pyplot as plt

INPUT_DATA = "data/faces/"  # 原始数据，暂时只生成带标签的数据
INPUT_DATA_EXTRA = "data/extra_data/images/"  # 更多的数据
INPUT_LABEL = "data/extra_data/tags.csv"  # 数据标签
OUTPUT_DATA = 'data/anime_face.h5'  # 将整理后的图片数据通过numpy的格式保存
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64


def create_file_list(root_dir):
    print("开始读取数据集：" + root_dir)
    # 获取一个子目录中所有的图片文件
    extensions = ['jpg']  # 列出所有扩展名, 'jpeg', 'JPG', 'JPEG'
    file_list = []
    # 针对不同的扩展名，将其文件名加入文件列表
    for extension in extensions:
        # INPUT_DATA是存放图片的文件夹
        # file_glob形如"INPUT_DATA/dir_name/*.extension"
        file_glob = os.path.join(root_dir, '*.' + extension)
        # extend()的作用是将glob.glob(file_glob)加入file_list
        # glob.glob()返回所有匹配的文件路径列表,此处返回的是所有在INPUT_DATA/dir_name文件夹中，且扩展名是extension的文件
        file_list.extend(glob.glob(file_glob))
    print("文件名列表制作完毕")
    # 包含有存有路径的string的list
    return file_list


def create_image_list(file_list, width, height):
    # 初始化数据集
    anime_images = []

    # 将file_list中的图片文件一条一条进行数据处理
    # 注意此时file_list已经变成了一个基本单位为字符串的list，list中的每个字符串存储的是一个图片的完整文件名（含路径），
    processing_image_count = 0  # 跟踪处理进度
    for file_name in file_list:
        # 使用PIL读取图片
        raw_image = Image.open(file_name)
        image = raw_image.resize((width, height), Image.NEAREST)  # Image.ANTIALIAS高质量，这里先不转换为float要不存的太慢
        image = np.array(image)  # 准换为ndarray形式
        anime_images.append(image)
        # 跟踪图片处理进度
        if processing_image_count % 200 == 0:
            print("正在读取第" + str(processing_image_count) + "张图片")
        processing_image_count = processing_image_count + 1
    # current_label += 1  # 注意这一行在上一个for外面，在最外层for里面；作用是在进入最外层for的下一轮循环之前，将"当前标签"加一，以表示下一个图片文件夹
    print("本数据集读取完毕")
    return anime_images


def create_label_list(file_dist):
    # 初始化数组和字典
    tags = []  # 为了生成独热编码存的数字标签，一个标签对应一个图片
    tags_v = []  # 返回的独热编码
    hair_color = {}  # 存出现过的头发颜色
    eyes_color = {}  # 存出现过的眼睛颜色
    l = open(file_dist).readlines()
    # 如果字典里没有，以字典长度创建一个新编码，否则引用原编码
    print("开始进行独热编码")
    for s in l:
        s = s.split(',')[1]
        s = s.split()
        h = s[0]
        e = s[2]
        x = []
        if h in hair_color:
            x.append(hair_color[h])
        else:
            hair_color[h] = len(hair_color)
            x.append(hair_color[h])
        if e in eyes_color:
            x.append(eyes_color[e])
        else:
            eyes_color[e] = len(eyes_color)
            x.append(eyes_color[e])
        tags.append(x)
    # 转换成独热编码
    for t in tags:
        h_v = np.zeros(len(hair_color))
        e_v = np.zeros(len(eyes_color))
        h_v[t[0]] = 1
        e_v[t[1]] = 1
        tags_v.append(np.concatenate((h_v, e_v)))
    tags_v = np.array(tags_v)
    print("独热编码完成")
    return tags_v, hair_color, eyes_color


def shuffle_image_label(anime_images, anime_tags):
    print("开始打乱数据集")

    # 将训练数据随机打乱以获得更好的训练效果
    state = np.random.get_state()  # 获取随机生成器np.random的状态
    np.random.shuffle(anime_images)  # 进行打乱操作，如果对象是多维矩阵，只对第一维进行打乱操作
    np.random.set_state(state)  # 将之前随机生成器的状态设置为现在随机生成器的状态，目的是让下面一行对标签的打乱和上一行图片的打乱一致
    np.random.shuffle(anime_tags)
    np.random.set_state(state)

    print("数据集处理完毕！")

    return anime_images, anime_tags


def load_h5py_to_np(path):
    h5_file = h5py.File(path, 'r')
    print('打印一下h5py中有哪些关键字', h5_file.keys())
    permutation = np.random.permutation(len(h5_file['labels']))
    shuffled_image = h5_file['image'][:][permutation, :, :, :]
    shuffled_label = h5_file['labels'][:][permutation]
    print('经过打乱之后数据集中的标签顺序是:\n', shuffled_label, len(h5_file['labels']))
    return shuffled_image, shuffled_label


if __name__ == '__main__':
    file_list = create_file_list(INPUT_DATA_EXTRA)
    anime_images = create_image_list(file_list, IMAGE_WIDTH, IMAGE_HEIGHT)
    anime_tags, hair_color, eyes_color = create_label_list(INPUT_LABEL)
    anime_images, anime_tags = shuffle_image_label(anime_images, anime_tags)
    f = h5py.File('data/anime_face.h5', 'w')
    f['image'] = anime_images
    f['labels'] = anime_tags
    f.close()
