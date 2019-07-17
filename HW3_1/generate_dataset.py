import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from PIL import Image

INPUT_DATA = "data/faces/"     # 原始数据
INPUT_DATA_EXTRA = "data/extra_data/images/" # 更多的数据
OUTPUT_DATA = 'data/anime_face.npy'     # 将整理后的图片数据通过numpy的格式保存

# 读取数据并将数据分割成训练数据、验证数据和测试数据
def create_image_lists(sess):
    # sub_dirs用于存储INPUT_DATA下的全部子文件夹目录
    #sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]  # os.walk() 方法用于通过在目录树中游走输出在目录中的文件名，向上或者向下
    #is_root_dir = True

    # 初始化各个数据集
    # training_images = []
    # training_labels = []
    # testing_images = []
    # testing_labels = []
    # validation_images = []
    # validation_labels = []
    # current_label = 0   # 在接下来的for循环中，第一次循环时值为0，每次循环结束时加一
    # count = 1   # 循环计数器

    # 初始化数据集
    anime_images=[]

    # 对每个在sub_dirs中的子文件夹进行操作
    count = 0
    for sub_dir in [INPUT_DATA, INPUT_DATA_EXTRA]:
        # # 直观上感觉这个条件结构是多此一举，暂时不分析为什么要加上这个语句
        # if is_root_dir:
        #     is_root_dir = False
        #     continue    # 继续下一轮循环，下一轮就无法进入条件分支而是直接执行下列语句

        print("开始读取数据集：" + str(count))
        count += 1

        # 获取一个子目录中所有的图片文件
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']     # 列出所有扩展名
        file_list = []
        # os.path.basename()返回path最后的文件名。若path以/或\结尾，那么就会返回空值
        # dir_name = os.path.basename(sub_dir)    # 返回子文件夹的名称（sub_dir是包含文件夹地址的串，去掉其地址，只保留文件夹名称）
        # 针对不同的扩展名，将其文件名加入文件列表
        for extension in extensions:
            # INPUT_DATA是数据集的根文件夹，其下有五个子文件夹，每个文件夹下是一种花的照片；
            # dir_name是这次循环中存放所要处理的某种花的图片的文件夹的名称
            # file_glob形如"INPUT_DATA/dir_name/*.extension"
            file_glob = os.path.join(sub_dir, '*.' + extension)
            # extend()的作用是将glob.glob(file_glob)加入file_list
            # glob.glob()返回所有匹配的文件路径列表,此处返回的是所有在INPUT_DATA/dir_name文件夹中，且扩展名是extension的文件
            file_list.extend(glob.glob(file_glob))
        # 猜想这句话的意思是，如果file_list是空list，则不继续运行下面的数据处理部分，而是直接进行下一轮循环，
        # 即换一个子文件夹继续操作
        if not file_list: continue

        print("文件名列表制作完毕，开始读取图片文件")

        # 将file_list中的图片文件一条一条进行数据处理
        # 注意此时file_list已经变成了一个基本单位为字符串的list，list中的每个字符串存储的是一个图片的完整文件名（含路径），
        # 这些图片所属的文件夹就是这一轮循环的sub_dir
        processing_image_count = 0 # 跟踪处理进度
        for file_name in file_list:
            # 以下两行是读文件常用语句
            image_raw_data = gfile.FastGFile(file_name, 'rb').read()
            image = tf.image.decode_jpeg(image_raw_data)
            # 如果图片数据的类型不是float32，则转换之
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            # 调整图片的尺寸,将其化为64*64，以便inception-v3模型来处理
            image = tf.image.resize_images(image, [64, 64])
            image_value = sess.run(image)   # 提示：sess.run(image)返回image的计算结果;
            # 至此， image_value类型是299*299的float32型矩阵，代表当前循环所处理的图片文件

            # 随机划分数据集，通过生成一个0-99的随机数chance来决定当前循环中的图片文件划入验证集、测试集还是训练集
            # np.random.randint(100)作用是随机生成在0-99间的一个数（此函数还可以指定返回的尺寸，比如可以指定返回一个x*y的矩阵，未指定尺寸则返回一个数）
            # chance = np.random.randint(100)
            # if chance < validation_percentage:
            #     validation_images.append(image_value)   # 由于一共有3670张图片，这样最终的validation_images的尺寸大致是(3670*validation_percentage%)*229*229*3
            #     validation_labels.append(current_label)     # 由于一共有3670张图片，这样最终的validation_labels的尺寸大致是(3670*validation_percentage%)*1
            # elif chance < (testing_percentage + validation_percentage):
            #     testing_images.append(image_value)
            #     testing_labels.append(current_label)
            # else:
            #     training_images.append(image_value)
            #     training_labels.append(current_label)
            anime_images.append(image_value)
            if processing_image_count % 200 == 0:
                print("正在读取第" + str(processing_image_count) + "张图片")
            processing_image_count = processing_image_count + 1
        # current_label += 1  # 注意这一行在上一个for外面，在最外层for里面；作用是在进入最外层for的下一轮循环之前，将"当前标签"加一，以表示下一个图片文件夹
        print("本数据集读取完毕")

    print("开始打乱数据集")

    # 将训练数据随机打乱以获得更好的训练效果
    # 注意这里已经跳出了for循环，此时的training_image尺寸大致是图片数量，五万张左右*64*64*3
    # training_labels尺寸大致是(734*(100-validition_percentage-testing_percentage)%)*1(废话)
    # state = np.random.get_state()   # 获取随机生成器np.random的状态
    np.random.shuffle(anime_images)      # 进行打乱操作，如果对象是多维矩阵，只对第一维进行打乱操作
    # np.random.set_state(state)      # 将之前随机生成器的状态设置为现在随机生成器的状态，目的是让下面一行对标签的打乱和上一行图片的打乱一致
    # np.random.shuffle(training_labels) #没有labels

    print("数据集处理完毕！")

    return np.asarray([anime_images])

def main():
    with tf.Session() as sess:
        processed_data = create_image_lists(sess)
        np.save(OUTPUT_DATA, processed_data)

if __name__ == '__main__':
    main()
