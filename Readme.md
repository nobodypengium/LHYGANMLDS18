# LHYGANMLDS18
学习李宏毅老师的的网课 MLDS18中GAN的部分
http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS18.html
# HW3-1参照资源
仅用于学习目的，建议去原作者那里查看
## 创建TFRecord的数据集
https://blog.csdn.net/sinat_34474705/article/details/78966064
## 动漫图片脸部识别
https://github.com/nagadomi/lbpcascade_animeface
## HW3-1参考代码
https://github.com/CodePlay2016/GAN_learn
https://github.com/GatzZ/MLDS18
https://github.com/d31003/MLDS_2019spring/tree/master/HW3/3-1
# 目前调通了的
WGAN-GP WGAN(使用weight-clip) CGAN(接受单输入和多输入版本的)
## WGAN(weight-clip) 3000epoch
![WGAN(weight-clip) 3000epoch](https://www.picgd.com/images/2019/07/26/319aeb5bedf36277179e3aeec31f13d7.png)
## WGAN-GP 4000epoch
![WGAN-GP 4000epoch](https://www.picgd.com/images/2019/07/26/244158b8bb499ff1680afc86b1633db7.png)
## CGAN 单输入 9500epoch
0. 请使用HW3_1中generate_dataset_onelabel.py来生成数据集
1. 有些图片跟标签不符，看了看原数据集好多标注错误
2. 图片比较糊，可能与Dense层参数过少有关，如果显存大请自行增加显存
![CGAN 9500epoch](https://www.picgd.com/images/2019/07/30/8f8042eb2866650e12178ed39c20d112.png)
## CGAN 多输入 19000epoch
0. 请使用HW3_1中generate_dataset_v3来生成数据集
1. 有些图片跟标签不符，看了看原数据集好多标注错误
2. 图片比较糊，可能与Dense层参数过少有关，如果显存大请自行增加显存
![CGAN 19000epoch](https://www.picgd.com/images/2019/07/30/aca81426814a67cd6565612b8bd57d6c.png)
GAN和EBGAN还没调通，能运行但是效果emmmmm 令人难以捉摸
