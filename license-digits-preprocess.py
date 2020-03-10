import os
import time

import numpy as np
from PIL import Image
from sklearn.manifold import TSNE

SIZE = 1280
WIDTH = 32
HEIGHT = 40
NUM_CLASSES = 34
iterations = 1000

SAVER_DIR = "train-saver/digits/"

LETTERS_DIGITS = (
"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P",
"Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z")
license_num = ""


time_begin = time.time()

# 第一次遍历图片目录是为了获取图片总数
print('第一次遍历图片目录是为了获取图片总数')
input_count = 0
for i in range(0, NUM_CLASSES):
    dir = './train_images/training-set/%s/' % i  # 这里可以改成你自己的图片目录，i为分类标签
    for rt, dirs, files in os.walk(dir):
        for filename in files:
            input_count += 1

# 定义对应维数和各维长度的数组
input_images = np.array([[0] * SIZE for i in range(input_count)])
input_labels = np.array([[0] * NUM_CLASSES for i in range(input_count)])

# 第二次遍历图片目录是为了生成图片数据和标签
print('第二次遍历图片目录是为了生成图片数据和标签')
index = 0
for i in range(0, NUM_CLASSES):
    dir = './train_images/training-set/%s/' % i  # 这里可以改成你自己的图片目录，i为分类标签
    for rt, dirs, files in os.walk(dir):
        for filename in files:
            filename = dir + filename
            img = Image.open(filename)
            width = img.size[0]
            height = img.size[1]
            for h in range(0, height):
                for w in range(0, width):
                    # 通过这样的处理，使数字的线条变细，有利于提高识别准确率
                    if img.getpixel((w, h)) > 230:
                        input_images[index][w + h * width] = 0
                    else:
                        input_images[index][w + h * width] = 1
            input_labels[index][i] = 1
            index += 1

# 第一次遍历图片目录是为了获取图片总数
print('第一次遍历图片目录是为了获取图片总数')
val_count = 0
for i in range(0, NUM_CLASSES):
    dir = './train_images/validation-set/%s/' % i  # 这里可以改成你自己的图片目录，i为分类标签
    for rt, dirs, files in os.walk(dir):
        for filename in files:
            val_count += 1

# 定义对应维数和各维长度的数组
val_images = np.array([[0] * SIZE for i in range(val_count)])
val_labels = np.array([[0] * NUM_CLASSES for i in range(val_count)])

# 第二次遍历图片目录是为了生成图片数据和标签
print('第二次遍历图片目录是为了生成图片数据和标签')
index = 0
for i in range(0, NUM_CLASSES):
    dir = './train_images/validation-set/%s/' % i  # 这里可以改成你自己的图片目录，i为分类标签
    for rt, dirs, files in os.walk(dir):
        for filename in files:
            filename = dir + filename
            img = Image.open(filename)
            width = img.size[0]
            height = img.size[1]
            for h in range(0, height):
                for w in range(0, width):
                    # 通过这样的处理，使数字的线条变细，有利于提高识别准确率
                    if img.getpixel((w, h)) > 230:
                        val_images[index][w + h * width] = 0
                    else:
                        val_images[index][w + h * width] = 1
            val_labels[index][i] = 1
            index += 1

print('t-SEN降维处理')
X_ = TSNE(n_components=3, init='pca').fit_transform(input_images)
print('存储降维后的digits-input-images数据')
np.save('preprocess_data/digits/digits_input_images_tSEN_3.npy', X_)

print('保存预处理数据')
np.save('preprocess_data/digits/digits_input_images.npy', input_images)
np.save('preprocess_data/digits/digits_val_images.npy', val_images)
np.save('preprocess_data/digits/digits_input_labels.npy', input_labels)
np.save('preprocess_data/digits/digits_val_labels.npy', val_labels)