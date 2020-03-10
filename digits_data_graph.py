import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from mpl_toolkits.mplot3d import Axes3D

LETTERS_DIGITS = (
"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P",
"Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z")

input_labels = np.load('preprocess_data/digits/digits_input_labels.npy')

#将One-hot热变化转为原始格式
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(np.array(LETTERS_DIGITS).reshape([-1, 1]))
decode = enc.inverse_transform(input_labels)
targets = decode.flatten()

#加载t-SEN数据
X_ = np.load('preprocess_data/digits/digits_input_images_tSEN_3.npy')


def graph_2d():
    fig = plt.figure()
    for i, t in enumerate(set(targets)):
        idx = targets == t
        plt.scatter(X_[idx, 0], X_[idx, 1])
    plt.show()

def graph_3d():
    fig = plt.figure()
    ax = Axes3D(fig)
    # 添加坐标轴(顺序是Z, Y, X)
    for i, t in enumerate(set(targets)):
        idx = targets == t
        ax.scatter(X_[idx, 0], X_[idx, 1], X_[idx, 2])
    plt.show()

graph_3d()