import sys
import os
import time
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

LETTERS_DIGITS = (
"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P",
"Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z")

input_labels = np.load('preprocess_data/digits/digits_input_labels.npy')

enc = OneHotEncoder(handle_unknown='ignore')
# leter = np.array(LETTERS_DIGITS)
# print(enumerate(set(leter)))

enc.fit(np.array(LETTERS_DIGITS).reshape([-1, 1]))
decode = enc.inverse_transform(input_labels)
targets = decode.flatten()
print(decode)


leter = np.array(LETTERS_DIGITS)
print(leter)
print(list(enumerate(leter)))

X_ = np.load('preprocess_data/digits/digits_input_images_tSEN.npy')

fig = plt.figure()
for i, t in enumerate(set(targets)):
    idx = targets == t
    plt.scatter(X_[idx, 0], X_[idx, 1])
plt.show()