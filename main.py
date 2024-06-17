import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# pre-pocessing

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x - pixel data // image // handwritten number
# y - classification // actiual value // number

#normalizing 0 - 255 to 0 - 1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
