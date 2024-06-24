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


# neural network

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(10, activation = 'softmax')) #output layer

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)
# epochs - how many times the same data will be seen

model.export('handwritten.model')


# load the model

# model = tf.keras.layers.TFSMLayer('handwritten.model', call_endpoint='serving_default')
# model = tf.keras.models.load_model('handwritten.model')

loss, accuracy = model.evaluate(x_test, y_test)
print("loss and accuracy:  ")
print(loss)
print(accuracy)

image_number = 4
while os.path.isfile(f"digits/{image_number}.png"):
    try:    
        img = cv2.imread(f"digits/{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"this digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number += 1


