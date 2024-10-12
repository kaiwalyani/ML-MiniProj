#how to train a multilayered perceptrom 
import os
os.environ['KMP_DUPLICATE_LIB_OK'] ='True'
import cv2 #its about computer vision to load and process images
import numpy as np #will be used to working with numpy arrays
import matplotlib.pyplot as plt #used for visualization
import tensorflow as tf #used for machine learning part

mnist = tf.keras.datasets.mnist #while training data we use label data we already know how it is 
(x_train, y_train), (x_test, y_test) = mnist.load_data() #x is the pixel data like handwrittern and y is classification (no.)the digit

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))#relu= rectified linear unit
model.add(tf.keras.layers.Dense(128, activation='relu'))#relu= rectified linear unit
model.add(tf.keras.layers.Dense(10, activation='softmax'))#softmax makes sure that all the o/p all the 10 neurons add to 1, 

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics = ['accuracy'])

model.fit(x_train, y_train, epochs=3)

model.save('handwritten.model')

loss, accuracy = model.evaluate(x_test, y_test)
print(loss)
print(accuracy)

model = tf.keras.models.load_model('handwritten.model')

image_num = 1
while os.path.isfile(f"Digits/digit{image_num}.png"):
  try:
    img = cv2.imread(f"Digits/digit{image_num}.png")[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f"The number is probably {np.argmax(prediction)}")
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
  except:
    print("Error!")
  finally:
    image_num += 1