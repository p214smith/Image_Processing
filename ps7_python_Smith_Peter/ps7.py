import numpy as np
from skimage.feature import hog
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn import svm
from skimage import exposure
from matplotlib import gridspec
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers
from keras.preprocessing import image
car = cv.imread("./input/p1/car.jpg")
features, car_features = hog(car,cells_per_block=(2,2),channel_axis=2,visualize=True)
fig, axs = plt.subplots(1,2)
axs[0].imshow(car)
axs[0].set_title("input image")
car_features = exposure.rescale_intensity(car_features, in_range=(0,10))
axs[1].imshow(car_features,cmap=plt.cm.gray)
axs[1].set_title("Histogram of Oriented Gradients")
plt.show()
print(features.shape)
y_train = np.zeros(100,np.uint8)
for i in range(50):
    y_train[i + 50] = 1
X_train = np.zeros((100,1188))
for i in range(50):
    string = "./input/p1/train_imgs/0_" + str(i + 1) +".jpg"
    img = cv.imread(string)
    features, cfeat = hog(img,cells_per_block=(2,2),channel_axis=2,visualize=True)
    features = np.reshape(features,(1,1188))
    X_train[i,:] = features
for i in range(50):
    string = "./input/p1/train_imgs/1_" + str(i + 1) +".jpg"
    img = cv.imread(string)
    features, cfeat = hog(img,cells_per_block=(2,2),channel_axis=2,visualize=True)
    features = np.reshape(features,(1,1188))
    X_train[i+50,:] = features
print("Shape of X_train =" + str(X_train.shape))
print("Shape of y_train =" + str(y_train.shape))
model = svm.LinearSVC(dual='auto')
model.fit(X_train,y_train)
k = 5
l = 2
for kk in range(k):
    for ll in range(l):
        string = "./input/p1/test_imgs/" + str(ll) + "_0" + str(kk + 1) + ".jpg"
        img = cv.imread(string)
        i , j , k= img.shape
        i = i - 96
        j = j - 32
        scores = np.zeros((i,j))
        for ii in range(i):
            for jj in range(j):
                window = img[ii:ii+96,jj:jj+32,:]
                features, cfeat = hog(window,cells_per_block=(2,2),channel_axis=2,visualize=True)
                features = np.reshape(features,(1,1188))
                scores[ii,jj] = model.decision_function(features)[0]
        idx = np.unravel_index(np.argmax(scores,axis=None),scores.shape)
        print(scores[idx])
        if scores[idx] > 0.0:
            img = cv.rectangle(img,(idx[1],idx[0]),(idx[1]+32,idx[0]+96),color=(255,0,0),thickness=2)
        string = "./output/ps7-1-d-" + str(kk + 1 + 5*ll) + ".png"
        cv.imwrite(string,img)
        
ds_train_ = image_dataset_from_directory('./input/p2/train_imgs',
                                        labels='inferred',label_mode='categorical',
                                        image_size=[32,32],batch_size=100,shuffle=True)
ds_test_ = image_dataset_from_directory('./input/p2/train_imgs',
                                        labels='inferred',label_mode='categorical',
                                        image_size=[32,32],batch_size=100,shuffle=False)

model = keras.Sequential([layers.Conv2D(filters=32,kernel_size=5,activation="relu",padding='same',input_shape=[32,32,3]),
                          layers.BatchNormalization(),
                          layers.MaxPool2D(),
                          layers.Conv2D(filters=32,kernel_size=5,activation="relu",padding='same'),
                          layers.BatchNormalization(),
                          layers.MaxPool2D(),
                          layers.Conv2D(filters=64,kernel_size=5,activation="relu",padding='same'),
                          layers.BatchNormalization(),
                          layers.MaxPool2D(),
                          layers.Flatten(),
                          layers.Dense(units=27,activation="relu"),
                          layers.Dense(units=3,activation='softmax')
                          ])
model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(ds_train_,epochs=10)
plt.plot(history.history['accuracy'],label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0,1])
plt.legend(loc='lower right')
plt.show()
scores = model.evaluate(ds_test_,verbose=0)
print('Accuracy on testing data: {}% \n Error on testing data: {}% \n'.format(scores[1], 1 - scores[1]))
print(model.predict(ds_test_))
fig, axs = plt.subplots(3,2)
im1 = cv.imread('./input/p2/display_imgs/image254.png')
im2 = cv.imread('./input/p2/display_imgs/image547.png')
im3 = cv.imread('./input/p2/display_imgs/image869.png')
im4 = cv.imread('./input/p2/display_imgs/image888.png')
im5 = cv.imread('./input/p2/display_imgs/image1820.png')
im6 = cv.imread('./input/p2/display_imgs/image2072.png')
classes = ["Airplane","Automobile","Truck"]
im1 = np.expand_dims(im1,axis=0)
im2 = np.expand_dims(im2,axis=0)
im3 = np.expand_dims(im3,axis=0)
im4 = np.expand_dims(im4,axis=0)
im5 = np.expand_dims(im5,axis=0)
im6 = np.expand_dims(im6,axis=0)

pred1 = model.predict(im1,verbose=0)
print(pred1)
class_ID = np.argmax(pred1)
title = 'predicted' + classes[class_ID]
axs[0,0].imshow(tf.squeeze(im1))
axs[0,0].axis('off')
axs[0,0].set_title('predicted ' + classes[class_ID])

pred1 = model.predict(im2,verbose=0)
print(pred1)
class_ID = np.argmax(pred1)
title = 'predicted' + classes[class_ID]
axs[0,1].imshow(tf.squeeze(im2))
axs[0,1].axis('off')
axs[0,1].set_title('predicted ' + classes[class_ID])

pred1 = model.predict(im3,verbose=0)
print(pred1)
class_ID = np.argmax(pred1)
title = 'predicted' + classes[class_ID]
axs[1,0].imshow(tf.squeeze(im3))
axs[1,0].axis('off')
axs[1,0].set_title('predicted ' + classes[class_ID])

pred1 = model.predict(im4,verbose=0)
print(pred1)
class_ID = np.argmax(pred1)
title = 'predicted' + classes[class_ID]
axs[1,1].imshow(tf.squeeze(im4))
axs[1,1].axis('off')
axs[1,1].set_title('predicted ' + classes[class_ID])

pred1 = model.predict(im5,verbose=0)
print(pred1)
class_ID = np.argmax(pred1)
title = 'predicted' + classes[class_ID]
axs[2,0].imshow(tf.squeeze(im5))
axs[2,0].axis('off')
axs[2,0].set_title('predicted ' + classes[class_ID])

pred1 = model.predict(im6,verbose=0)
print(pred1)
class_ID = np.argmax(pred1)
title = 'predicted' + classes[class_ID]
axs[2,1].imshow(tf.squeeze(im6))
axs[2,1].axis('off')
axs[2,1].set_title('predicted ' + classes[class_ID])
plt.show()
