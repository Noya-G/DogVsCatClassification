import os
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils  import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, log_loss
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2, glob, random
import warnings
from datetime import datetime
count= 0
warnings.filterwarnings("ignore")

trainDir = #Enter here the directory of the train set
testDir = #Enter here the directory of the test set

dataset = []
mapping = {"dog": 0,"cat": 1}




# dog = glob.glob(pathTest + "Dog/*.jpg")
# cat = glob.glob(pathTest + "Cat/*.jpg")
#
# for im in trainDir:
#     image = load_img(os.path.join(path, im), grayscale=False,
#                      color_mode='rgb', target_size=(64, 64))

for file in os.listdir(trainDir):
    if file != '.DS_Store':
        path = os.path.join(trainDir,file)
        for im in os.listdir(path):
                try:
                    # print(file+" "+im)
                    image = load_img(os.path.join(path,im),
                                     color_mode='rgb', target_size=(64,64))
                    image=img_to_array(image)
                    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image=image/255
                    dataset.append([image,mapping[file]])
                except Exception as e: print(e)


counter=0
testset = []
for file in os.listdir(testDir):
    if file != '.DS_Store':
        path = os.path.join(testDir,file)
        for im in os.listdir(path):
            try:
                image = load_img(os.path.join(path,im),
                                     color_mode='rgb', target_size=(64,64))
                image=img_to_array(image)
                # image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                image=image/255
                testset.append([image,mapping[file]])
                counter+=1
            except Exception as e: print(e)

print(f"test set size {counter}\n\n")
data,labels0 = zip(*dataset)
test,tlabels0 = zip(*testset)

cats=0
dogs=0

for i in labels0:
    if i == 0:
        cats+=1
    elif i == 1:
        dogs+=1

len_labels0  = len(labels0)
print(f"Total no. of images in dataset ar {len_labels0}")
print(f"Number of images classified as cat: {cats}\n"
      f"Number of images classified as dogs:{dogs}\n")

cats=0
dogs=0

for i in tlabels0:
    if i == 0:
        cats+=1
    elif i == 1:
        dogs+=1

len_tlabels0  = len(tlabels0)
print(f"Total no. of images in test set ar {len_tlabels0}")
print(f"Number of images classified as cat: {cats}\n"
      f"Number of images classified as dogs:{dogs}\n")

labels1 = to_categorical(labels0)
data = np.array(data)
labels = np.array(labels1)
print(f"Data shape: {data.shape}\n"
      f"Table shape:{labels}")


tlabels1 = to_categorical(tlabels0)
test = np.array(data)
tlabels = np.array(labels1)
print(f"Data shape: {test.shape}\n"
      f"Table shape:{tlabels}")

data2 = data.reshape(-1,64,64,3)
test2 = test.reshape(-1,64,64,3)
print(f"Data after reshape: {data2.shape}\n"
      f"Test after reshape:{test2.shape}")

trainx, valx,trainy, valy = train_test_split(data,labels,test_size=0.2,random_state=44)

print(trainx.shape)
print(valx.shape)
print(trainy.shape)
print(valy.shape)

datagen = ImageDataGenerator(horizontal_flip=True,
                             vertical_flip=True,
                             rotation_range=0.2,
                             zoom_range=0.2,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.1,
                             fill_mode="nearest")

model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(64,64,3), activation='relu'))
model.add(MaxPooling2D((5,5)))

model.add(Conv2D(32,(3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(units=512, activation= 'relu'))

model.add(Dense(units=2, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

his = model.fit(datagen.flow(trainx,trainy,batch_size=256),
                validation_data=(valx,valy),epochs=100)

y_pred=model.predict(valx)
pred = np.argmax(y_pred, axis=1)
ground = np.argmax(valy, axis=1)
print(classification_report(ground,pred))
get_acc = his.history['accuracy']
value_acc = his.history['val_accuracy']
get_loss =  his.history['loss']
validation_loss = his.history['val_loss']

epochs = range(len(get_acc))
plt.plot(epochs, get_acc,'r',label='Accuracy of Training Data')
plt.plot(epochs, value_acc,'b',label='Accuracy of Validation Data')
plt.title('Training vs Validation Accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()

epochs = range(len(get_loss))
plt.plot(epochs,get_loss,'r',label='Loss of Training Data')
plt.plot(epochs,validation_loss,'b',label='Loss of Validation Data')
plt.title('Training vs Validation Loss')
plt.legend(loc=0)
plt.figure()
plt.show()

img_dog = load_img("/Users/noyagendelman/Desktop/IMG_3242.jpeg",target_size=(64,64))
image = img_to_array(img_dog)
image = image/255.0
predictionImage = np.array(image)
predictionImage = np.expand_dims(image,axis=0)

reverse_mapping = {0:'dog', 1:'cat'}

def mapper(value):
    return reverse_mapping[value]


print(test.shape)
prediction2 = model.predict(test)
print(prediction2.shape)


print(f"model accuracy: {get_acc[0]}")

now = str(datetime.now())
now = now.replace(":","").replace(" ","").replace("-","")

path = os.path.join("/Users/noyagendelman/Desktop/archive/models/DogsVSCats",
                    now)
model.save_weights(path)

