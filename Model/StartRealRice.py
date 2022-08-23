from numpy.core.fromnumeric import shape
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os,cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten 
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tqdm import tqdm
from os import listdir
from os.path import isfile ,join
import matplotlib.pyplot as plt
import pandas as pd

train = 'train2/' #ใช้ชื่อ train แบบเอาจริง/ ไม่ได้เพราะ cv2. มันอ่านภาษาไทยไม่ออก มันจะError
test = 'test/'
size = 128
num_classes = 7
train_path_data = [train+k for k in listdir(train) if listdir(join(train, k ))]
test_path_data = [test+k for k in listdir(test) if listdir(join(test, k))]
'''
def img4data(path):
    img_name = []
    img_data = []
    c = 0
    for img_path in (path):
        for items in tqdm(os.listdir(img_path)):
            files = os.path.join(img_path, items)
            c+=1
            check = img_path.split('/')[1]
            if check == 'forest':
                img_name.append([1,0,0])
            elif check == 'sea':
                img_name.append([[0,0,1]])
            elif check == 'bird':
                img_name.append([0,1,0])
            RawImg = cv2.imread(files, cv2.COLOR_BGR2RGB)
            RawImg = cv2.resize(RawImg, (size,size))
            img_data.append(RawImg)
    return img_data, img_name

'''
'''
sheath rot disease = กาบใบเน่่า

Sheath blight Disease = กาบใบแห้ง

Bacterial Blight Disease = ขอบใบแห้ง

Leaf Scald Disease = ใบวงสีน้ำตาล

Narrow Brown Spot Disease = ใบขีดสีน้ำตาล

Brown Spot Disease = ใบจุดน้ำตาล

Dirty Panicle Disease = เมล็ดด่าง

'''
def img2data(path):
  rawImgs = []
  labels = []
  c = 0
  for imagePath in (path):
      for item in tqdm(os.listdir(imagePath)): #os.listdir(คือ os.listdirectory) คือการไล่หาไฟล์ที่มีอยู่ในโฟลเดอร์ tqdmคือแสดงสถานะ
          file = os.path.join(imagePath, item) #นำ imagePath และ item มาเชื่อมกัน 
          #print(file)
          l = imagePath.split('/')[1] #ทำป้ายใส่ชื่อ โดยเอาสมาชิคแต่ละตัวใน imagePath ออกมาเอาชื่อตัวแรกเท่านั้น ตย. Tawan.Sugoi.jpg จะเอาแค่ Tawan มาเท่านั้น
          #print("_____________________________________________________")
          #print(imagePath)
          #print(item)
          #print(file)
          #print(l)
          if l == 'Sheath Rot Disease':
            labels.append([1,0,0,0,0,0,0])         
          elif l == 'Sheath blight Disease':
            labels.append([0,1,0,0,0,0,0])
          elif l == 'Bacterial Blight Disease':
            labels.append([0,0,1,0,0,0,0])
          elif l == 'Leaf Scald Disease':
            labels.append([0,0,0,1,0,0,0])
          elif l == 'Narrow Brown Spot Disease':
            labels.append([0,0,0,0,1,0,0])
          elif l == 'Brown Spot Disease':
            labels.append([0,0,0,0,0,1,0])
          elif l == 'Dirty Panicle Disease':
            labels.append([0,0,0,0,0,0,1])
          img = cv2.imread(file , cv2.COLOR_BGR2RGB)
          #cv2.imshow('test',img)
          #cv2.waitKey(1)
          img = cv2.resize(img ,(size,size))
          rawImgs.append(img)
          c+=1
          #print("checked")
          #print("_____________________________________________________")
          #print(rawImgs)
          #print(labels)
          #print(c)
  return rawImgs, labels

x_train,y_train= img2data(train_path_data)
#y_train = img2data(train_path_data)
x_test,y_test = img2data(test_path_data)

#print(x_train)
#print("________________")
#print(y_train)
#print("________________")

'''
print("Here 1 ")
#print(x_train)
print("____________________________________________________________________________________________")
print(y_train)
print("____________________________________________________________________________________________")
'''

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

#print(x_train.shape,y_train.shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
  
print("here 2 ")
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
print("__________________________________________________________________________________________")
print(x_train[0])
print("__________________________________________________________________________________________")

model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64,(3,3), activation='relu', input_shape=(size, size, 3)),
        #tf.keras.layers.AveragePooling2D(pool_size=(2,2)),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(128,(3,3), activation='relu', input_shape=(size, size, 3)),
        #tf.keras.layers.AveragePooling2D(pool_size=(2,2)),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Dropout(0.30),

        tf.keras.layers.Conv2D(128,(3,3), activation='relu', input_shape=(size, size, 3)),
        #tf.keras.layers.AveragePooling2D(pool_size=(2,2)),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Dropout(0.40),
        
        

        tf.keras.layers.Dense(16),
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(num_classes, activation='softmax')  
    ])
model.summary()
optimizers = tf.keras.optimizers.Adam(learning_rate=0.001)
#tf.keras.layers.AveragePooling2D(pool_size=(2,2)),
#tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
model.compile(optimizer=optimizers, loss='categorical_crossentropy'
                , metrics= ['accuracy'])


batch_size = 75
epochs = 10
early_stop = EarlyStopping(monitor='val_loss', patience=2)
history = model.fit(x_train, y_train ,
                    batch_size=batch_size, 
                    epochs=epochs ,
                    validation_data=(x_test, y_test),
                    callbacks=[early_stop],
                    shuffle=True)

'''                    
#softmax for one hot . . # sigmoid for 0/1
tf.keras.optimizers.Adam(learning_rate=0.001) 
tf.keras.optimizers.Adadelta(learning_rate=0.001) 
tf.keras.optimizers.Adagrad(learning_rate=0.001) 
tf.keras.optimizers.Adamax(learning_rate=0.001) 
tf.keras.optimizers.Ftrl(learning_rate=0.001) 
tf.keras.optimizers.Nadam(learning_rate=0.001) 
tf.keras.optimizers.RMSprop(learning_rate=0.001) 
tf.keras.optimizers.SGD(learning_rate=0.001) 
'''





plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

result = model.evaluate(x_test, y_test, verbose = 0) 

print('Test loss:', result[0]) 
print('Test accuracy:', result[1])

print('Do you want to save this model? (y/n)')
Ask_b4_save = input()
if (Ask_b4_save == 'y'):
  mode_save = model.save(input("Insert the model's name : ")+ '.h5')
  print('done')

elif (Ask_b4_save == 'n'):
  print('your model is not saved')

#test

'''
testpath = 'test/'
testImg = [testpath+f for f in listdir(testpath) if listdir(join(testpath, f))]
rimg = []
for imagePath in (testImg):
    for item in (os.listdir(imagePath)):
        file = os.path.join(imagePath, item)
        if item.split('.')[0] != "":
           
          img = cv2.imread(file , cv2.COLOR_BGR2RGB)
          ori = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
          img = cv2.resize(img ,(size,size))
          rimg = np.array(img)
          rimg = rimg.astype('float32')
          rimg /= 255
          rimg = np.reshape(rimg ,(1,128,128,3))
          predict = model.predict(rimg)
          label = ['Sheath Rot Disease','Sheath blight Disease','Bacterial Blight Disease','Leaf Scald Disease','Narrow Brown Spot Disease','Brown Spot Disease','Dirty Panicle Disease']
          result = label[np.argmax(predict)]
          #print(predict)
          print("__________________________________________________________________________________________________________")
          print('real:'+str(item))
          print('predict:'+str(result))
          print("__________________________________________________________________________________________________________")
          
          #cv2.putText(img, 'real:'+str(item), (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255,0,))
          #cv2.putText(img, 'predict'+str(result), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255,0,0))
          #plt.imshow(ori)
          #plt.show()
        


'''