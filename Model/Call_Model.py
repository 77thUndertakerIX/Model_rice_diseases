from tensorflow import keras
import tensorflow as tf 
import os 
from os.path import join
from os import listdir
import numpy as np
import cv2

size = 128

model = keras.models.load_model('Better_model.h5')
model.summary

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