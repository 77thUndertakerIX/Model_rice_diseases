import cv2
import os 
from os.path import isfile,join
from os import listdir, name
from PIL import Image

file = "train2demo/"
#path_file = [file+k for k in listdir(file) if listdir(join(file, k ))]







def find ():
    listdirect = os.listdir(file)
    for i in range(len(listdirect)):
        #print(listdirect[i])
        pic_path = file+listdirect[i]
        #print(pic_path)
        img = cv2.imread(pic_path)
        zoom_factor = 10
        name = listdirect[i]
    return img,zoom_factor,str(name)



#pic_path = img

def zoom(img, zoom_factor):
    
    resized = cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)
    
    cv2.imshow("Resized image", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imwrite('test.jpg',resized)
    

x,y,z= find()
zoom(x,y)
