from PIL import Image
from os import listdir, name
from os.path import join
import os
import cv2


Folder_name = 'Dirty Panicle Disease'

file = "train2demo/"
Rename_file = [file+k for k in listdir(file) if listdir(join(file, k ))]
#print(Rename_file)

def Rotate (Rename_file):
    for i in range(len(Rename_file)):
        loop = 4
        #print(Rename_file[i].split('/')[1])
        if(Rename_file[i].split('/')[1] ==  Folder_name):
            print("Found!")
            #print(os.listdir(Rename_file[i]))
            count = len(os.listdir(Rename_file[i]))
            name = os.listdir(Rename_file[i])
            #print(name)
            #print(count)
            slash = "/"
            for k in range(count):
                Rotate_path_pic = os.path.join(Rename_file[i]+slash+name[k])
                print(Rotate_path_pic)
                print(k)
                theta = 90
                for l in range(loop):
                    print(l)
                
                    if(theta < 360 ):
                        JPG = ".jpg"
                        dot = '.'
                        print(str(name[k])+str(theta)+str(JPG))
                        print(str(name[k].split('.')[0])+dot+str(name[k].split('.')[1])+dot+str(theta)) #EX. Sheath rot Disease + . + 99 + . + 270 ==> Sheath Rot Disease.99.270
                        im = Image.open(Rotate_path_pic)
                        im = im.convert("RGB")
                        
                        
                        print(theta)
                        im.rotate(theta).save(str(name[k].split('.')[0])+dot+str(name[k].split('.')[1])+dot+str(theta)+JPG)
                        theta= theta + 90 
                
                pass
        else : 
            pass
    pass

Rotate(Rename_file)