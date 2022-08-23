import os 
from os import listdir
from os.path import isfile,join

file = "train2demo/"
Rename_file = [file+k for k in listdir(file) if listdir(join(file, k ))]
print(Rename_file)

#Renamefile == filepath

def find(filepath):
    for i in range(len(Rename_file)):# i คือ จำนวนโฟลเดอร์ ใน Renamefile เช่น tran/.... train/.... รวมเป็น 2 ตัว
        pic = os.listdir(filepath[i]) # ดูในโฟลเดอร์ตัวที่ i ว่ามีรูปไรมั่ง
    
        count = len(pic) #นับจำนวนในlist pic ว่ามีเท่าไหร่
        
        for k in range(count): # ค่า count  คือค่าใน count
            dot = '.'
            JPG = '.jpg'
            print('__________________________________')
            print(k)
            old_name = Rename_file[i] + "/" + pic[k] # Ex. train/Sheath Rot disease/(ชื่อเก่า)
            print(old_name)
            new_name = Rename_file[i] + "/" + filepath[i].split('/')[1]+str(dot)+str(k)+JPG ## Ex. train/Sheath Rot disease/ ชื่อจาก file path มาแยกออก (train/Sheath Rot disease เหลือแค่ Sheath Rot disease)
            print(new_name)
            
            if os.path.isfile(new_name):
                    
                print("This file is already exists")
                print('__________________________________')
                pass
            else:
                # Rename the file
                os.rename(old_name, new_name)
            
        
    pass

find(Rename_file)