import tensorflow as tf

import random
from PIL import Image
import cv2
import numpy as np
from util import *
import math
import matplotlib.pyplot as plt
import os
from target_utils import target_encoder
#解决中文路径
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv_img

class RandomTarget_dataset(tf.keras.utils.Sequence):
    def __init__(self,root: str,
                 batch_size: int,
                 bg_root,
                 bg_r=96,
                 bg_w=128,
                 box_r=32,
                 img_size=(22, 28),
                 img_num=None,
                 Chinese_path=False,
                 valid=False):
        self.paths = self.get_paths_and_labels(root, class_num=3)
        random.shuffle(self.paths)
        if valid:
            self.paths = self.paths[:len(self.paths)//5]
        self.batch_size = batch_size
        self.bg_r = bg_r
        self.bg_w = bg_w
        self.bg_paths = [f'{bg_root}/{p}' for p in os.listdir(bg_root)]
        self.img_num=img_num
        self.Chinese_path=Chinese_path
        self.box_r = box_r
        self.img_size = img_size
        self.target_d = round(self.box_r*(2**0.5))#目标之间的最近距离
        self.encoder = target_encoder()
    def get_paths_and_labels(self,root, class_num)->list:
        #针对猪肺默认双层目录读取
        img_paths = []
        if class_num == 3:
            for cls in os.listdir(root):
                path_1 = f"{root}/{cls}"
                for s_cls in os.listdir(path_1):
                    path_2 = f"{path_1}/{s_cls}"
                    for img_path in os.listdir(path_2):
                        img_paths.append(f"{path_2}/{img_path}")
            return img_paths

    def creat_random_data(self,path,bg_img_path):

        bg_size = (self.bg_w, self.bg_r)

        if self.Chinese_path:
            bg_img = cv_imread(bg_img_path)
        else:
            bg_img = cv2.imread(bg_img_path)
        bg_img = cv2.resize(bg_img, bg_size)

        if self.img_num==None:
            img_num = random.randint(2, 4)
        else:
            img_num=self.img_num

        target_centers = []  # 图片中心的偏移
        if self.Chinese_path:
            img = cv_imread(path)
        else:
            img = cv2.imread(path)
        #去除边框
        img = remove_frame(img)

        l=3#边界距离
        img0 = np.zeros((self.bg_r + self.box_r, self.bg_w + self.box_r, 3), dtype=np.uint8)

        for i in range(img_num):
            x, y = random.randint(l, self.bg_w-1-l), random.randint(l, self.bg_r-1-l)
            t =0
            while judge_point(target_centers, x, y,self.target_d) != True:
                x, y = random.randint(l, self.bg_w-1-l), random.randint(l, self.bg_r-1-l)
                t+=1
                if t>=50:
                    break
            if t>=50:
                break
            size = random.randint(self.img_size[0],self.img_size[1])//2*2+1
            img_rotate = random_rotate(img)
            img_rotate = cv2.resize(img_rotate, (size, size))

            _x = x+self.box_r//2-(size-1)//2
            x_ = x+self.box_r//2+(size-1)//2+1
            _y = y+self.box_r//2-(size-1)//2
            y_ = y+self.box_r//2+(size-1)//2+1
            img0[_y:y_,_x:x_,:] = img_rotate[:, :, :]
            target_centers.append([x, y,size])
        # 创建一张黑色画布,尺寸要比目标尺寸大，最后截取中间部分，这样可以包含边界不完整情况


        # for point in target_centers:
        #     size = random.randint(self.img_size[0],self.img_size[1])//2*2+1
        #     img_rotate = random_rotate(img)
        #     img_rotate = cv2.resize(img_rotate, (size, size))
        #     x,y = point
        #     _x = x+self.box_r//2-(size-1)//2
        #     x_ = x+self.box_r//2+(size-1)//2+1
        #     _y = y+self.box_r//2-(size-1)//2
        #     y_ = y+self.box_r//2+(size-1)//2+1
        #     img0[_y:y_,_x:x_,:] = img_rotate[:, :, :]
        img0 =  img0[self.box_r//2:self.box_r//2+self.bg_r,self.box_r//2:self.box_r//2+self.bg_w]

        mask = get_mask(img0)
        bg_img = cv2.bitwise_and(bg_img, bg_img, mask=mask)
        img1 = cv2.add(bg_img, img0)
        # label = self.encode(target_centers,self.box_r)
        label = self.encoder.encode(target_centers)
        """
        opencv 为bgr要转rgb
        """
        img1 = img1[:,:,::-1]
        return np.array(img1,dtype=np.float32),np.array(label,dtype=np.float32)
    # def encode(self,target_centers,box_r):
    #     label = np.zeros(36, dtype=np.float32)  # 0:24 为目标中心相对格子中心x,y坐标偏移量， 24:为12个格子内是否有目标
    #
    #     for point in target_centers:
    #         x, y,_ = point
    #         box_idex = x // box_r + 4 * (y // box_r)
    #         x_offset, y_offset = x % box_r - box_r / 2, y % box_r - box_r / 2#相对格子中心
    #         # x_offset, y_offset = x % box_r, y % box_r #相对格子左上角
    #         x_offset, y_offset = x_offset / box_r, y_offset / box_r  # 归一化
    #         label[2 * box_idex], label[2 * box_idex + 1] = x_offset, y_offset
    #         label[24 + box_idex] = 1
    #     return label
    def on_epoch_end(self):
        random.shuffle(self.paths)
    def __len__(self):
        return math.ceil(len(self.paths) / self.batch_size)
    def __getitem__(self, item):
        batch = self.paths[item * self.batch_size:(item + 1) * self.batch_size]
        imgs = []
        labels = []
        for path in batch:
            bg_img_path = self.bg_paths[random.randint(0,len(self.bg_paths)-1)]
            img,label=self.creat_random_data(path,bg_img_path)
            imgs.append(img)
            labels.append(label)

        return np.array(imgs,dtype=np.float32),np.array(labels,dtype=np.float32)

class Fast_dataset(tf.keras.utils.Sequence):
    def __init__(self,imgs_path:str,labels_path,batch_size,input_size=(96,128)):
        self.imgs_path = imgs_path
        self.labels_path = labels_path
        self.batch_size = batch_size
        self.imgs,self.labels = self.get_paths_and_labels(self.imgs_path,self.labels_path)
        self.input_r,self.input_w = input_size
    def get_paths_and_labels(self,imgs_path,labels_path):
        imgs = os.listdir(imgs_path)
        labels0 = np.load(labels_path)
        labels =[]
        for path in imgs:
            idx = int(path[:-4])
            labels.append(labels0[idx-1])
        return imgs,labels
    def __len__(self):
        return math.ceil(len(self.imgs) / self.batch_size)
    def __getitem__(self, item):
        batch_img_path = self.imgs[item * self.batch_size:(item + 1) * self.batch_size]
        batch_labels = self.labels[item * self.batch_size:(item + 1) * self.batch_size]
        imgs = []
        for path in batch_img_path:
            img = cv2.imread(f'{self.imgs_path}/{path}')
            img = cv2.resize(img,(self.input_w,self.input_r))
            img = img[:,:,::-1]
            imgs.append(img)
        return np.array(imgs, dtype=np.float32), np.array(batch_labels, dtype=np.float32)
if __name__=='__main__':
    batch_size =1
    img_num=3
    dataset = RandomTarget_dataset(root = r'C:\Project\python\dataset\加框后的JPEG图',
                          batch_size=batch_size,bg_r=96,bg_w=128,
                          bg_root=r'C:\Project\python\dataset\background',
                          img_num=img_num,
                          Chinese_path=True)
    # dataset = Fast_dataset(
    #     imgs_path='../dataset/valid/images',
    #     labels_path='../dataset/valid/labels.npy',
    #     batch_size=batch_size
    # )
    print(dataset[0][0])
    for (img,label) in dataset:
        print(img.shape)
        img1 = np.array(img[0],dtype=np.uint8)
        #绘制网格
        for i in range(3):
            for j in range(4):
                y, x = 32 * i, 32 * j
                img1 = cv2.rectangle(img1, (x, y), (x + 32, y + 32), color=(0, 0, 255))
        img_pos=decode(label[0])
        print(label[0])
        print(img_pos)
        for point in img_pos:
            x, y = point
            cv2.circle(img1, (int(x), int(y)), radius=2, color=(0, 255, 0))
        plt.figure()
        plt.imshow(img1)
        plt.show()
