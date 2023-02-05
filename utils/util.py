import cv2
import os
import numpy as np
import imutils
import random
from random import randint
import matplotlib.pyplot as plt

def random_rotate(img):
    r,w,_ = img.shape
    angle = random.randint(0, 361)
    # mask = np.ones(img.shape[:2],dtype=np.uint8)*255
    img1 = imutils.rotate_bound(img, angle)
    # mask = imutils.rotate_bound(mask,angle)
    img1 = cv2.resize(img1, (w,r))
    return img1
def get_mask(img):
    r, w, _ = img.shape
    img_mask = img.copy()
    img_mask = cv2.cvtColor(img_mask,cv2.COLOR_BGR2GRAY)
    _,mask = cv2.threshold(img_mask,50,255,cv2.THRESH_BINARY_INV)
    mask = cv2.resize(mask,(w,r))
    return mask
def encode(img_center_points,box_r = 32):
    label = np.zeros(36,dtype=np.float32) # 0:24 为目标中心相对格子中心x,y坐标偏移量， 24:为12个格子内是否有目标

    for point in img_center_points:
        x, y = point
        box_idex = x // box_r + 4 * (y // box_r)
        x_offset, y_offset = x % box_r - box_r / 2, y % box_r - box_r / 2#相对格子中心
        # x_offset, y_offset = x % box_r, y % box_r  # 相对格子左上角
        x_offset, y_offset = x_offset / box_r, y_offset / box_r  # 归一化
        label[2 * box_idex], label[2 * box_idex + 1] = x_offset, y_offset
        label[24 + box_idex] = 1

    return label
def decode(label,box_r = 32):
    judge_conf = label[24:]
    pos =[]

    for idx,point in enumerate(judge_conf):
        if point>=0.5:
            x_offset,y_offset = label[2*idx],label[2*idx+1]
            x,y = idx%4*32+x_offset*box_r+box_r/2,idx//4*32+y_offset*box_r+box_r/2#相对格子中心
            #x, y = idx % 4 * 32 + x_offset * box_r, idx // 4 * 32 + y_offset * box_r  # 相对格子左上角
            pos.append([x,y])
    return np.round(pos)

#判断生成的坐标是否太靠近
def judge_point(img_pos,x,y,d):
    for point in img_pos:
        x1,y1,_ =point
        if abs(x1-x)<=d and abs(y1-y)<=d:

            return False
    return True
def remove_frame(img):
    img = img[40:376,40:376,:]
    return img
if __name__=="__main__":
    # bg_img_path = r'C:\Users\du\Desktop\17_ai\make_dataset\blue_background\00000.jpg'
    # path = r'C:\Users\du\Desktop\17_ai\make_dataset\dataset_3_5\animals\cat\cat10.jpg'
    #
    # bg_r,bg_w = 96,128
    # bg_size = (bg_w,bg_r)
    # bg_img = cv2.imread(bg_img_path)
    #
    # bg_img = cv2.resize(bg_img, bg_size)
    #
    # img_num = random.randint(0,2)
    #
    # img_pos=[]
    # for i in range(img_num):
    #     x,y = random.randint(0,bg_w-32),random.randint(0,bg_r-32)
    #     while judge_point(img_pos,x,y)!=True:
    #         x,y = random.randint(0,bg_w-32),random.randint(0,bg_r-32)
    #     img_pos.append([x,y])
    #
    # img = cv2.imread(path)
    #
    #
    # img0 = np.zeros((bg_r,bg_w,3),dtype=np.uint8)
    #
    # img_center_points = []
    # for point in img_pos:
    #     x,y = point
    #     r, w = random.randint(20, 30), random.randint(20, 30)
    #     img = cv2.resize(img, (w, r))
    #     img = random_rotate(img)
    #     img0[y:y+r,x:x+w,:] = img[:,:,:]
    #     img_center_points.append([x+w//2,y+r//2])
    #
    # mask = get_mask(img0)
    # bg_img =cv2.bitwise_and(bg_img,bg_img,mask=mask)
    # img1 = cv2.add(bg_img,img0)
    # label = encode(img_center_points)
    # for i in range(3):
    #     for j in range(4):
    #         y,x = 32*i,32*j
    #         img1 = cv2.rectangle(img1,(x,y),(x+32,y+32),color=(0,0,255))
    # for point in img_center_points:
    #     x,y = point
    #     cv2.circle(img1,(x,y),radius=2,color=(0,255,0))
    # print(label[0:24])
    # print(label[24:])
    # cv2.imwrite('./test.jpg',img1)
    # plt.figure()
    # # plt.imshow(img0[:,:,::-1])
    # plt.imshow(img1[:,:,::-1])
    # plt.show()
    pass
