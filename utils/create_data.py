from Mydataloader import RandomTarget_dataset
import numpy as np
import cv2
import os
from util import *
def create_data(root,create_num,img_num,overwrite):
    batch_size = 1
    if overwrite:
        t=0
        labels=[]
    else:

        t, labels = check(root)
    while t<create_num:
        dataset = RandomTarget_dataset(root = r'C:\Project\python\dataset\加框后的JPEG图',
                              batch_size=batch_size,bg_r=96,bg_w=128,
                              bg_root=r'C:\Project\python\dataset\background',
                              img_num=img_num,
                              Chinese_path=True,
                                valid=True)
        # save_data(dataset,f'{root}/images',train_num)
        for (img, label) in dataset:
            img1 = np.array(img[0], dtype=np.uint8)
            labels.append(label[0])

            # for i in range(3):
            #     for j in range(4):
            #         y, x = 32 * i, 32 * j
            #         img1 = cv2.rectangle(img1, (x, y), (x + 32, y + 32), color=(0, 0, 255))
            # img_pos = decode(label[0])
            # print(label[0])
            # print(img_pos)
            # for point in img_pos:
            #     x, y = point
            #     cv2.circle(img1, (int(x), int(y)), radius=2, color=(0, 255, 0))
            # plt.figure()
            # plt.imshow(img1)
            # plt.show()

            """
            opencv 为bgr要转rgb
            """
            img1 = img1[:, :, ::-1]
            t += 1
            print(t)
            cv2.imwrite(f'{root}/images/{t}.jpg', img1)

            if t >= create_num:
                break
    labels = np.array(labels)
    np.save(f'{root}/labels.npy',labels)
def check(root):
    t = len(os.listdir(f'{root}/images'))
    labels = np.load(f'{root}/labels.npy')
    labels = labels.tolist()
    print(t,len(labels))
    assert t==len(labels)
    return t,labels

if __name__ == '__main__':

    t= 1
    train_num=100000
    valid_num=1000
    test_num=10
    labels =[]
    train_root = '../dataset/train'
    valid_root = '../dataset/valid'
    test_root = '../dataset/test'
    overwrite = True
    img_num = 3
    # create_data(root=train_root, create_num=train_num, img_num=img_num, overwrite=overwrite)
    # create_data(root=valid_root,create_num=valid_num,img_num=img_num,overwrite=overwrite)
    create_data(root=test_root,create_num=test_num,img_num=img_num,overwrite=overwrite)

