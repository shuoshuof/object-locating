import cv2
import os
import numpy as np
import imutils
import random

class target_encoder(object):
    def __init__(self,
                 input_size:tuple=(96,128),
                 FeatureMap_shape:tuple=(3,4),
                 box_size=40,
                 pred_size=False):

        self.input_size = input_size
        self.FeatureMap_shape = FeatureMap_shape
        assert self.input_size[0]/self.FeatureMap_shape[0] == self.input_size[1]/self.FeatureMap_shape[1]

        #相邻格子中心的距离
        self.box_d = self.input_size[0]/self.FeatureMap_shape[0]
        #box_size应比box_d略大，这样能解决边界问题
        self.box_size = box_size
        self.box_centers = self.get_box_centers()
        #是否预测图片尺寸
        self.pred_size = pred_size
    def get_box_centers(self):
        box_centers = []
        for y in range(self.FeatureMap_shape[0]):
            centers = []
            for x in range(self.FeatureMap_shape[1]):
                cx = self.box_d/2+x*self.box_d-0.5
                cy = self.box_d/2+y*self.box_d-0.5
                centers.append((cx,cy))
            box_centers.append(centers)
        return box_centers
    def neighbor_box_match(self,x,y,size,label):
        x_offset, y_offset = x % self.box_d - self.box_d / 2, y % self.box_d - self.box_d / 2
        box_x_idx,box_y_idx = x // self.box_d,y // self.box_d
        c_x,c_y = self.box_centers[box_y_idx][box_x_idx]

        neighbor_boxs = [
            ((1 if x_offset>0 else -1),0),
            (0,(1 if y_offset > 0 else -1)),
            ((1 if x_offset>0 else -1),(1 if y_offset > 0 else -1))
        ]

        def check(neighbor_box,box_x_idx,box_y_idx):
            box_x_idx, box_y_idx = neighbor_box[0]+box_x_idx,neighbor_box[1]+box_y_idx
            return box_x_idx>=0 and box_y_idx>=0 and box_y_idx<=2 and box_x_idx<=3

        filter(lambda x:check(x,box_x_idx,box_y_idx),neighbor_boxs)#去除越界数据

        for box in neighbor_boxs:
            x_limit = c_x+box[0]*(self.box_d-self.box_size//2)
            y_limit = c_y+box[1]*(self.box_d-self.box_size//2)
            if abs(x-x_limit)>=self.box_d-self.box_size//2 or  abs(y-y_limit)>=self.box_d-self.box_size//2:
                label[24+box_x_idx+box_y_idx*self.FeatureMap_shape[1]] = 1


    def FirstPrinciple_matching(self,target_centers,label):

        for point in target_centers:
            x, y,size = point
            box_x_idx, box_y_idx = int(x // self.box_d), int(y // self.box_d)
            box_cx,box_cy = self.box_centers[box_y_idx][box_x_idx]
            x_offset, y_offset = (x-box_cx)/self.box_size,(y-box_cy)/self.box_size
            #先验格子在label里的下标
            box_idx = box_x_idx+box_y_idx*self.FeatureMap_shape[1]
            label[24+box_idx] = 1
            label[2*box_idx],label[2*box_idx+1] = x_offset, y_offset
        return label

    def SecondPrinciple_matching(self,target_centers,label):

        for box_y_idx in range(self.FeatureMap_shape[0]):
            for box_x_idx in range(self.FeatureMap_shape[1]):
                box_cx, box_cy = self.box_centers[box_y_idx][box_x_idx]
                box_idx = box_x_idx + box_y_idx * self.FeatureMap_shape[1]
                #如果该格子已被占用,跳过
                if label[24+box_idx]:
                    continue
                x_matched,y_matched = None,None
                min_d = 100000

                #当前格子与目标中心匹配
                for point in target_centers:
                    x, y, size = point
                    dx = x-box_cx
                    dy = y-box_cy
                    if abs(dx)<=self.box_size//2 and abs(dy)<=self.box_size//2:
                        d = dx**2+dy**2
                        if d<=min_d:
                            min_d = d
                            x_matched,y_matched = x,y
                if x_matched is not None:
                    label[24 + box_idx] = 1
                    x_offset, y_offset = (x_matched - box_cx) / self.box_size, (y_matched - box_cy) / self.box_size
                    label[2 * box_idx], label[2 * box_idx + 1] = x_offset, y_offset
        return label

    def encode(self,target_centers):
        # 0:24 为目标中心相对格子中心x,y坐标偏移量， 24:为12个格子内是否有目标
        label = np.zeros(36, dtype=np.float32)
        label = self.FirstPrinciple_matching(target_centers,label)
        label = self.SecondPrinciple_matching(target_centers,label)

        return label
class target_decoder(object):
    def __init__(self,
                 input_size: tuple = (96, 128),
                 FeatureMap_shape: tuple = (3, 4),
                 box_size=40,
                 pred_size=False):
        self.input_size = input_size
        self.FeatureMap_shape = FeatureMap_shape
        assert self.input_size[0]/self.FeatureMap_shape[0] == self.input_size[1]/self.FeatureMap_shape[1]

        #相邻格子中心的距离
        self.box_d = self.input_size[0]/self.FeatureMap_shape[0]
        #box_size应比box_d略大，这样能解决边界问题
        self.box_size = box_size
        #是否预测图片尺寸
        self.pred_size = pred_size
    def NMS(self,label):
        pass
    def decode(self,label):
        pass
if __name__ == '__main__':
    encoder = target_encoder()
    target_centers =[[31.5,31.5,2],
                     [63.5,63.5,3]
                     ]
    label = encoder.encode(target_centers)
    print(label)
