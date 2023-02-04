from tensorflow import keras
from regression_model import test_dataset
import numpy as np
import matplotlib.pyplot as plt
import cv2

def predict(model,dataset,input_r,input_w):
    for (img, label) in dataset:
        img1 = np.array(img[0], dtype=np.uint8)
        # 绘制网格
        y = model.predict(img)
        x_pre,y_pre =y[0]
        print(y)
        for i in range(3):
            for j in range(4):
                y, x = 32 * i, 32 * j
                img1 = cv2.rectangle(img1, (x, y), (x + 32, y + 32), color=(0, 0, 255))
        print(label[0])
        x, y = label[0]
        cv2.circle(img1, (int(x*input_w), int(y*input_r)), radius=2, color=(0, 255, 0))


        cv2.circle(img1, (int(x_pre*input_w), int(y_pre*input_r)), radius=2, color=(255, 0, 0))
        plt.figure()
        plt.imshow(img1[:, :, ::-1])
        plt.show()

if __name__ == '__main__':
    model = keras.models.load_model('models_save/2023_01_26_23_36_33/model_61_0.0001.h5')
    model.summary()

    input_r, input_w = 96, 128
    batch_size = 1
    valid_dataset = test_dataset(root=r'C:\Project\python\dataset\加框后的JPEG图',
                                 batch_size=batch_size, bg_r=input_r, bg_w=input_w,
                                 bg_root=r'C:\Project\python\dataset\background',
                                 Chinese_path=True,
                                 valid=True
                                 )
    predict(model,valid_dataset,input_r, input_w)