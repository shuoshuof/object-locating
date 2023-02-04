import numpy as np
from tensorflow import keras
import tensorflow as tf
from utils.Mydataloader import Fast_dataset
from utils.util import *
import cv2

class MyTensorboardCallback(keras.callbacks.Callback):
    def __init__(self,log_dir:str,test_dir:str='./dataset/test',input_size=(96,128)):
        super(MyTensorboardCallback,self).__init__()
        self.writer = tf.summary.create_file_writer(log_dir)
        self.dataset = Fast_dataset(
            imgs_path=f'{test_dir}/images',
            labels_path=f'{test_dir}/labels.npy',
            batch_size=10,
            input_size=input_size
        )


    def test_model(self):
        imgs, labels = self.dataset[0]
        y_pred = self.model.predict(imgs)

        imgs = np.array(imgs, np.uint8)

        imgs_=[]
        for img,label,y_pred in zip(imgs,labels,y_pred):
            img = cv2.resize(img,(128,96))
            for i in range(3):
                for j in range(4):
                    y, x = 32 * i, 32 * j
                    img = cv2.rectangle(img, (x, y), (x + 32, y + 32), color=(0, 0, 255))

            img_pos = decode(label)
            for point in img_pos:
                x, y = point
                cv2.circle(img, (int(x), int(y)), radius=2, color=(0, 255, 0))

            img_pos_pred = decode(y_pred)
            for point in img_pos_pred:
                x, y = point
                cv2.circle(img, (int(x), int(y)), radius=2, color=(255, 0, 0))
            imgs_.append(img)
        return np.array(imgs_)
    def on_epoch_end(self, epoch, logs=None):
        loss = logs['loss']
        val_loss = logs['val_loss']
        recall = logs['recall']
        imgs= self.test_model()
        with self.writer.as_default():
            tf.summary.scalar("loss", loss,step=epoch)
            tf.summary.scalar("val_loss",val_loss,step=epoch)
            tf.summary.scalar("recall", recall, step=epoch)
            tf.summary.image("test images",imgs,max_outputs=10,step=epoch)
            self.writer.flush()


if __name__ == '__main__':
    tensorboard_callback = MyTensorboardCallback('logs')