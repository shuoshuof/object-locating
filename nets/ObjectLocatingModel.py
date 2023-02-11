import tensorflow as tf
from tensorflow import keras
from nets.backbone import mobilenet_backbone

'''
art QQVGA为160x120，考虑96x128输入
仅需要一个feature map 3x4
预测点偏移 x_offset,y_offset 3x4x2
判断是否有目标板 4x3x1 二分类激活函数用sigmod
如果需要改变输入尺寸
    输入需要满足的要求：
    r/3=2^n
    w/4=2^m
'''


class ShuoShuoNet:
    def __init__(self,
                 input_shape: tuple = (96, 128, 3),
                 alpha=0.35,
                 FeatureMap_shape: tuple = (3, 4),
                 backbone=None,
                 ):

        self.Input_shape = input_shape

        self.Input_layer = keras.layers.Input(shape=self.Input_shape)
        self.rescale_layer = keras.layers.Rescaling(scale=1. / 127.5, offset=-1)

        self.backbone = mobilenet_backbone(
            input_shape = self.Input_shape,
            alpha = alpha,
            FeatureMap_shape = FeatureMap_shape,
        )  if backbone is None else backbone


        self.coord_offset =  keras.layers.Conv2D(2,kernel_size=(3,3), padding='same')
        self.offset_flatten = keras.layers.Flatten()

        self.target_conf = keras.layers.Conv2D(1,kernel_size=(3,3),padding='same')
        self.conf_flatten = keras.layers.Flatten()
        self.conf_activations = tf.keras.activations.sigmoid
    def model(self):
        inputs = self.Input_layer
        x=  self.rescale_layer(inputs)

        x = self.backbone(x)

        #回归头
        coord_outputs = self.coord_offset(x)
        coord_outputs = self.offset_flatten(coord_outputs)

        #分类头
        conf_outputs = self.target_conf(x)
        conf_outputs = self.conf_flatten(conf_outputs)
        conf_outputs = self.conf_activations(conf_outputs)

        outputs = keras.layers.Concatenate()([coord_outputs,conf_outputs])
        model = keras.Model(inputs, outputs)
        return model
if __name__=='__main__':
    model  = ShuoShuoNet().model()

    model.summary()