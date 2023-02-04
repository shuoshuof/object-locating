import tensorflow as tf
from tensorflow import keras
from nets.backbone import mobilenetv2_backbone

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

def mini_ssd(input_shape=(96,128,3),alpha=1.0):

    inputs = keras.layers.Input(shape=input_shape)
    rescale_layer = keras.layers.Rescaling(scale=1. / 127.5, offset=-1)

    x = rescale_layer(inputs)
    x = mobilenetv2_backbone(x,alpha=alpha)

    #对坐标偏移预测
    pos_offest = keras.layers.Conv2D(2,kernel_size=(3,3), padding='same')(x)#(3,4,96)->(3,4,2)
    pos_offest = keras.layers.Flatten()(pos_offest)#(24)


    #判断是否有目标板
    target_conf = keras.layers.Conv2D(1,kernel_size=(3,3),padding='same')(x)#(3,4,96)->(3,4,1)
    target_conf = keras.layers.Flatten()(target_conf)#(12)
    target_conf = tf.keras.activations.sigmoid(target_conf)

    outputs = keras.layers.Concatenate()([pos_offest,target_conf])#(36)

    model = keras.Model(inputs,outputs)
    return model

if __name__=='__main__':
    model = mini_ssd()
    model.summary()
    model.save("./mini_ssd.h5")