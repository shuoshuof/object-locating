from tensorflow import keras

training = None

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def _conv_block(inputs,filters,kernel_size,strides):
    x = keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,padding='same',strides=strides)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(6.)(x)
    return x
def _bottleneck(inputs, c_output, kernel_size, t, alpha, strides, r=False):
    """
    c_output:输出通道数
    t:1*1拓展倍数
    s:步长
    r:是否进行残差连接
    """
    _,_,_,c = inputs.shape
    expand_channel = c*t
    output_channel = int(c_output*alpha)

    x = _conv_block(inputs,expand_channel,(1,1),(1,1))

    x = keras.layers.DepthwiseConv2D(kernel_size=kernel_size,strides=strides, depth_multiplier=1, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(6.)(x)

    x = keras.layers.Conv2D(output_channel,(1,1),strides=(1,1),padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    if r:
        x = keras.layers.Add()([x,inputs])
    return x
def _inverted_residual_block(inputs, filters, kernel_size, t, alpha, strides, n):
    """
    filters： 输出通道数
    t：1*1拓展倍数
    n：重复次数
    """

    x = _bottleneck(inputs,filters,kernel_size,t,alpha,strides,r=False)

    for i in range(1,n):
        x = _bottleneck(x,filters,kernel_size,t,alpha,strides=(1,1),r = True)
    return x
def mobilenetv2_backbone(inputs,alpha=1.0):

    """
    输入需要满足的要求：
    r//3=2^n
    w//4=2^n
    """

    first_filters = _make_divisible(32 * alpha, 8)
    x = _conv_block(inputs, first_filters, (3, 3), strides=(2, 2)) #(96,128,3)->(48,64,32)

    x = _inverted_residual_block(x, 16, (3, 3), t=1, alpha=alpha, strides=(1,1), n=1)#(48,64,32)->(48,64,16)
    x = _inverted_residual_block(x, 24, (3, 3), t=6, alpha=alpha, strides=(2,2), n=1)#(48,64,16)->(24,32,24)
    x = _inverted_residual_block(x, 32, (3, 3), t=6, alpha=alpha, strides=(2,2), n=1)#(24,32,24)->(12,16,32)
    x = _inverted_residual_block(x, 64, (3, 3), t=6, alpha=alpha, strides=(2,2), n=1)#(12,16,32)->(6,8,64)

    # x = _inverted_residual_block(x, 96, (3, 3), t=6, alpha=alpha, strides=(1,1), n=1)#(6,8,64)->(6,8,96)

    # x = _inverted_residual_block(x, 96, (3, 3), t=6, alpha=alpha, strides=(2,2), n=1)#(6,8,96)->(3,4,96)

    # x = _inverted_residual_block(x, 160, (3, 3), t=6, alpha=alpha, strides=2, n=3)
    # x = _inverted_residual_block(x, 320, (3, 3), t=6, alpha=alpha, strides=1, n=1)

    return x