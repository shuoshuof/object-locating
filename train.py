from utils.Mydataloader import RandomTarget_dataset,Fast_dataset
from training.callbacks import *
from training.loss_function import SSD_loss
from training.metrics import Recall
import time
from nets.ObjectLocatingModel import ShuoShuoNet

if __name__=='__main__':
    batch_size =512
    img_num=3
    BN_momentum=0.99
    input_size = (48,64)
    input_shape = (input_size[0], input_size[1], 3)
    # train_dataset = RandomTarget_dataset(root = r'C:\Project\python\dataset\加框后的JPEG图',
    #                       batch_size=batch_size,bg_r=96,bg_w=128,
    #                       bg_root=r'C:\Project\python\dataset\background',
    #                       img_num=img_num,
    #                       Chinese_path=True)
    # valid_dataset = RandomTarget_dataset(root = r'C:\Project\python\dataset\加框后的JPEG图',
    #                       batch_size=batch_size,bg_r=96,bg_w=128,
    #                       bg_root=r'C:\Project\python\dataset\background',
    #                       img_num=img_num,
    #                       Chinese_path=True,
    #                       valid=True
    #                      )
    train_dataset = Fast_dataset(
        imgs_path='./dataset/train/images',
        labels_path='./dataset/train/labels.npy',
        batch_size=batch_size,
        input_size=input_size
    )
    valid_dataset =  Fast_dataset(
        imgs_path='./dataset/valid/images',
        labels_path='./dataset/valid/labels.npy',
        batch_size=batch_size,
        input_size=input_size
    )


    model = ShuoShuoNet(input_shape=input_shape,
                        alpha=0.35,
                        FeatureMap_shape=(3,4)
                        ).model()
    for layer in model.layers:
        if type(layer) == type(keras.layers.BatchNormalization()):
            layer.momentum = BN_momentum
    model.summary()
    loss_fn = SSD_loss()
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=loss_fn,
                  metrics=[Recall()])

    save_path = './models_save/%s' % (time.strftime('%Y_%m_%d_%H_%M_%S'))

    save_weights = keras.callbacks.ModelCheckpoint(save_path + "/model_{epoch:02d}_{val_loss:.4f}.h5",
                                                   save_best_only=True, monitor='val_loss')
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=1)
    tensorboard_callback = MyTensorboardCallback('logs',input_size=input_size)

    callback_list=[save_weights,reduce_lr,early_stop,tensorboard_callback]
    hist = model.fit(train_dataset,
                     epochs=200,
                     workers=8,
                     validation_data=valid_dataset,
                     callbacks=callback_list
                     )



