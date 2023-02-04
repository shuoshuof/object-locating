# %%
from tensorflow import keras
from nets.backbone import mobilenetv2_backbone
from utils.Mydataloader import RandomTarget_dataset
from utils.util import *
from training.callbacks import MyTensorboardCallback
import time
# %%
def decodes(labels,r,w):
    coords =[]
    for label in labels:
        pos = decode(label)
        coords.append(pos[0])
    return coords

# %%
class test_dataset(RandomTarget_dataset):
    def __init__(self,root: str, batch_size: int, bg_r,bg_w, bg_root,Chinese_path=False,valid=False):
        super().__init__(root, batch_size, bg_r, bg_w,bg_root,img_num=1,Chinese_path=Chinese_path,valid=valid)
    def encode(self,img_pos):
        x,y = img_pos[0][0],img_pos[0][1]
        label=np.array([x/self.bg_w,y/self.bg_r])
        return label
    def __getitem__(self, item):
        batch = self.paths[item * self.batch_size:(item + 1) * self.batch_size]
        imgs = []
        labels = []
        for path in batch:
            bg_img_path = self.bg_paths[random.randint(0,len(self.bg_paths)-1)]
            img,label=self.creat_random_data(path,bg_img_path)
            imgs.append(img)
            labels.append(label)
        # labels = decodes(labels,)
        return np.array(imgs,dtype=np.float32),np.array(labels,dtype=np.float32)

# %%



def net(input_shape:tuple,alpha=1.0):
    inputs = keras.layers.Input(shape=input_shape)

    rescale_layer = keras.layers.Rescaling(scale=1./127.5, offset=-1)

    x = rescale_layer(inputs)
    x = mobilenetv2_backbone(x,alpha=alpha)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(2,activation='sigmoid')(x)

    model= keras.Model(inputs,outputs)
    for layer in model.layers:
        if type(layer) == type(keras.layers.BatchNormalization()):
            layer.momentum = 0.9
    return model
def show_dataset(dataset:test_dataset,input_r,input_w):

    for (img, label) in dataset:
        print(img.shape)
        print(label)
        img1 = np.array(img[0], dtype=np.uint8)
        # 绘制网格
        for i in range(3):
            for j in range(4):
                y, x = 32 * i, 32 * j
                img1 = cv2.rectangle(img1, (x, y), (x + 32, y + 32), color=(0, 0, 255))
        print(label[0])
        x, y = label[0]
        cv2.circle(img1, (int(x*input_w), int(y*input_r)), radius=2, color=(0, 255, 0))
        plt.figure()
        plt.imshow(img1[:, :, ::-1])
        plt.show()
if __name__ == '__main__':
    input_r, input_w = 96, 128
    batch_size = 128
    train_dataset = test_dataset(root=r'C:\Project\python\dataset\加框后的JPEG图',
                           batch_size=batch_size, bg_r=input_r, bg_w=input_w,
                           bg_root=r'C:\Project\python\dataset\background',
                           Chinese_path=True,
                           )
    valid_dataset = test_dataset(root=r'C:\Project\python\dataset\加框后的JPEG图',
                           batch_size=batch_size, bg_r=input_r, bg_w=input_w,
                           bg_root=r'C:\Project\python\dataset\background',
                           Chinese_path=True,
                           valid=True
                           )

    # show_dataset(dataset,input_r, input_w)
    save_path = './models_save/%s' % (time.strftime('%Y_%m_%d_%H_%M_%S'))
    # tensorboard_callback = keras.callbacks.TensorBoard(
    #                                                      histogram_freq=1,
    #                                                      write_graph=True,
    #                                                      write_images=True,
    #                                                      update_freq='epoch',
    #                                                      profile_batch=2,
    #                                                      embeddings_freq=1
    #                             )
    tensorboard_callback = MyTensorboardCallback(log_dir='logs')
    save_weights = keras.callbacks.ModelCheckpoint(save_path + "/model_{epoch:02d}_{val_loss:.4f}.h5",
                                                   save_best_only=True, monitor='val_mse')
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_mse', factor=0.2, patience=10, verbose=1)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_mse', patience=15, verbose=1)
    model = net((96, 128, 3))
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.MeanSquaredError(),
                  metrics=["mse"])
    hist = model.fit(train_dataset,
                     epochs=100,
                     workers=8,
                     validation_data=valid_dataset,
                     callbacks=[tensorboard_callback,save_weights,reduce_lr,early_stop]

                     )