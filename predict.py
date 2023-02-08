from tensorflow import keras
from utils.Mydataloader import RandomTarget_dataset,Fast_dataset
from utils.util import *
from utils.target_utils import target_decoder
def predict(model,dataset):
    for (img, label) in dataset:
        img1 = np.array(img[0], dtype=np.uint8)
        print(label)
        result_show(img1,label[0],target_decoder(),model)



if __name__ =='__main__':

    model = keras.models.load_model('models_save/2023_02_08_00_07_17/model_66_0.0175.h5',compile=False)
    model.summary()
    batch_size =1
    img_num=3
    input_size = (96,128)
    valid_dataset = RandomTarget_dataset(root = r'C:\Project\python\dataset\加框后的JPEG图',
                          batch_size=batch_size,
                          bg_r=96,
                          bg_w=128,
                          bg_root=r'C:\Project\python\dataset\background',
                          img_num=img_num,
                          Chinese_path=True,
                          valid=True
                         )
    # valid_dataset =  Fast_dataset(
    #     imgs_path='./dataset/valid/images',
    #     labels_path='./dataset/valid/labels.npy',
    #     batch_size=batch_size,
    #     input_size=input_size
    # )
    predict(model,valid_dataset)