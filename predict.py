from tensorflow import keras
from utils.Mydataloader import RandomTarget_dataset,Fast_dataset
from utils.util import *
from utils.target_utils import target_decoder
def predict(model,dataset):
    for (img, label) in dataset:
        img1 = np.array(img[0], dtype=np.uint8)
        # img1 =cv2.resize(img1,(128,96))
        # #绘制网格
        # for i in range(3):
        #     for j in range(4):
        #         y, x = 32 * i, 32 * j
        #         img1 = cv2.rectangle(img1, (x, y), (x + 32, y + 32), color=(255, 0, 0))
        # img_pos=decode(label[0])
        # print(label[0])
        # print(img_pos)
        # for point in img_pos:
        #     x, y = point
        #     cv2.circle(img1, (int(x), int(y)), radius=1, color=(0, 255, 0))
        #
        # pred = model.predict(img)
        # pred_pos = decode(pred[0])
        # for point in pred_pos:
        #     x, y = point
        #     cv2.circle(img1, (int(x), int(y)), radius=1, color=(0, 0, 255))
        #
        # plt.figure()
        # plt.imshow(img1)
        # plt.show()
        print(label)
        result_show(img1,label[0],target_decoder(),model)



if __name__ =='__main__':

    model = keras.models.load_model('models_save/2023_02_03_00_35_25/model_40_0.0260.h5',compile=False)
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