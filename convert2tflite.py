import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from tensorflow import keras
from keras.models import load_model
from utils.Mydataloader import RandomTarget_dataset,Fast_dataset
import json
import time
import os
from utils.target_utils import *
# font = {'family': 'SimHei', "size": 24}
# matplotlib.rc('font', **font)
# input_size =128
# def get_confusion_matrix(label,y_pred,cls):
#     Confusion_Matrix = confusion_matrix(label, y_pred)
#     cls_sum = Confusion_Matrix.sum(axis=0)
#     Confusion_Matrix = Confusion_Matrix/cls_sum#归一化
#     print(np.around(Confusion_Matrix,3))
#     Confusion_Matrix=Confusion_Matrix*255
#     plt.matshow(Confusion_Matrix, cmap=plt.cm.Greens)
#     plt.xlabel('predict')
#     plt.ylabel('true')
#     xlocations = np.array(range(len(cls)))
#     print(cls)
#     print(xlocations)
#     plt.xticks(xlocations, cls, rotation=0)
#     plt.yticks(xlocations, cls)
#     plt.show()
def result_show(img,label,decoder,pred):
    pred_pos = decoder.decode(pred)
    img = cv2.resize(img,(128,96))
    for point in pred_pos:
        x, y = point
        cv2.circle(img, (int(x), int(y)), radius=1, color=(255, 0, 0))
    for i in range(3):
        for j in range(4):
            y, x = 32 * i, 32 * j
            img = cv2.rectangle(img, (x, y), (x + 32, y + 32), color=(0, 0, 255))
    img_pos = decoder.decode(label)
    for point in img_pos:
        x, y = point
        cv2.circle(img, (int(x), int(y)), radius=2, color=(0, 255, 0))

    plt.figure()
    plt.imshow(img)
    plt.show()
def tflite_pre(modelpath,dataset_root,batch_size=1,input_size:tuple = (96,128)):
    test_root = dataset_root


    interpreter = tf.lite.Interpreter(model_path=modelpath)

    input_index = interpreter.get_input_details()
    output_index = interpreter.get_output_details()
    valid_dataset =  Fast_dataset(
        imgs_path=f'{test_root}/images',
        labels_path=f'{test_root}/labels.npy',
        batch_size=batch_size,
        input_size=input_size
    )
    for (img,label) in valid_dataset:
        # img1 = np.array(img[0], dtype=np.uint8)
        start = time.time()
        interpreter.resize_tensor_input(input_index[0]['index'], (1, input_size[0], input_size[1], 3))
        interpreter.resize_tensor_input(output_index[0]['index'], (1, 36))
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_index[0]['index'], np.array(img, dtype=np.float32))
        interpreter.invoke()
        output = interpreter.get_tensor(output_index[0]['index'])
        end = time.time()
        print(float(end-start))
        img1 = np.array(img[0], dtype=np.uint8)
        img1 = cv2.resize(img1,(128,96))
        result_show(img1,label[0],target_decoder(),pred=output[0])

    # get_confusion_matrix(labels,predictions,valid_dataset.class_indices)
    # return acc/sum
def convert_to_tf_lite(model_path,valid_input_size:tuple,valid_path=None):
    model = load_model(model_path,compile=False)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.post_training_quantize=True
    tflite_model = converter.convert()

    start_time = time.strftime('%m_%d_%H_%M')
    name = model_path.split('/')[-1][:-3]

    save_root = "./tflite_model/%s_" % start_time +'_' + name
    os.makedirs(save_root)

    save_path = save_root+"/"+str(valid_input_size)+name+'.tflite'

    with open(save_path, 'wb') as f:
        f.write(tflite_model)
    # if valid_path!=None:
    #     tflite_pre(modelpath=save_path,
    #                  dataset_root=valid_path,
    #                  input_size=valid_input_size)
    #     remark = {'model_path':model_path,'input_size':valid_input_size,'valid_path':valid_path,'acc':acc}
    #     with open(save_root+"/remark.json",'w',encoding='utf-8') as f:
    #         json.dump(remark,f)
input_size = (48,64)
# convert_to_tf_lite(model_path='models_save/2023_02_07_00_34_58/model_70_0.0246.h5',
#                    valid_input_size=input_size,
#                    valid_path='./dataset/valid')
tflite_pre(modelpath='./tflite_model/02_07_11_59__model_70_0.0246/(48, 64)model_70_0.0246.tflite',
           dataset_root='./dataset/valid',
            input_size=input_size
           )