from tensorflow import keras
import tensorflow as tf
"""
因为分类头使用的是sigmoid，
所以就不存在什么map了
直接看召回率，准确率
"""
class Recall(keras.metrics.Metric):
    def __init__(self,name='recall',**kwargs):
        super(Recall,self).__init__(name=name,**kwargs)
        self.recall = self.add_weight(name='recall',**kwargs)
        self.batch_num  = self.add_weight(name='batch_num',**kwargs)
        self.recall.assign(0.0)
        self.batch_num.assign(0.0)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Recall = TP/(TP+FN)
        """
        y_true = tf.convert_to_tensor(y_true,dtype=tf.float32)

        y_true_conf = y_true[:,24:]
        y_pred_conf = y_pred[:,24:]

        y_true_conf = tf.where(y_true_conf>=0.5,1,0)
        y_pred_conf = tf.where(y_pred_conf>=0.5,1,0)

        TP = tf.multiply(y_true_conf,y_pred_conf)
        TP = tf.reduce_sum(TP)
        TP_FN = tf.reduce_sum(y_true_conf)

        recall = TP/TP_FN

        recall = tf.cast(recall,dtype=tf.float32)
        self.batch_num.assign_add(1.0)
        self.recall.assign_add(recall)
    def result(self):
        return self.recall/self.batch_num
    def reset_state(self):

        self.recall.assign(0.0)
        self.batch_num.assign(0.0)

if __name__ == '__main__':
    me = Recall()
    y_true = tf.random.normal((4,36),dtype=tf.float32)
    y_pred = tf.random.normal((4,36),dtype=tf.float32)
    me.update_state(y_true,y_pred)
