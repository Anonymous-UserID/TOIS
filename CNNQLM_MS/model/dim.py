#!/usr/bin/env python
#encoding=utf-8

import tensorflow as tf
import numpy as np
from multiply import ComplexMultiply
import math
from scipy import linalg
# point_wise obbject
from numpy.random import RandomState
rng = np.random.RandomState(23455)
from keras import initializers
# from complexnn.dense import ComplexDense
# from complexnn.utils import GetReal
from keras import backend as K
import math
from model.CNNQLM_II import QA_quantum

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      # 计算参数的均值，并使用tf.summary.scaler记录
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)

      # 计算参数的标准差
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      # 用直方图记录参数的分布
      tf.summary.histogram('histogram', var)

class CNNQLM_Dim(QA_quantum):
    def __init__(
      self, opt):

    	super().__init__(opt)
          
    def Position_Embedding(self,position_size):
        batch_size=self.batch_size
        seq_len = self.vocab_size
        position_j = 1. / tf.pow(10000., 2 * tf.range(position_size, dtype=tf.float32) / position_size)
        position_embedding = tf.expand_dims(position_j, 0)

        return position_embedding

    def add_embeddings(self):
        with tf.name_scope("embedding"):
            if self.is_Embedding_Needed:
                W = tf.Variable(np.array(self.embeddings),name="W" ,dtype="float32",trainable = self.trainable )
                W_complex_pos=tf.Variable(self.Position_Embedding(self.embedding_size),name = 'W',trainable = True)
                # W_complex_pos=tf.Variable(np.array(self.embeddings_complex),name = 'W' ,dtype="float32",trainable = True)
            else:
                W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),name="W",trainable = self.trainable)
#                 W_complex = tf.Variable(tf.random_uniform([self.vocab_size, 1], 0, 2*math.pi),name="W",trainable = self.trainable)
                W_complex_pos=tf.Variable(np.array(self.embeddings_complex),name = 'W' ,dtype="float32",trainable = True)
            self.embedding_W = W
            W_complex_pos = tf.tile(W_complex_pos,[self.vocab_size,1])
            self.embedding_W_pos=W_complex_pos
            self.overlap_W =  tf.get_variable('overlap_w', shape=[3, self.embedding_size],initializer = tf.random_normal_initializer())#tf.Variable(tf.random_uniform([3, self.extend_feature_dim], -1.0, 1.0),name="W",trainable = True)
            # self.position_W = tf.Variable(tf.random_uniform([31,self.embedding_size], 0,+2*math.pi),name = 'W',trainable = False)

        self.embedded_chars_q,self.embedding_chars_q_complex = self.concat_embedding(self.question,self.q_position,self.max_input_left,self.q_overlap)
        self.embedded_chars_a,self.embedding_chars_a_complex= self.concat_embedding(self.answer,self.a_position,self.max_input_right,self.a_overlap)

    def build_graph(self):
        self.create_placeholder()
        self.add_embeddings()
        self.density_weighted()
        self.joint_representation()
        # self.trace_represent()
        self.convolution()
        self.pooling_graph()
        self.feed_neural_work()
        self.create_loss()   

if __name__ == '__main__':
    cnn = QA_quantum(max_input_left = 33,
                max_input_right = 40,
                vocab_size = 5000,
                embedding_size = 50,
                batch_size = 3,
                embeddings = None,
                embeddings_complex=None,
                dropout_keep_prob = 1,
                filter_sizes = [40],
                num_filters = 65,
                l2_reg_lambda = 0.0,
                is_Embedding_Needed = False,
                trainable = True,
                overlap_needed = False,
                pooling = 'max',
                position_needed = False)
    cnn.build_graph()
    input_x_1 = np.reshape(np.arange(3*33),[3,33])
    input_x_2 = np.reshape(np.arange(3 * 40),[3,40])
    input_y = np.ones((3,2))

    input_overlap_q = np.ones((3,33))
    input_overlap_a = np.ones((3,40))
    q_posi = np.ones((3,33))
    a_posi = np.ones((3,40))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {
            cnn.question:input_x_1,
            cnn.answer:input_x_2,
            cnn.input_y:input_y,
            cnn.q_overlap:input_overlap_q,
            cnn.a_overlap:input_overlap_a,
            cnn.q_position:q_posi,
            cnn.a_position:a_posi
        }

        see,question,answer,scores = sess.run([cnn.embedded_chars_q,cnn.question,cnn.answer,cnn.scores],feed_dict)
        print (see)

# regularizer1 = tf.contrib.layers.l2_regularizer(self.l2_reg_lambda)
            
#             W1 = tf.get_variable( "W_hidden1",
#                 shape=[3*self.num_filters,self.hidden_num],
#                 initializer = tf.contrib.layers.xavier_initializer(),
#                 regularizer=regularizer1)
#             b1 = tf.get_variable('b_hidden1', shape=[self.hidden_num],initializer = tf.random_normal_initializer(),regularizer=regularizer1)
#             self.para.append(W1)
#             self.para.append(b1)
#             self.hidden_output_pos = tf.nn.tanh(tf.nn.xw_plus_b(self.represent_pos, W1, b1, name = "hidden_output"))

#             W2 = tf.get_variable(
#                 "W_outpu2t",
#                 shape = [self.hidden_num, 2],
#                 initializer = tf.contrib.layers.xavier_initializer(),
#                 regularizer=regularizer1)
#             b2 = tf.get_variable('b_output2', shape=[2],initializer = tf.random_normal_initializer(),regularizer=regularizer1)
#             self.para.append(W2)
#             self.para.append(b2)

#             self.logits_pos = tf.nn.xw_plus_b(self.hidden_output_pos, W2, b2, name = "scores")
#             self.scores_pos = tf.nn.softmax(self.logits_pos)
#             self.predictions_pos = tf.argmax(self.scores_pos, 1, name = "predictions")

#             regularizer = tf.contrib.layers.l2_regularizer(self.l2_reg_lambda)
            
#             W3 = tf.get_variable( "W_hidden3",
#                 #shape=[102,self.hidden_num],
#                 shape=[3*self.num_filters,self.hidden_num],
#                 initializer = tf.contrib.layers.xavier_initializer(),
#                 regularizer=regularizer)
#             b3 = tf.get_variable('b_hidden3', shape=[self.hidden_num],initializer = tf.random_normal_initializer(),regularizer=regularizer)
#             self.para.append(W3)
#             self.para.append(b3)
            
#             self.hidden_output_neg = tf.nn.tanh(tf.nn.xw_plus_b(self.represent_neg, W3, b3, name = "hidden_output"))
#             W4 = tf.get_variable(
#                 "W_output3",
#                 shape = [self.hidden_num, 2],
#                 initializer = tf.contrib.layers.xavier_initializer(),
#                 regularizer=regularizer)
#             b4 = tf.get_variable('b_output3', shape=[2],initializer = tf.random_normal_initializer(),regularizer=regularizer)
#             self.para.append(W4)
#             self.para.append(b4)

#             self.logits_neg = tf.nn.xw_plus_b(self.hidden_output_neg, W4, b4, name = "scores")
#             self.scores_neg = tf.nn.softmax(self.logits_neg)
#             self.predictions_neg = tf.argmax(self.scores_neg, 1, name = "predictions")
