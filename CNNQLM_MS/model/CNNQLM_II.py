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
from keras import backend as K
import math
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

class QA_quantum(object):
    def __init__(self,opt):
        self.dropout_keep_prob = opt.dropout_keep_prob
        self.num_filters = opt.num_filters
        self.embeddings = opt.embeddings
        self.embeddings_complex = opt.embeddings_complex
        self.embedding_size = opt.embedding_size
        self.overlap_needed = opt.overlap_needed
        self.vocab_size = opt.vocab_size
        self.trainable = opt.trainable
        self.filter_sizes = opt.filter_sizes
        self.pooling = opt.pooling
        self.position_needed = opt.position_needed
        if self.overlap_needed:
            self.total_embedding_dim = opt.embedding_size + opt.extend_feature_dim
        else:
            self.total_embedding_dim = opt.embedding_size
        if self.position_needed:
            self.total_embedding_dim = self.total_embedding_dim + opt.extend_feature_dim
        self.batch_size = opt.batch_size
        self.l2_reg_lambda = opt.l2_reg_lambda
        self.para = []
        self.max_input_left = opt.max_input_left
        self.max_input_right = opt.max_input_right
        self.hidden_num = opt.hidden_num[0]
        self.extend_feature_dim = opt.extend_feature_dim
        self.is_Embedding_Needed = opt.is_Embedding_Needed
        self.rng = 23455
    def create_placeholder(self):
        self.question = tf.placeholder(tf.int32,[self.batch_size,self.max_input_left],name = 'input_question')
        self.answer = tf.placeholder(tf.int32,[self.batch_size,self.max_input_right],name = 'input_answer')
        self.input_y = tf.placeholder(tf.float32, [self.batch_size,2], name = "input_y")
        self.q_position = tf.placeholder(tf.int32,[self.batch_size,self.max_input_left],name = 'q_position')
        self.a_position = tf.placeholder(tf.int32,[self.batch_size,self.max_input_right],name = 'a_position')
        self.overlap = tf.placeholder(tf.float32,[self.batch_size,2],name = 'a_position')
        self.q_overlap = tf.placeholder(tf.int32,[None,self.max_input_left],name = 'q_position')
        self.a_overlap = tf.placeholder(tf.int32,[None,self.max_input_right],name = 'a_position')

    def density_weighted(self):
        self.weighted_q = tf.Variable(tf.ones([1,self.max_input_left,1,1]) , name = 'weighted_q')
        self.weighted_q=tf.nn.softmax(self.weighted_q,1)
        self.para.append(self.weighted_q)
        self.weighted_a = tf.Variable(tf.ones([1,self.max_input_right,1,1]) , name = 'weighted_a')
        self.weighted_a=tf.nn.softmax(self.weighted_a,1)
        self.para.append(self.weighted_a)
    def Position_Embedding(self,position_size):
        batch_size=self.batch_size
        seq_len = self.vocab_size
        position_j = 1. / tf.pow(10000., 2 * tf.range(position_size, dtype=tf.float32) / position_size)
        position_j = tf.expand_dims(position_j, 0)

        position_i=tf.range(tf.cast(seq_len,tf.float32), dtype=tf.float32) + 1
        position_i=tf.expand_dims(position_i,1)

       
        position_ij = tf.matmul(position_i, position_j)
        position_embedding = position_ij

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
            self.embedding_W_pos=W_complex_pos
            self.overlap_W =  tf.get_variable('overlap_w', shape=[3, self.embedding_size],initializer = tf.random_normal_initializer())#tf.Variable(tf.random_uniform([3, self.extend_feature_dim], -1.0, 1.0),name="W",trainable = True)
            # self.position_W = tf.Variable(tf.random_uniform([31,self.embedding_size], 0,+2*math.pi),name = 'W',trainable = False)

        self.embedded_chars_q,self.embedding_chars_q_complex = self.concat_embedding(self.question,self.q_position,self.max_input_left,self.q_overlap)

        self.embedded_chars_a,self.embedding_chars_a_complex= self.concat_embedding(self.answer,self.a_position,self.max_input_right,self.a_overlap)
    def joint_representation(self):
        self.density_q_real,self.density_q_imag = self.density_matrix(self.embedded_chars_q,self.embedding_chars_q_complex,self.weighted_q)
        self.density_a_real,self.density_a_imag = self.density_matrix(self.embedded_chars_a,self.embedding_chars_a_complex,self.weighted_a)
        self.M_qa_real=tf.matmul(self.density_q_real,self.density_a_real)+tf.matmul(self.density_q_imag,self.density_a_imag)
        self.M_qa_imag=tf.matmul(self.density_q_imag,self.density_a_real)-tf.matmul(self.density_q_real,self.density_a_imag)
    def direct_representation(self):

        self.embedded_q = tf.reshape(self.embedded_chars_q,[-1,self.max_input_left,self.total_embedding_dim])
        self.embedded_a = tf.reshape(self.embedded_chars_a,[-1,self.max_input_right,self.total_embedding_dim])
        reverse_a = tf.transpose(self.embedded_a,[0,2,1])
        self.M_qa = tf.matmul(self.embedded_q,reverse_a)

    def density_matrix(self,sentence_matrix,sentence_matrix_complex,sentence_weighted):
        self.input_real=tf.expand_dims(sentence_matrix,-1)
        self.input_imag=tf.expand_dims(sentence_matrix_complex,-1)
        input_real_transpose = tf.transpose(self.input_real, perm = [0,1,3,2])
        input_imag_transpose = tf.transpose(self.input_imag, perm = [0,1,3,2])
        q_a_real_real = tf.matmul(self.input_real,input_real_transpose)
        q_a_real_imag = tf.matmul(self.input_imag,input_imag_transpose)
        q_a_real = q_a_real_real-q_a_real_imag
        q_a_imag_real=tf.matmul(self.input_imag,input_real_transpose)
        q_a_imag_imag=tf.matmul(self.input_real,input_imag_transpose)
        q_a_imag = q_a_imag_real+q_a_imag_imag
        return tf.reduce_sum(tf.multiply(q_a_real,sentence_weighted),1),tf.reduce_sum(tf.multiply(q_a_imag,sentence_weighted),1)

    def set_weight(self,num_unit,dim):
        input_dim = (self.total_embedding_dim - self.filter_sizes[0] + 1) * self.num_filters * dim
        unit=num_unit
        kernel_shape = [input_dim,unit]
        fan_in_f=np.prod(kernel_shape)
        s = np.sqrt(1. / fan_in_f)
        rng=RandomState(23455)
        modulus_f=rng.rayleigh(scale=s,size=kernel_shape)
        phase_f=rng.uniform(low=-np.pi,high=np.pi,size=kernel_shape)
        real_init=modulus_f*np.cos(phase_f)
        imag_init=modulus_f*np.sin(phase_f)
        real_kernel=tf.Variable(real_init,name='real_kernel')
        real_kernel=tf.to_float(real_kernel)
        imag_kernel=tf.Variable(imag_init,name='imag_kernel')
        imag_kernel=tf.to_float(imag_kernel)
        return real_kernel,imag_kernel
    def feed_neural_work(self):
        with tf.name_scope('regression'):
            #W = tf.Variable(tf.zeros(shape = [(self.total_embedding_dim - self.filter_sizes[0] + 1) * self.num_filters * 2,2]),name = 'W') 
            #修改
            # self.represent=tf.concat([self.represent_imag,self.represent_real],1)
            # self.real_kernel,self.imag_kernel=self.set_weight(664,2)
            # cat_kernels_4_real = tf.concat([self.real_kernel, -self.imag_kernel],axis=-1)
            # cat_kernels_4_imag = tf.concat([self.imag_kernel, self.real_kernel],axis=-1)
            # cat_kernels_4_complex = tf.concat([cat_kernels_4_real, cat_kernels_4_imag],axis=0)
            # self.full_join_real_1=tf.matmul(self.represent,cat_kernels_4_complex)
            #修改
            #之前
            # print(self.full_join_real_1)
            # exit()
            # self.full_join_real_1=tf.matmul(self.represent_real,self.real_kernel_1)-tf.matmul(self.represent_imag,self.imag_kernel_1)
            # self.full_join_imag_1=tf.matmul(self.represent_real,self.imag_kernel_1)+tf.matmul(self.represent_imag,self.real_kernel_1)
            # self.real_kernel_2,self.imag_kernel_2=self.set_weight(1,1)
            # # cat_kernels_4_real = tf.concat([self.real_kernel, -self.imag_kernel],axis=-1)
            # # cat_kernels_4_imag = tf.concat([self.imag_kernel, self.real_kernel],axis=-1)
            # # cat_kernels_4_complex = tf.concat([cat_kernels_4_real, cat_kernels_4_imag],axis=0)
            # self.full_join_real_2=tf.matmul(self.full_join_real_1,self.real_kernel_2)-tf.matmul(self.full_join_imag_1,self.imag_kernel_2)
            # self.full_join_imag_2=tf.matmul(self.full_join_real_1,self.imag_kernel_2)+tf.matmul(self.full_join_imag_1,self.real_kernel_2)
            # b = tf.get_variable('b_hidden', shape=[2],initializer = tf.random_normal_initializer())
            # self.logits=tf.concat([self.full_join_real_2,self.full_join_imag_2],1)+b
            # self.logits = tf.nn.xw_plus_b(self.full_join_real_1, W, b, name = "scores")
            # self.concat_out = tf.matmul(self.represent, cat_kernels_4_complex)
            #之前
            regularizer = tf.contrib.layers.l2_regularizer(self.l2_reg_lambda)
            W = tf.get_variable( "W_hidden",
                #shape=[102,self.hidden_num],
                shape=[self.represent.shape[-1],self.hidden_num],
                initializer = tf.contrib.layers.xavier_initializer(),
                regularizer=regularizer)
            b = tf.get_variable('b_hidden', shape=[self.hidden_num],initializer = tf.random_normal_initializer(),regularizer=regularizer)
            self.para.append(W)
            self.para.append(b)
            self.hidden_output = tf.nn.tanh(tf.nn.xw_plus_b(self.represent, W, b, name = "hidden_output"))
            #self.hidden_output=tf.nn.dropout(self.hidden_output, self.dropout_keep_prob, name="hidden_output_drop")
            W = tf.get_variable(
                "W_output",
                shape = [self.hidden_num, 2],
                initializer = tf.contrib.layers.xavier_initializer(),
                regularizer=regularizer)
            b = tf.get_variable('b_output', shape=[2],initializer = tf.random_normal_initializer(),regularizer=regularizer)
            self.para.append(W)
            self.para.append(b)
            self.logits = tf.nn.xw_plus_b(self.hidden_output, W, b, name = "scores")
            self.scores = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(self.scores, 1, name = "predictions")
    
    def focal_loss(self,logits, labels, gamma=2):
        '''
        :param logits:  [batch_size, n_class]
        :param labels: [batch_size]
        :return: -(1-y)^r * log(y)
        '''
        softmax = tf.reshape(tf.nn.softmax(logits), [-1])  # [batch_size * n_class]
        labels = tf.range(0, logits.shape[0]) * logits.shape[1] + labels
        prob = tf.gather(softmax, labels)
        weight = tf.pow(tf.subtract(1., prob), gamma)
        loss = -tf.reduce_mean(tf.multiply(weight, tf.log(prob)))
        return loss

    def create_loss(self):
        l2_loss = tf.constant(0.0)
        for p in self.para:
            l2_loss += tf.nn.l2_loss(p)
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.input_y)
            pi_regularization = tf.reduce_sum(self.weighted_q) - 1 + tf.reduce_sum(self.weighted_a) - 1
            self.loss = tf.reduce_mean(losses)+self.l2_reg_lambda*l2_loss+0.0001*tf.nn.l2_loss(pi_regularization)
            #self.loss = tf.reduce_mean(losses)
            # print(self.input_y)
            # self.loss = self.focal_loss(self.logits,self.input_y)
            

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            # self.predictions = tf.cast(self.predictions,dtype=tf.int32)
            # correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
    def concat_embedding(self,words_indice,position_indice,sentence_length,overlap_indice):
        embedded_chars_q = tf.nn.embedding_lookup(self.embedding_W,words_indice)
        embedded_chars_q=tf.nn.dropout(embedded_chars_q, self.dropout_keep_prob, name="hidden_output_drop")
        # embeddings_overlap=tf.nn.embedding_lookup(self.overlap_W,overlap_indice)
        # embedded_chars_q=tf.reduce_sum([embedded_chars_q,embeddings_overlap],0)
        embedding_chars_q_phase=tf.nn.embedding_lookup(self.embedding_W_pos,words_indice)
        # self.embedding_chars_q_phase=tf.nn.embedding_lookup(self.embedding_W_pos,position_indice)
        
        pos=tf.expand_dims(position_indice,2)
        pos=tf.cast(pos,tf.float32)
        embedding_chars_q_phase=tf.multiply(pos,embedding_chars_q_phase)

        embedding_chars_q_phase=tf.nn.dropout(embedding_chars_q_phase, self.dropout_keep_prob, name="hidden_output_drop")
        [embedded_chars_q, embedding_chars_q_phase] = ComplexMultiply()([embedding_chars_q_phase,embedded_chars_q])
        #embedded_chars_q=tf.complex(embedded_chars_q, embedding_chars_q_phase)
        return embedded_chars_q,embedding_chars_q_phase
    def convolution(self):
        #initialize my conv kernel
        self.kernels_real = []
        self.kernels_imag=[]
        for i,filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('conv-pool-%s' % filter_size):
                filter_shape = [filter_size,filter_size,1,self.num_filters]
                input_dim=2
                fan_in = np.prod(filter_shape[:-1])
                fan_out = (filter_shape[-1] * np.prod(filter_shape[:2]))
                s=1./fan_in
                rng=RandomState(23455)
                modulus=rng.rayleigh(scale=s,size=filter_shape)
                phase=rng.uniform(low=-np.pi,high=np.pi,size=filter_shape)
                W_real=modulus*np.cos(phase)
                W_imag=modulus*np.sin(phase)
                W_real = tf.Variable(W_real,dtype = 'float32')
                W_imag = tf.Variable(W_imag,dtype = 'float32')
                self.kernels_real.append(W_real)
                self.kernels_imag.append(W_imag)
                # self.para.append(W_real)
                # self.para.append(W_imag)
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        # self.qa_real = self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_real)-self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_imag)
        # print(self.qa_real)
        # self.qa_imag = self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_real)+self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_imag)
        # print(self.qa_imag)
        # self.qa_real_0 = tf.nn.relu(self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_real)[0]+tf.expand_dims(self.M_qa_real,-1))-tf.nn.relu(self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_imag)[0]+tf.expand_dims(self.M_qa_real,-1))
        # self.qa_real_1 = tf.nn.relu(self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_real)[1]+tf.expand_dims(self.M_qa_real,-1))-tf.nn.relu(self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_imag)[1]+tf.expand_dims(self.M_qa_real,-1))
        # self.qa_real_2 = tf.nn.relu(self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_real)[2]+tf.expand_dims(self.M_qa_real,-1))-tf.nn.relu(self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_imag)[2]+tf.expand_dims(self.M_qa_real,-1))
        # self.qa_real_3 = tf.nn.relu(self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_real)[3]+tf.expand_dims(self.M_qa_real,-1))-tf.nn.relu(self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_imag)[3]+tf.expand_dims(self.M_qa_real,-1))
        # self.qa_imag_0 = tf.nn.relu(self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_real)[0]+tf.expand_dims(self.M_qa_imag,-1))+tf.nn.relu(self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_imag)[0]+tf.expand_dims(self.M_qa_imag,-1))   
        # self.qa_imag_1 = tf.nn.relu(self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_real)[1]+tf.expand_dims(self.M_qa_imag,-1))+tf.nn.relu(self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_imag)[1]+tf.expand_dims(self.M_qa_imag,-1))  
        # self.qa_imag_2 = tf.nn.relu(self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_real)[2]+tf.expand_dims(self.M_qa_imag,-1))+tf.nn.relu(self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_imag)[2]+tf.expand_dims(self.M_qa_imag,-1)) 
        # self.qa_imag_3 = tf.nn.relu(self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_real)[3]+tf.expand_dims(self.M_qa_imag,-1))+tf.nn.relu(self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_imag)[3]+tf.expand_dims(self.M_qa_imag,-1))   
        self.qa_real_0 = (self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_real)[0])-(self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_imag)[0])
        self.qa_real_1 = (self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_real)[1])-(self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_imag)[1])
        self.qa_real_2 = (self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_real)[2])-(self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_imag)[2])
        self.qa_real_3 = (self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_real)[3])-(self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_imag)[3])
        self.qa_imag_0 = (self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_real)[0])+(self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_imag)[0])
        self.qa_imag_1 = (self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_real)[1])+(self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_imag)[1])
        self.qa_imag_2 = (self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_real)[2])+(self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_imag)[2])
        self.qa_imag_3 = (self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_real)[3])+(self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_imag)[3] )
    def max_pooling(self,conv):
        pooled = tf.nn.max_pool(
                    conv,
                    # ksize = [1, 8, 8, 1],
                    ksize = [1, 11, 11, 1],
                    # ksize = [1, 6, 6, 1],
                    strides = [1, 1, 1, 1],
                    padding = 'VALID',
                    name = "pool")
        return pooled
    def avg_pooling(self,conv):
        pooled = tf.nn.avg_pool(
                    conv,
                    ksize = [1, self.max_input_left, self.max_input_right, 1],
                    strides = [1, 1, 1, 1],
                    padding = 'VALID',
                    name = "pool")
        return pooled
    def pooling_graph(self):
        with tf.name_scope('pooling'):

      
            # raw_pooling_real = tf.contrib.layers.flatten(tf.reduce_mean(self.qa_real,1))
            # col_pooling_real = tf.contrib.layers.flatten(tf.reduce_mean(self.qa_real,2))
            # self.represent_real = tf.concat([raw_pooling_real,col_pooling_real],1)
            

            self.represent_real_0 = self.max_pooling(self.qa_real_0)
            self.represent_real_1 = self.max_pooling(self.qa_real_1)
            self.represent_real_2 = self.max_pooling(self.qa_real_2)
            self.represent_real_3 = self.max_pooling(self.qa_real_3)
            # print(self.represent_real)
            self.represent_real_0 =tf.reshape(self.represent_real_0 ,[self.batch_size,-1])
            self.represent_real_1 =tf.reshape(self.represent_real_1 ,[self.batch_size,-1])
            self.represent_real_2 =tf.reshape(self.represent_real_2 ,[self.batch_size,-1])
            self.represent_real_3 =tf.reshape(self.represent_real_3 ,[self.batch_size,-1])
            # print(self.represent_real)
            self.represent_img_0 = self.max_pooling(self.qa_imag_0)
            self.represent_img_1 = self.max_pooling(self.qa_imag_1)
            self.represent_img_2 = self.max_pooling(self.qa_imag_2)
            self.represent_img_3 = self.max_pooling(self.qa_imag_3)
            # self.represent_img=tf.reshape(self.represent_img,[self.batch_size,-1])
            # print(self.represent_img)
            self.represent_img_0 =tf.reshape(self.represent_img_0 ,[self.batch_size,-1])
            self.represent_img_1 =tf.reshape(self.represent_img_1 ,[self.batch_size,-1])
            self.represent_img_2 =tf.reshape(self.represent_img_2 ,[self.batch_size,-1])
            self.represent_img_3 =tf.reshape(self.represent_img_3 ,[self.batch_size,-1])
            w_0 = tf.Variable(tf.zeros(shape = [self.batch_size,tf.shape(self.represent_real_0)[-1]],name = 'W'))
            w_1 = tf.Variable(tf.zeros(shape = [self.batch_size,tf.shape(self.represent_real_1)[-1]],name = 'W'))
            w_2 = tf.Variable(tf.zeros(shape = [self.batch_size,tf.shape(self.represent_real_2)[-1]],name = 'W'))
            w_3 = tf.Variable(tf.zeros(shape = [self.batch_size,tf.shape(self.represent_real_3)[-1]],name = 'W'))
            # w = tf.Variable(tf.zeros(shape = [1,141120],name = 'W'))
            self.represent_real_W_0=tf.multiply(tf.multiply(self.represent_real_0,w_0),self.represent_img_0)
            self.represent_real_W_1=tf.multiply(tf.multiply(self.represent_real_1,w_1),self.represent_img_1)
            self.represent_real_W_2=tf.multiply(tf.multiply(self.represent_real_2,w_2),self.represent_img_2)
            self.represent_real_W_3=tf.multiply(tf.multiply(self.represent_real_3,w_3),self.represent_img_3)
            # print(self.represent_real_W)
            # raw_pooling_imag = tf.contrib.layers.flatten(tf.reduce_mean(self.qa_imag,1))
            # col_pooling_imag = tf.contrib.layers.flatten(tf.reduce_mean(self.qa_imag,2))
            # self.represent_imag = tf.concat([raw_pooling_imag,col_pooling_imag],1)
            self.represent = tf.concat([self.represent_real_0,self.represent_real_1,self.represent_real_2,self.represent_real_3,
                                        self.represent_img_0,self.represent_img_1,self.represent_img_2,self.represent_img_3,
                                        self.represent_real_W_0,self.represent_real_W_1,self.represent_real_W_2,self.represent_real_W_3,
                                        self.overlap],1)
            # self.represent = tf.concat([self.represent_real,self.represent_img,self.represent_real_W],1)
            #self.represent=tf.nn.dropout(self.represent, 0.4, name="hidden_output_drop")

            print(self.represent)

    def wide_convolution(self,embedding,kernel):
        cnn_outputs = []
        for i,filter_size in enumerate(self.filter_sizes):
            conv = tf.nn.conv2d(
                    embedding,
                    kernel[i],
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    name="conv-1"
            )
            cnn_outputs.append(conv)
        cnn_reshaped = tf.concat(cnn_outputs,3)
        return cnn_reshaped
    def narrow_convolution(self,embedding,kernel):
        cnn_outputs = []
        for i,filter_size in enumerate(self.filter_sizes):
            conv = tf.nn.conv2d(
                    embedding,
                    kernel[i],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="conv-1"
            )
            cnn_outputs.append(conv)
        # cnn_reshaped = tf.concat(cnn_outputs,3)
        # return cnn_outputs
        return cnn_outputs[0], cnn_outputs[1] , cnn_outputs[2] , cnn_outputs[3]
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
