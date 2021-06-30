#!/usr/bin/env python
#encoding=utf-8

import tensorflow as tf
import numpy as np
# point_wise obbject
rng = np.random.RandomState(23455)
class NNQLM_II(object):
    def __init__(self, opt):
        self.dropout_keep_prob = opt.dropout_keep_prob
        self.num_filters = opt.num_filters
        self.embeddings = opt.embeddings
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
        self.hidden_num = opt.hidden_num
        self.extend_feature_dim = opt.extend_feature_dim
        self.is_Embedding_Needed = opt.is_Embedding_Needed
        self.rng = 23455
    def create_placeholder(self):
        self.question = tf.placeholder(tf.int32,[self.batch_size,self.max_input_left],name = 'input_question')
        self.answer = tf.placeholder(tf.int32,[self.batch_size,self.max_input_right],name = 'input_answer')
        self.input_y = tf.placeholder(tf.float32, [self.batch_size,2], name = "input_y")
        self.q_overlap = tf.placeholder(tf.int32,[None,self.max_input_left],name = 'q_feature_embeding')
        self.a_overlap = tf.placeholder(tf.int32,[None,self.max_input_right],name = 'a_feature_embeding')
        self.q_position = tf.placeholder(tf.int32,[self.batch_size,self.max_input_left],name = 'q_position')
        self.a_position = tf.placeholder(tf.int32,[self.batch_size,self.max_input_right],name = 'a_position')
    def density_weighted(self):
        self.weighted_q = tf.Variable(tf.ones([1,self.max_input_left,1,1]) , name = 'weighted_q')
        self.para.append(self.weighted_q)
        self.weighted_a = tf.Variable(tf.ones([1,self.max_input_right,1,1]) , name = 'weighted_a')
        self.para.append(self.weighted_a)
    def add_embeddings(self):

        # Embedding layer for both CNN
        with tf.name_scope("embedding"):
            if self.is_Embedding_Needed:
                W = tf.Variable(np.array(self.embeddings),name="W" ,dtype="float32",trainable = self.trainable )
            else:
                W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),name="W",trainable = self.trainable)
            self.embedding_W = W
            self.overlap_W =  tf.get_variable('overlap_w', shape=[3, self.embedding_size],initializer = tf.random_normal_initializer())#tf.Variable(tf.random_uniform([3, self.extend_feature_dim], -1.0, 1.0),name="W",trainable = True)
            self.position_W = tf.Variable(tf.random_uniform([300,self.embedding_size], -1.0, 1.0),name = 'W',trainable = True)
            # self.para.append(self.embedding_W)
            # self.para.append(self.overlap_W)

        #get embedding from the word indices
        self.embedded_chars_q = self.concat_embedding(self.question,self.q_overlap,self.q_position)

        self.embedded_chars_a = self.concat_embedding(self.answer,self.a_overlap,self.a_position)
    def joint_representation(self):
        self.density_q = self.density_matrix(self.embedded_chars_q,self.weighted_q)
        self.density_a = self.density_matrix(self.embedded_chars_a,self.weighted_a)
        self.M_qa = tf.matmul(self.density_q,self.density_a)
    def direct_representation(self):

        self.embedded_q = tf.reshape(self.embedded_chars_q,[-1,self.max_input_left,self.total_embedding_dim])
        self.embedded_a = tf.reshape(self.embedded_chars_a,[-1,self.max_input_right,self.total_embedding_dim])
        reverse_a = tf.transpose(self.embedded_a,[0,2,1])
        self.M_qa = tf.matmul(self.embedded_q,reverse_a)

    def trace_represent(self):
        self.density_diag = tf.matrix_diag_part(self.M_qa)
        self.density_trace = tf.expand_dims(tf.trace(self.M_qa),-1)
        self.match_represent = tf.concat([self.density_diag,self.density_trace],1)
    #construct the density_matrix
    def density_matrix(self,sentence_matrix,sentence_weighted):
        # print sentence_matrix
        # print tf.nn.l2_normalize(sentence_matrix,2)

        self.norm = tf.nn.l2_normalize(sentence_matrix,2)
        # self.norm = tf.nn.softmax(sentence_matrix,2)
        # self.norm = sentence_matrix
        # print tf.reduce_sum(norm,2)
        reverse_matrix = tf.transpose(self.norm, perm = [0,1,3,2])
        q_a = tf.matmul(self.norm,reverse_matrix)
        # return tf.reduce_sum(tf.matmul(self.norm,reverse_matrix), 1)
        return tf.reduce_sum(tf.multiply(q_a,sentence_weighted),1)


    def feed_neural_work(self):     
        with tf.name_scope('regression'):  
            W = tf.Variable(tf.zeros(shape = [self.represent.shape[-1],2]),name = 'W') 
            b = tf.Variable(tf.zeros([2]),name = 'b')
            self.para.append(W)
            self.para.append(b)
            self.logits = tf.nn.xw_plus_b(self.represent, W, b, name = "scores")
            self.scores = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(self.scores, 1, name = "predictions")

            
    def create_loss(self):
        l2_loss = tf.constant(0.0)
        for p in self.para:
            l2_loss += tf.nn.l2_loss(p)
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


    def concat_embedding(self,words_indice,overlap_indice,position_indice):
        embedded_chars_q = tf.nn.embedding_lookup(self.embedding_W,words_indice)
        position_embedding = tf.nn.embedding_lookup(self.position_W,position_indice)
        overlap_embedding_q = tf.nn.embedding_lookup(self.overlap_W,overlap_indice)
        if not self.overlap_needed :
            if not self.position_needed:
                return tf.expand_dims(embedded_chars_q,-1)
            else:
                return tf.expand_dims(tf.reduce_sum([embedded_chars_q,position_embedding],0),-1)
        else:
            if not self.position_needed:
                return  tf.expand_dims(tf.reduce_sum([embedded_chars_q,overlap_embedding_q],0),-1)
            else:
                return tf.expand_dims(tf.reduce_sum([embedded_chars_q,overlap_embedding_q,position_embedding],0),-1)
        
    def convolution(self):
        #initialize my conv kernel
        self.kernels = []
        for i,filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('conv-pool-%s' % filter_size):
                filter_shape = [filter_size,filter_size,1,self.num_filters]
                fan_in = np.prod(filter_shape[:-1])
                fan_out = (filter_shape[-1] * np.prod(filter_shape[:2]))
                W_bound = np.sqrt(6. / (fan_in + fan_out))
                W = tf.Variable(np.asarray(rng.uniform(low = -W_bound,high = W_bound,size = filter_shape),dtype = 'float32'))
                b = tf.Variable(tf.constant(0.0, shape = [self.num_filters]), name = "b")
                self.kernels.append((W,b))
                self.para.append(W)
                self.para.append(b)
        self.num_filters_total = self.num_filters * len(self.filter_sizes)

        self.qa_real_0,self.qa_real_1,self.qa_real_2,self.qa_real_3 = (self.narrow_convolution(tf.expand_dims(self.M_qa,-1),self.kernels))
        
    def max_pooling(self,conv):
        pooled = tf.nn.max_pool(
                    conv,
                    # ksize = [1, self.total_embedding_dim, self.total_embedding_dim, 1],
                    ksize = [1, 11, 11, 1],
                    strides = [1, 1, 1, 1],
                    padding = 'VALID',
                    name = "pool")
        return pooled
    def pooling_graph(self):
        with tf.name_scope('pooling'):
      
            # pooling = self.max_pooling(self.qa)
            # print self.pooling
            # self.represent = tf.reshape(pooling,[-1,self.num_filters * len(self.filter_sizes)])
            # raw_pooling = tf.contrib.layers.flatten(tf.reduce_max(self.qa,1))

            # col_pooling = tf.contrib.layers.flatten(tf.reduce_max(self.qa,2))
            # self.represent = tf.concat([raw_pooling,col_pooling],1)
            self.represent_real_0 = self.max_pooling(self.qa_real_0)
            self.represent_real_1 = self.max_pooling(self.qa_real_1)
            self.represent_real_2 = self.max_pooling(self.qa_real_2)
            self.represent_real_3 = self.max_pooling(self.qa_real_3)
            # print(self.represent_real)
            self.represent_real_0 =tf.reshape(self.represent_real_0 ,[self.batch_size,-1])
            self.represent_real_1 =tf.reshape(self.represent_real_1 ,[self.batch_size,-1])
            self.represent_real_2 =tf.reshape(self.represent_real_2 ,[self.batch_size,-1])
            self.represent_real_3 =tf.reshape(self.represent_real_3 ,[self.batch_size,-1])
            self.represent = tf.concat([self.represent_real_0,self.represent_real_1,self.represent_real_2,self.represent_real_3],1)

    def wide_convolution(self,embedding):
        cnn_outputs = []
        for i,filter_size in enumerate(self.filter_sizes):
            conv = tf.nn.conv2d(
                    embedding,
                    self.kernels[i][0],
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    name="conv-1"
            )

            h = tf.nn.tanh(tf.nn.bias_add(conv, self.kernels[i][1]), name="relu-1")
            cnn_outputs.append(h)
        cnn_reshaped = tf.concat(cnn_outputs,3)
        return cnn_reshaped
    def narrow_convolution(self,embedding,kernel):
        cnn_outputs = []
        for i,filter_size in enumerate(self.filter_sizes):
            conv = tf.nn.conv2d(
                    embedding,
                    kernel[i][0],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="conv-1"
            )
            h = tf.nn.tanh(tf.nn.bias_add(conv, self.kernels[i][1]), name="relu-1")
            cnn_outputs.append(h)
        # cnn_reshaped = tf.concat(cnn_outputs,3)
        # return cnn_outputs
        return cnn_outputs[0], cnn_outputs[1] , cnn_outputs[2] , cnn_outputs[3]    
    def build_graph(self):
        self.create_placeholder()
        self.add_embeddings()
        self.density_weighted()
        self.joint_representation()
        # self.direct_representation()
        self.trace_represent()
        self.convolution()
        self.pooling_graph()
        # self.interact()
        self.feed_neural_work()
        self.create_loss()

if __name__ == '__main__':
    cnn = QA_quantum(max_input_left = 33,
                max_input_right = 40,
                vocab_size = 5000,
                embedding_size = 50,
                batch_size = 3,
                embeddings = None,
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
    input_x_1 = np.reshape(np.arange(3 * 33),[3,33])
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
        # print see

       
