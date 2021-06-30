import argparse,os,re
import configparser
from tensorflow import flags
import config

class Params(object):
    def __init__(self,q_max_sent_length,a_max_sent_length,alphabet,embeddings,embeddings_complex):
    	FLAGS = config.flags.FLAGS
    	self.max_input_left = q_max_sent_length
    	self.max_input_right = a_max_sent_length
    	self.vocab_size = len(alphabet)
    	self.embedding_size = FLAGS.embedding_dim
    	self.batch_size = FLAGS.batch_size
    	self.embeddings = embeddings
    	self.embeddings_complex = embeddings_complex
    	self.dropout_keep_prob = FLAGS.dropout_keep_prob
    	self.filter_sizes = list(map(int, FLAGS.filter_sizes.split(",")))
    	self.num_filters = FLAGS.num_filters
    	self.l2_reg_lambda = FLAGS.l2_reg_lambda
    	self.is_Embedding_Needed = True
    	self.trainable = FLAGS.trainable
    	self.overlap_needed = FLAGS.overlap_needed,
    	self.position_needed = FLAGS.position_needed,
    	self.pooling = FLAGS.pooling,
    	self.hidden_num = FLAGS.hidden_num,
    	self.extend_feature_dim = FLAGS.extend_feature_dim
    	self.model = FLAGS.modelName

def parse_opt(q_max_sent_length,a_max_sent_length,alphabet,embeddings,embeddings_complex):
    opt = Params(q_max_sent_length,a_max_sent_length,alphabet,embeddings,embeddings_complex)
    return opt

