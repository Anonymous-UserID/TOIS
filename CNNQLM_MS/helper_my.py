# -*- coding:utf-8-*-
import numpy as np
import random,os,math
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.keyedvectors import KeyedVectors
import sklearn
import multiprocessing
import time
import pickle 
from collections import defaultdict
import evaluation
import string
from nltk import stem
from tqdm import tqdm
import chardet
import re
import config
from functools import wraps
import nltk
from nltk.corpus import stopwords
from numpy.random import seed
import math

seed(1234)
FLAGS = config.flags.FLAGS
# FLAGS._parse_flags()
FLAGS.flag_values_dict()
dataset = FLAGS.data
isEnglish = FLAGS.isEnglish
model_dim = FLAGS.embedding_dim
UNKNOWN_WORD_IDX = 0
is_stemmed_needed = False
# stopwords=stopwords.words("english")
# stopwords = { word.decode("utf-8") for word in open("model/chStopWordsSimple.txt").read().split()}


def cut(sentence, isEnglish=isEnglish):
    if isEnglish:
        tokens =sentence.lower().split()
        # tokens = [word for word in sentence.split() if word not in stopwords]
    else:
        # words = jieba.cut(str(sentence))
        tokens = [word for word in sentence.split() if word not in stopwords]
    return tokens
# def cut(sentence, isEnglish=isEnglish):
#     if isEnglish:
#         # tokens = sentence.lower().split()
#         tokens = nltk.tokenize.word_tokenize(sentence)
#         # tokens = [word for word in sentence.split() if word not in stopwords]
#     else:
#         # words = jieba.cut(str(sentence))
#         tokens = [word for word in sentence.split() if word not in stopwords]
#     return tokens
#print( tf.__version__)


def log_time_delta(func):
    @wraps(func)
    def _deco(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        print("%s runed %.2f seconds" % (func.__name__, delta))
        return ret
    return _deco


class Alphabet(dict):
    def __init__(self, start_feature_id=1):
        self.fid = start_feature_id

    def add(self, item):
        idx = self.get(item, None)
        if idx is None:
            idx = self.fid
            self[item] = idx
      # self[idx] = item
            self.fid += 1
        return idx

    def dump(self, fname):
        with open(fname, "w") as out:
            for k in sorted(self.keys()):
                out.write("{}\t{}\n".format(k, self[k]))
def getSubVectors_complex_uniform(max_sentence, dim=50):
    embedding = np.zeros((max_sentence, dim))
    for i in range(max_sentence):
        embedding[i] = np.random.uniform(+((2*math.pi)/30)*i, +((2*math.pi)/30)*(i+1), dim)
    return embedding

@log_time_delta
def prepare(cropuses, max_sent_length=31,is_embedding_needed=False, dim=model_dim, fresh=False):
    vocab_file = str(dataset) +'_voc.pkl'
    d = dict()
    if os.path.exists('v2d.txt'):
        with open('v2d.txt', 'rb') as dd:
            v2d = pickle.load(dd)
    if os.path.exists(vocab_file):
    	with open(vocab_file,'rb') as f:
    		alphabet = pickle.load(f)
    else:
        alphabet = Alphabet(start_feature_id=0)
        alphabet.add('[UNKNOW]')
        alphabet.add('END')
        count = 0
        for corpus in cropuses:
            for texts in [corpus["question"].unique(), corpus["answer"]]:
                for sentence in tqdm(texts):
                    count += 1
                    if count % 10000 == 0:
                        print (count)
                    tokens = cut(sentence)
                    for token in set(tokens):
                        d[token] = d.get(token,0) + 1
                        alphabet.add(token)
        print (len(alphabet.keys()))
        with open(vocab_file,'wb') as f:
        	pickle.dump(alphabet,f)
        with open('v2d.txt', 'wb') as dd:
            pickle.dump(d,dd)
    if is_embedding_needed:
        # sub_vec_file = '../embedding/sub_vector'
        # if os.path.exists(sub_vec_file) and not fresh:
        #     sub_embeddings = pickle.load(open(sub_vec_file, 'r'))
        # else:
        if isEnglish:
            if dim == 50:
                fname = "../embedding/aquaint+wiki.txt.gz.ndim=50.bin"
                embeddings_1 = KeyedVectors.load_word2vec_format(fname, binary=True)
                sub_embeddings,count,total = getSubVectors(embeddings_1,alphabet,dim,v2d)
                # fname = "../embedding/glove.6B.50d.txt"
                # embeddings_1 = load_text_vec(alphabet, fname, embedding_size=dim)
                # sub_embeddings,count,total = getSubVectorsFromDict(embeddings_1, alphabet, dim)
                # print('{}/{} = {}'.format(count,total,count/total))
                # exit()
                embedding_complex = getSubVectors_complex_uniform(max_sent_length,dim)
            else:
                # fname = "../../Class/embedding/glove_300d.txt"
                # embeddings_1 = load_text_vec(alphabet, fname, embedding_size=dim)
                # sub_embeddings = getSubVectorsFromDict(embeddings_1, alphabet, dim)
                # embedding_complex = getSubVectors_complex_random(alphabet,dim)
                fname = "../embedding/glove.42B.300d.txt"
                embeddings_1 = load_text_vec(alphabet, fname, embedding_size=dim)
                sub_embeddings = getSubVectorsFromDict(embeddings_1, alphabet, dim)
                sub_embeddings,count,total = getSubVectorsFromDict(embeddings_1, alphabet, dim)
                print('{}/{} = {}'.format(count,total,count/total))
                exit()
                embedding_complex = getSubVectors_complex_uniform(max_sent_length,dim)
        else:
            fname = 'model/wiki.ch.text.vector'
            embeddings = load_text_vec(alphabet, fname, embedding_size=dim)
            sub_embeddings = getSubVectorsFromDict(
                embeddings, alphabet, dim)
        # pickle.dump(sub_embeddings, open(sub_vec_file, 'wb'))
        # print (len(alphabet.keys()))
        # embeddings = load_vectors(vectors,alphabet.keys(),layer1_size)
        # embeddings = KeyedVectors.load_word2vec_format(fname, binary=True)
        # sub_embeddings = getSubVectors(embeddings,alphabet)
        #print(sub_embeddings[49240],embedding_complex[49240])
        #exit()
        return alphabet, sub_embeddings,embedding_complex
    else:
        return alphabet
        

def get_lookup_table(embedding_params):
    id2word = embedding_params['id2word']
    word_vec = embedding_params['word_vec']
    lookup_table = []

    # Index 0 corresponds to nothing
    lookup_table.append([0]* embedding_params['wvec_dim'])
    for i in range(1, len(id2word)):
        word = id2word[i]
        wvec = [0]* embedding_params['wvec_dim']
        if word in word_vec:
            wvec = word_vec[word]
        # print(wvec)
        lookup_table.append(wvec)

    lookup_table = np.asarray(lookup_table)
    return(lookup_table)
def getSubVectors(vectors, vocab, dim=model_dim,v2d=None):
    embedding = np.zeros((len(vocab), dim))
    temp_vec = 0
    count = 0
    total = len(vocab)
    unk = np.random.uniform(-0.5, +0.5, vectors.syn0.shape[1])
    for word in vocab:
        if word in vectors.vocab:
            
            if v2d[word] >=3 :
                count += 1
                embedding[vocab[word]] = vectors.word_vec(word)
            else:
                embedding[vocab[word]] = unk
        else:
            embedding[vocab[word]
                      ] = np.random.uniform(-0.5, +0.5, vectors.syn0.shape[1])
    return embedding,count,total
def transform(flag):
    if flag == 0:
        return [1,0]


    else:
        return [0,1]
'''def getSubVectors(vectors, vocab, dim=50):
    print ('embedding_size:', vectors.syn0.shape[1])
    embedding = np.zeros((len(vocab), vectors.syn0.shape[1]))
    temp_vec = 0
    for word in vocab:
        if word in vectors.vocab:
            embedding[vocab[word]] = vectors.word_vec(word)
        else:
            # .tolist()
            embedding[vocab[word]
                      ] = np.random.uniform(-0.25, +0.25, vectors.syn0.shape[1])
        embedding_ave=np.sum(embedding[vocab[word]])/dim
        embedding[vocab[word]]=embedding[vocab[word]]-embedding_ave
        temp_vec += embedding[vocab[word]]
    temp_vec /= len(vocab)
    for index, _ in enumerate(embedding):
        embedding[index] -= temp_vec
    return embedding'''
def load_text_vec(alphabet, filename="", embedding_size=100):
    vectors = {}
    with open(filename,encoding='utf-8') as f:
        i = 0
        for line in f:
            i += 1
            if i % 100000 == 0:
                print ('epch %d' % i)
            items = line.strip().split(' ')
            if len(items) == 2:
                vocab_size, embedding_size = items[0], items[1]
                print (vocab_size, embedding_size)
            else:
                word = items[0]
                if word in alphabet:
                    vectors[word] = items[1:]
    print ('embedding_size', embedding_size)
    print ('done')
    print ('words found in wor2vec embedding ', len(vectors.keys()))
    return vectors


def getSubVectorsFromDict(vectors, vocab, dim=300):
    file = open('missword', 'w')
    embedding = np.zeros((len(vocab), dim))
    count = 0
    total = len(vocab)
    for word in vocab:

        if word in vectors:
            count += 1
            embedding[vocab[word]] = vectors[word]
        else:
            # if word in names:
            #     embedding[vocab[word]] = vectors['谁']
            # else:
            file.write(word + '\n')
            # vectors['[UNKNOW]'] #.tolist()
            embedding[vocab[word]] = np.random.uniform(-0.25, +0.25, dim)
    file.close()
    # print ('word in embedding', count)
    return embedding,count,total


@log_time_delta
def get_overlap_dict(df, alphabet, q_len=40, a_len=40):
    d = dict()
    for question in df['question'].unique():
        group = df[df['question'] == question]
        answers = group['answer']
        for ans in answers:
            q_overlap, a_overlap = overlap_index(question, ans, q_len, a_len)
            d[(question, ans)] = (q_overlap, a_overlap)
    return d
# calculate the overlap_index


def overlap_index(question, answer, q_len, a_len, stopwords=[]):
    qset = set(cut(question))
    aset = set(cut(answer))

    q_index = np.zeros(q_len)
    a_index = np.zeros(a_len)

    overlap = qset.intersection(aset)
    for i, q in enumerate(cut(question)[:q_len]):
        value = 1
        if q in overlap:
            value = 2
        q_index[i] = value
    for i, a in enumerate(cut(answer)[:a_len]):
        value = 1
        if a in overlap:
            value = 2
        a_index[i] = value
    return q_index, a_index


def position_index(sentence, length):
    index = np.zeros(length)
    raw_len = len(cut(sentence))
    index[:min(raw_len, length)] = range(1, min(raw_len + 1, length + 1))
    # print index
    return index


def encode_to_split(sentence, alphabet, max_sentence=40):
    indices = []
    tokens = cut(sentence)
    for word in tokens:
        indices.append(alphabet[word])
    while(len(indices) < max_sentence):
        indices += indices[:(max_sentence - len(indices))]
    # results=indices+[alphabet["END"]]*(max_sentence-len(indices))
    return indices[:max_sentence]

def overlap_score(question,answer):
    question_word=cut(question)
    answer_word=cut(answer)
    same_num=0
    for w in question_word:
        if w in answer_word:
            same_num = same_num+1
        else:
            same_num += 0
    return [same_num,1]



# def load(dataset = dataset, filter = False):
#     data_dir = "../data/" + dataset
#     datas = []  
#     for data_name in ['train.txt','dev.txt','test.txt']:
#         if data_name=='train.txt':
#             data_file = os.path.join(data_dir,data_name)
#             data = pd.read_csv(data_file,header = None,sep="\t",names=["question","answer","flag"],quoting =3).fillna("WASHINGTON")
#             if filter == True:
#                 datas.append(removeUnanswerdQuestion(data))
#             else:
#                 datas.append(data)
#         if data_name=='dev.txt':
#             data_file = os.path.join(data_dir,data_name)
#             data = pd.read_csv(data_file,header = None,sep="\t",names=["question","answer","flag"],quoting =3).fillna("WASHINGTON")
#             if filter == True:
#                 datas.append(removeUnanswerdQuestion(data))
#             else:
#                 datas.append(data)
#         if data_name=='test.txt':
#             data_file = os.path.join(data_dir,data_name)
#             data = pd.read_csv(data_file,header = None,sep="\t",names=["question","answer","flag"],quoting =3).fillna("WASHINGTON")
#             if filter == True:
#                 datas.append(removeUnanswerdQuestion(data))
#             else:
#                 datas.append(data)
    
#     sub_file = os.path.join(data_dir,'submit.txt')
#     return tuple(datas)

def load(dataset=dataset, filter=False):
    data_dir = "../new_data/" + dataset
    datas = []
    for data_name in ['train.txt', 'test.txt']:
        data_file = os.path.join(data_dir, data_name)
        data = pd.read_csv(data_file, header=None, sep="\t", names=[
                           "qid", "aid", "question", "answer", "flag"], quoting=3).fillna('N')
        if filter == True:
            datas.append(removeUnanswerdQuestion(data))
        else:
            datas.append(data)
    data_file = os.path.join(data_dir, "dev.txt")
    dev = pd.read_csv(data_file, header=None, sep="\t", names=[
                      "question", "answer", "flag"], quoting=3).fillna('N')
    datas.append(dev)
    # submit = pd.read_csv(sub_file,header = None,sep = "\t",names = ['question','answer'],quoting = 3)
    # datas.append(submit)
    return tuple(datas)

def load_ms(dataset=dataset, filter=False):
    data_dir = "../new_data/" + dataset
    datas = []
    for data_name in ['train_all.txt', "dev_clean.txt",'testtt.txt']:
        data_file = os.path.join(data_dir, data_name)
        data = pd.read_csv(data_file, header=None, sep="\t", names=[
                           "question", "answer", "flag"], quoting=3).fillna('N')
        if filter == True:
            datas.append(removeUnanswerdQuestion(data))
        else:
            datas.append(data)
    return tuple(datas)

def load_wiki(dataset=dataset, filter=False):
    data_dir = "../new_data/" + dataset
    datas = []
    for data_name in ['train.txt', "dev.txt",'test.txt']:
        data_file = os.path.join(data_dir, data_name)
        data = pd.read_csv(data_file, header=None, sep="\t", names=[
                           "question", "answer", "flag"], quoting=3).fillna('N')
        if filter == True:
            datas.append(removeUnanswerdQuestion(data))
        else:
            datas.append(data)
    return tuple(datas)

def removeUnanswerdQuestion(df):
    counter = df.groupby("question").apply(lambda group: sum(group["flag"]))
    questions_have_correct = counter[counter > 0].index
    counter = df.groupby("question").apply(
        lambda group: sum(group["flag"] == 0))
    questions_have_uncorrect = counter[counter > 0].index
    counter = df.groupby("question").apply(lambda group: len(group["flag"]))
    questions_multi = counter[counter > 1].index

    return df[df["question"].isin(questions_have_correct) & df["question"].isin(questions_have_correct) & df["question"].isin(questions_have_uncorrect)].reset_index()


def batch_gen_with_single(df, alphabet, batch_size=10, q_len=33, a_len=40, overlap_dict=None,name = 'QA_quantum'):
    pairs = []
    
    for index, row in df.iterrows():
        quetion = encode_to_split(
            row["question"], alphabet, max_sentence=q_len)
        answer = encode_to_split(row["answer"], alphabet, max_sentence=a_len)
        if overlap_dict == None:
            q_overlap, a_overlap = overlap_index(
                row["question"], row["answer"], q_len, a_len)
        else:
            q_overlap, a_overlap = overlap_dict[(
                row["question"], row["answer"])]      
        q_position = position_index(row['question'], q_len)
        a_position = position_index(row['answer'], a_len)
        overlap=overlap_score(row['question'],row['answer'])
        if name =='QA_quantum' or name =='CNNQLM_I'  or name =='CNNQLM_I_Flat' or name =='CNNQLM_Vocab' or name =='CNNQLM_Dim' :
            pairs.append((quetion, answer, q_position, a_position,overlap,q_overlap,a_overlap))
            input_num = 7    
        else:
            pairs.append((quetion, answer, q_position, a_position,q_overlap,a_overlap))
            input_num = 6
    # n_batches= int(math.ceil(df["flag"].sum()*1.0/batch_size))
    n_batches = int(len(pairs) * 1.0 / batch_size)
    # pairs = sklearn.utils.shuffle(pairs,random_state =132)
    for i in range(0, n_batches):
        batch = pairs[i * batch_size:(i + 1) * batch_size]

        yield [[pair[j] for pair in batch] for j in range(input_num)]
    batch = pairs[n_batches * batch_size:] + [pairs[n_batches *
                                                    batch_size]] * (batch_size - len(pairs) + n_batches * batch_size)
    yield [[pair[i] for pair in batch] for i in range(input_num)]
    
def batch_gen_with_point_wise(df, alphabet, batch_size=10, overlap_dict=None, q_len=33, a_len=40,name='QA_quantum'):
    # inputq inputa intput_y overlap
    pairs = []
    for index, row in df.iterrows():
        question = encode_to_split(
            row["question"], alphabet, max_sentence=q_len)
        answer = encode_to_split(row["answer"], alphabet, max_sentence=a_len)
        if overlap_dict == None:
            q_overlap, a_overlap = overlap_index(
                row["question"], row["answer"], q_len, a_len)
        else:
            q_overlap, a_overlap = overlap_dict[(
                row["question"], row["answer"])]
        q_position = position_index(row['question'], q_len)
        a_position = position_index(row['answer'], a_len)
        label = transform(row["flag"])
        overlap=overlap_score(row['question'],row['answer'])
        if name =='QA_quantum' or name =='CNNQLM_I'  or name =='CNNQLM_I_Flat' or name =='CNNQLM_Vocab' or name =='CNNQLM_Dim' :
        #pairs.append((question, answer, label, q_overlap,a_overlap, q_position, a_position))
            pairs.append((question, answer, label,q_position, a_position,overlap,q_overlap,a_overlap))
            input_num = 8
        else:
            pairs.append((question, answer, label, q_overlap,a_overlap, q_position, a_position))
            input_num = 7
    # n_batches= int(math.ceil(df["flag"].sum()*1.0/batch_size))
    n_batches = int(len(pairs) * 1.0 / batch_size)
    pairs = sklearn.utils.shuffle(pairs, random_state=121)

    for i in range(0, n_batches):
        batch = pairs[i * batch_size:(i + 1) * batch_size]
        yield [np.array([pair[i] for pair in batch]) for i in range(input_num)]
    batch = pairs[n_batches * batch_size:] + [pairs[n_batches *
                                                    batch_size]] * (batch_size - len(pairs) + n_batches * batch_size)
    yield [np.array([pair[i] for pair in batch]) for i in range(input_num)]




def test_my(sentence,alphabet,q_len):
    question=encode_to_split(sentence,alphabet,max_sentence=q_len)

    return question

if __name__ == '__main__':
    train, test, dev = load(FLAGS.data, filter=FLAGS.clean)
    q_max_sent_length = max(
        map(lambda x: len(x), train['question'].str.split()))
    a_max_sent_length = max(map(lambda x: len(x), train['answer'].str.split()))
    alphabet, embeddings = prepare(
        [train, test, dev], dim=FLAGS.embedding_dim, is_embedding_needed=True, fresh=True)
