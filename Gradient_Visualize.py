import sys
sys.path.append('/Users/tomoki/Mypython/R-net')
from functools import wraps
import threading

from tensorflow.python.platform import tf_logging as logging

from params import Params
import numpy as np
import tensorflow as tf
from process import *
import evaluate
from sklearn.model_selection import train_test_split

from chainer import Variable, Chain
import chainer.links as L
import numpy as np
import chainer
from chainer import functions as F
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import link
from chainer import reporter
from layers import *

indices = load_target("../data/trainset/" + Params.target_dir)
print("Loading question data...")
q_word_ids, _ = load_word("../data/trainset/" + Params.q_word_dir)
q_char_ids, q_char_len, q_word_len = load_char("../data/trainset/" + Params.q_chars_dir)

# Load passage data
print("Loading passage data...")
p_word_ids, _ = load_word("../data/trainset/" + Params.p_word_dir)
p_char_ids, p_char_len, p_word_len = load_char("../data/trainset/" + Params.p_chars_dir)

# Get max length to pad
p_max_word = np.max(p_word_len)
p_max_char = max_value(p_char_len)
q_max_word = np.max(q_word_len)
q_max_char = max_value(q_char_len)

# pad_data
print("Preparing training data...")
p_word_ids = pad_data(p_word_ids,p_max_word)
q_word_ids = pad_data(q_word_ids,q_max_word)
p_char_ids = pad_char_data(p_char_ids,p_max_char,p_max_word)
q_char_ids = pad_char_data(q_char_ids,q_max_char,q_max_word)

# to numpy
indices = np.reshape(np.asarray(indices,np.int32),(-1,2))
p_word_len = np.reshape(np.asarray(p_word_len,np.int32),(-1,1))
q_word_len = np.reshape(np.asarray(q_word_len,np.int32),(-1,1))
p_char_len = pad_data(p_char_len,p_max_word)
q_char_len = pad_data(q_char_len,q_max_word)


n_layers = 1
batch_size = 32
vocab_size = 2196018
char_vocab_size = 95
unit_size = 50

glove = np.memmap("../" + Params.data_dir + "glove.np",dtype = np.float32,mode = "r")
glove = np.asarray(np.reshape(glove,(Params.vocab_size,300)))
embed_w = L.EmbedID(vocab_size, 300, initialW = glove, ignore_label  = 0)
dict_ = pickle.load(gzip.open("../" + Params.data_dir + "dictionary.pkl.gz","r"))


def output_f1_EM_score(indices_batch, R_net_model, p_w_s_batch):
    batch_y_s = indices_batch.T[0]
    batch_y_f = indices_batch.T[1]
    answer_s = np.argmax(R_net_model.y[0].data, axis = 1)
    answer_f = np.argmax(R_net_model.y[1].data, axis = 1)
    exact_match_score = 0
    F1_score = 0
    for EM_index, (s_index, f_index) in enumerate(zip(answer_s, answer_f)):
        pred_ind = p_w_s_batch[EM_index][s_index:f_index+1]
        pred_word = dict_.ind2word(pred_ind)
        answer_ind = p_w_s_batch[EM_index][batch_y_s[EM_index]:batch_y_f[EM_index] + 1]
        answer_word = dict_.ind2word(answer_ind)
        exact_match_score += np.float(answer_word == pred_word)
        F1_score += evaluate.f1_score(pred_word, answer_word)
    return  exact_match_score/(EM_index + 1), F1_score/(EM_index + 1)

class R_net_Model_grad(layers.R_net_Model):
    def __call__(self, p_w_s_batch, p_c_s_batch, q_w_s_batch, q_c_s_batch):
        self.xpws = [Variable(F.forget(embed_w,item).data) for item in p_w_s_batch]
        #xpws = [F.embed_id(Variable(item), glove) for item in p_w_s_batch]
        self.xpcs = [Variable(F.concat(
            self.model2(
                Variable(np.zeros((n_layers * 2, len(items),50)).astype(np.float32)), 
                [Variable(np.array(item).astype(np.int32)) for item in items]), axis = 1).data)
                for items in p_c_s_batch]
        self.concat_input_p = [F.concat((self.xpcs[index], self.xpws[index])) for index in xrange(self.batch_size)]
        _,self.u_ps = self.model3.l1(
            Variable(np.zeros((n_layers * 2, self.batch_size, self.unit_size/2)).astype(np.float32)),
            self.concat_input_p)
        self.xqws = [Variable(F.forget(embed_w, item).data) for item in q_w_s_batch]
        #xqws = [F.embed_id(Variable(item), glove) for item in q_w_s_batch]
        self.xqcs = [Variable(F.concat(
                self.model2( 
                Variable(np.zeros((n_layers * 2, len(items),50)).astype(np.float32)), 
                [Variable(np.array(item).astype(np.int32)) for item in items]), axis = 1).data)
        for items in q_c_s_batch]
        self.concat_input_q = [F.concat((self.xqcs[index], self.xqws[index])) for index in xrange(self.batch_size)]
        _, self.u_qs = self.model3.l1(
            Variable(np.zeros((n_layers * 2, self.batch_size, self.unit_size/2)).astype(np.float32)),
            self.concat_input_q)
        #return self.model_GARNN(u_qs, u_ps)
        self.vtp_list, self.WqUqj = self.model_GARNN(self.u_qs, self.u_ps)
        self.hps = self.model_SMARNN(self.vtp_list)
        hta_new_b_list = self.model_OL(self.WqUqj, self.u_qs,self.hps)
        return hta_new_b_list


class Classifier(L.Classifier):
    def __call__(self, p_w_s_batch, p_c_s_batch, q_w_s_batch, q_c_s_batch, indices_batch,embed_w):
        self.y = None
        self.loss = None
        self.accuracy = None
        batch_y_s = indices_batch.T[0]
        batch_y_f = indices_batch.T[1]
        self.y = self.predictor(p_w_s_batch, p_c_s_batch, q_w_s_batch, q_c_s_batch,embed_w)
        self.loss = self.lossfun(self.y[0], batch_y_s) + self.lossfun(self.y[1], batch_y_f)
        chainer.reporter.report({'loss': self.loss}, self)
        #exact match
        answer_s = np.argmax(self.y[0].data, axis = 1)
        answer_f = np.argmax(self.y[1].data, axis = 1)
        self.exact_match_score = np.float(((answer_s == batch_y_s) & (answer_f == batch_y_f)).sum())/answer_s.shape[0]
        return self.loss


import copy
glove = 0
unit_size = 150
output_size = 150
batch_size = 1
p_w_s_batch = [np.array(x).astype(np.int32) for x in  p_word_ids[0:batch_size]]
p_c_s_batch = p_char_ids[0:batch_size]
q_w_s_batch = [np.array(x).astype(np.int32) for x in  q_word_ids[0:batch_size]]
q_c_s_batch = q_char_ids[0:batch_size]

#test
R_net_model = Classifier(R_net_Model_grad(
    n_layers, vocab_size, char_vocab_size, unit_size, output_size, batch_size))
chainer.serializers.load_npz("model_5",R_net_model)
#optimizer = chainer.optimizers.Adam()
#optimizer.setup(R_net_model)
output = R_net_model.predictor(p_w_s_batch, p_c_s_batch, q_w_s_batch, q_c_s_batch)
answer_s = F.argmax(output[0], axis = 1)
answer_f = F.argmax(output[1], axis = 1)
y_pred_s = copy.deepcopy(output[0].data)
y_pred_s[:,answer_s.data[0]] = 0
loss_s = F.mean_absolute_error(output[0], y_pred_s)
R_net_model.cleargrads()
loss_s.backward()
place_index = 0
w_place_weights_s = []
for place_index in range(len(R_net_model.predictor.xpws[0])):
	w_place_weights_s.append(np.dot(R_net_model.predictor.xpws[0].grad[place_index], 
	R_net_model.predictor.xpws[0].data[place_index]))

np.argsort(abs(np.array(w_place_weights_s)))[-1::-1]

c_place_weights_s = []
for place_index in range(len(R_net_model.predictor.xpcs[0])):
	c_place_weights_s.append(np.dot(R_net_model.predictor.xpcs[0].grad[place_index], 
	R_net_model.predictor.xpcs[0].data[place_index]))

np.argsort(abs(np.array(c_place_weights_s)))[-1::-1]
np.argsort(output[0].data, axis = 1).T[-1::-1].T
#R_net_model.predictor.embed_w.cleargrads()
#optimizer.update()
