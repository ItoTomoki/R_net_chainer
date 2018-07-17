import sys
#sys.path.append('/Users/tomoki/Mypython/R-net')
sys.path.append('./R-net')
from functools import wraps
import threading

#from tensorflow.python.platform import tf_logging as logging

from params import Params
import numpy as np
#import tensorflow as tf
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


indices = load_target("../data/trainset/" + Params.target_dir)
print("Loading question data...")
q_word_ids, _ = load_word("../data/trainset/" + Params.q_word_dir)
q_char_ids, q_char_len, q_word_len = load_char("../data/trainset/" + Params.q_chars_dir)

# Load passage data
print("Loading passage data...")
p_word_ids, _ = load_word("../data/trainset/" + Params.p_word_dir)
p_char_ids, p_char_len, p_word_len = load_char("../data/trainset/" + Params.p_chars_dir)

# Get max length to pad
p_max_word = np.max(p_word_len) + 1
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

# pad_data

n_layers = 1
batch_size = 32
vocab_size = 2196018
char_vocab_size = 95
unit_size = 50
#p_w_s_batch = [np.array(x).astype(np.int32) for x in  p_word_ids[0:batch_size]]
#p_c_s_batch = p_char_ids[0:batch_size]
#q_w_s_batch = [np.array(x).astype(np.int32) for x in  q_word_ids[0:batch_size]]
#q_c_s_batch = q_char_ids[0:batch_size]

glove = np.memmap("../" + Params.data_dir + "glove.np",dtype = np.float32,mode = "r")
glove = np.asarray(np.reshape(glove,(Params.vocab_size,300)))
glove_c = np.memmap("../" + Params.data_dir + "glove_char.np",dtype = np.float32,mode = "r")
glove_c = np.asarray(np.reshape(glove_c,(char_vocab_size,300)))
embed_w = L.EmbedID(vocab_size, 300, initialW = glove, ignore_label  = 0)
embed_w_c = L.EmbedID(char_vocab_size, 300, initialW = glove_c, ignore_label  = 0)

dict_ = pickle.load(gzip.open("../" + Params.data_dir + "dictionary.pkl.gz","r"))
#indices_batch = indices_train_random[batch_index:batch_index + batch_size]
#indices_batch = indices_valid[batch_index:batch_index + batch_size]
def output_f1_EM_score(indices_batch, R_net_model, p_w_s_batch):
    batch_y_s = indices_batch.T[0]
    batch_y_f = indices_batch.T[1]
    answer_s = np.argmax(chainer.cuda.to_cpu(R_net_model.y[0].data), axis = 1)
    answer_f = np.argmax(chainer.cuda.to_cpu(R_net_model.y[1].data), axis = 1)
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

class BiRNN_2(chainer.Chain):
    def __init__(self, n_layer, n_vocab, n_units, dropout, cudnn, initialW = None):
        super(BiRNN_2, self).__init__(
            #embed=L.EmbedID(n_vocab, 300, initialW = initialW, ignore_label  = 0),
            l1=L.NStepBiGRU(n_layer, 400, n_units/2,
                           dropout)
            #l2=L.Linear(n_units/2, 10),
        )
    def __call__(self, hx, xs):
        #xs = [self.embed(item) for item in xs]
        hy, ys = self.l1(hx, xs)
        return hy, ys

class BiRNN_1(chainer.Chain):
    def __init__(self, n_layer, n_units, n_hidden, dropout, cudnn, initialW = None):
        super(BiRNN_1, self).__init__(
            l1=L.NStepBiGRU(n_layer,n_units, n_hidden,
                           dropout)
        )
    def __call__(self, hx, xs):
        with chainer.no_backprop_mode():
            xs = [Variable(xp.array(F.forget(embed_w_c, item).data)) for item in xs] 
        #xs = [self.embed(item) for item in xs]
        #print xs[0].shape, hx.shape
        hy, _ = self.l1(hx, xs)
        return hy

class BiRNN(chainer.Chain):
    def __init__(self, n_layer, n_vocab, n_units, dropout, cudnn, initialW = None):
        super(BiRNN, self).__init__(
            embed=L.EmbedID(n_vocab, 50, initialW = initialW, ignore_label  = 0),
            l1=L.NStepBiGRU(n_layer, 50, 50,
                           dropout)
            #l2=L.Linear(n_units/2, 10),
        )
    def __call__(self, hx, xs):
        xs = [self.embed(item) for item in xs]
        hy, _ = self.l1(hx, xs)
        return hy

class GARNN(chainer.Chain):
    def __init__(self, n_layer, input_dim, n_units, batch_size, dropout = 0.2, cudnn = False):
        super(GARNN, self).__init__(
            W_up=L.Linear(input_dim, n_units),
            W_vp=L.Linear(n_units, n_units),
            W_uq = L.Linear(input_dim, n_units),
            W_v = L.Linear(n_units,1),
            W_g = L.Linear(input_dim * 2, input_dim * 2),
            W_gru = L.StatefulGRU(input_dim * 2, n_units)
            )
        self.batch_size = batch_size
    def __call__(self, u_qs, u_ps):
        self.W_gru.h = Variable(xp.zeros((self.batch_size, u_qs[0].shape[1])).astype(np.float32))
        WqUqj = F.stack([self.W_uq(uq) for uq in u_qs])
        WpUps = F.stack([self.W_up(up) for up in u_ps])
        batch_W_v = F.broadcast_to(self.W_v.W, [self.batch_size, self.W_v.W.shape[1]])
        u_ps_stack = F.stack(u_ps)
        u_qs_stack = F.stack(u_qs)
        vtp_list = []
        for index in range(WpUps.shape[1]):
            WvVp_batch = F.transpose(F.broadcast_to(self.W_vp(self.W_gru.h), 
            [WqUqj.shape[1], WqUqj.shape[0], WqUqj.shape[2]]),(1,0,2))
            s_tj = F.batch_matmul(
                #(F.tanh(WqUqj + WvVp_batch + F.stack([WpUps[:,index,:]] * WqUqj.shape[1], axis = 1))),
                (F.tanh(WqUqj + WvVp_batch + 
                    F.transpose(F.broadcast_to(WpUps[:,index,:],
                        [WqUqj.shape[1], WqUqj.shape[0], WqUqj.shape[2]]),(1,0,2)))),
                batch_W_v).reshape(self.batch_size, q_max_word)
            at = F.softmax(s_tj)
            ct = F.batch_matmul(u_qs_stack, at, transa = True).reshape(self.batch_size, u_qs[0].shape[1])
            gt_batch = F.sigmoid(self.W_g(F.concat([u_ps_stack[:,index,:], ct])))
            g_input = gt_batch * F.concat([u_ps_stack[:,index,:], ct])
            vtp_list.append(self.W_gru(g_input))
        WpUps = 0
        WvVp_batch = 0
        return vtp_list, WqUqj

#self-matching attention
class SMARNN(chainer.Chain):
    def __init__(self, n_layer, input_dim, n_units, batch_size, dropout = 0.2, cudnn = False):
        super(SMARNN, self).__init__(
            W_vp= L.Linear(input_dim, n_units),
            W_vpa =L.Linear(input_dim, n_units),
            W_v = L.Linear(n_units,1), 
            W_f_gru = L.StatefulGRU(input_dim + n_units, n_units),
            W_b_gru = L.StatefulGRU(input_dim + n_units, n_units)
            )
        self.batch_size = batch_size
    def __call__(self, vtp_list):
        WpVp = F.stack([self.W_vp(vtp) for vtp in vtp_list], axis = 1)
        WpVp2 = F.stack([self.W_vpa(vtp) for vtp in vtp_list], axis = 1)
        htp_new_f_list = []
        batch_W_v = F.broadcast_to(self.W_v.W, [self.batch_size, self.W_v.W.shape[1]])
        for index in range(WpVp2.shape[1]):
            s_tj = F.batch_matmul(
                    #(F.tanh(WpVp + F.stack([WpVp2[:,index,:]] * WpVp.shape[1], axis = 1))),
                    (F.tanh(WpVp + 
                        F.transpose(F.broadcast_to(WpVp2[:,index,:], 
                            [WpVp.shape[1], WpVp.shape[0], WpVp.shape[2]]), (1,0,2)))),
                    batch_W_v).reshape(self.batch_size, WpVp.shape[1])
            at = F.softmax(s_tj)
            ct = F.batch_matmul(F.stack(vtp_list, axis = 1), 
                at, transa = True).reshape(self.batch_size, vtp_list[0].shape[1])
            htp_new_f_list.append(self.W_f_gru(F.concat([WpVp2[:,index,:],ct])))
        htp_new_b_list = []
        for index in range(WpVp2.shape[1])[-1::-1]:
            s_tj = F.batch_matmul(
                #(F.tanh(WpVp[:,-1::-1,:] + F.stack([WpVp2[:,index,:]] * WpVp_b.shape[1], axis = 1))),
                (F.tanh(WpVp[:,-1::-1,:] + 
                    F.transpose(F.broadcast_to(WpVp2[:,index,:], 
                        [WpVp.shape[1], WpVp.shape[0], WpVp.shape[2]]), (1,0,2)))),
                batch_W_v).reshape(self.batch_size, WpVp.shape[1])
            at = F.softmax(s_tj)
            ct = F.batch_matmul(F.stack(vtp_list[-1::-1], axis = 1), 
            	                at, transa = True).reshape(self.batch_size, vtp_list[0].shape[1])
            htp_new_b_list.append(self.W_b_gru(F.concat([WpVp2[:,index,:],ct])))
        hps = F.transpose(F.concat([F.transpose(F.stack(htp_new_f_list, axis = 1)), 
                F.transpose(F.stack(htp_new_b_list[-1::-1], axis = 1))
                ], axis = 0))
        WpVp = 0
        WpVp_b = 0
        WpVp2 = 0
        return hps

#hps in [batch * sentence length * dimention]

#Output Layer
class OutputLayer(chainer.Chain):
    def __init__(self, n_layer, input_dim, n_units, batch_size, dropout = 0.2, cudnn = False):
        super(OutputLayer, self).__init__(
            Wp_h = L.Linear(input_dim, n_units),
            Wa_h =L.Linear(n_units, n_units),
            W_v = L.Linear(input_dim,1), 
            #W_vQVQ = L.Linear(input_dim, q_max_word), 
            #W_rq = L.Linear(input_dim,n_units),
            W_f_gru = L.StatefulGRU(input_dim, n_units))
        self.n_units = n_units
        self.batch_size = batch_size
    def __call__(self, WqUqj, u_qs,hps, put_zero = False):
        #differ from paper p5: eq(11)
        #s_j = F.batch_matmul(
        #F.tanh(WqUqj + F.stack([self.W_vQVQ.W] * batch_size, axis = 0)),
        #F.broadcast_to(self.W_v.W, [batch_size, self.W_v.W.shape[1]]),
        #).reshape(batch_size, WqUqj.shape[1])
        #it correspond to V_r^Q = zeros() in eq(11)
        s_j = F.batch_matmul(
            F.tanh(WqUqj), 
            F.broadcast_to(self.W_v.W, [self.batch_size, self.W_v.W.shape[1]]),
            ).reshape(self.batch_size, WqUqj.shape[1])
        a_i = F.softmax(s_j)
        rQ = F.batch_matmul(F.stack(u_qs), a_i, transa = True).reshape(self.batch_size, u_qs[0].shape[1])
        self.W_f_gru.h =  rQ
        WpHp = F.stack([self.Wp_h(hp) for hp in hps])
        hta_new_b_list = []
        for index in range(2):
            s_tj = F.batch_matmul(
            F.tanh(WpHp + F.stack([self.W_f_gru.h] * WpHp.shape[1], axis = 1)),
            F.broadcast_to(self.W_v.W, [self.batch_size, self.W_v.W.shape[1]])
                ).reshape(self.batch_size, WpHp.shape[1])
            if ((put_zero == True) & (index == 1)):
                mask = np.ones()
                for r_index, pt_index in enumerate(pt):
                    mask[r][:pt_index] = 0
                s_tj = s_tj * Variable(mask)
            hta_new_b_list.append(s_tj)
            at = F.softmax(s_tj)
            pt = F.argmax(at, axis = 1)
            ct = F.batch_matmul(hps, at, transa = True).reshape(self.batch_size, self.n_units)
            hta_new = self.W_f_gru(ct)
        return hta_new_b_list


class R_net_Model_gpu(chainer.Chain):
    def __init__(self, n_layer, vocab_size, char_vocab_size, unit_size, output_size, 
    	batch_size = 32,dropout = 0.2, cudnn = True):
        super(R_net_Model_gpu, self).__init__(
            model2 = BiRNN_1(1, 300, 50, 0.2, cudnn = cudnn),
            model3 = BiRNN_2(1, char_vocab_size, unit_size, 0.2, cudnn = cudnn),
            #modelq_2 = BiRNN(1, char_vocab_size, 100, 0.5, False),
            #modelq_3 = BiRNN(1, char_vocab_size, 2 * 100, 0.5, False),
            model_GARNN = GARNN(1, unit_size, unit_size, batch_size, cudnn = cudnn),
            model_SMARNN = SMARNN(1, unit_size, unit_size/2, batch_size, cudnn = cudnn),
            model_OL = OutputLayer(1, unit_size, output_size, batch_size, cudnn = cudnn)
            )
        self.unit_size = unit_size
        self.batch_size = batch_size
    def __call__(self, p_w_s_batch, p_c_s_batch, q_w_s_batch, q_c_s_batch):
        with chainer.no_backprop_mode():
            xpws = [Variable(xp.array(F.forget(embed_w,item).data)) for item in p_w_s_batch]
        #xpws = [F.embed_id(Variable(item), glove) for item in p_w_s_batch]
        #xpcs = [F.concat(
            #self.model2(
                #Variable(xp.zeros((n_layers * 2, len(items),50)).astype(np.float32)), 
                #[Variable(np.array(item)) for item in items]), axis = 1)
                #for items in p_c_s_batch]
        xpcs = [F.concat(
            self.model2(
                Variable(xp.zeros((n_layers * 2, len(items),50)).astype(np.float32)), 
                [item for item in items]), axis = 1)
                for items in p_c_s_batch]
        concat_input = [F.concat((xpcs[index], xpws[index])) for index in xrange(self.batch_size)]
        _,u_ps = self.model3.l1(
            Variable(xp.zeros((n_layers * 2, self.batch_size, self.unit_size/2)).astype(np.float32)),
            concat_input)
        with chainer.no_backprop_mode():
            xqws = [Variable(xp.array(F.forget(embed_w, item).data)) for item in q_w_s_batch]
        #xqws = [F.embed_id(Variable(item), glove) for item in q_w_s_batch]
        xqcs = [F.concat(
                self.model2( 
                Variable(xp.zeros((n_layers * 2, len(items),50)).astype(np.float32)), 
                [item for item in items]), axis = 1)
        for items in q_c_s_batch]
        concat_input = [F.concat((xqcs[index], xqws[index])) for index in xrange(self.batch_size)]
        _, u_qs = self.model3.l1(
            Variable(xp.zeros((n_layers * 2, self.batch_size, self.unit_size/2)).astype(np.float32)),
            concat_input)
        #return self.model_GARNN(u_qs, u_ps)
        vtp_list, WqUqj = self.model_GARNN(u_qs, u_ps)
        hps = self.model_SMARNN(vtp_list)
        hta_new_b_list = self.model_OL(WqUqj, u_qs,hps)
        return hta_new_b_list


class Classifier(L.Classifier):
    def __call__(self, p_w_s_batch, p_c_s_batch, q_w_s_batch, q_c_s_batch, indices_batch):
        self.y = None
        self.loss = None
        self.accuracy = None
        batch_y_s = xp.array(indices_batch.T[0])
        batch_y_f = xp.array(indices_batch.T[1])
        self.y = self.predictor(p_w_s_batch, p_c_s_batch, q_w_s_batch, q_c_s_batch)
        self.loss = self.lossfun(self.y[0], batch_y_s) + self.lossfun(self.y[1], batch_y_f)
        chainer.reporter.report({'loss': self.loss}, self)
        #exact match
        answer_s = np.argmax(self.y[0].data, axis = 1)
        answer_f = np.argmax(self.y[1].data, axis = 1)
        self.exact_match_score = np.float(((answer_s == batch_y_s) & (answer_f == batch_y_f)).sum())/answer_s.shape[0]
        return self.loss

#glove = 0
unit_size = 100
output_size = 100
batch_size = 32

#p_w_s_batch_xp = [xp.array(x).astype(np.int32) for x in  p_w_s_batch]
#p_c_s_batch_xp = xp.array(p_c_s_batch)
#q_w_s_batch_xp = [np.array(x).astype(np.int32) for x in  q_word_ids[0:batch_size]]
#q_c_s_batch_xp = q_char_ids[0:batch_size]

#test

R_net_model = Classifier(R_net_Model_gpu(
    n_layers, vocab_size, char_vocab_size, unit_size, output_size, batch_size, cudnn = False))
#R_net_model.to_gpu()
#optimizer = chainer.optimizers.Adam()
#optimizer.setup(R_net_model)
#output = R_net_model.predictor(p_w_s_batch, p_c_s_batch, q_w_s_batch, q_c_s_batch)
#loss = R_net_model(p_w_s_batch, p_c_s_batch, q_w_s_batch, q_c_s_batch, indices_train_random[batch_index:batch_index + batch_size])
#R_net_model.cleargrads()
#loss.backward()
#loss.unchain_backward()
#R_net_model.predictor.model2.embed.W.grad
#optimizer.update()

#import numpy as xp
#train_data_size = 3000
#train_data_size = 70000
#valid_data_size = 2000

indices_valid = load_target("../data/devset/" + Params.target_dir)
print("Loading question data...")
q_word_ids_valid, _ = load_word("../data/devset/" + Params.q_word_dir)
q_char_ids_valid, q_char_len_valid, q_word_len_valid = load_char("../data/devset/" + Params.q_chars_dir)

# Load passage data
print("Loading passage data...")
p_word_ids_valid, _ = load_word("../data/devset/" + Params.p_word_dir)
p_char_ids_valid, p_char_len_valid, p_word_len_valid = load_char("../data/devset/" + Params.p_chars_dir)

# pad_data
print("Preparing training data...")
p_w_s_valid = pad_data(p_word_ids_valid,p_max_word)
q_w_s_valid = pad_data(q_word_ids_valid,q_max_word)
p_c_s_valid = pad_char_data(p_char_ids_valid,p_max_char,p_max_word)
q_c_s_valid = pad_char_data(q_char_ids_valid,q_max_char,q_max_word)
indices_valid = np.reshape(np.asarray(indices_valid,np.int32),(-1,2))
valid_data_size = len(indices_valid)

indices_train =  indices[0:-1]
p_w_s_train = [np.array(x).astype(np.int32) for x in  p_word_ids[0:-1]]
p_c_s_train = p_char_ids[0:-1]
q_w_s_train = [np.array(x).astype(np.int32) for x in  q_word_ids[0:-1]]
q_c_s_train = q_char_ids[0:-1]
train_data_size = len(indices_train)


#train
#indices_train =  indices[indices.T[1] != 0][0:train_data_size]
#p_w_s_train = [np.array(x).astype(np.int32) for x in  p_word_ids[indices.T[1] != 0][0:train_data_size]]
#p_c_s_train = p_char_ids[indices.T[1] != 0][0:train_data_size]
#q_w_s_train = [np.array(x).astype(np.int32) for x in  q_word_ids[indices.T[1] != 0][0:train_data_size]]
#q_c_s_train = q_char_ids[indices.T[1] != 0][0:train_data_size]
#valid
#indices_valid = indices[indices.T[1] != 0][-valid_data_size:]
#p_w_s_valid = [np.array(x).astype(np.int32) for x in  p_word_ids[indices.T[1] != 0][-valid_data_size:]]
#p_c_s_valid = p_char_ids[indices.T[1] != 0][-valid_data_size:]
#q_w_s_valid = [np.array(x).astype(np.int32) for x in  q_word_ids[indices.T[1] != 0][-valid_data_size:]]
#q_c_s_valid = q_char_ids[indices.T[1] != 0][-valid_data_size:]
#if you wan to handle the problem which answer length is longer than 1
#train
#indices_train =  indices[indices.T[0] != indices.T[1]][0:train_data_size]
#p_w_s_train = [np.array(x).astype(np.int32) for x in  p_word_ids[indices.T[0] != indices.T[1]][0:train_data_size]]
#p_c_s_train = p_char_ids[indices.T[0] != indices.T[1]][0:train_data_size]
#q_w_s_train = [np.array(x).astype(np.int32) for x in  q_word_ids[indices.T[0] != indices.T[1]][0:train_data_size]]
#q_c_s_train = q_char_ids[indices.T[0] != indices.T[1]][0:train_data_size]
#valid
#indices_valid = indices[indices.T[0] != indices.T[1]][-valid_data_size:]
#p_w_s_valid = [np.array(x).astype(np.int32) for x in  p_word_ids[indices.T[0] != indices.T[1]][-valid_data_size:]]
#p_c_s_valid = p_char_ids[indices.T[0] != indices.T[1]][-valid_data_size:]
#q_w_s_valid = [np.array(x).astype(np.int32) for x in  q_word_ids[indices.T[0] != indices.T[1]][-valid_data_size:]]
#q_c_s_valid = q_char_ids[indices.T[0] != indices.T[1]][-valid_data_size:]
import time
import cupy as xp
output_size = 100
unit_size = 100
batch_size  =32
n_layers = 1
R_net_model = Classifier(R_net_Model_gpu(
    n_layers, vocab_size, char_vocab_size, unit_size, output_size, batch_size, cudnn = True))
chainer.serializers.load_npz("model_gpu2",R_net_model)
chainer.serializers.load_npz("model_gpu3",R_net_model)
#R_net_model.to_gpu()
optimizer = chainer.optimizers.Adam()
optimizer.setup(R_net_model)
first_time = time.time()
num_epoch  = 10
loss_sum_train_list = []
for epoch in range(4,num_epoch):
    R_net_model.to_gpu()
    loss_sum_train = 0
    loss_sum_valid = 0
    exact_match_score_sum = 0
    F1score_sum = 0
    random_index = np.random.permutation(train_data_size)
    p_w_s_train_random = np.array(p_w_s_train)[random_index]
    p_c_s_train_random = p_c_s_train[random_index]
    q_w_s_train_random = np.array(q_w_s_train)[random_index]
    q_c_s_train_random = q_c_s_train[random_index]
    indices_train_random = indices_train[random_index]
    for batch_index in range(0,train_data_size, batch_size)[0:-1]:
        p_w_s_batch = p_w_s_train_random[batch_index:batch_index + batch_size]
        p_c_s_batch = p_c_s_train_random[batch_index:batch_index + batch_size]
        q_w_s_batch = q_w_s_train_random[batch_index:batch_index + batch_size]
        q_c_s_batch= q_c_s_train_random[batch_index:batch_index + batch_size]
        R_net_model.cleargrads()
        print "forward", epoch, batch_index, time.time() - first_time
        loss = R_net_model(p_w_s_batch, p_c_s_batch, q_w_s_batch, q_c_s_batch, 
            indices_train_random[batch_index:batch_index + batch_size])
        print loss.data
        #print "backward", time.time() - first_time
        R_net_model.cleargrads()
        loss.backward()
        loss.unchain_backward()
        #print "update", time.time() - first_time
        optimizer.update()
        print "EM and F1_score(batch): ", output_f1_EM_score(
            indices_train_random[batch_index:batch_index + batch_size], 
            R_net_model, p_w_s_batch)
        loss_sum_train += chainer.cuda.to_cpu(loss.data) * len(p_c_s_batch)
        loss = 0
    print "train_sum_loss:", loss_sum_train/(train_data_size - train_data_size % batch_size)
    loss_sum_train_list.append(loss_sum_train/(train_data_size - train_data_size % batch_size))
    for batch_index in range(0,valid_data_size, batch_size)[0:-1]:
        p_w_s_batch = p_w_s_valid[batch_index:batch_index + batch_size]
        p_c_s_batch = p_c_s_valid[batch_index:batch_index + batch_size]
        q_w_s_batch = q_w_s_valid[batch_index:batch_index + batch_size]
        q_c_s_batch= q_c_s_valid[batch_index:batch_index + batch_size]
        with chainer.using_config('train', False):
            with chainer.no_backprop_mode():
                loss = R_net_model(p_w_s_batch, p_c_s_batch, q_w_s_batch, q_c_s_batch, 
                    indices_valid[batch_index:batch_index + batch_size])
                loss_sum_valid += chainer.cuda.to_cpu(loss.data) * len(p_c_s_batch)
            loss = 0
        EM, F1score = output_f1_EM_score(
            indices_valid[batch_index:batch_index + batch_size], R_net_model, p_w_s_batch)
        exact_match_score_sum += EM * batch_size
        F1score_sum += F1score * batch_size
        #print "EM and F1_score(batch): ", output_f1_EM_score(indices_batch, R_net_model, p_w_s_batch)
    print "valid:", time.time() - first_time
    print "loss:", loss_sum_valid/(valid_data_size - valid_data_size % batch_size), 
    print "EM:", exact_match_score_sum/(valid_data_size - valid_data_size % batch_size)
    print "F1:", F1score_sum/(valid_data_size - valid_data_size % batch_size)
    R_net_model.to_cpu()
    chainer.serializers.save_npz("model_gpu" + str(epoch), R_net_model, compression=True)

R_net_model = Classifier(R_net_Model_gpu(
    n_layers, vocab_size, char_vocab_size, unit_size, output_size, batch_size, cudnn = True))
chainer.serializers.load_npz("model_gpu1",R_net_model)




import time
import numpy as xp
output_size = 100
unit_size = 100
batch_size  =32
n_layers = 1
R_net_model = Classifier(R_net_Model_gpu(
    n_layers, vocab_size, char_vocab_size, unit_size, output_size, batch_size, cudnn = True))
#chainer.serializers.load_npz("model_gpu2",R_net_model)
chainer.serializers.load_npz("model_gpu3",R_net_model)
#R_net_model.to_gpu()
optimizer = chainer.optimizers.Adam()
optimizer.setup(R_net_model)
first_time = time.time()
num_epoch  = 5
loss_sum_train_list = []

def evaluate_score(R_net_model, p_w_s_valid,q_w_s_valid,
    p_c_s_valid ,q_c_s_valid,indices_valid,valid_data_size):
    import numpy as xp
    for epoch in range(0,1):
        loss_sum_valid = 0
        exact_match_score_sum = 0
        F1score_sum = 0
        for batch_index in range(0,valid_data_size, batch_size)[0:-1]:
            p_w_s_batch = p_w_s_valid[batch_index:batch_index + batch_size]
            p_c_s_batch = p_c_s_valid[batch_index:batch_index + batch_size]
            q_w_s_batch = q_w_s_valid[batch_index:batch_index + batch_size]
            q_c_s_batch= q_c_s_valid[batch_index:batch_index + batch_size]
            with chainer.using_config('train', False):
                with chainer.no_backprop_mode():
                    loss = R_net_model(p_w_s_batch, p_c_s_batch, q_w_s_batch, q_c_s_batch, 
                    indices_valid[batch_index:batch_index + batch_size])
                    loss_sum_valid += loss.data * len(p_c_s_batch)
                    loss = 0
            EM, F1score = output_f1_EM_score(
                indices_valid[batch_index:batch_index + batch_size], R_net_model, p_w_s_batch)
            exact_match_score_sum += EM * batch_size
            F1score_sum += F1score * batch_size
            print batch_index, ": EM and F1_score(batch): ", EM, F1score
    print "valid:", time.time() - first_time
    print "loss:", loss_sum_valid/(valid_data_size - valid_data_size % batch_size), 
    print "EM:", exact_match_score_sum/(valid_data_size - valid_data_size % batch_size)
    print "F1:", F1score_sum/(valid_data_size - valid_data_size % batch_size)
    return (loss_sum_valid/(valid_data_size - valid_data_size % batch_size), 
            exact_match_score_sum/(valid_data_size - valid_data_size % batch_size),
            F1score_sum/(valid_data_size - valid_data_size % batch_size))
    
    
result_1 = evaluate_score(R_net_model, p_w_s_valid,q_w_s_valid,
    p_c_s_valid ,q_c_s_valid,indices_valid,valid_data_size)




