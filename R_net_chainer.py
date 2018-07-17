import sys
#sys.path.append('/Users/tomoki/Mypython/R-net')
sys.path.append('./R-net')
from functools import wraps
import threading

from tensorflow.python.platform import tf_logging as logging

from params import Params
import numpy as np
import tensorflow as tf
from process import *
from sklearn.model_selection import train_test_split

import sys
from chainer import Variable, Chain
import chainer.links as L
import numpy as np
import chainer
from chainer import functions as F

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




class NStepBiLSTM(chainer.links.NStepLSTM):
    def __init__(self, n_layers, in_size, out_size, dropout, **kwargs):
        NStepLSTMBase.__init__(
            self, n_layers, in_size, out_size, dropout,
            use_bi_direction=True, **kwargs)


class BiRNN(chainer.Chain):
    def __init__(self, n_layer, n_vocab, n_units, dropout, cudnn):
        super(BiRNN, self).__init__(
            embed=L.EmbedID(n_vocab, n_units, ignore_label  = 0),
            l1=L.NStepBiGRU(n_layer, n_units, n_units,
                           dropout),
            l2=L.Linear(n_units, 10),
        )
    def __call__(self, hx, xs):
        xs = [self.embed(item) for item in xs]
        hy, ys = self.l1(hx, xs)
        return hy, ys

n_layers = 1
batch_size = 32
vocab_size = 2196018
char_vocab_size = 95

p_w_s_batch = [np.array(x).astype(np.int32) for x in  p_word_ids[0:32]]
p_c_s_batch = p_char_ids[0:32]


model = BiRNN(n_layers, vocab_size, 100, 0.5, False)
xpws = [model.embed(item) for item in p_w_s_batch]
#hx = Variable(np.zeros((n_layers * 2, batch_size,100)).astype(np.float32))
#cx = Variable(np.zeros((n_layers * 2, batch_size,100)).astype(np.float32))
#hy, cy, ys = model.l1(hx, cx, xpws)

model2 = BiRNN(1, char_vocab_size, 50, 0.5, False)
xpcs = [F.concat(
			model2(
				Variable(np.zeros((n_layers * 2, len(items),50)).astype(np.float32)), 
				[np.array(item).astype(np.int32) for item in items])[0], axis = 1)
				for items in p_c_s_batch]

concat_input = [F.concat((xpcs[index], xpws[index])) for index in xrange(batch_size)]
model3 = BiRNN(1, char_vocab_size, 2 * 100, 0.5, False)
hu_p,u_ps = model3.l1(
			Variable(np.zeros((n_layers * 2, batch_size,2 * 100)).astype(np.float32)),
			concat_input)


q_w_s_batch = [np.array(x).astype(np.int32) for x in  q_word_ids[0:32]]
q_c_s_batch = q_char_ids[0:32]


modelq_1 = BiRNN(n_layers, vocab_size, 100, 0.5, False)
xqws = [modelq_1.embed(item) for item in q_w_s_batch]

modelq_2 = BiRNN(1, char_vocab_size, 50, 0.5, False)
xqcs = [F.concat(
			modelq_2( 
			Variable(np.zeros((n_layers * 2, len(items),50)).astype(np.float32)), 
			[np.array(item).astype(np.int32) for item in items])[0], axis = 1)
			for items in q_c_s_batch]

concat_input = [F.concat((xqcs[index], xqws[index])) for index in xrange(batch_size)]
modelq_3 = BiRNN(1, char_vocab_size, 2 * 100, 0.5, False)
_, u_qs = modelq_3.l1(
			Variable(np.zeros((n_layers * 2, batch_size,2 * 100)).astype(np.float32)),
			concat_input)


#Gated Attention Based RNN
class GARNN(chainer.Chain):
    def __init__(self, n_layer, input_dim, n_units, dropout = 0.2, cudnn = False):
        super(GARNN, self).__init__(
            W_up=L.Linear(input_dim, n_units),
            W_vp=L.Linear(n_units, n_units),
            W_uq = L.Linear(input_dim, n_units),
            W_v = L.Linear(n_units,1),
            W_g = L.Linear(input_dim * 2, input_dim * 2),
            W_gru = L.StatefulGRU(input_dim * 2, n_units)
        )
    #def __call__(self, hx, xs):
        #s_jt = 
        #return hy, ys

model_GARNN = GARNN(1, u_qs[0].shape[1], 100)
model_GARNN.W_gru.h = Variable(np.zeros((batch_size, 100)).astype(np.float32))
WqUqj = F.stack([model_GARNN.W_uq(uq) for uq in u_qs])
WvVp_batch = F.transpose(F.stack([F.transpose(model_GARNN.W_vp(model_GARNN.W_gru.h))] * 37, axis = 1))
WpUps = F.stack([model_GARNN.W_up(up) for up in u_ps])
vtp_list = []
for index in range(WpUps.shape[1]):
	WqUqj = F.stack([model_GARNN.W_uq(uq) for uq in u_qs])
	s_tj = F.batch_matmul(
		(F.tanh(WqUqj + WvVp_batch + F.stack([WpUps[:,index,:]] * WqUqj.shape[1], axis = 1))),
			F.broadcast_to(model_GARNN.W_v.W, [batch_size, model_GARNN.W_v.W.shape[1]])
			).reshape(batch_size, 37)
	at = F.softmax(s_tj)
	ct = F.batch_matmul(F.stack(u_qs), at, transa = True).reshape(batch_size, u_qs[0].shape[1])
	gt_batch = F.sigmoid(model_GARNN.W_g(F.concat([F.stack(u_ps)[:,index,:], ct])))
	g_input = gt_batch * F.concat([F.stack(u_ps)[:,index,:], ct])
	vtp_list.append(model_GARNN.W_gru(g_input))

#self-matching attention
class SMARNN(chainer.Chain):
    def __init__(self, n_layer, input_dim, n_units, dropout = 0.2, cudnn = False):
        super(SMARNN, self).__init__(
            W_vp= L.Linear(input_dim, n_units),
            W_vpa =L.Linear(n_units, n_units),
            W_v = L.Linear(n_units,1), 
            W_f_gru = L.StatefulGRU(input_dim * 2, n_units),
            W_b_gru = L.StatefulGRU(input_dim * 2, n_units)
        )

model_SMARNN = SMARNN(1, 100, 100)
WpVp = F.stack([model_SMARNN.W_vp(vtp) for vtp in vtp_list], axis = 1)
WpVp2 = F.stack([model_SMARNN.W_vpa(vtp) for vtp in vtp_list], axis = 1)

htp_new_f_list = []
for index in range(WpHp2.shape[1]):
	s_tj = F.batch_matmul(
		(F.tanh(WpVp + F.stack([WpVp2[:,index,:]] * WpVp.shape[1], axis = 1))),
			F.broadcast_to(model_SMARNN.W_v.W, [batch_size, model_SMARNN.W_v.W.shape[1]])
			).reshape(batch_size, WpVp.shape[1])
	at = F.softmax(s_tj)
	ct = F.batch_matmul(F.stack(vtp_list, axis = 1), at, transa = True).reshape(batch_size, vtp_list[0].shape[1])
	htp_new_f_list.append(model_SMARNN.W_f_gru(F.concat([WpVp2[:,index,:],ct])))

htp_new_b_list = []
for index in range(WpVp2.shape[1])[-1::-1]:
	s_tj = F.batch_matmul(
		(F.tanh(WpVp + F.stack([WpVp2[:,index,:]] * WpVp.shape[1], axis = 1))),
			F.broadcast_to(model_SMARNN.W_v.W, [batch_size, model_SMARNN.W_v.W.shape[1]])
			).reshape(batch_size, WpVp.shape[1])
	at = F.softmax(s_tj)
	ct = F.batch_matmul(F.stack(vtp_list, axis = 1), at, transa = True).reshape(batch_size, vtp_list[0].shape[1])
	htp_new_b_list.append(model_SMARNN.W_b_gru(F.concat([WpVp2[:,index,:],ct])))

hps = F.transpose(F.concat([F.transpose(F.stack(htp_new_f_list, axis = 1)), 
			    F.transpose(F.stack(htp_new_b_list[-1::-1], axis = 1))
			    ], axis = 0))
#hps in [batch * sentence length * dimention]

#Output Layer
class OutputLayer(chainer.Chain):
    def __init__(self, n_layer, input_dim, n_units, dropout = 0.2, cudnn = False):
        super(OutputLayer, self).__init__(
            Wp_h = L.Linear(input_dim, n_units),
            Wa_h =L.Linear(n_units, n_units),
            W_v = L.Linear(n_units,1), 
            W_vQVQ = L.Linear(100, 37), 
            W_f_gru = L.StatefulGRU(input_dim, n_units)
        )

model_OL = OutputLayer(1, 200, u_qs[0].shape[1])
s_j = F.batch_matmul(
		F.tanh(WqUqj + F.stack([model_OL.W_vQVQ.W] * batch_size, axis = 0)),
		F.broadcast_to(model_OL.W_v.W, [batch_size, model_OL.W_v.W.shape[1]]),
		).reshape(batch_size, WqUqj.shape[1])
a_i = F.softmax(s_j)
rQ = F.batch_matmul(F.stack(u_qs), a_i, transa = True).reshape(batch_size, u_qs[0].shape[1])
model_OL.W_f_gru.h =  rQ

WpHp = F.stack([model_OL.Wp_h(hp) for hp in hps])
hta_new_b_list = []
for index in range(2):
	s_tj = F.batch_matmul(
		F.tanh(WpHp + F.stack([model_OL.W_f_gru.h] * WpHp.shape[1], axis = 1)),
		F.broadcast_to(model_OL.W_v.W, [batch_size, model_OL.W_v.W.shape[1]])
			).reshape(batch_size, WpHp.shape[1])
	at = F.softmax(s_tj)
	pt = F.argmax(at, axis = 1)
	ct = F.batch_matmul(hps, at).reshape(batch_size, hps.shape[1])
	h_new = model_OL.W_f_gru(ct)
	hta_new_b_list.append(s_tj)

	