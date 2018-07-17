import sys
#sys.path.append('/Users/tomoki/Mypython/R-net')
sys.path.append('./R-net')
from functools import wraps
import threading

from tensorflow.python.platform import tf_logging as logging

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


n_layers = 1
batch_size = 32
vocab_size = 2196018
char_vocab_size = 95
unit_size = 50
p_w_s_batch = [np.array(x).astype(np.int32) for x in  p_word_ids[0:batch_size]]
p_c_s_batch = p_char_ids[0:batch_size]
q_w_s_batch = [np.array(x).astype(np.int32) for x in  q_word_ids[0:batch_size]]
q_c_s_batch = q_char_ids[0:batch_size]

glove = np.memmap("../" + Params.data_dir + "glove.np",dtype = np.float32,mode = "r")
glove = np.asarray(np.reshape(glove,(Params.vocab_size,300)))
embed_w = L.EmbedID(vocab_size, 300, initialW = glove, ignore_label  = 0)

dict_ = pickle.load(gzip.open("../" + Params.data_dir + "dictionary.pkl.gz","r"))
#indices_batch = indices_train_random[batch_index:batch_index + batch_size]
#indices_batch = indices_valid[batch_index:batch_index + batch_size]


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

class MGRU(chainer.Chain):
    def __init__(self, n_layer, input_dim, n_units, batch_size, dropout = 0.2, cudnn = False):
        super(MGRU, self).__init__(
            W_up=L.Linear(input_dim, n_units/2),
            W_vp=L.Linear(n_units/2, n_units/2),
            W_uq = L.Linear(input_dim, n_units/2),
            W_v = L.Linear(n_units/2,1),
            #W_g = L.Linear(input_dim * 2, input_dim * 2),
            W_gru_f = L.StatefulGRU(input_dim * 2, n_units/2),
            W_gru_b = L.StatefulGRU(input_dim * 2, n_units/2)
            )
        self.batch_size = batch_size
        self.n_units = n_units
    def __call__(self, u_qs, u_ps):
        self.W_gru_f.h = Variable(np.zeros((self.batch_size, self.n_units/2)).astype(np.float32))
        self.W_gru_b.h = Variable(np.zeros((self.batch_size, self.n_units/2)).astype(np.float32))
        WqUqj = F.stack([self.W_uq(uq) for uq in u_qs])
        #WvVp_batch = F.transpose(F.stack([F.transpose(self.W_vp(self.W_gru_f.h))] * q_max_word, axis = 1))
        WpUps = F.stack([self.W_up(up) for up in u_ps])
        batch_ups = F.stack(u_ps)
        batch_uqs = F.stack(u_qs)
        batch_W_v = F.broadcast_to(self.W_v.W, [self.batch_size, self.W_v.W.shape[1]])
        vtp_list_f = []
        for index in range(WpUps.shape[1]):
            WvVp_batch = F.transpose(F.broadcast_to(self.W_vp(self.W_gru_f.h), 
                [WqUqj.shape[1], WqUqj.shape[0], WqUqj.shape[2]]),(1,0,2))
            s_tj = F.batch_matmul(
                (F.tanh(WqUqj + WvVp_batch + 
                    #F.stack([WpUps[:,index,:]] * (WqUqj.shape[1]), axis = 1)
                    F.transpose(F.broadcast_to(WpUps[:,index,:], 
                        [WqUqj.shape[1],WqUqj.shape[0], WqUqj.shape[2]]), (1,0,2))
                )),
                batch_W_v).reshape(self.batch_size, q_max_word)
            at = F.softmax(s_tj)
            ct = F.batch_matmul(batch_uqs, at, transa = True).reshape(self.batch_size, u_qs[0].shape[1])
            vtp_list_f.append(self.W_gru_f(F.concat([batch_ups[:,index,:], ct])))
        vtp_list_b = []
        #WvVp_batch_b = F.transpose(F.stack([F.transpose(self.W_vp(self.W_gru_b.h))] * q_max_word, axis = 1))
        for index in range(WpUps.shape[1])[-1::-1]:
            WvVp_batch_b = F.transpose(F.broadcast_to(self.W_vp(self.W_gru_b.h), 
                [WqUqj.shape[1], WqUqj.shape[0], WqUqj.shape[2]]),(1,0,2))   
            s_tj = F.batch_matmul(
                (F.tanh(WqUqj[:,-1::-1,:] + WvVp_batch_b + 
                   #F.stack([WpUps[:,index,:]] * (WqUqj.shape[1]), axis = 1)
                   F.transpose(F.broadcast_to(WpUps[:,index,:], 
                    [WqUqj.shape[1],WqUqj.shape[0], WqUqj.shape[2]]), (1,0,2))
                )),
                batch_W_v).reshape(self.batch_size, q_max_word)
            at = F.softmax(s_tj)
            ct = F.batch_matmul(batch_uqs[:,-1::-1,:], at, transa = True).reshape(self.batch_size, u_qs[0].shape[1])
            vtp_list_b.append(self.W_gru_b(F.concat([batch_ups[:,index,:], ct])))
        #WpUps = 0
        #WvVp_batch = 0
        #WqUqj  = 0
        return F.concat([F.stack(vtp_list_f, axis = 1), 
                         F.stack(vtp_list_b[-1::-1], axis = 1)], axis = 2)
        
#Match_gru_model.predictor.MGRU
#hps in [batch * sentence length * dimention]

#Output Layer
class PointLayer(chainer.Chain):
    def __init__(self, n_layer, input_dim, n_units, batch_size, dropout = 0.2, cudnn = False):
        super(PointLayer, self).__init__(
            Wp_h = L.Linear(input_dim, n_units),
            Wa_h =L.Linear(n_units, n_units),
            W_v = L.Linear(input_dim,1), 
            #W_vQVQ = L.Linear(input_dim, q_max_word), 
            #W_rq = L.Linear(input_dim,n_units),
            W_f_gru = L.StatefulGRU(input_dim, n_units))
        self.n_units = n_units
        self.batch_size = batch_size
    def __call__(self, hps, put_zero = False):
        #s_j = F.batch_matmul(
            #F.tanh(WqUqj), 
            #F.broadcast_to(self.W_v.W, [self.batch_size, self.W_v.W.shape[1]]),
            #).reshape(self.batch_size, WqUqj.shape[1])
        #a_i = F.softmax(s_j)
        #rQ = F.batch_matmul(F.stack(u_qs), a_i, transa = True).reshape(self.batch_size, u_qs[0].shape[1])
        #self.W_f_gru.h =  rQ
        self.W_f_gru.h =  Variable(np.zeros((self.batch_size, self.n_units)).astype(np.float32))
        WpHp = F.stack([self.Wp_h(hp) for hp in hps])
        hta_new_b_list = []
        for index in range(2):
            s_tj = F.batch_matmul(
            F.tanh(WpHp + F.stack([self.W_f_gru.h] * WpHp.shape[1], axis = 1)),
            F.broadcast_to(self.W_v.W, [self.batch_size, self.W_v.W.shape[1]])
                ).reshape(self.batch_size, WpHp.shape[1])
            #if ((put_zero == True) & (index == 1)):
                #mask = np.ones()
                #for r_index, pt_index in enumerate(pt):
                    #mask[r][:pt_index] = 0
                #s_tj = s_tj * Variable(mask)
            hta_new_b_list.append(s_tj)
            at = F.softmax(s_tj)
            pt = F.argmax(at, axis = 1)
            ct = F.batch_matmul(hps, at, transa = True).reshape(self.batch_size, self.n_units)
            hta_new = self.W_f_gru(ct)
        return hta_new_b_list


class Match_GRU_Model(chainer.Chain):
    def __init__(self, n_layer, vocab_size, char_vocab_size, unit_size, output_size, 
    	batch_size = 32,dropout = 0.2, cudnn = False):
        super(Match_GRU_Model, self).__init__(
            model2 = BiRNN(1, char_vocab_size, 50, 0.5, False),
            model3 = BiRNN_2(1, char_vocab_size, unit_size, 0.5, False),
            model_MGRU = MGRU(1, unit_size, unit_size, batch_size),
            #model_SMARNN = SMARNN(1, unit_size, unit_size/2, batch_size),
            model_OL = PointLayer(1, unit_size, output_size, batch_size)
            )
        self.unit_size = unit_size
        self.batch_size = batch_size
    def __call__(self, p_w_s_batch, p_c_s_batch, q_w_s_batch, q_c_s_batch):
        with chainer.no_backprop_mode():
            xpws = [Variable(F.forget(embed_w,item).data) for item in p_w_s_batch]
        xpcs = [F.concat(
            self.model2(
                Variable(np.zeros((n_layers * 2, len(items),50)).astype(np.float32)), 
                [Variable(np.array(item).astype(np.int32)) for item in items]), axis = 1)
                for items in p_c_s_batch]
        concat_input_p = [F.concat((xpcs[index], xpws[index])) for index in xrange(self.batch_size)]
        _,u_ps = self.model3.l1(
            Variable(np.zeros((n_layers * 2, self.batch_size, self.unit_size/2)).astype(np.float32)),
            concat_input_p)
        with chainer.no_backprop_mode():
            xqws = [Variable(F.forget(embed_w, item).data) for item in q_w_s_batch]
        xqcs = [F.concat(
                self.model2( 
                Variable(np.zeros((n_layers * 2, len(items),50)).astype(np.float32)), 
                [Variable(np.array(item).astype(np.int32)) for item in items]), axis = 1)
        for items in q_c_s_batch]
        concat_input_q = [F.concat((xqcs[index], xqws[index])) for index in xrange(self.batch_size)]
        _, u_qs = self.model3.l1(
            Variable(np.zeros((n_layers * 2, self.batch_size, self.unit_size/2)).astype(np.float32)),
            concat_input_q)
        #return u_qs, u_ps
        hps = self.model_MGRU(u_qs, u_ps)
        #return vtp_list
        #hps = self.model_SMARNN(vtp_list)
        hta_new_b_list = self.model_OL(hps)
        return hta_new_b_list



class Classifier(L.Classifier):
    def __call__(self, p_w_s_batch, p_c_s_batch, q_w_s_batch, q_c_s_batch, indices_batch):
        self.y = None
        self.loss = None
        self.accuracy = None
        batch_y_s = indices_batch.T[0]
        batch_y_f = indices_batch.T[1]
        self.y = self.predictor(p_w_s_batch, p_c_s_batch, q_w_s_batch, q_c_s_batch)
        self.loss = self.lossfun(self.y[0], batch_y_s) + self.lossfun(self.y[1], batch_y_f)
        chainer.reporter.report({'loss': self.loss}, self)
        #exact match
        answer_s = np.argmax(self.y[0].data, axis = 1)
        answer_f = np.argmax(self.y[1].data, axis = 1)
        self.exact_match_score = np.float(((answer_s == batch_y_s) & (answer_f == batch_y_f)).sum())/answer_s.shape[0]
        return self.loss

def output_f1_EM_score(indices_batch, model, p_w_s_batch):
    batch_y_s = indices_batch.T[0]
    batch_y_f = indices_batch.T[1]
    answer_s = np.argmax(model.y[0].data, axis = 1)
    answer_f = np.argmax(model.y[1].data, axis = 1)
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


glove = 0
unit_size = 100
output_size = 100
batch_size = 32
#test
Match_gru_model = Classifier(Match_GRU_Model(
    n_layers, vocab_size, char_vocab_size, unit_size, output_size, batch_size))

#u_qs, u_ps = Match_gru_model.predictor(p_w_s_batch, p_c_s_batch, q_w_s_batch, q_c_s_batch)
#vtp_list = Match_gru_model.predictor(p_w_s_batch, p_c_s_batch, q_w_s_batch, q_c_s_batch)
#hps = F.stack(vtp_list)
optimizer = chainer.optimizers.Adam()
optimizer.setup(Match_gru_model)
#output = R_net_model.predictor(p_w_s_batch, p_c_s_batch, q_w_s_batch, q_c_s_batch)
loss = Match_gru_model(p_w_s_batch, p_c_s_batch, q_w_s_batch, q_c_s_batch, indices[0:batch_size])
Match_gru_model.cleargrads()
loss.backward()
loss.unchain_backward()
#Match_gru_model.predictor.model_MGRU.W_vp.W.grad
optimizer.update()

#train_data_size = 3000
#valid_data_size = 2000
#train
indices_train =  indices[indices.T[1] != 0][0:-1]
p_w_s_train = [np.array(x).astype(np.int32) for x in  p_word_ids[indices.T[1] != 0][0:-1]]
p_c_s_train = p_char_ids[indices.T[1] != 0][0:-1]
q_w_s_train = [np.array(x).astype(np.int32) for x in  q_word_ids[indices.T[1] != 0][0:-1]]
q_c_s_train = q_char_ids[indices.T[1] != 0][0:-1]
train_data_size = len(indices_train)

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
Match_gru_model = Classifier(Match_GRU_Model(
    n_layers, vocab_size, char_vocab_size, unit_size, output_size, batch_size))

#chainer.serializers.load_npz("Match_GRU_model_0", Match_gru_model)
optimizer = chainer.optimizers.Adam()
optimizer.setup(Match_gru_model)
first_time = time.time()
num_epoch  = 5
for epoch in range(0,num_epoch):
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
        Match_gru_model.cleargrads()
        print "forward(MGRU)", epoch, batch_index, time.time() - first_time
        with chainer.using_config('train', True):
            loss = Match_gru_model(p_w_s_batch, p_c_s_batch, q_w_s_batch, q_c_s_batch, 
                indices_train_random[batch_index:batch_index + batch_size])
        print loss.data
        #print "backward", time.time() - first_time
        Match_gru_model.cleargrads()
        loss.backward()
        loss.unchain_backward()
        #print "update", time.time() - first_time
        optimizer.update()
        print "EM and F1_score(batch): ", output_f1_EM_score(
            indices_train_random[batch_index:batch_index + batch_size], 
            Match_gru_model, p_w_s_batch)
        loss_sum_train += loss.data * len(p_c_s_batch)
        loss = 0
    print "train_sum_loss:", loss_sum_train/(train_data_size - train_data_size % batch_size)
    for batch_index in range(0,valid_data_size, batch_size)[0:-1]:
        p_w_s_batch = np.array(p_w_s_valid)[batch_index:batch_index + batch_size]
        p_c_s_batch = p_c_s_valid[batch_index:batch_index + batch_size]
        q_w_s_batch = np.array(q_w_s_valid)[batch_index:batch_index + batch_size]
        q_c_s_batch= q_c_s_valid[batch_index:batch_index + batch_size]
        with chainer.using_config('train', False):
            with chainer.no_backprop_mode():
                loss = Match_gru_model(p_w_s_batch, p_c_s_batch, q_w_s_batch, q_c_s_batch, 
                    indices_valid[batch_index:batch_index + batch_size])
                loss_sum_valid += loss.data * len(p_c_s_batch)
                loss = 0
        EM, F1score = output_f1_EM_score(
            indices_valid[batch_index:batch_index + batch_size], Match_gru_model, p_w_s_batch)
        exact_match_score_sum += EM * batch_size
        F1score_sum += F1score * batch_size
        #print "EM and F1_score(batch): ", output_f1_EM_score(indices_batch, Match_gru_model, p_w_s_batch)
    print "valid:", time.time() - first_time
    print "loss:", loss_sum_valid/(valid_data_size - valid_data_size % batch_size), 
    print "EM:", exact_match_score_sum/(valid_data_size - valid_data_size % batch_size)
    print "F1:", F1score_sum/(valid_data_size - valid_data_size % batch_size)
    chainer.serializers.save_npz("Match_GRU_model_" + str(epoch), Match_gru_model, compression=True)
