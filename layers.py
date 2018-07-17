import sys
sys.path.append('/Users/tomoki/Mypython/R-net')
from functools import wraps
import evaluate
import threading
from params import Params
import numpy as np
from process import *
from chainer import Variable, Chain
import chainer.links as L
import chainer
from chainer import functions as F

p_max_word = 200
p_max_char = 37
q_max_word = 200
q_max_char = 37
n_layers = 1
batch_size = 32
vocab_size = 2196018
char_vocab_size = 95

glove = np.memmap("../" + Params.data_dir + "glove.np",dtype = np.float32,mode = "r")
glove = np.asarray(np.reshape(glove,(Params.vocab_size,300)))
glove_c = np.memmap("../" + Params.data_dir + "glove_char.np",dtype = np.float32,mode = "r")
glove_c = np.asarray(np.reshape(glove_c,(char_vocab_size,300)))
embed_w = L.EmbedID(vocab_size, 300, initialW = glove, ignore_label  = 0)
embed_w_c = L.EmbedID(char_vocab_size, 300, initialW = glove_c, ignore_label  = 0)

dict_ = pickle.load(gzip.open("../" + Params.data_dir + "dictionary.pkl.gz","r"))

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
        if cudnn == False:
            import numpy as xp
        else:
            import cupy as xp
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
        if cudnn == True:
            import cupy as xp
        else:
            import numpy as xp
    def __call__(self, u_qs, u_ps):
        self.W_gru_f.h = Variable(xp.zeros((self.batch_size, self.n_units/2)).astype(np.float32))
        self.W_gru_b.h = Variable(xp.zeros((self.batch_size, self.n_units/2)).astype(np.float32))
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
        if cudnn == True:
            import cupy as xp
        else:
            import numpy as xp
    def __call__(self, hps, put_zero = False):
        #s_j = F.batch_matmul(
            #F.tanh(WqUqj), 
            #F.broadcast_to(self.W_v.W, [self.batch_size, self.W_v.W.shape[1]]),
            #).reshape(self.batch_size, WqUqj.shape[1])
        #a_i = F.softmax(s_j)
        #rQ = F.batch_matmul(F.stack(u_qs), a_i, transa = True).reshape(self.batch_size, u_qs[0].shape[1])
        #self.W_f_gru.h =  rQ
        self.W_f_gru.h =  Variable(xp.zeros((self.batch_size, self.n_units)).astype(np.float32))
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
            model2 = BiRNN_1(1, 300, 50, dropout, cudnn = cudnn),
            model3 = BiRNN_2(1, char_vocab_size, unit_size, dropout, cudnn),
            model_MGRU = MGRU(1, unit_size, unit_size, batch_size),
            #model_SMARNN = SMARNN(1, unit_size, unit_size/2, batch_size),
            model_OL = PointLayer(1, unit_size, output_size, batch_size, dropout, cudnn)
            )
        self.unit_size = unit_size
        self.batch_size = batch_size
        if cudnn == False:
            import numpy as xp
        else:
            import cupy as xp
    def __call__(self, p_w_s_batch, p_c_s_batch, q_w_s_batch, q_c_s_batch):
        with chainer.no_backprop_mode():
            xpws = [Variable(xp.array(F.forget(embed_w,item).data)) for item in p_w_s_batch]
        xpcs = [F.concat(
            self.model2(
                Variable(xp.zeros((n_layers * 2, len(items),50)).astype(np.float32)), 
                [item for item in items]), axis = 1)
                for items in p_c_s_batch]
        concat_input_p = [F.concat((xpcs[index], xpws[index])) for index in xrange(self.batch_size)]
        _,u_ps = self.model3.l1(
            Variable(xp.zeros((n_layers * 2, self.batch_size, self.unit_size/2)).astype(np.float32)),
            concat_input_p)
        with chainer.no_backprop_mode():
            xqws = [Variable(xp.array(F.forget(embed_w, item).data)) for item in q_w_s_batch]
        xqcs = [F.concat(
                self.model2( 
                Variable(xp.zeros((n_layers * 2, len(items),50)).astype(np.float32)), 
                [item for item in items]), axis = 1)
        for items in q_c_s_batch]
        concat_input_q = [F.concat((xqcs[index], xqws[index])) for index in xrange(self.batch_size)]
        _, u_qs = self.model3.l1(
            Variable(xp.zeros((n_layers * 2, self.batch_size, self.unit_size/2)).astype(np.float32)),
            concat_input_q)
        #return u_qs, u_ps
        hps = self.model_MGRU(u_qs, u_ps)
        #return vtp_list
        #hps = self.model_SMARNN(vtp_list)
        hta_new_b_list = self.model_OL(hps)
        return hta_new_b_list

class R_net_Model_gpu(chainer.Chain):
    def __init__(self, n_layer, vocab_size, char_vocab_size, unit_size, output_size, 
        batch_size = 32,dropout = 0.2, cudnn = False):
        super(R_net_Model_gpu, self).__init__(
            model2 = BiRNN_1(1, 300, 50, dropout, cudnn = cudnn),
            model3 = BiRNN_2(1, char_vocab_size, unit_size, dropout, cudnn = cudnn),
            #modelq_2 = BiRNN(1, char_vocab_size, 100, 0.5, False),
            #modelq_3 = BiRNN(1, char_vocab_size, 2 * 100, 0.5, False),
            model_GARNN = GARNN(1, unit_size, unit_size, batch_size, cudnn = cudnn),
            model_SMARNN = SMARNN(1, unit_size, unit_size/2, batch_size, cudnn = cudnn),
            model_OL = OutputLayer(1, unit_size, output_size, batch_size, cudnn = cudnn)
            )
        self.unit_size = unit_size
        self.batch_size = batch_size
        if cudnn == False:
            import numpy as xp
        else:
            import cupy as xp
    def __call__(self, p_w_s_batch, p_c_s_batch, q_w_s_batch, q_c_s_batch):
        with chainer.no_backprop_mode():
            xpws = [Variable(xp.array(F.forget(embed_w,item).data)) for item in p_w_s_batch]
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
        xqcs = [F.concat(
                self.model2( 
                Variable(xp.zeros((n_layers * 2, len(items),50)).astype(np.float32)), 
                [item for item in items]), axis = 1)
        for items in q_c_s_batch]
        concat_input = [F.concat((xqcs[index], xqws[index])) for index in xrange(self.batch_size)]
        _, u_qs = self.model3.l1(
            Variable(xp.zeros((n_layers * 2, self.batch_size, self.unit_size/2)).astype(np.float32)),
            concat_input)
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


def output_f1_EM_score(indices_batch, R_net_model, p_w_s_batch, dict_ = dict_):
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

