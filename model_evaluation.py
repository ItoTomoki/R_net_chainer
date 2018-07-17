import sys
#sys.path.append('/Users/tomoki/Mypython/R-net')
sys.path.append('./R-net')
from functools import wraps
import threading
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
from layers import *
import time
import numpy as xp

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

# Get max length to pad
p_max_word = 200
p_max_char = 37
q_max_word = 200
q_max_char = 37



n_layers = 1
batch_size = 32
vocab_size = 2196018
char_vocab_size = 95

indices_valid = load_target("../data/devset/" + Params.target_dir)
print("Loading question data...")
q_word_ids_valid, _ = load_word("../data/devset/" + Params.q_word_dir)
q_char_ids_valid, q_char_len_valid, q_word_len_valid = load_char("../data/devset/" + Params.q_chars_dir)

# Load passage data
print("Loading passage data...")
p_word_ids_valid, _ = load_word("../data/devset/" + Params.p_word_dir)
p_char_ids_valid, p_char_len_valid, p_word_len_valid = load_char("../data/devset/" + Params.p_chars_dir)

# pad_data
print("Preparing evaluation data...")
p_w_s_valid = pad_data(p_word_ids_valid,p_max_word)
q_w_s_valid = pad_data(q_word_ids_valid,q_max_word)
p_c_s_valid = pad_char_data(p_char_ids_valid,p_max_char,p_max_word)
q_c_s_valid = pad_char_data(q_char_ids_valid,q_max_char,q_max_word)
indices_valid = np.reshape(np.asarray(indices_valid,np.int32),(-1,2))
valid_data_size = len(indices_valid)


output_size = 100
unit_size = 100
batch_size  =32
n_layers = 1
Match_gru_model = Classifier(Match_GRU_Model(
    n_layers, vocab_size, char_vocab_size, unit_size, output_size, batch_size))

chainer.serializers.load_npz("Match_GRU_model_1",Match_gru_model)
#R_net_model.to_gpu()
#optimizer = chainer.optimizers.Adam()
#optimizer.setup(Match_gru_model)
first_time = time.time()
loss_sum_train_list = []
result_1 = evaluate_score(Match_gru_model, p_w_s_valid,q_w_s_valid,
    p_c_s_valid ,q_c_s_valid,indices_valid,valid_data_size)

