# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 15:22:11 2018

@author: ChoiHongSeok
"""

import function_collection as fc
import custom_collection as cc

import os
import numpy as np
import time
import theano
import theano.tensor as T
import lasagne

import mkl
mkl.get_max_threads()

from lasagne.regularization import regularize_layer_params_weighted, l2

""" Hyperparameter """
epochs = 10
batch_unit = 25
init_lr = 0.0002
decay = 0.9
lamda = 0.0005
cut_len = 40

""" Data preparation """
current_path = os.getcwd()
data_path = current_path+'/data/'
word_w2v_file = 'emW_5987_w2v.bin'

print current_path
print data_path
print "data processing..."

mod_vocab, w_freq, w2i, i2w, max_len, emW = fc.load_all(data_path + word_w2v_file)

tst_dict = fc.read_data(data_path+'preprocessed_tst.txt')
dev_dict = fc.read_data(data_path+'preprocessed_dev.txt')
trn_dict = fc.read_data(data_path+'preprocessed_trn.txt')

trn_mtrx_dict = fc.data_gen(dev_dict, i2w, w2i, max_len, cut_len, batch_unit=batch_unit, ispermute=False)
dev_mtrx_dict = fc.data_gen(dev_dict, i2w, w2i, max_len, cut_len, batch_unit=len(dev_dict['id']), ispermute=False) # 316
tst_mtrx_dict = fc.data_gen(tst_dict, i2w, w2i, max_len, cut_len, batch_unit=len(tst_dict['id']), ispermute=False) # 444

""" build network """
L = lasagne.layers
SH = lasagne.layers.get_output_shape
OUT = lasagne.layers.get_output
NL = lasagne.nonlinearities

def layer_LSTM_share(r_word_lstm,l_word_lstm):
    r_word_lstm.params = l_word_lstm.params 
    r_word_lstm.cell_init = l_word_lstm.cell_init
    r_word_lstm.hid_init = l_word_lstm.hid_init 
    r_word_lstm.b_cell = l_word_lstm.b_cell
    r_word_lstm.b_forgetgate = l_word_lstm.b_forgetgate 
    r_word_lstm.b_ingate = l_word_lstm.b_ingate 
    r_word_lstm.b_outgate = l_word_lstm.b_outgate 
    r_word_lstm.W_cell_to_forgetgate=l_word_lstm.W_cell_to_forgetgate
    r_word_lstm.W_cell_to_ingate=l_word_lstm.W_cell_to_ingate
    r_word_lstm.W_cell_to_outgate=l_word_lstm.W_cell_to_outgate
    r_word_lstm.W_hid_to_cell=l_word_lstm.W_hid_to_cell
    r_word_lstm.W_hid_to_forgetgate=l_word_lstm.W_hid_to_forgetgate
    r_word_lstm.W_hid_to_ingate=l_word_lstm.W_hid_to_ingate
    r_word_lstm.W_hid_to_outgate=l_word_lstm.W_hid_to_outgate
    r_word_lstm.W_in_to_cell=l_word_lstm.W_in_to_cell
    r_word_lstm.W_in_to_forgetgate=l_word_lstm.W_in_to_forgetgate
    r_word_lstm.W_in_to_ingate=l_word_lstm.W_in_to_ingate
    r_word_lstm.W_in_to_outgate=l_word_lstm.W_in_to_outgate

def layer_encoder_clone(r_em,r_in_mask,input_layer,layer_encoder):
    input_layers = L.get_all_layers(input_layer)
    clone_layer = layer_encoder(r_em,r_in_mask)
    clone_layers = L.get_all_layers(clone_layer)
    for i in xrange(len(input_layers)):
        try:        
            if layer_check(input_layers[i])=='LSTM':
                layer_LSTM_share(clone_layers[i],input_layers[i])
            elif layer_check(input_layers[i])=='W':
                clone_layers[i].params = input_layers[i].params
                clone_layers[i].W = input_layers[i].W
            else:
                clone_layers[i].params = input_layers[i].params
                clone_layers[i].W = input_layers[i].W
                clone_layers[i].b = input_layers[i].b
        except AttributeError:
            continue
    return clone_layer

def layer_check(input_layer):
    input_class = input_layer.__class__
    if input_class==L.LSTMLayer:
        return 'LSTM'
    elif input_class==L.EmbeddingLayer:
        return 'W'
    else:
        return 'W&b'

""" build the ESIM (Chen et al., 2017) network """
print "build ESIM..."
IS_STATIC="static"
lstm_units = 300

"sentence 1"
l_in_sent = L.InputLayer(shape=(None,cut_len),input_var=T.imatrix())
l_in_mask = L.InputLayer(shape=(None,cut_len))
l_em = L.EmbeddingLayer(l_in_sent,input_size=len(w2i),output_size=300, W=emW)
if IS_STATIC=="static":
    l_em.params[l_em.W].remove('trainable')

"sentence 2"
r_in_sent = L.InputLayer(shape=(None,cut_len),input_var=T.imatrix())
r_in_mask = L.InputLayer(shape=(None,cut_len))
r_em = L.EmbeddingLayer(r_in_sent,input_size=len(w2i),output_size=300, W=l_em.W)
if IS_STATIC=="static":
    r_em.params[r_em.W].remove('trainable')

def layer_encoder_module(l_em,l_in_mask):
    l_lstm_f = L.LSTMLayer(l_em,num_units=lstm_units/2,mask_input=l_in_mask)
    l_lstm_b = L.LSTMLayer(l_em,num_units=lstm_units/2,backwards=True,mask_input=l_in_mask)
    l_lstm = L.ConcatLayer([l_lstm_f,l_lstm_b],axis=2)
    return l_lstm

l_mask_dim = L.DimshuffleLayer(l_in_mask,(0,1,'x'))
r_mask_dim = L.DimshuffleLayer(r_in_mask,(0,1,'x'))

l_lstm = layer_encoder_module(l_em,l_in_mask)
r_lstm = layer_encoder_clone(r_em,r_in_mask,l_lstm,layer_encoder_module)

l_matrix_xy = L.ConcatLayer(cut_len*[L.DimshuffleLayer(l_lstm,(0,1,'x',2))],axis=2)
r_matrix_xy = L.ConcatLayer(cut_len*[L.DimshuffleLayer(r_lstm,(0,1,'x',2))],axis=2)
r_matrix_yx = L.DimshuffleLayer(r_matrix_xy,(0,2,1,3))

lm_matrix_xy = L.ConcatLayer(cut_len*[l_mask_dim],axis=2)
rm_matrix_xy = L.ConcatLayer(cut_len*[r_mask_dim],axis=2)
rm_matrix_yx = L.DimshuffleLayer(rm_matrix_xy,(0,2,1))

s_matrix = L.ElemwiseMergeLayer([l_matrix_xy,r_matrix_yx],T.mul) # mul or sub
mask_matrix = L.DimshuffleLayer(L.ElemwiseMergeLayer([lm_matrix_xy,rm_matrix_yx],T.mul),(0,1,2,'x'))
sent_matrix = cc.Custom_merge([s_matrix,mask_matrix],axis=3)

attn_temp1 = L.DimshuffleLayer(cc.Tensor_func_Layer(cc.Sum_axis_Layer(sent_matrix,axis=3),T.exp),(0,1,2,'x'))
attn_temp2 = L.ElemwiseMergeLayer([attn_temp1, mask_matrix],T.mul)
attn_matrix_r = cc.Softmax_axis_Layer(attn_temp2,axis=2) # Error
attn_matrix_l = cc.Softmax_axis_Layer(attn_temp2,axis=1) # Error

l_lstm_var = cc.Sum_axis_Layer(cc.Custom_merge([r_matrix_yx,attn_matrix_r],axis=3),axis=2)
r_lstm_var = cc.Sum_axis_Layer(cc.Custom_merge([l_matrix_xy,attn_matrix_l],axis=3),axis=1)

l_m_1 = l_lstm
l_m_2 = l_lstm_var
l_m_3 = L.ElemwiseMergeLayer([l_lstm,l_lstm_var],T.sub)
l_m_4 = L.ElemwiseMergeLayer([l_lstm,l_lstm_var],T.mul)
l_m = L.ConcatLayer([l_m_1,l_m_2,l_m_3,l_m_4],axis=2)

r_m_1 = r_lstm
r_m_2 = r_lstm_var
r_m_3 = L.ElemwiseMergeLayer([r_lstm,r_lstm_var],T.sub)
r_m_4 = L.ElemwiseMergeLayer([r_lstm,r_lstm_var],T.mul)
r_m = L.ConcatLayer([r_m_1,r_m_2,r_m_3,r_m_4],axis=2)

l_mm = L.DenseLayer(l_m, num_units=lstm_units, nonlinearity=NL.rectify, num_leading_axes=2)
r_mm = L.DenseLayer(r_m, num_units=lstm_units, nonlinearity=NL.rectify, num_leading_axes=2, W=l_mm.W, b=l_mm.b)

l_lstm_final = layer_encoder_module(l_mm,l_in_mask)
r_lstm_final = layer_encoder_clone(r_mm,r_in_mask,l_lstm_final,layer_encoder_module)
lstm_final = L.ConcatLayer([l_lstm_final,r_lstm_final],axis=2)

l_lstm_max = L.ReshapeLayer(L.MaxPool1DLayer(L.DimshuffleLayer(l_lstm_final,(0,2,1)),(cut_len)),([0],[1]))
r_lstm_max = L.ReshapeLayer(L.MaxPool1DLayer(L.DimshuffleLayer(r_lstm_final,(0,2,1)),(cut_len)),([0],[1]))

l_avg_numer = cc.Sum_axis_Layer(cc.Custom_merge([l_lstm_final,l_mask_dim],axis=2),axis=1)
l_avg_denom = cc.Tensor_func_Layer(cc.Sum_axis_Layer(l_mask_dim,axis=1),T.inv)
l_lstm_avg = cc.Custom_merge([l_avg_numer,l_avg_denom],axis=1)

r_avg_numer = cc.Sum_axis_Layer(cc.Custom_merge([r_lstm_final,r_mask_dim],axis=2),axis=1)
r_avg_denom = cc.Tensor_func_Layer(cc.Sum_axis_Layer(r_mask_dim,axis=1),T.inv)
r_lstm_avg = cc.Custom_merge([r_avg_numer,r_avg_denom],axis=1)

l_sent_vec = L.ConcatLayer([l_lstm_avg,l_lstm_max],axis=1)
r_sent_vec = L.ConcatLayer([r_lstm_avg,r_lstm_max],axis=1)
sent_vec = L.ConcatLayer([l_sent_vec,r_sent_vec],axis=1)

# ESIM's final output vectors
final_output = theano.function([l_in_sent.input_var,   r_in_sent.input_var,
                                l_in_mask.input_var,   r_in_mask.input_var],
                                L.get_output(sent_vec),
                                allow_input_downcast=True, on_unused_input='ignore')

# Pre-trained ESIM parameters
load_param = fc.load_all(data_path+'nli_pretraining_params.bin')
mod_param = load_param
mod_param.insert(0,emW)
L.set_all_param_values(l_lstm_final,mod_param)
L.set_all_param_values(r_lstm_final,mod_param)

""" build GIST network """
print "build GIST team system network..."
lstm_units = 100

# LSTM Mean and Max
def layer_encoder_module(l_em,l_in_mask):
    l_lstm_f = L.LSTMLayer(l_em,num_units=lstm_units/2,mask_input=l_in_mask)
    l_lstm_b = L.LSTMLayer(l_em,num_units=lstm_units/2,backwards=True,mask_input=l_in_mask)
    l_lstm = L.ConcatLayer([l_lstm_f,l_lstm_b],axis=2)
    l_sum_vec = cc.Sum_axis_Layer(cc.Custom_merge([l_lstm,L.DimshuffleLayer(l_in_mask,(0,1,'x'))],axis=2),axis=1)
    l_mean_vec = cc.Custom_merge([l_sum_vec,L.DimshuffleLayer(cc.INVLayer(cc.Sum_axis_Layer(l_in_mask,axis=1)),(0,'x'))],axis=1)
    l_lstm_max = L.ReshapeLayer(L.MaxPool1DLayer(L.DimshuffleLayer(l_lstm,(0,2,1)),(cut_len)),([0],[1]))
    l_mean_max = L.ConcatLayer([l_mean_vec,l_lstm_max],axis=1)
    return l_mean_max

# To get initial weight
def get_init_params():
    l_in_sent1 = L.InputLayer(shape=(None,cut_len),input_var=T.imatrix())
    l_in_mask1 = L.InputLayer(shape=(None,cut_len))
    l_em1 = L.EmbeddingLayer(l_in_sent1,input_size=len(w2i),output_size=300, W=emW)
    if IS_STATIC=="static":
        l_em1.params[l_em1.W].remove('trainable')
        
    l_in_sent2 = L.InputLayer(shape=(None,cut_len),input_var=T.imatrix())
    l_in_mask2 = L.InputLayer(shape=(None,cut_len))
    l_em2 = L.EmbeddingLayer(l_in_sent2,input_size=len(w2i),output_size=300, W=l_em1.W)
    if IS_STATIC=="static":
        l_em2.params[l_em2.W].remove('trainable')
        
    l_in_sent3 = L.InputLayer(shape=(None,cut_len),input_var=T.imatrix())
    l_in_mask3 = L.InputLayer(shape=(None,cut_len))
    l_em3 = L.EmbeddingLayer(l_in_sent3,input_size=len(w2i),output_size=300, W=emW)
    if IS_STATIC=="static":
        l_em3.params[l_em3.W].remove('trainable')
        
    l_in_sent4 = L.InputLayer(shape=(None,cut_len),input_var=T.imatrix())
    l_in_mask4 = L.InputLayer(shape=(None,cut_len))
    l_em4 = L.EmbeddingLayer(l_in_sent4,input_size=len(w2i),output_size=300, W=l_em1.W)
    if IS_STATIC=="static":
        l_em4.params[l_em4.W].remove('trainable')
    
    l_lstm_w0 = layer_encoder_module(l_em1,l_in_mask1)
    l_lstm_w1 = layer_encoder_clone(l_em2,l_in_mask2,l_lstm_w0,layer_encoder_module)
    l_lstm_cl = layer_encoder_module(l_em3,l_in_mask3)
    l_lstm_rs = layer_encoder_module(l_em4,l_in_mask4)
    
    claim_w0_in = L.InputLayer(shape=(None,1200),input_var=T.fmatrix())
    w0_reason_in = L.InputLayer(shape=(None,1200),input_var=T.fmatrix())
    
    claim_w1_in = L.InputLayer(shape=(None,1200),input_var=T.fmatrix())
    w1_reason_in = L.InputLayer(shape=(None,1200),input_var=T.fmatrix())
    
    w0_w1_in = L.InputLayer(shape=(None,1200),input_var=T.fmatrix())
    w1_w0_in = L.InputLayer(shape=(None,1200),input_var=T.fmatrix())
    
    w0_compare_concat = L.ConcatLayer([claim_w0_in,w0_reason_in,w0_w1_in,l_lstm_cl,l_lstm_w0,l_lstm_rs],axis=1)
    w1_compare_concat = L.ConcatLayer([claim_w1_in,w1_reason_in,w1_w0_in,l_lstm_cl,l_lstm_w1,l_lstm_rs],axis=1)
    
    w0_compare_dense = L.DenseLayer(w0_compare_concat, 600, nonlinearity=NL.rectify)
    w1_compare_dense = L.DenseLayer(w1_compare_concat, 600, nonlinearity=NL.rectify, W=w0_compare_dense.W, b=w0_compare_dense.b)
    
    w0_compare_final = L.DenseLayer(w0_compare_dense, 1, nonlinearity=None, num_leading_axes=1)
    w1_compare_final = L.DenseLayer(w1_compare_dense, 1, nonlinearity=None, num_leading_axes=1, W=w0_compare_final.W, b=w0_compare_final.b)
    
    w0_w1_final = L.ConcatLayer([w0_compare_final,w1_compare_final],axis=1)
    
    w0orw1_layer = L.NonlinearityLayer(w0_w1_final,nonlinearity=NL.softmax)
    return L.get_all_param_values(w0orw1_layer)

# For Warrant0
l_in_sent1 = L.InputLayer(shape=(None,cut_len),input_var=T.imatrix())
l_in_mask1 = L.InputLayer(shape=(None,cut_len))
l_em1 = L.EmbeddingLayer(l_in_sent1,input_size=len(w2i),output_size=300, W=emW)
if IS_STATIC=="static":
    l_em1.params[l_em1.W].remove('trainable')

# For Warrant1    
l_in_sent2 = L.InputLayer(shape=(None,cut_len),input_var=T.imatrix())
l_in_mask2 = L.InputLayer(shape=(None,cut_len))
l_em2 = L.EmbeddingLayer(l_in_sent2,input_size=len(w2i),output_size=300, W=l_em1.W)
if IS_STATIC=="static":
    l_em2.params[l_em2.W].remove('trainable')

# For Claim
l_in_sent3 = L.InputLayer(shape=(None,cut_len),input_var=T.imatrix())
l_in_mask3 = L.InputLayer(shape=(None,cut_len))
l_em3 = L.EmbeddingLayer(l_in_sent3,input_size=len(w2i),output_size=300, W=emW)
if IS_STATIC=="static":
    l_em3.params[l_em3.W].remove('trainable')

# For Reason   
l_in_sent4 = L.InputLayer(shape=(None,cut_len),input_var=T.imatrix())
l_in_mask4 = L.InputLayer(shape=(None,cut_len))
l_em4 = L.EmbeddingLayer(l_in_sent4,input_size=len(w2i),output_size=300, W=l_em1.W)
if IS_STATIC=="static":
    l_em4.params[l_em4.W].remove('trainable')

# Mean and Max pooling, W0 and W1 is sharing the parameters
l_lstm_w0 = layer_encoder_module(l_em1,l_in_mask1)
l_lstm_w1 = layer_encoder_clone(l_em2,l_in_mask2,l_lstm_w0,layer_encoder_module)
l_lstm_cl = layer_encoder_module(l_em3,l_in_mask3)
l_lstm_rs = layer_encoder_module(l_em4,l_in_mask4)

# For ESIM inputs
claim_w0_in = L.InputLayer(shape=(None,1200),input_var=T.fmatrix())
w0_reason_in = L.InputLayer(shape=(None,1200),input_var=T.fmatrix())

claim_w1_in = L.InputLayer(shape=(None,1200),input_var=T.fmatrix())
w1_reason_in = L.InputLayer(shape=(None,1200),input_var=T.fmatrix())

w0_w1_in = L.InputLayer(shape=(None,1200),input_var=T.fmatrix())
w1_w0_in = L.InputLayer(shape=(None,1200),input_var=T.fmatrix())

w0_compare_concat = L.ConcatLayer([claim_w0_in,w0_reason_in,w0_w1_in,l_lstm_cl,l_lstm_w0,l_lstm_rs],axis=1)
w1_compare_concat = L.ConcatLayer([claim_w1_in,w1_reason_in,w1_w0_in,l_lstm_cl,l_lstm_w1,l_lstm_rs],axis=1)

w0_compare_dense = L.DenseLayer(w0_compare_concat, 600, nonlinearity=NL.rectify)
w1_compare_dense = L.DenseLayer(w1_compare_concat, 600, nonlinearity=NL.rectify, W=w0_compare_dense.W, b=w0_compare_dense.b)

w0_compare_final = L.DenseLayer(w0_compare_dense, 1, nonlinearity=None, num_leading_axes=1)
w1_compare_final = L.DenseLayer(w1_compare_dense, 1, nonlinearity=None, num_leading_axes=1, W=w0_compare_final.W, b=w0_compare_final.b)

w0_w1_final = L.ConcatLayer([w0_compare_final,w1_compare_final],axis=1)

w0orw1_layer = L.NonlinearityLayer(w0_w1_final,nonlinearity=NL.softmax)

layers = {w0_compare_final:0.0, w0_compare_dense:1.0}
l2_penalty = regularize_layer_params_weighted(layers, l2)

predictions_test = L.get_output(w0orw1_layer, deterministic=True)
predictions_train = L.get_output(w0orw1_layer, deterministic=False)

print "Compile the theano function..."
preds = theano.function([l_in_sent1.input_var,    l_in_mask1.input_var,
                         l_in_sent2.input_var,    l_in_mask2.input_var,
                         l_in_sent3.input_var,    l_in_mask3.input_var,
                         l_in_sent4.input_var,    l_in_mask4.input_var,
                         claim_w0_in.input_var,   w0_reason_in.input_var,
                         claim_w1_in.input_var,   w1_reason_in.input_var,
                         w0_w1_in.input_var,      w1_w0_in.input_var],
                         predictions_test,
                         allow_input_downcast=True, on_unused_input='ignore')

target_values = T.ivector('target_values')
layer_cost = lasagne.objectives.categorical_crossentropy(predictions_train, target_values).mean() + lamda*l2_penalty
all_params = L.get_all_params(w0orw1_layer, trainable=True)
layer_lr = T.scalar()
layer_updates = lasagne.updates.adam(layer_cost, all_params, layer_lr)

train = theano.function([l_in_sent1.input_var,    l_in_mask1.input_var,
                         l_in_sent2.input_var,    l_in_mask2.input_var,
                         l_in_sent3.input_var,    l_in_mask3.input_var,
                         l_in_sent4.input_var,    l_in_mask4.input_var,
                         claim_w0_in.input_var,   w0_reason_in.input_var,
                         claim_w1_in.input_var,   w1_reason_in.input_var,
                         w0_w1_in.input_var,      w1_w0_in.input_var,
                         layer_lr, target_values],
                         layer_cost, updates=layer_updates, 
                         allow_input_downcast=True, on_unused_input='ignore')

init_params = L.get_all_param_values(w0orw1_layer)

def sentence_six(mtrx_dict,i=0):
    sent_vec1 = final_output(mtrx_dict['claim'][i],mtrx_dict['W0'][i],
                             mtrx_dict['claim_mask'][i],mtrx_dict['W0_mask'][i])
    sent_vec2 = final_output(mtrx_dict['W0'][i],mtrx_dict['reason'][i],
                             mtrx_dict['W0_mask'][i],mtrx_dict['reason_mask'][i])
    sent_vec3 = final_output(mtrx_dict['claim'][i],mtrx_dict['W1'][i],
                             mtrx_dict['claim_mask'][i],mtrx_dict['W1_mask'][i])
    sent_vec4 = final_output(mtrx_dict['W1'][i],mtrx_dict['reason'][i],
                             mtrx_dict['W1_mask'][i],mtrx_dict['reason_mask'][i])
    sent_vec5  = final_output(mtrx_dict['W0'][i],mtrx_dict['W1'][i],
                              mtrx_dict['W0_mask'][i],mtrx_dict['W1_mask'][i])
    sent_vec6 = final_output(mtrx_dict['W1'][i],mtrx_dict['W0'][i],
                             mtrx_dict['W1_mask'][i],mtrx_dict['W0_mask'][i])    
    return (sent_vec1,sent_vec2,sent_vec3,sent_vec4,sent_vec5,sent_vec6)

def acc_print(mtrx_dict,rtnaslist=0):
    acc_list = []
    train_batchs = np.shape(mtrx_dict['id'])[0]
    for i in xrange(train_batchs):
        sent_vec1,sent_vec2,sent_vec3,sent_vec4,sent_vec5,sent_vec6 = sentence_six(mtrx_dict,i)
        acc_list += list(np.argmax(preds(mtrx_dict['W0'][i],mtrx_dict['W0_mask'][i],
                                        mtrx_dict['W1'][i],mtrx_dict['W1_mask'][i],
                                        mtrx_dict['claim'][i],mtrx_dict['claim_mask'][i],
                                        mtrx_dict['reason'][i],mtrx_dict['reason_mask'][i],
                                        sent_vec1,sent_vec2,sent_vec3,sent_vec4,sent_vec5,sent_vec6),axis=1))
    acc_list = np.array(acc_list,dtype='int32')
    accuracy = (mtrx_dict['W0orW1'].flatten()==acc_list).mean()    
    if rtnaslist==0:
        return accuracy
    else:
        return acc_list

if __name__=="__main__":
    start_time = time.time()
    lr = init_lr
    init_param = get_init_params()
    L.set_all_param_values(w0orw1_layer,init_param)
    print "lamda :",lamda, "  decay:",decay, "  init_lr:", round(lr,6)
    print "init_dev_accuracy : ", acc_print(dev_mtrx_dict)
    print "init_tst_accuracy : ", acc_print(tst_mtrx_dict)
    for epoch in xrange(epochs):
        train_acc_list = []
        trn_mtrx_dict = fc.data_gen(trn_dict, i2w, w2i, max_len, cut_len, batch_unit=batch_unit, ispermute=True)
        train_batchs = np.shape(trn_mtrx_dict['id'])[0]
        for i in xrange(train_batchs):
            mtrx_dict = trn_mtrx_dict
            sent_vec1,sent_vec2,sent_vec3,sent_vec4,sent_vec5,sent_vec6 = sentence_six(trn_mtrx_dict,i)
            train(mtrx_dict['W0'][i],mtrx_dict['W0_mask'][i],
                  mtrx_dict['W1'][i],mtrx_dict['W1_mask'][i],
                  mtrx_dict['claim'][i],mtrx_dict['claim_mask'][i],
                  mtrx_dict['reason'][i],mtrx_dict['reason_mask'][i],
                  sent_vec1,sent_vec2,sent_vec3,sent_vec4,sent_vec5,sent_vec6, 
                  lr, trn_mtrx_dict['W0orW1'][i])
        trn_mtrx_dict = fc.data_gen(trn_dict, i2w, w2i, max_len, cut_len, batch_unit=110, ispermute=False)
        trn_accuracy = acc_print(trn_mtrx_dict)
        dev_accuracy = acc_print(dev_mtrx_dict)
        tst_accuracy = acc_print(tst_mtrx_dict)
        print "epoch:", epoch+1, "  trn:", round(trn_accuracy,4), "  dev:", round(dev_accuracy,4), "  tst:", round(tst_accuracy,4), "  lr:",round(lr,6)
        lr=decay*lr
    print "time : ", round((time.time() - start_time)/60.0, 3), "m"
    print "training done"

