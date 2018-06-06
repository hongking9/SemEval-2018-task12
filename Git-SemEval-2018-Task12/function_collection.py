# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 02:49:18 2017

@author: ChoiHongSeok
"""

import pickle
import numpy as np

def dump_all(all_data, data_path):
    with open(data_path, 'wb') as handle:
        pickle.dump(all_data, handle)

def load_all(data_path):
    with open(data_path, 'rb') as handle:
        load_all = pickle.loads(handle.read())
    return load_all

def read_data(path):
    f = open(path,'r')
    lines = f.readlines()
    sid, W0, W1, W0orW1, reason, claim, debateTitle, debateInfo = [],[],[],[],[],[],[],[]
    for line in lines:
        line = line.strip('/n').strip(' ')
        if line[0]=='#' : continue
        row_array = line.split('\t')
        sid.append(row_array[0])
        W0.append(row_array[1])
        W1.append(row_array[2])
        W0orW1.append(int(row_array[3]))
        reason.append(row_array[4])
        claim.append(row_array[5])
        debateTitle.append(row_array[6])
        debateInfo.append(row_array[7])
    f.close()
    
    data_dict = {}
    data_dict['id'] = sid
    data_dict['W0'] = W0
    data_dict['W1'] = W1
    data_dict['W0orW1'] = W0orW1
    data_dict['reason'] = reason
    data_dict['claim'] = claim
    data_dict['debateTitle'] = debateTitle
    data_dict['debateInfo'] = debateInfo
    return data_dict

def data_gen(data_dict, i2w, w2i, max_len, cut_len, batch_unit, ispermute):
    Not_sentence_key = ['id', 'W0orW1']
    if max_len>cut_len : max_len=cut_len
    data_size = len(data_dict['id'])
    batchs_num = int(data_size/batch_unit)
    if ispermute:
        permute = np.random.permutation(np.arange(data_size))
    else:
        permute = np.arange(data_size)
    prmt_dict = {}
    for key in data_dict.iterkeys():
        prmt_dict[key] = []    
    for pi in permute[:batchs_num*batch_unit]:
        for key in data_dict.iterkeys():
            prmt_dict[key].append(data_dict[key][pi])            
    mtrx_dict = {}
    mtrx_dict['id'] = prmt_dict['id']
    mtrx_dict['W0orW1'] = np.array(prmt_dict['W0orW1'],'int32')
    for key in data_dict.iterkeys():
        if key in Not_sentence_key : continue
        mtrx_dict[key] = np.zeros(shape=(batchs_num*batch_unit, max_len),dtype='int32')
        mtrx_dict[key+'_mask'] = np.zeros(shape=(batchs_num*batch_unit, max_len),dtype='int32')    
    for i in xrange(batchs_num*batch_unit):
        for key in data_dict.iterkeys():
            if key in Not_sentence_key : continue
            words = prmt_dict[key][i].split()
            for wi, ww in enumerate(words):
                if wi >= max_len : break
                mtrx_dict[key][i][wi] = w2i[ww]
                mtrx_dict[key+'_mask'][i][wi] = 1    
    mtrx_dict['id'] = np.reshape(mtrx_dict['id'], (batchs_num,batch_unit))
    mtrx_dict['W0orW1'] = np.reshape(mtrx_dict['W0orW1'], (batchs_num,batch_unit))
    for key in data_dict.iterkeys():
        if key in Not_sentence_key : continue
        mtrx_dict[key] = np.reshape(mtrx_dict[key], (batchs_num,batch_unit,max_len))
        mtrx_dict[key+'_mask'] = np.reshape(mtrx_dict[key+'_mask'], (batchs_num,batch_unit,max_len))
    return mtrx_dict
