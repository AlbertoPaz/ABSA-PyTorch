# -*- coding: utf-8 -*-
# file: lcrs.py
# author: albertopaz <aj.paz167@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention
import torch
import torch.nn as nn
import torch.nn.functional as F


class LCRS(nn.Module):
    '''
    hyperparameters
    lr: 0.1
    L2: 1e-5
    dropout: 0.5
    SGD: 0.9 momentum
    embeddings: GloVe 300, out-vocabulary: U(-0.1, 0.1)
    bias: 0 
    
    TO DO , USE EMBED DIM INSTEAD OF HIDDEN DIM
    '''
    def __init__(self, embedding_matrix, opt, memory_weighter = 'no'):
        super(LCRS, self).__init__()
        self.opt = opt
        self.memory_weighter = memory_weighter
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.blstm_l = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, rnn_type = 'LSTM')
        self.blstm_c = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, rnn_type = 'LSTM')
        self.blstm_r = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, rnn_type = 'LSTM')
        self.dense = nn.Linear(opt.hidden_dim*4, opt.polarities_dim)
        # target to context attention
        self.t2c_l_attention = Attention(opt.hidden_dim, score_function='bi_linear')
        self.t2c_r_attention = Attention(opt.hidden_dim, score_function='bi_linear')
        # context to target attention
        self.c2t_l_attention = Attention(opt.hidden_dim, score_function='bi_linear')
        self.c2t_r_attention = Attention(opt.hidden_dim, score_function='bi_linear')
             
    def locationed_memory(self, memory_l, memory_r, x_l_len, x_c_len, x_r_len, dep_offset):
        position = 'linear'
        memory_len = x_l_len + x_c_len + x_r_len
        # loop over samples in the batch
        for i in range(memory_l.size(0)):      
            # Loop over memory slices 
            for idx in range(memory_len[i]):   
                aspect_start = x_l_len[i] + 1             # INCORRECT: ASSUME x_l_len = 0 THEN aspect_start = 0
                aspect_end = x_l_len[i] + x_c_len[i]
                if idx < aspect_start:  # left locationed memory
                    if position == 'linear':
                        l = aspect_start.item() - idx 
                        memory_l[i][idx] *= (1-float(l)/int(memory_len[i]))
                    elif position == 'dependency':
                        memory_l[i][idx] *= (1-float(dep_offset[i][idx])/int(memory_len[i]))
                elif idx > aspect_end:  # right locationed memory
                    if position == 'linear':
                        l = idx - aspect_end.item()
                        memory_r[i][idx] *= (1-float(l)/int(memory_len[i]))
                    elif position == 'dependency':
                        memory_r[i][idx] *= (1-float(dep_offset[i][idx])/int(memory_len[i]))
               
        return memory_l, memory_r
    
    def forward(self, inputs):
        # raw indices for left, center, and right parts
        x_l, x_c, x_r, dep_offset = inputs[0], inputs[1], inputs[2], inputs[3]
        x_l_len = torch.sum(x_l != 0, dim=-1)
        x_c_len = torch.sum(x_c != 0, dim=-1)
        x_r_len = torch.sum(x_r != 0, dim=-1)
        
        # embedding layer
        x_l, x_c, x_r = self.embed(x_l), self.embed(x_c), self.embed(x_r)
       
        # Memory module:
        # ----------------------------
        # left memory
        if x_l_len == 0: memory_l = x_l
        if x_l_len > 0 : memory_l, (_, _) = blstm_l(x_l, x_l_len)
        # center memory
        if x_c_len == 0: memory_c = x_c
        if x_c_len > 0 : memory_c, (_, _) = blstm_c(x_c, x_c_len)
        # right memory
        if x_r_len == 0: memory_r = x_r
        if x_r_len > 0 : memory_r, (_, _) = blstm_r(x_r, x_r_len)
        
        # Target-Aware memory
        
        # locationed-memory
        if self.memory_weighter == 'position': 
            memory_l, memory_r = self.locationed_memory(memory_l, memory_r, x_l_len, 
                                                        x_c_len, x_r_len, dep_offset)
        # context-attended-memory
        if self.memory_weighter == 'cam': 
            pass
        # ----------------------------
        
        # Aspect vector representation
        x_c_len = torch.tensor(x_c_len, dtype=torch.float).to(self.opt.device)
        v_c = torch.sum(memory_c, dim=1)
        v_c = torch.div(v_c, x_c_len.view(x_c_len.size(0), 1))
      
    
        # Rotatory attention:
        # ----------------------------
        # [1] Target2Context Attention
        v_l = self.t2c_l_attention(memory_l, v_c).squeeze(dim=1)           # left vector representation
        v_r = self.t2c_r_attention(memory_r, v_c).squeeze(dim=1)           # Right vector representation

        # [2] Context2Target Attention
        v_c_l = self.c2t_l_attention(memory_c, v_l).squeeze(dim=1)  # Left-aware target
        v_c_r = self.c2t_r_attention(memory_c, v_r).squeeze(dim=1)  # Right-aware target
        # ----------------------------
        
        # sentence representation
        v_s = torch.cat((v_l, v_c_l, v_c_r, v_r), dim=-1)    # dim : (1, 800)
        # v_s = torch.cat((v_l, v_c_l, v_c_r, v_r), dim = 0)     # dim : (4, 300)
        
        # Classifier 
        out = self.dense(v_s)
        out = F.softmax(out, dim=-1)   
        
        return out
