# -*- coding: utf-8 -*-
# file: cabasc.py
# author: albertopaz <aj.paz167@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.attention import Attention
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.squeeze_embedding import SqueezeEmbedding
from layers.dynamic_rnn import DynamicLSTM         

# Flip a tensor along a given dimension
# https://github.com/pytorch/pytorch/issues/229
def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

# Content Attention Attention Model for ABSA
class Cabasc(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(Cabasc, self).__init__()
        self.opt = opt
        self.dropout = nn.Dropout(0.5)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        if opt.freeze_embeddings == "yes":
            self.embed.weight.requieres_grad = False 
        self.squeeze_embedding = SqueezeEmbedding(batch_first=True)         
        self.linear1 = nn.Linear(3*opt.embed_dim, opt.embed_dim)
        self.linear2 = nn.Linear(opt.embed_dim, 1, bias=False)    
        self.mlp = nn.Linear(opt.embed_dim, opt.embed_dim)                           
        self.dense = nn.Linear(opt.embed_dim, opt.polarities_dim)  
        if opt.model_name == 'cabasc':                  
            self.rnn_l = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, rnn_type = 'GRU') 
            self.rnn_r = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, rnn_type = 'GRU')
            self.mlp_l = nn.Linear(opt.hidden_dim, 1)
            self.mlp_r = nn.Linear(opt.hidden_dim, 1)
    
    
    def forward(self, inputs):
        
        text_raw_indices, aspect_indices, x_l, x_r = inputs[0], inputs[1], inputs[2], inputs[3]
        memory_len = torch.sum(text_raw_indices != 0, dim = -1)
        aspect_len = torch.sum(aspect_indices != 0, dim = -1)
        left_len = torch.sum(x_l != 0, dim = -1)
        
        # aspect representation
        nonzeros_aspect = torch.tensor(aspect_len, dtype=torch.float).to(self.opt.device)
        aspect = self.embed(aspect_indices)
        aspect = torch.sum(aspect, dim=1)
        v_a = torch.div(aspect, nonzeros_aspect.unsqueeze(1)).unsqueeze(1) # batch_size x 1 x embed_dim
        
        # memory module
        memory = self.embed(text_raw_indices)
        memory = self.squeeze_embedding(memory, memory_len)
        
        # sentence representation         
        nonzeros_memory = memory_len.float()
        v_s = torch.sum(memory, dim = 1)
        v_s = torch.div(v_s, nonzeros_memory.unsqueeze(1)).unsqueeze(1) # batch_size x 1 x embed_dim
        
        # weighted memory        
        if self.opt.model_name == 'base_model': 
            memory = self.locationed_memory(memory, memory_len, left_len, aspect_len) 
        elif self.opt.model_name == 'cabasc': 
            memory = self.context_attention(x_l, x_r, memory, memory_len, aspect_len)
            v_s = torch.sum(memory, dim = 1)                                             
            v_s = torch.div(v_s, nonzeros_memory.view(nonzeros_memory.size(0),1))  
            v_s = v_s.unsqueeze(dim=1)
        
        # content attention module
        v_ts = self.content_attention(memory, v_a, v_s)      
       
        # classifier
        v_s = self.dropout(v_s)
        v_ns = v_ts + v_s                                
        v_ns = v_ns.view(v_ns.size(0), -1)
        v_ms = F.tanh(self.mlp(v_ns))
        out = self.dense(v_ms)
        out = F.softmax(out, dim=-1)   
        
        return out

    def content_attention(self, memory, v_a, v_s):
        '''
        sentence-level content based attention module (attention # 2)
        '''
        memory_chunks = memory.chunk(memory.size(1), dim=1)
        c = []
        
        for memory_chunk in memory_chunks: # batch_size x 1 x embed_dim
            c_i = self.linear1(torch.cat([memory_chunk, v_a, v_s], dim=1).view(memory_chunk.size(0), -1))
            c_i = self.linear2(torch.tanh(c_i)) # batch_size x 1
            c.append(c_i)
        alpha = F.softmax(torch.cat(c, dim=1), dim=1) # batch_size x seq_len
        v_ts = torch.matmul(memory.transpose(1, 2), alpha.unsqueeze(-1)).transpose(1, 2)       
        
        return v_ts
    
    
    def context_attention(self, x_l, x_r, memory, memory_len, aspect_len):
        '''
        context based attention module (attention # 1)
        '''
        
        left_len, right_len = torch.sum(x_l != 0, dim=-1), torch.sum(x_r != 0, dim=-1)
        x_l, x_r = self.embed(x_l), self.embed(x_r) 
        context_l, (_, _) =  self.rnn_l(x_l, left_len)     # left, right context : (batch size, max_len, embedds)
        context_r, (_, _) =  self.rnn_r(x_r, right_len)           
        
        # Attention weights : (batch_size, max_batch_len, 1) 
        attn_l = F.sigmoid(self.mlp_l(context_l)) + 0.5
        attn_r = F.sigmoid(self.mlp_r(context_r)) + 0.5
        
        # apply weights one sample at a time
        for i in range(memory.size(0)):                                         # doing opposite directions as the paper
            aspect_start = (left_len[i] - aspect_len[i]).item()
            aspect_end = left_len[i] 
            # remove zeros and flip to get scores in the original order
            attn_r_reversed = attn_r[i][:aspect_end] 
            attn_r_reversed = flip(attn_r_reversed, 0)    
            # attention weights for each element in the sentence
            for idx in range(memory_len[i]):
                if idx < aspect_start: 
                    memory[i][idx] *= attn_r_reversed[idx]             
                elif idx < aspect_end: 
                    memory[i][idx] *= (attn_r_reversed[idx] + attn_l[i][idx - aspect_start])/2
                else: 
                    memory[i][idx] *= attn_l[i][idx - aspect_start]       
          
        return memory
    
    
    # based on the absolute distance to the first border word of the aspect
    def locationed_memory(self, memory, memory_len, left_len, aspect_len):
        '''
        position based attention module (attention # 1)
        '''
        
        for i in range(memory.size(0)):
            for idx in range(memory_len[i]):
                aspect_start = left_len[i] - aspect_len[i]
                aspect_end = left_len[i] 
                if idx < aspect_start: 
                    l = aspect_start.item() - idx                   
                elif idx <= aspect_end: 
                    l = 0
                else: 
                    l = idx - aspect_end.item()
                memory[i][idx] *= (1-float(l)/int(memory_len[i]))
               
        return memory