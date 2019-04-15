# -*- coding: utf-8 -*-
# file: data_utils.py
# author: albertopaz <aj.paz167@gmail.com>
# Copyright (C) 2019. All Rights Reserved.

import re
import os
import pickle
import numpy as np
import spacy
import itertools
import pandas as pd
from tqdm import tqdm

from dan_parser import Parser
from torch.utils.data import Dataset

def make_dependency_aware(dataset, raw_data_path, boc):
    dat_fname = re.findall(r'datasets/(.*?)\..',raw_data_path)[0].lower()
    dan_data_path = os.path.join('data/dan/dan_{}.dat'.format( dat_fname))
    if os.path.exists(dan_data_path):
        print('loading dan inputs:', dat_fname)
        awareness = pickle.load(open(dan_data_path, 'rb'))
    else:
        awareness = []
        print('parsing...')
        dp = Parser(boc = boc)
        for i in tqdm(dataset):
            x = dp.parse(i['text_string'], i['aspect_position'][0], 
                         i['aspect_position'][1])
            awareness.append(x)
            
        pickle.dump(awareness, open(dan_data_path, 'wb'))
        
    # merge regular inputs dictionary with the dan dictionary
    dataset_v2 = [{**dataset[i], **awareness[i]} for i in range(len(dataset))]
        
    return dataset_v2


# creates a file with the concepts found accorss all datasets
def build_boc(fnames, dat_fname):
    boc_path = os.path.join('data/embeddings', dat_fname)
    affective_space_path = 'data/embeddings/affectivespace/affectivespace.csv' 
    if os.path.exists(boc_path):
        print('loading bag of concepts:', dat_fname)
        boc = pickle.load(open(boc_path, 'rb'))
    else:
        dan = Parser()
        concepts = []
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                
                doc = dan.nlp(text_raw)
                for tok in doc: 
                    c = dan.extract_concepts(tok, in_bag = False)
                    if c != None: concepts.append(c)
    
        concepts = list(itertools.chain(*concepts))
        concepts = [i['text'] for i in concepts]
        concepts = list(set(concepts))
        print('total concepts found: ', len(concepts))
        
        # keep only the ones that match with affective space 
        affective_space = pd.read_csv(affective_space_path, header = None)
        bag_of_concepts = []
        for key in list(affective_space[0]):
            if key in concepts: bag_of_concepts.append(key)
        print('total concepts keept: ', len(bag_of_concepts))
        
        text = " ".join(bag_of_concepts)
        boc = Tokenizer(max_seq_len = 5)
        boc.fit_on_text(text)
        boc.tokenize = None # remove spacy model
        pickle.dump(boc, open(boc_path, 'wb'))
        
    return boc

    
def build_tokenizer(fnames, max_seq_len, dat_fname):
    tokenizer_path = os.path.join('data/embeddings', dat_fname)
    if os.path.exists(tokenizer_path):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(tokenizer_path, 'rb'))
    else:
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        tokenizer.tokenize = None
        pickle.dump(tokenizer, open(tokenizer_path, 'wb'))

    return tokenizer


def _load_word_vec(path, word2idx=None):
    word_vec = {}
    if path[-3:] == 'csv':
        fin = pd.read_csv(path, header = None)
        for row in range(len(fin)):
            vec = fin.iloc[row].values
            if word2idx is None or vec[0] in word2idx.keys():
                word_vec[vec[0]] = np.asarray(vec[1:], dtype = 'float32')
    else:
        fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
        for line in fin:
            tokens = line.rstrip().split()
            if word2idx is None or tokens[0] in word2idx.keys():
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    embed_path = os.path.join('data/embeddings', dat_fname)
    if os.path.exists(embed_path):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(embed_path, 'rb'))
    else:
        print('loading word vectors...')
        if dat_fname.split('_')[0] == '100':
            embedding_matrix = np.zeros((len(word2idx) + 2, 100))  # idx 0 and len(word2idx)+1 are all-zeros
            fname = 'data/embeddings/affectivespace/affectivespace.csv'
        else: 
            embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
            fname = 'data/embeddings/glove/glove.42B.300d.txt'
        
        word_vec = _load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embed_path, 'wb'))
    return embedding_matrix


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1
        self.init_tokenizer()

    def init_tokenizer(self):
        self.tokenize = spacy.load("en_core_web_sm") 
        for p in ['parser', 'ner', 'tagger']: self.tokenize.remove_pipe(p)
        
    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = [w.text for w in self.tokenize(text)]
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = [w.text for w in self.tokenize(text)]
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)
    

class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer, boc = None):
        print('reading {} data'.format(fname))
        tokenizer.init_tokenizer()
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = []
        for i in tqdm(range(0, len(lines), 3)):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()

            text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            text_raw_without_aspect_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            text_left_indices = tokenizer.text_to_sequence(text_left)
            text_left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
            text_right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
            text_right_with_aspect_indices = tokenizer.text_to_sequence(" " + aspect + " " + text_right, reverse=True)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            left_context_len = np.sum(text_left_indices != 0)
            aspect_len = np.sum(aspect_indices != 0)
            text_string = text_string = text_left + ' ' + aspect + ' ' + text_right
            aspect_position = [left_context_len.item(), (left_context_len + aspect_len).item()]
            #aspect_position = torch.tensor([left_context_len.item(), (left_context_len + aspect_len - 1).item()])
            polarity = int(polarity) + 1
                   
            data = {
                'text_raw_indices': text_raw_indices,
                'text_raw_without_aspect_indices': text_raw_without_aspect_indices,
                'text_left_indices': text_left_indices,
                'text_left_with_aspect_indices': text_left_with_aspect_indices,
                'text_right_indices': text_right_indices,
                'text_right_with_aspect_indices': text_right_with_aspect_indices,
                'aspect_indices': aspect_indices,
                'polarity': polarity,
                'text_string': text_string,  
                'aspect_position': aspect_position,
            }

            all_data.append(data)
        self.data = all_data
        tokenizer = None
        
        if boc != None:
            self.data = make_dependency_aware(self.data, fname, boc)

        print('all input prepared (✓)')

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
