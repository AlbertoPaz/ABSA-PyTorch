# -*- coding: utf-8 -*-
# file: parser.py
# author: albertopaz <aj.paz167@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import numpy as np
import spacy
import en_core_web_sm
import networkx as nx
from spacy.tokenizer import Tokenizer                 # Inconsistent tokenizer TEMPORAL SOLUTION = whitespace tokenizer


# Enforce fully connected tree (one root per sample)
# https://github.com/explosion/spaCy/issues/1850
def one_sentence_per_doc(doc):
    doc[0].sent_start = True
    for i in range(1, len(doc)):
        doc[i].sent_start = False
    return doc


class Parser():
    '''
    Arguments:
        - one_root:   True: enforces one root per sentence
        - simple_tok: True: white space tokenizer. False: english specific tokenizer
        - dep_split:  True: Excecutes dependency based split
    '''
    def __init__(self, one_root = True, simple_tok = True):
        # load english language model
        self.nlp = en_core_web_sm.load()    
        # enforce one tree per sentence
        if one_root: self.nlp.add_pipe(one_sentence_per_doc, before='parser')  
        # disabled name entity recognizer for more speed
        self.nlp.remove_pipe('ner')                          
        # white space tokenizer
        if simple_tok: self.nlp.tokenizer = Tokenizer(self.nlp.vocab)         
            
            
    def forward(self, text, aspect_indices, text_left_with_aspect_indices):
        # find target positions
        start, end = self.targetPositions(aspect_indices, text_left_with_aspect_indices)     
        # make dependency graph
        doc, graph, aspect_nodes = self.makeGraph(text, start, end)  
        # dependency-based word positions
        offset, weights = self.shortestPaths(doc, graph, aspect_nodes)
        # dependency based context splitting
        aspect_term = doc[start:end] 
        dep_splits = self.splitTree(doc, aspect_term)
        
        return offset, weights, dep_splits
            
    
    def targetPositions(self, aspect_indices, text_left_with_aspect_indices):
        # get the target starting and ending position
        aspect_len = np.count_nonzero(aspect_indices)
        left_len = np.count_nonzero(text_left_with_aspect_indices)
        start, end = left_len - aspect_len, left_len
        
        return start , end
     
    
    def makeGraph(self, text, aspect_start, aspect_end):
        '''
        Syntactic dependency parser. Returns:   
            doc          -- sequence of token objects
            graph        -- network containing the dependency tree, where nodes = (token--index)
            aspect_nodes -- nodes in the graph corresponding to the aspect term 
        '''
        # create doc object  
        doc = self.nlp(text)     
        
        # Load spacy's dependency tree into a networkx graph
        edges, aspect_nodes = [], []     
        for token in doc:
            # identify nodes corresponding to the aspect
            if token.i > aspect_start and token.i <= aspect_end: 
                aspect_nodes.append('{}-{}'.format(token.lower_,token.i))
            # create edges
            for child in token.children:
                edges.append(('{}-{}'.format(token.lower_,token.i),
                              '{}-{}'.format(child.lower_,child.i)))

        return doc, nx.Graph(edges), aspect_nodes
    
    
    def shortestPaths(self, doc, graph, aspect_nodes):
        '''
        Find the shortest path in the dependency graph, from each word to the aspect. Returns:
            offset       -- distance / displacement between the word and the aspect vector
            weights      -- dependency-based position weights vector
        '''
        weights, offset = np.zeros(len(doc)), np.zeros(len(doc))     
        for tok in doc:
            l = 100
            # shortest path to the closest aspect word
            for aspect in aspect_nodes:
                try:path_len = nx.shortest_path_length(graph, source= '{}-{}'.format(tok.lower_,tok.i) , target = aspect)
                except: path_len = len(doc)                      # if not connected (one_root = false), assign max length                   
                if path_len < l : l = path_len
            offset[tok.i] = float(l)/len(doc)   
            weights[tok.i] = 1 - float(l)/len(doc)
            
        return offset, weights
    
    
    def splitTree(self, doc, aspect_term):
        '''
        Split the sentence into aspect and context based on the dependency tree. Returns:
            dep_splits[0] -- text corresponding to the words in the subtree of the aspect term (aspect term)
            dep_splits[1] -- text out of the subtree, to the left of the aspect (left context) 
            dep_splits[2] -- text out of the subtree, to right of the aspect (right context)
        '''
        # find head of the aspect term:
        # the word with the biggest subtree is the head
        max_len = 0
        for j in aspect_term:
            c = len([i for i in j.subtree])
            if c > max_len: aspect, max_len = j, c

        # list of tokens down the syntactic subtree of the aspect
        right_list = [ i for i in aspect.subtree ]
        
        # text corresponding to the new aspect and context : 
        dep_aspect, left_context, right_context = '', '', ''
        for i in right_list:                          # aspect subtree ( word order may change )
            dep_aspect += i.lower_ + ' '           
            
        context = 'left'
        for i in doc:                                 # context ( preserve original word order ) 
            if context == 'left':
                if i not in right_list: left_context += i.lower_ + ' '
                if i in right_list: context = 'right'
            if context == 'right':
                if i not in right_list: right_context += i.lower_ + ' '
        
        dep_context = left_context + right_context
        dep_splits = [dep_aspect, left_context, right_context]
                
        ''' note : 
        to preserve the order...
        for i in doc:
            if i in right_list: dep_aspect += i.lower_ + ' '
            else: dep_context += i.lower_ + ' ' 
        '''

        return dep_splits
   

