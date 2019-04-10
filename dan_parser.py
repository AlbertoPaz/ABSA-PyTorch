# -*- coding: utf-8 -*-
# file: parser.py
# author: albertopaz <aj.paz167@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import spacy
import pandas as pd
from tqdm import tqdm

# Enforce fully connected tree (one root per sample)
# https://github.com/explosion/spaCy/issues/1850
def one_sentence_per_doc(doc):
    doc[0].sent_start = True
    for i in range(1, len(doc)):
        doc[i].sent_start = False
    return doc


class Parser(object):
    '''
    Concept parser python class
    '''
    # returns the concept and the position in the dependency tree where it has been identified
    # based on: Dependency-Based Semantic Parsing for Concept-Level Text Analysis
    def __init__(self, one_root = True):
        self.nlp = spacy.load("en_core_web_sm")   
        self.nlp.remove_pipe('ner')  
        if one_root: 
            self.nlp.add_pipe(one_sentence_per_doc, before='parser')   
            
    def extract_concepts(self, tok, in_bag = True):
        concepts = []
        
        # VERB head token rules
        if tok.head.pos_ == 'VERB':
            # Joint Subject noun and Adjective complement rule
            if tok.dep_ == 'nsubj':
              for c in tok.head.children:
                if c.dep_ == 'acomp':
                    concepts.append({'pos': max(c.i, tok.i), 
                                     'text': c.lemma_ + '_' + tok.lemma_})

            # Direct nominal objects     
            if tok.dep_ == 'dobj':
              concepts.append({'pos': max(tok.i, tok.head.i), 
                               'text': tok.head.lemma_ + '_' + tok.lemma_})
    
            # Adjective and clausal complements Rules
            if tok.dep_ == 'acomp':
                concepts.append({'pos': max(tok.i, tok.head.i), 
                                 'text': tok.head.lemma_ + '_' + tok.lemma_})
              
            # Open clausal complements
            if tok.dep_ == 'xcomp':
                concepts.append({'pos': max(tok.head.i, tok.i), 
                                 'text':tok.head.lemma_ + '_' + tok.lemma_})
                for t in tok.children:
                    if t.dep_ == 'dobj':
                        concepts.append({'pos': max(tok.head.i, tok.i, t.i), 
                                         'text':tok.head.lemma_ + '_' 
                                         + tok.lemma_ + '_' + t.lemma_})
        # Dependency-based
        # negation : problems with dont 
        if tok.dep_ == 'neg':
            concepts.append({'pos': max(tok.i, tok.head.i), 
                             'text': tok.lemma_ + '_' + tok.head.lemma_})
 
        # Adjectival, adverbial and participial modification
        elif tok.dep_ == 'amod':
            concepts.append({'pos': max(tok.i, tok.head.i),
                             'text':tok.lemma_+ '_' + tok.head.lemma_})

        # Prepositional phrases 
        elif tok.dep_ == 'prep':
            for c in tok.children:
                if c.dep_ == 'pobj':
                    concepts.append({'pos': max(tok.head.i, tok.i, c.i),
                                     'text': tok.head.lemma_ + '_' 
                                     + tok.lemma_ + '_' + c.lemma_})

        # Adverbial clause modifier
        elif tok.dep_ == 'advcl':
            concepts.append({'pos': max( tok.i, tok.head.i),
                             'text':tok.lemma_ + '_' + tok.head.lemma_})

        # Noun Compound Modifier
        elif tok.dep_ == 'compound':
            concepts.append({'pos': max(tok.i, tok.head.i), 
                             'text': tok.lemma_ + '_' + tok.head.lemma_})
  
        if len(concepts) > 0:
            # TODO: function to choose only one concept
            return concepts
        
        
    # select only one of the triggered concepts 
    def one_concept(concepts_lists):
        concept = concepts_lists[0]
        return concept

    # extract all concepts on a sentence
    def find_concepts_in_sentence(self, text):
        concepts = {}
        doc = self.nlp(text)
        for tok in doc: 
            c = self.extract_concepts(tok)
            if c != None:
                for i in c:
                    concepts.append(i['text'])
                    
        return ' '.join(concepts)
          

# tree object from stanfordnlp/treelstm
class Tree(object):
    '''
    tree object from stanfordnlp/treelstm
    '''
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def paint(self, level = 0):
        print(level * ' ├──', self.idx)
        level = level + 1
        for idx in range(self.num_children):
            self.children[idx].paint(level)
        
        
        
def read_tree(parents):
    trees = dict()
    root = None
    for i in range(1, len(parents) + 1):
        if i - 1 not in trees.keys() and parents[i - 1] != -1:
            idx = i
            prev = None
            while True:
                parent = parents[idx - 1]
                if parent == -1:
                    break
                tree = Tree()
                if prev is not None:
                    tree.add_child(prev)
                trees[idx - 1] = tree
                tree.idx = idx - 1
                if parent - 1 in trees.keys():
                    trees[parent - 1].add_child(tree)
                    break
                elif parent == 0:
                    root = tree
                    break
                else:
                    prev = tree
                    idx = parent
    print('\n', ' '.join(str(p-1) for p in parents))
    root.paint()
    return root

        

