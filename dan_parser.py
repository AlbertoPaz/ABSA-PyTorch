# -*- coding: utf-8 -*-
# file: parser.py
# author: albertopaz <aj.paz167@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import networkx as nx
import numpy as np
import spacy

# Enforce fully connected tree (one root per sample)
# https://github.com/explosion/spaCy/issues/1850
def one_sentence_per_doc(doc):
    doc[0].sent_start = True
    for i in range(1, len(doc)):
        doc[i].sent_start = False
    return doc

# DAN dependency based parser
class Parser(object):
    def __init__(self, one_root = True, boc = None):
        '''
        Initialize the DAN dependency based parser python class
        '''
        self.nlp = spacy.load("en_core_web_sm")   
        self.nlp.remove_pipe('ner')  
        if one_root: 
            self.nlp.add_pipe(one_sentence_per_doc, before='parser')  
        self.bag_of_concepts = boc
            
        
    def parse(self, text, aspect_start, aspect_end):
        '''
        Parse a text to obtain and returns:
            tree_positions, concepts, supersized aspect, and dependency_tree
        '''
        doc = self.nlp(text)
        aspect_term = ' '.join(a.text for a in doc[aspect_start:aspect_end])
        
        parents, concepts = [], []
        edges, aspect_nodes = [], [] 
        subtree_aspect, compound_aspect, noun_chunk_aspect = None, None, None
        
        for tok in doc:
            # get parents and extract concepts 
            parents.append(tok.head.i + 1 if tok.dep_ != 'ROOT' else 0) # +1 to be consistent with stan
            concepts.append(self.extract_concepts(tok))
            
            # identify nodes corresponding to the aspect and super size it
            if (tok.i >= aspect_start) and (tok.i < aspect_end):
                aspect_nodes.append('{}-{}'.format(tok.lower_, tok.i))
                if subtree_aspect == None:
                    subtree_aspect = self.subtree_supersize(tok)
                if compound_aspect == None:
                    compound_aspect = self.compound_supersize(tok, aspect_term)
                
            # load spacy dependency tree into a networkx graph
            for c in tok.children:
                edges.append(('{}-{}'.format(tok.lower_, tok.i),
                              '{}-{}'.format(c.lower_, c.i)))
            
        # consolidate output
        noun_chunk_aspect = self.noun_chunk_supersize(doc, aspect_start, aspect_end)
        position_vector = self.shortest_path(doc, nx.Graph(edges), aspect_nodes)
        concept_vector = self.make_concept_vector(doc, concepts)  
        tree = self.read_tree(parents)
        
        dan_inputs = {
                'tree_positions': position_vector,
                'concepts': concept_vector,
                'compound_aspect': compound_aspect,
                'subtree_aspect': subtree_aspect,
                'noun_chunk_aspect': noun_chunk_aspect, 
                'dependency_tree': tree 
                      }
            
        return dan_inputs     
    
    
    def shortest_path(self, doc, graph, aspect_nodes):
        '''
        Find the shortest path in the dependency graph from each word to the aspect
        '''
        distance_to_aspect = np.zeros(len(doc))
        for tok in doc:
            path_len = []
            for aspect in aspect_nodes: 
                try: path_len.append(nx.shortest_path_length(graph, 
                                                   source='{}-{}'.format(tok.lower_, tok.i), 
                                                   target = aspect))
                except: print('critical error when calculating shortest paths!')
 
            distance_to_aspect[tok.i] = min(path_len)
        
        return distance_to_aspect 
        
        
    # concept to vector, max one concept per token 
    def make_concept_vector(self, doc, concepts):
        concept_vector = np.zeros(len(doc))
        concepts = list(filter(None, concepts))
        for c in concepts:
            if c[0]['text'] in self.bag_of_concepts.word2idx:     
                if concept_vector[c[0]['pos']] != 0:
                    # single concept - >
                    print('>>> overwriting ')
                concept_vector[c[0]['pos']] = self.bag_of_concepts.word2idx[c[0]['text']]
                
        return concept_vector 
        
    
    def extract_concepts(self, tok):
        '''
        Dependency based semantic parser for concept-level text analysis (simplified)
        '''
        concepts = []
        # TODO: add bigram based rules (?) 
        # check : (tok.pos, tok.nbor(1).pos)
        
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
            return concepts
            
        
    def noun_chunk_supersize(self, doc, aspect_start, aspect_end):
      '''
      supersize the aspect based on its adjacency to other noun phrases 
      '''
      aspect_indices = set(range(aspect_start, aspect_end))
      selected = None
      candidates = []
      
      for nc in doc.noun_chunks:
        nc_start = nc.root.left_edge.i
        nc_end = nc.root.right_edge.i+1
        nc_indices = list(range(nc_start, nc_end))
        
        # find new nc2 in the intersection of nc with aspect
        if bool(set(aspect_indices) & set(nc_indices)):
          for nc2 in [ nc, doc[min(aspect_start, nc_start) :max(aspect_end,nc_end)]]:
            candidates.append(nc2)
            
      if len(candidates) > 0:
        selected = max(candidates, key=len)
        if selected[0].dep_ == 'det':
          selected = selected[1:]
        selected = ' '.join(s.text for s in selected) 
        
      return selected 
            
    
    def subtree_supersize(self, aspect_token):
      '''
      supersize the aspect based on prepositional dependency relations (down)
      '''
      subtree = None      
      for c in list(aspect_token.children):
        if c.dep_ == 'prep':
          subtree = [t for t in aspect_token.subtree]
          if subtree[0].dep_ == 'det':  # remove det relations at the begining
            subtree = subtree[1:]
          subtree = ' '.join(s.text for s in subtree)  
    
      return subtree
    
    
    def compound_supersize(self, aspect_token, aspect_term):
      '''
      supersize the aspect based on compound dependency relations (up)
      '''
      compound = None
      if aspect_token.dep_ == 'compound':
        if aspect_token.head.text not in aspect_term:
          compound = aspect_term + ' ' +  aspect_token.head.text
    
      return compound 


    def read_tree(self, parents):
        '''
        converts list of parents into a tree object 
        '''
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
        #print('\n', ' '.join(str(p-1) for p in parents))
        #root.paint()
        return root

    
          

class Tree(object):
    '''
    Tree object from stanfordnlp/treelstm
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
        
        

