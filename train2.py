# -*- coding: utf-8 -*-
# file: train.py
# author: alberto <aj.paz167@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import argparse
import math
import os

import utils
from data_utils import (build_tokenizer, build_embedding_matrix, ABSADataset,
                        build_boc)

from models import LSTM, IAN, MemNet, RAM, TD_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/restaurant/base_model', type=str, help = 'Directory containing params.json')


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        tokenizer = build_tokenizer(
            fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
            max_seq_len=opt.max_seq_len,
            dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
        embedding_matrix = build_embedding_matrix(
            word2idx=tokenizer.word2idx,
            embed_dim=opt.embed_dim,
            dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
        if opt.model_name != 'base_model':
            boc = build_boc(' ', dat_fname='bag_of_concepts.dat')
            affective_matrix = build_embedding_matrix( word2idx=boc.word2idx,
                embed_dim=100,dat_fname='100_concept_embeddings.dat')
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
        else: 
            boc = None
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
        
        trainset = ABSADataset(opt.dataset_file['train'], tokenizer, boc)
        testset = ABSADataset(opt.dataset_file['test'], tokenizer, boc)
        
        self.train_data_loader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True)
        self.test_data_loader = DataLoader(dataset=testset, batch_size=opt.batch_size, shuffle=False)

        if opt.device.type == 'cuda':
            print("cuda memory allocated:", torch.cuda.memory_allocated(device=opt.device.index))
        self._print_args()
        

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))


    def _reset_params(self):
        for child in self.model.children():
            for p in child.parameters():
                if p.requires_grad:
                    if len(p.shape) > 1:
                        self.opt.initializer(p)
                    else:
                        stdv = 1. / math.sqrt(p.shape[0])
                        torch.nn.init.uniform_(p, a=-stdv, b=stdv)


    def _train(self, criterion, optimizer, max_test_acc_overall=0):
        writer = SummaryWriter(log_dir=self.opt.logdir)
        max_test_acc = 0
        max_f1 = 0
        global_step = 0
        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            n_correct, n_total = 0, 0
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1

                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = sample_batched['polarity'].to(self.opt.device)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total

                    test_acc, f1 = self._evaluate_acc_f1()
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                        if test_acc > max_test_acc_overall:
                            path = '{0}/{1}_{2}_acc{3}'.format(self.opt.model_dir,
                                                               self.opt.model_name, 
                                                               self.opt.dataset, 
                                                               round(test_acc, 4))
                            torch.save(self.model.state_dict(), path)
                            print('>> saved: ' + path)                      
                    if f1 > max_f1:
                        max_f1 = f1
                    writer.add_scalar('loss', loss, global_step)
                    writer.add_scalar('acc', train_acc, global_step)
                    writer.add_scalar('test_acc', test_acc, global_step)
                    print('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'.format(loss.item(), train_acc, test_acc, f1))

        writer.close()
        return max_test_acc, max_f1

    def _evaluate_acc_f1(self):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                t_inputs = [t_sample_batched[col].to(opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(opt.device)
                t_outputs = self.model(t_inputs)

                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_test_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return test_acc, f1

    def run(self, repeats=1):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        max_test_acc_overall = 0
        max_f1_overall = 0
        for i in range(repeats):
            print('repeat: ', i)
            self._reset_params()
            max_test_acc, max_f1 = self._train(criterion, optimizer, max_test_acc_overall=max_test_acc_overall)
            print('max_test_acc: {0}     max_f1: {1}'.format(max_test_acc, max_f1))
            max_test_acc_overall = max(max_test_acc, max_test_acc_overall)
            max_f1_overall = max(max_f1, max_f1_overall)
            print('#' * 100)
        print("max_test_acc_overall:", max_test_acc_overall)
        print("max_f1_overall:", max_f1_overall)


if __name__ == '__main__':
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path),  'No json configuration file found at {}'.format(json_path)
    opt = utils.Params(json_path)
    
    model_classes = {
        'base_model': Cabasc, 
        'cabasc': Cabasc
    }
    dataset_files = {
        'twitter': {
            'train': 'data/datasets/Twitter_Train.raw',
            'test': 'data/datasets/Twitter_Test.raw'
        },
        'restaurant': {
            'train': 'data/datasets/Restaurants_Train.xml.seg',
            'test': 'data/datasets/Restaurants_Test.xml.seg'
        },
        'laptop': {
            'train': 'data/datasets/Laptops_Train.xml.seg',
            'test': 'data/datasets/Laptops_Test.xml.seg'
        }
    }
    input_colses = {
        'base_model': ['text_raw_indices', 'aspect_indices', 'text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
        'cabasc': ['text_raw_indices', 'aspect_indices', 'text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
        'lstm': ['text_raw_indices'],
        'td_lstm': ['text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
        'atae_lstm': ['text_raw_indices', 'aspect_indices'],
        'ian': ['text_raw_indices', 'aspect_indices'],
        'memnet': ['text_raw_without_aspect_indices', 'aspect_indices'],
        'ram': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
        'tnet_lf': ['text_raw_indices', 'aspect_indices', 'aspect_in_text'],
        'aoa': ['text_raw_indices', 'aspect_indices'],
        'mgan': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }

    opt.model_class = model_classes[opt.model_name]
    opt.model_dir = args.model_dir 
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # build common concept embeds for all dataset
    if os.path.exists('data/embeddings/bag_of_concepts.dat') == False:
        print('building the bag of concepts...')
        print('')
        bag_of_concepts = build_boc(
        fnames=[dataset_files[k][t] for t in ['train','test'] for k in dataset_files],
                                      dat_fname='bag_of_concepts.dat')

    ins = Instructor(opt)
    ins.run(1)  # _reset_params in every repeat
