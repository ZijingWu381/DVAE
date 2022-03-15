#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Software dvae-speech
Copyright Inria
Year 2020
Contact : xiaoyu.bie@inria.fr
License agreement in LICENSE.txt
"""

import os
import random
import numpy as np
import scipy.io as sio
import torch
from torch.utils import data

def build_dataloader(cfg):


    # Load dataset params for RATRUN subset
    data_path = cfg.get('User', 'data_path')
    dataset_name = cfg.get('DataFrame', 'dataset_name')
    batch_size = cfg.getint('DataFrame', 'batch_size')
    split1 = cfg.getint('DataFrame', 'split1')
    split2 = cfg.getint('DataFrame', 'split2')
    shuffle = cfg.getboolean('DataFrame', 'shuffle')
    num_workers = cfg.getint('DataFrame', 'num_workers')
    sequence_len = cfg.getint('DataFrame', 'sequence_len')
    use_random_seq = cfg.getboolean('DataFrame', 'use_random_seq')

    # Training dataset
    train_dataset = RatHippocampus(data_path, sequence_len, [0, split1])
    val_dataset = RatHippocampus(data_path, sequence_len, [split1, split2])

    train_num = train_dataset.__len__()
    val_num = val_dataset.__len__()

    # Create dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                                shuffle=shuffle, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                shuffle=shuffle, num_workers=num_workers)

    return train_dataloader, val_dataloader, train_num, val_num


class RatHippocampus(data.Dataset):
    """
    Customize a dataset of speech sequences for Pytorch
    at least the three following functions should be defined.
    """
    def __init__(self, data_path, sequence_len, splits, name='RATRUN'):

        super().__init__()
        
        # data parameters
        self.data_path = data_path
        self.sequence_len = sequence_len
        self.splits = splits
        self.x, self.u = self.load_data()

    def __len__(self):
        """
        arguments should not be modified
        Return the total number of samples
        """
        return len(self.x)

    def __getitem__(self, index):
        """
        input arguments should not be modified
        torch data loader will use this function to read ONE sample of data from a list that can be indexed by
        parameter 'index'
        """

        sample = torch.FloatTensor(self.x[index])
        if self.sequence_len > len(sample):
            s_len, x_dim = sample.shape
            zeros = torch.zeros(self.sequence_len - s_len, x_dim)
            sample = torch.cat([sample, zeros], 0)
        elif self.sequence_len <= len(sample):
            sample = sample[:self.sequence_len]

        # if self.sequence_len <= self.min_seq_len:
        #     sample = sample[:self.sequence_len]
        # elif self.sequence_len >= self.max_seq_len:
        #     s_len, x_dim = sample.shape
        #     zeros = torch.zeros(self.sequence_len - s_len, x_dim)
        #     sample = torch.cat([sample, zeros], 0)
        #     assert sample.shape[0] == self.max_seq_len

        return sample

    def load_data(self):
        # load data
        # rat_data = sio.loadmat("data/achilles_data/Achilles_data.mat")
        rat_data = sio.loadmat(self.data_path)

        ## load trial information
        idx_split = rat_data["trial"][0]
        ## load spike data
        spike_by_neuron_use = rat_data["spikes"]
        ## load locations
        locations_vec = rat_data["loc"][0]

        u = np.array(
            np.array_split(
                np.hstack((locations_vec.reshape(-1, 1), np.zeros((locations_vec.shape[0], 2)))), idx_split[1:-1],
                axis=0
            )
        )
        x = np.array(np.array_split(spike_by_neuron_use, idx_split[1:-1], axis=0))
        for ii in range(len(u)):
            u[ii][:, int(ii % 2) + 1] = 1

        # get max and min sequence length
        self.max_seq_len = np.max([len(trial) for trial in x])  # 351
        self.min_seq_len = np.min([len(trial) for trial in x])  # 70
        assert self.min_seq_len == 70

        u = u[self.splits[0]: self.splits[1]]
        x = x[self.splits[0]: self.splits[1]]
        return x, u