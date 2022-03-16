#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt
"""
import datetime

import scipy.io as sio


import os
import sys
import argparse

from matplotlib import ticker
from tqdm import tqdm
import torch
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from dvae.learning_algo import LearningAlgorithm
from dvae.learning_algo_ss import LearningAlgorithm_ss
from dvae.utils.eval_metric import compute_median, EvalMetrics
from dvae.utils.random_seeders import set_random_seeds

set_random_seeds(666)

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # Basic config file
        self.parser.add_argument('--ss', action='store_true', help='schedule sampling')
        self.parser.add_argument('--cfg', type=str, default=None, help='config path')
        self.parser.add_argument('--saved_dict', type=str, default=None, help='trained model dict')
        self.parser.add_argument('--date', type=str, default=None, help='date and time when save training')
        # Dataset
        self.parser.add_argument('--test_dir', type=str, default='./data/clean_speech/wsj0_si_et_05', help='test dataset')
        # Restuls directory
        self.parser.add_argument('--ret_dir', type=str, default='./data/tmp', help='tmp dir for audio reconstruction')
    def get_params(self):
        self._initial()
        self.opt = self.parser.parse_args()
        params = vars(self.opt)
        return params

params = Options().get_params()

if params['ss']:
    learning_algo = LearningAlgorithm_ss(params=params)
else:
    learning_algo = LearningAlgorithm(params=params)
learning_algo.build_model()
dvae = learning_algo.model
dvae.load_state_dict(torch.load(params['saved_dict'], map_location='cpu'))
eval_metrics = EvalMetrics(metric='all')
dvae.eval()
cfg = learning_algo.cfg
print('Total params: %.2fM' % (sum(p.numel() for p in dvae.parameters()) / 1000000.0))


# Load configs
data_path = cfg.get('User', 'data_path')
sequence_len = cfg.getint('DataFrame', 'sequence_len')
dataset_name = cfg.get('DataFrame', 'dataset_name')

saved_root = cfg.get('User', 'saved_root')
z_dim = cfg.getint('Network','z_dim')
tag = cfg.get('Network', 'tag')
date = '2022-03-' + params["date"]
filename = "{}_{}_{}_z_dim={}".format(dataset_name, date, tag, z_dim)
save_dir = os.path.join(saved_root, filename) + '/'

rat_data = sio.loadmat(data_path)

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
x_all = np.array(np.array_split(spike_by_neuron_use, idx_split[1:-1], axis=0))
trial_ls = [len(x) for x in x_all]
num_trial = len(x_all)

for ii in range(len(u)):
    u[ii][:, int(ii % 2) + 1] = 1

# add zero samples by sequence length, will be remove when plotting
max_seq_len = np.max([len(trial) for trial in x_all])  # 351
min_seq_len = np.min([len(trial) for trial in x_all])  # 70
temp = torch.zeros((len(x_all), sequence_len, x_all[0].shape[1]))
for i, x in enumerate(x_all):
    sample = torch.FloatTensor(x)
    if sequence_len <= len(sample):
        sample = sample[:sequence_len]
    elif sequence_len > len(sample):
        s_len, x_dim = sample.shape
        zeros = torch.zeros(sequence_len - s_len, x_dim)
        sample = torch.cat([sample, zeros], 0)

    # if sequence_len <= min_seq_len:
    #     sample = sample[:sequence_len]
    # elif sequence_len >= max_seq_len:
    #     s_len, x_dim = sample.shape
    #     zeros = torch.zeros(sequence_len - s_len, x_dim)
    #     sample = torch.cat([sample, zeros], 0)
    #     assert sample.shape[0] == max_seq_len

    temp[i] = sample

x_all = temp
x_all = x_all.permute(1, 0, 2)

with torch.no_grad():
    outputs = dvae.inference(x_all)
    _, z_mean, _ = outputs
    z_mean = z_mean.permute(1, 0, 2).reshape(-1, 2).numpy()


def get_tc_rd(y, hd, hd_bins):  # compute empirical tunning curve of data
    tuning_curve = np.zeros((len(hd_bins) - 1, y.shape[1]))
    for ii in range(len(hd_bins) - 1):
        data_pos = (hd >= hd_bins[ii]) * (hd <= hd_bins[ii + 1])
        tuning_curve[ii, :] = y[data_pos, :].mean(axis=0)
    return tuning_curve

## posterior mean
# We need the direction information for hue
# and the location information for shade
# So we restore u_all, which should only be used
# for these two purposes from now.


temp = []
ind = 0
for ii in range(num_trial):
    length = min(trial_ls[ii], sequence_len)
    z_m = z_mean[ind:ind+length]
    temp.append(z_m)
    ind = ind + sequence_len

z_mean = np.concatenate(temp)



locations_vec = rat_data['loc'][0]

u_all = np.array(
    np.array_split(np.hstack((locations_vec.reshape(-1, 1), np.zeros((locations_vec.shape[0], 2)))), idx_split[1:-1],
                   axis=0))

temp = []
for u in u_all:
    temp.append(u[:sequence_len])
u_all = temp

for ii in range(len(u_all)):
    u_all[ii][:, int(ii % 2) + 1] = 1;


ll = 11
hd_bins = np.linspace(0, 1.6, ll)
select = np.concatenate(u_all)[:, 1] == 1
print(z_mean.shape)
# print(u_all.shape)
tc1 = get_tc_rd(z_mean[select], np.concatenate(u_all)[select, 0], hd_bins)
# plt.plot(np.concatenate(u_all)[select, 0], color='r')

select = np.concatenate(u_all)[:, 2] == 1
tc2 = get_tc_rd(z_mean[select], np.concatenate(u_all)[select, 0], hd_bins)
# plt.plot(np.concatenate(u_all)[select, 0], color='b')

dis_mat = np.zeros((len(tc1), len(tc2)))
for jj in range(len(tc1)):
    dis_mat[jj] = np.sqrt(np.square(tc1[jj] - tc2).sum(axis=-1))

ll = 5000
fig = plt.figure(figsize=(5.5, 4))
ax = plt.subplot(111)
# fig.add_subplot(111, projection='3d')
fsz = 14


## learn locations
select = np.concatenate(u_all)[:ll, 1] == 1

im = ax.scatter(
    z_mean[:ll][select][:, 0],
    z_mean[:ll][select][:, 1],
    s=1,
    c=np.concatenate(u_all)[:ll][select, 0],
    cmap="Reds",
    vmin=0,
    vmax=1.6,
)
ax.plot(tc1[:, 0], tc1[:, 1], c="black")
cbar = plt.colorbar(im)
cbar.ax.tick_params(labelsize=14)
tick_locator = ticker.MaxNLocator(nbins=5)
cbar.locator = tick_locator
cbar.update_ticks()

## learn locations
select = np.concatenate(u_all)[:ll][:, 1] == 0

im = ax.scatter(
    z_mean[:ll][select][:, 0],
    z_mean[:ll][select][:, 1],
    s=1,
    c=np.concatenate(u_all)[:ll][select, 0],
    cmap="Blues",
    vmin=0,
    vmax=1.6,
)
ax.plot(tc2[:, 0], tc2[:, 1], c="black")
cbar = plt.colorbar(im)
cbar.ax.tick_params(labelsize=14)
tick_locator = ticker.MaxNLocator(nbins=5)
cbar.locator = tick_locator
cbar.update_ticks()
ax.set_xlabel("Latent 1", fontsize=fsz)
ax.set_ylabel("Latent 2", fontsize=fsz)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.setp(ax.get_xticklabels(), fontsize=fsz)
plt.setp(ax.get_yticklabels(), fontsize=fsz)

ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4, min_n_ticks=4, prune=None))
ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, min_n_ticks=4, prune=None))

plt.savefig(save_dir + "z")
plt.show()
