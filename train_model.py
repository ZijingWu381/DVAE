#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt
"""

import sys
import os
import shutil
import socket
import datetime
import pickle
from configparser import ConfigParser
import build_model

# import plot functions and disabhle the figure display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def train_model(config_file):
    
    # Build model using config_file
    model_class = build_model(config_file)
    model, optimizer = model_class.build_net()
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.print_model()
    
    # Create dataloader
    train_dataloader, val_dataloader, train_num, val_num = model_class.build_dataloader()
    
    # Create python list for loss
    train_loss = np.zeros((model_class.epochs,))
    val_loss = np.zeros((model_class.epochs,))
    best_val_loss = np.inf
    cpt_patience = 0

    for epoch in range(model_class.epochs):

        start_time = datetime.datetime.now()
        model.train()

        # Batch training
        for batch_idx, batch_data in enumerate(train_dataloader):
            batch_data = batch_data.to(model_class.device)
            recon_batch_data, mean, logvar, z = model(batch_data)
            loss = model_class.loss_function(recon_batch_data, batch_data, mean, logvar)
            loss.backward()
            train_loss[epoch] += loss.item()
            optimizer.step()
        
        # Cross validation
        for batch_idx, batch_data in enumerate(val_dataloader):
            batch_data = batch_data.to(model_class.device)
            recon_batch_data, mean, logvar, z = model(batch_data)
            loss = model_class.loss_function(recon_batch_data, batch_data, mean, logvar)
            val_loss[epoch] += loss.item()

        # Early stop patiance
        if val_loss[epoch] < best_val_loss:
            best_val_loss = val_loss[epoch]
            cpt_patience = 0
            best_state_dict = model.state_dict()
            cur_best_epoch = epoch
        else:
            cpt_patience += 1


        train_loss[epoch] = train_loss[epoch] / train_num
        val_loss[epoch] = val_loss[epoch] / val_num

        end_time = datetime.datetime.now()
        interval = (end_time - start_time).seconds / 60
        print('=====> Epoch: {} train loss: {:.4f} val loss {:.4f} training time {:.2f}m'.format(epoch, train_loss[epoch], val_loss[epoch], interval))

        # Stop traning if early-stop triggers
        if cpt_patience == model_class.early_stop_patience:
            print('===== Early stop patience achieved =====')
            break

        # Save model parameters regularly
        if epoch % model_class.save_frequency == 0:
            save_file = os.path.join(model_class.save_dir, 
                                     model_class.model_name + '_epoch' + str(cur_best_epoch) + '.pt')
            torch.save(best_state_dict, save_file)
    
    # Save the final weights of network with the best validation loss
    train_loss = train_loss[:epoch+1]
    val_loss = val_loss[:epoch+1]
    save_file = os.path.join(model_class.save_dir, 
                             model_class.model_name + '_final_epoch' + str(cur_best_epoch) + '.pt')
    torch.save(best_state_dict, save_file)
    
    # Save the training loss and validation loss
    loss_file = os.path.join(save_dir, 'loss_model.pckl')
    with open(loss_file, 'wb') as f:
        pickle.dump([train_loss, val_loss], f)

    # Save the model parameters
    save_cfg = os.path.join(model_class.save_dir, 'config.ini')
    shutil.copy(config_file, save_cfg)

    # Save the loss figure
    plt.clf()
    plt.plot(train_loss, '--o')
    plt.plot(val_loss, '--x')
    plt.legend(('train loss', 'val loss'))
    plt.xlabel('epochs')
    loss_figure_file = os.path.join(model_class.save_dir, 'loss.pdf')
    plt.savefig(loss_figure_file)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        config_file = sys.argv[1]
        train_model(config_file)
    else:
        print("Please indiquate config file")
