import os
import sys
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
from dataset import TrainSet,ValidSet ,TestSet
from utils import Logger
from train import train_epoch
from validation import val_epoch
import test

if __name__ == '__main__':
    opt = parse_opts()
    if opt.root_path != '':
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    opt.arch ='{}-{}'.format(opt.model_name,opt.model_depth)
    print(opt)

    model,parameters = generate_model(opt)
    print(model)
   # model_data = torch.load(opt.model_name)
   # assert opt.arch == model_data['arch']
   # model.load_state_dict(model_data['state_dict'])
   # model.eval()
    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()

    if not opt.no_train:
        training_data = TrainSet()
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size = opt.batch_size,
            shuffle = True,
            num_workers = opt.n_threads,
            pin_memory =True)
        train_logger = Logger(
            os.path.join(opt.result_path,'train.log'),
            ['epoch','loss','acc','lr']
        )
        train_batch_logger = Logger(
            os.path.join(opt.result_path,'train_batch.log'),
            ['epoch','batch','iter','loss','acc','lr']
        )

        optimizer = optim.Adam(
            parameters,
            lr = opt.learning_rate,
            weight_decay = opt.weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience =opt.lr_patience)
    if not opt.no_val:
        validation_data = ValidSet()
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size = opt.batch_size,
            shuffle = False,
            num_workers = opt.n_threads,
            pin_memory=True
        )
        val_logger =  Logger(
            os.path.join(opt.result_path,'val.log'),['epoch','loss','acc']
        )

    if opt.resume_path:
        print('loading checkpoint{}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch==checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])


    print('run')
    for i in range(opt.begin_epoch,opt.n_epochs+1):
        if not opt.no_train:
            train_epoch(i,train_loader,model,criterion,optimizer,opt,
                        train_logger,train_batch_logger)
        if not opt.no_val:
            validation_loss = val_epoch(i,val_loader,model,criterion,opt,
                                         val_logger)
        if not opt.no_train and not opt.no_val:
            scheduler.step(validation_loss)

    if opt.test:
        test_data = TestSet()
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size = opt.batch_size,
            shuffle=False,
            num_workers = opt.n_threads,
            pin_memory=True
        )
        test.test(test_loader,model,opt,test_data.label)

