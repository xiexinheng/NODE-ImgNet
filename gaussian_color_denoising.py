#export PYTHONPATH="${PYTHONPATH}:/Users/cuiyu.he/Documents/XNODE_ImageDenoising"
#Note to cuiyu only: run using pt1 anaconda environemnt


#new part


#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import argparse
import logging
import os, time, glob
import numpy as np
import random

seed = 19960408
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------------
#                                 define parameters
# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument('--train_data', default='./data/1111111/', type=str, help='path of train data')  # 201807081928tcw
parser.add_argument('--test_dir', default='./data/Test/test_all', type=str, help='directory of test dataset')
#parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
parser.add_argument('--save_every', default=5, type=int,
                    help='save model at every x epoches')  # every x epoches save model
parser.add_argument('--pretrain', default=None, type=str, help='path of pre-trained model')
parser.add_argument('--only_test', default=False, type=bool, help='train and test or only test')
parser.add_argument('--Model_Path', default='./snapshot/save_BRDNet_sigma75_2022-10-14-21-21-48/model_epoch.pt', type=str, help='Path of model')

# new parameters
parser.add_argument('--train_file_path', default='./data/1111111/', type=str, help='path of train data')
#parser.add_argument('--train_file_path', default='./data/11150/', type=str, help='path of train data') #test
parser.add_argument('--scales', default=[1], help='scale')
parser.add_argument('--patch_size', default=60, type=int, help='patch size')
parser.add_argument('--patch_per_image', default=300, type=int, help='patch per image')
parser.add_argument('--test_file_path', default='./data/Test/test_all', type=str, help='directory of test dataset')
parser.add_argument('--load', default=False, help='file name to load')
parser.add_argument('--data_test', type=str, default='Set68', help='test dataset name')
parser.add_argument('--noise_type', type=str, default = 'Gaussian', help='image noise type')
parser.add_argument('--noise_level', type=float, default=50.0, help='noise_level for Gaussian type noise')
parser.add_argument('--patience', type=int, default=50, help='number of steps for early stopping') #dedault 1 by Yue
parser.add_argument('--test_only', default=False, type=bool, help='train and test or only test')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--decay', type=str, default='200',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--epochs', type=int, default=51,
                    help='number of epochs to train')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--loss', type=str, default = '1*MSE',
                    help='loss function configuration')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', type=str, default='test',
                    help='file name to save')
parser.add_argument('--save_gt', action='store_false',
                    help='save low-resolution and high-resolution images together')
parser.add_argument('--save_results', action='store_false',
                    help='save output results')
parser.add_argument('--save_dir', default = '', help='save_dir')
parser.add_argument('--save_models', action='store_false',
                    help='save all intermediate models')

parser.add_argument('--clean_file_path', default='./data/1111111/patchesdata1/clean/',
                    help='the path of the clean patches image. Valid when use PatchesData')
parser.add_argument('--noise_file_path', default='./data/1111111/patchesdata1/noise/',
                    help='the path of the clean patches image. Valid when use PatchesData')
parser.add_argument('--odeint_adjoint', default=False,
                    help='Enables debug mode')
parser.add_argument('--num_channel', type=int, default=3,
                    help='number of image channel(s), channel for gray image is 1 for color is 3')
parser.add_argument('--augmented_channels', type=int, default=0,
                    help='the number of zero channels to add')
parser.add_argument('--test', type=str, default='', help='test type')
parser.add_argument('--device', type=str, default='cuda', help='test type')
parser.add_argument('--N_t', type=int, default=9,
                    help='the number of times steps to show')
parser.add_argument('--model_name', default='CNN_XNODE_X', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=40, type=int, help='batch size')#### 24-10 change to banchnorm
parser.add_argument('--RHSX_model_name', default='_RHStest4_4_4444444', type=str, help='choose a type of RHSX model')
parser.add_argument('--BRDnet_name', default='no_NEWBNNet', type=str, help='choose a type of BRDnet model if we have')
parser.add_argument('--solver', type=str, default = 'f_random_euler', help='NODE solver')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--t', type=float, default=0.01, help='T')
parser.add_argument('--lr_patience', default=3, type=int, help='decay steps')
parser.add_argument('--decay_rate', default=0.5, type=float, help='decay rate')
parser.add_argument('--load_model', default=False, help='is load or not')
parser.add_argument('--load_model_path', default='./snapshot/save_CNN_XNODE_X_sigma50.0_2022-12-09-01-51-51/experiment/test/model/model_best.pt', help='the path of model')
parser.add_argument('--validation_file_path', default='./data/Test/McMaster', help='validation file path')
parser.add_argument('--percentage_data', default=1.0, help='percentage data we use')
parser.add_argument('--random_euler_random_seed', default=8000, help='f random euler random seed')
parser.add_argument('--is_blind', default=False, help='When train the model, blind noise or not')


args = parser.parse_args()

save_dir = './snapshot/save_' + 'Nt_' + str(args.N_t) + args.model_name + '_' + 'sigma' + str(args.noise_level) + '_' + time.strftime(
    "%Y-%m-%d-%H-%M-%S", time.localtime()) + '/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    # logD
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%Y %H:%M:%S',
                        filename=save_dir + 'info.log',
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-6s: %(levelname)-6s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.info(args)
args.save_dir = save_dir
args.data_test = args.data_test.split('+')


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
dev = torch.device(
    args.device) if torch.cuda.is_available() else torch.device("cpu") # use the GPU if we can



from src.model import NODEImgNet
from src.loss import Loss
from utils.utility import checkpoint
import torch
from src.training import NewTrainer, TrainerTest1, TrainerTest1_test_all_validation
from src.dataset import FullImgData_validation

torch.set_default_tensor_type(torch.FloatTensor)  #default data type for numpy tensor is FloatT

# setting to cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    is_save = False
    if args.load_model:
        _model = NODEImgNet(args)
        _model.load_state_dict(torch.load(args.load_model_path))
    else:
        _model = NODEImgNet(args)
    loader = FullImgData_validation(args, is_save)
    _loss = Loss(args)
    ckp = checkpoint(args)
    t = TrainerTest1_test_all_validation(args, loader, _model, _loss, device, ckp)

    while not t.terminate():
        t.train()
        t.test()
