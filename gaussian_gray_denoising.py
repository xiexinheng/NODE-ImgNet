import torch
import argparse
import logging
import os, time, glob
import numpy as np
import random

# --------------------------------------------------------------------------------
#                                 define parameters
# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument('--train_file_path',
                    default='./data/GaussianTrainingData/', type=str, help='Path of the training data')
parser.add_argument('--test_file_path',
                    default='./data/GaussianTestData/Gray/Set68', type=str, help='Path of test dataset')
parser.add_argument('--data_test', type=str, default='Set68', help='Name of the test dataset')
parser.add_argument('--validation_file_path',
                    default='./data/GaussianTestData/Gray/Set68', help='Path of the validation data')

parser.add_argument('--CUDA_number', type=str, default='0', help='Ordinal number of the CUDA device to be used')
parser.add_argument('--random_seed', type=int, default=100, help='Random seed to use for generating random numbers')

parser.add_argument('--patch_size', default=60, type=int, help='Patch size')
parser.add_argument('--patch_per_image', default=300, type=int, help='Number of patches per image')
parser.add_argument('--batch_size', default=40, type=int, help='batch Batch size')

parser.add_argument('--optimizer', default='ADAM',  choices=('SGD', 'ADAM', 'RMSprop'),
                    help='Optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--loss', type=str, default = '1*MSE', help='Loss function configuration')
parser.add_argument('--n_GPUs', type=int, default=1, help='Number of GPUs')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='Beta values for ADAM optimizer')
parser.add_argument('--epsilon', type=float, default=1e-8, help='Epsilon value for ADAM optimizer')
parser.add_argument('--print_every', type=int, default=100,
                    help='Number of batches to wait before logging the training status')

parser.add_argument('--save_dir', default='', help='Directory path to save the output files')
parser.add_argument('--save_models', action='store_false',  help='Save all intermediate models or not')
parser.add_argument('--save_results', default=True, help='Save output results')
parser.add_argument('--device', type=str, default='cuda', help='Device to be used for computation')

parser.add_argument('--model_name', default='NODEImgNet', type=str, help='Type of model to use')
parser.add_argument('--odeint_adjoint', default=False, help='For NODE, use odeint_adjoint or not')
parser.add_argument('--N_t', type=int, default=8, help='Number of time steps')
parser.add_argument('--ODE_vector_field', default='_ODEVectorField128', type=str, help='Vector field model to use')
parser.add_argument('--solver', type=str, default = 'f_random_euler', help='NODE solver to use')
parser.add_argument('--t', type=float, default=0.01, help='Value of T')
parser.add_argument('--random_euler_random_seed', default=8000, help='Random seed for f random euler')

parser.add_argument('--test_only', default=False, type=bool, help='Perform testing only or not')
parser.add_argument('--load_model', default=False, help='Load a pretrained model or not')
parser.add_argument('--load_model_path', default='./snapshot/save_CNN_XNODE_X_sigma50.0_2022-12-09-01-51-51/experiment/test/model/model_best.pt', help='Path of the pretrained model')

parser.add_argument('--noise_level', type=float, default=50.0, help='Noise level for Gaussian noise')
parser.add_argument('--is_blind', action='store_true', help='When train the model, blind noise or not')

parser.add_argument('--epochs', type=int, default=50, help='The largest number of epochs to train')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
parser.add_argument('--lr_patience', default=3, type=int, help='Number of epochs to wait for learning rate decay')
parser.add_argument('--decay_rate', default=0.5, type=float, help='Decay rate for learning rate')

parser.add_argument('--patience', type=int, default=50, help='Number of steps for early stopping')
parser.add_argument('--num_channel', type=int, default=1,
                    help='Number of image channels. Use 1 for grayscale and 3 for color')


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


os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_number
dev = torch.device(
    args.device) if torch.cuda.is_available() else torch.device("cpu") # use the GPU if we can
seed = args.random_seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

dev = torch.device(
    args.device) if torch.cuda.is_available() else torch.device("cpu") # use the GPU if we can



from src.model import NODEImgNet
from src.loss import Loss
from utils.utility import checkpoint
import torch
from src.training import ColorTrainer, GrayTrainer
from src.dataset import ColorDataLoader, GrayDataLoader

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
    loader = GrayDataLoader(args)
    _loss = Loss(args)
    ckp = checkpoint(args)
    t = GrayTrainer(args, loader, _model, _loss, device, ckp)

    while not t.terminate():
        t.train()
        t.test()
