from tkinter import N
import torch
import os
from torch import nn
from torchdiffeq import odeint, odeint_adjoint
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
# from src.dataset import fillt

def init_weights(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)  #let lay.weight under Uniform Distribution
        layer.bias.data.fill_(0) #output = sum (weights * inputs) + bias

class NODEImgNet(nn.Module):
    '''
    The generic CNN module for learning with Neural XODEs.

    This class wraps the `_F` field that acts like an RNN-Cell. This method simply initialises the hidden dynamics
    and computes the updated hidden dynamics through a call to `ode_int` using the `_F` as the function that
    computes the update.
    '''
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.t = args.t
        self.N_t = args.N_t
        self.num_channel = args.num_channel
        self.augmented_channels = args.augmented_channels
        self.solver = args.solver
        self.adjoint = args.odeint_adjoint
        self.save_models = args.save_models
        self.in_channel = args.num_channel
        self.augmented_channels = args.augmented_channels
        self.down_conv = nn.Conv2d(self.num_channel + self.augmented_channels, self.num_channel, kernel_size=3,
                                   padding=1, bias=False)
        self.ODE_rhs_x = globals()[self.args.RHSX_model_name](self.args)
        if self.args.load_model == False:
            self.ODE_rhs_x.apply(init_weights)
        self.dev = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")

    def forward(self, noised_image: torch.Tensor):
        out = self.prepare_data(
            noised_image)  # augment channels of zero, expand NI to have dim 4, and store channel number in NI.shape[1]
        # print('h0 shape={}\n'.format(h0.shape)) #h0 shape=torch.Size([25600, 20])
        odeint_solver = odeint
        #out = self.first_BNDNET_layer(out)
        #odeint_solver = odeint
        ####decoder + BRDNET_XNODE + encoder decoder (e.g., apply average polling ) #test#####
        #timesteps = torch.from_numpy(np.linspace(0, 1, self.N_t)).to(self.dev) #######test
        timesteps = torch.from_numpy(np.linspace(0, self.t, self.N_t)).to(self.dev) #test
        #timesteps = torch.from_numpy(np.linspace(0, 1, 6)).to(self.dev)  #test2
        #print(timesteps)
        outx = odeint_solver(func=self.ODE_rhs_x, y0=out, t=timesteps, method=self.solver)[-1]  # [N_t,shape[y0]] -> shape[y0]
        if self.augmented_channels > 0:
            outx = self.down_conv(outx)
        #out = torch.sub(noised_image, out) # XY test 1
        return outx

    def save(self, apath, epoch, is_best=False):
        save_dirs = [os.path.join(apath, 'model_latest.pt')]
        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pt'))
        if self.save_models:
            save_dirs.append(
                os.path.join(apath, 'model_{}.pt'.format(epoch))
            )
        for s in save_dirs:  # torch.save(model.state_dict(), PATH)
            torch.save(self.state_dict(), s)
            # print('model saving success')

    def prepare_data(self, NI: torch.Tensor):
        # augment channels of zero
        aug_size = list(NI.shape)
        aug_size[-3] = self.augmented_channels  # the channel number is assumed to be located in the last dimension
        aug_channels = torch.zeros(*aug_size, dtype=NI.dtype, layout=NI.layout, device=NI.device)
        NI_aug = torch.cat([NI, aug_channels], -3)
        return NI_aug