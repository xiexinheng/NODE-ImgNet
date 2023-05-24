from tkinter import N
import torch
import os
from torch import nn
from torchdiffeq import odeint, odeint_adjoint
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F


# from src.dataset import fillt


class _ODEVectorField128(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.Conv2D_6to128 = nn.Conv2d(in_channels=args.num_channel + 1, out_channels=128, kernel_size=3, stride=1,
                                       padding=1)
        self.BatchNorm2d_128_1 = nn.BatchNorm2d(128, eps=1e-3)
        self.Conv2D_128to128_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4,
                                           dilation=4)
        self.BatchNorm2d_128_2 = nn.BatchNorm2d(128, eps=1e-3)
        self.Conv2D_128to128_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4,
                                           dilation=4)
        self.BatchNorm2d_128_3 = nn.BatchNorm2d(128, eps=1e-3)
        self.Conv2D_128to128_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4,
                                           dilation=4)
        self.BatchNorm2d_128_4 = nn.BatchNorm2d(128, eps=1e-3)
        self.Conv2D_128to128_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4,
                                           dilation=4)
        self.BatchNorm2d_128_5 = nn.BatchNorm2d(128, eps=1e-3)
        self.Conv2D_128to128_5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4,
                                           dilation=4)
        self.BatchNorm2d_128_6 = nn.BatchNorm2d(128, eps=1e-3)
        self.Conv2D_128to128_6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4,
                                           dilation=4)
        self.BatchNorm2d_128_7 = nn.BatchNorm2d(128, eps=1e-3)
        self.Conv2D_128to128_7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,
                                           padding=1, dilation=1)
        self.BatchNorm2d_128_8 = nn.BatchNorm2d(128, eps=1e-3)
        self.Conv2D_128to3 = nn.Conv2d(in_channels=128, out_channels=args.num_channel, kernel_size=3, stride=1,
                                       padding=1)
        self.dev = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
        self.to(self.dev)

    def forward(self, t, h):
        # 1st layer, Conv+relu
        input = self.concat_t(h, t)  # 6
        output = self.Conv2D_6to128(input)
        output = F.relu(self.BatchNorm2d_128_1(output))
        output = self.Conv2D_128to128_1(output)
        output = F.relu(self.BatchNorm2d_128_2(output))
        output = self.Conv2D_128to128_2(output)
        output = F.relu(self.BatchNorm2d_128_3(output))
        output = self.Conv2D_128to128_3(output)
        output = F.relu(self.BatchNorm2d_128_4(output))
        output = self.Conv2D_128to128_4(output)
        output = F.relu(self.BatchNorm2d_128_5(output))
        output = self.Conv2D_128to128_5(output)
        output = F.relu(self.BatchNorm2d_128_6(output))
        output = self.Conv2D_128to128_6(output)
        output = F.relu(self.BatchNorm2d_128_7(output))
        output = self.Conv2D_128to128_7(output)
        output = F.relu(self.BatchNorm2d_128_8(output))
        output = self.Conv2D_128to3(output)
        return output

    def concat_t(self, h: torch.Tensor, t):
        h_shape = h.shape
        tt = torch.ones(h_shape[0], 1, h_shape[2], h_shape[3]).to(self.dev) * t
        out_ = torch.cat((h, tt), dim=1).float()  # shape =[bach_size, features+1, N_x, N_y]
        return out_


class NODEImgNet(nn.Module):
    '''
    The generic CNN module for learning with Neural ODEs.

    This class wraps the `_F` field that acts like an RNN-Cell. This method simply initialises the hidden dynamics
    and computes the updated hidden dynamics through a call to `ode_int` using the `_F` as the function that
    computes the update.
    '''

    def __init__(self, args):
        super().__init__()
        self.odeint_adjoint = args.odeint_adjoint
        self.t = args.t
        self.N_t = args.N_t
        self.solver = args.solver
        self.save_models = args.save_models
        self.ODE_vector_field = globals()[args.ODE_vector_field](args)
        self.dev = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")

    def forward(self, noise_image: torch.Tensor):
        if self.odeint_adjoint:
            odeint_solver = odeint_adjoint
        else:
            odeint_solver = odeint
        timesteps = torch.from_numpy(np.linspace(0, self.t, self.N_t + 1)).to(self.dev)
        out = odeint_solver(func=self.ODE_vector_field, y0=noise_image, t=timesteps, method=self.solver)[-1]
        return out

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
