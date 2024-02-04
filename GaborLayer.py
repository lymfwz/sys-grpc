import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import math
import warnings


class GaborConv2d(nn.Module):
    '''
    DESCRIPTION: an implementation of the Learnable Gabor Convolution (LGC) layer \n
    INPUTS: \n
    channel_in: should be 1 \n
    channel_out: number of the output channels \n
    kernel_size, stride, padding: 2D convolution parameters \n
    init_ratio: scale factor of the initial parameters (receptive filed) \n
    '''

    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=1, init_ratio=1, m=0.7):
        super(GaborConv2d, self).__init__()

        #assert channel_in == 1
        self.m = m

        self.channel_in = channel_in
        self.channel_out = channel_out

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.init_ratio = init_ratio

        self.kernel = 0

        if init_ratio <= 0:
            init_ratio = 1.0
            print('input error!!!, require init_ratio > 0.0, using default...')

        # initial parameters
        self._SIGMA = 9.2 * self.init_ratio
        self._FREQ = 0.057 / self.init_ratio
        self._GAMMA = 2.0

        # shape & scale of the Gaussian functioin:
        self.gamma = nn.Parameter(torch.FloatTensor([self._GAMMA]), requires_grad=True)
        self.sigma = nn.Parameter(torch.FloatTensor([self._SIGMA]), requires_grad=True)
        self.theta = nn.Parameter(torch.FloatTensor(torch.arange(0, channel_out).float()) * math.pi / channel_out,
                                  requires_grad=False)

        # frequency of the cosine envolope:
        self.f = nn.Parameter(torch.FloatTensor([self._FREQ]), requires_grad=True)
        self.psi = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)


    def genGaborBank(self, kernel_size, channel_in, channel_out, sigma, gamma, theta, f, psi):
        xmax = kernel_size // 2
        ymax = kernel_size // 2
        xmin = -xmax
        ymin = -ymax

        ksize = xmax - xmin + 1
        y_0 = torch.arange(ymin, ymax + 1).float()
        x_0 = torch.arange(xmin, xmax + 1).float()

        # [channel_out, channelin, kernel_H, kernel_W]
        y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1)
        x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize)

        x = x.float().to(sigma.device)
        y = y.float().to(sigma.device)

        # Rotated coordinate systems
        # [channel_out, <channel_in, kernel, kernel>], broadcasting
        x_theta = x * torch.cos(theta.view(-1, 1, 1, 1)) + y * torch.sin(theta.view(-1, 1, 1, 1))
        y_theta = -x * torch.sin(theta.view(-1, 1, 1, 1)) + y * torch.cos(theta.view(-1, 1, 1, 1))

        gb = -torch.exp(
            -0.5 * ((gamma * x_theta) ** 2 + y_theta ** 2) / (8 * sigma.view(-1, 1, 1, 1) ** 2)) \
             * torch.cos(2 * math.pi * f.view(-1, 1, 1, 1) * x_theta + psi.view(-1, 1, 1, 1))

        gb = gb - gb.mean(dim=[2, 3], keepdim=True)

        return gb

    def forward(self, x):
        kernel = self.genGaborBank(self.kernel_size, self.channel_in, self.channel_out, self.sigma, self.gamma,
                                   self.theta, self.f, self.psi)
        self.kernel = kernel
        #print("kernel",kernel.shape)

        out_normal = F.conv2d(x, kernel, stride=self.stride, padding=self.padding)

        if math.fabs(self.m - 0.0) < 1e-8:
            return out_normal
        else:
            #kernel_diff = torch.sum(torch.sum(kernel, dim=2), dim=2)
            kernel_diff = kernel.sum(2).sum(2)
            #print("1:",kernel_diff.shape)
            kernel_diff = kernel_diff[:, :, None, None]
            #print("2:",kernel_diff.shape)
            out_diff = F.conv2d(x, kernel_diff, stride=self.stride, padding=0)

            return out_normal - self.m * out_diff