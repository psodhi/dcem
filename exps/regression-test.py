#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import typing

import copy

import numpy as np
import numpy.random as npr

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable

import higher
import csv
import os
import datetime

import pickle as pkl
import rff

from dcem import dcem

import hydra

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, colorbar
from matplotlib import rc

from regression import plot_energy_landscape, EnergyNetRFF, UnrollEnergyGD, UnrollEnergyGN, EnergyModelGN

from setproctitle import setproctitle
setproctitle('regression')

plt.ion()
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

@hydra.main(config_path="regression-test.yaml", strict=True)
def main(cfg):
    import sys
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(mode='Verbose',
        color_scheme='Linux', call_pdb=1)
    print('Current dir: ', os.getcwd())

    device = torch.device("cuda") \
      if torch.cuda.is_available() else torch.device("cpu")

    Enet = EnergyNetRFF(1,1,128,1,128)
    print(cfg.model_path)
    Enet.load_state_dict(torch.load(cfg.model_path))
    Enet.eval()

    x_train = torch.linspace(0., 2.*np.pi, steps=cfg.n_samples).to(device)
    y_train = x_train*torch.sin(x_train)
    x = np.linspace(0., 2.*np.pi, num=cfg.n_samples)
    y = np.linspace(0.0, 0.01, num=1)

    X, Y = np.meshgrid(x, y)
    Xflat = torch.from_numpy(X.reshape(-1)).float().to(device).unsqueeze(1)
    Yflat = torch.from_numpy(Y.reshape(-1)).float().to(device).unsqueeze(1)
    Ygtflat = Xflat * torch.sin(Xflat)
    Yflat = Yflat + Ygtflat
    # Xflat = x_train.view(-1,1)
    # Yflat = y_train.view(-1,1)

    for unroll_iter in range(15):
      # model = UnrollEnergyGN(Enet, unroll_iter, 1.0)
      # model.eval()

      model = EnergyModelGN(Enet, 100, 1e9, 1e-3, 10.0)
      # model.eval()

      y_preds,_ = model(Xflat, Ygtflat)
      print(y_preds-Ygtflat)
      loss = F.mse_loss(input=y_preds, target=Ygtflat)
      converged = (torch.abs(y_preds - Ygtflat) < 1e-1).double()
      percent_converged = torch.sum(converged)/torch.numel(converged)
      print(f'Unroll Iter {unroll_iter}: Loss {loss:.5f}  Percent converged {percent_converged:.5f}')


    # plot_energy_landscape(x_train, y_train, Enet)
    # plt.show()
    # plt.pause(3)

    
    

if __name__ == '__main__':
    import sys
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(mode='Verbose',
        color_scheme='Linux', call_pdb=1)
    main()