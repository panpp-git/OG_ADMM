#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import torch
import util
import matplotlib.pyplot as plt
from data import fr
import matplotlib.font_manager as fm
from matplotlib import rcParams



##############################################################################################################
ogfreq_path="model/epoch_320_attnsoft.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#load models
fr_module, _, _, _, _ = util.load(ogfreq_path, 'og_bn', device)
fr_module.cpu()
fr_module.eval()
A=fr_module.Aori.AH.detach().numpy()
B=fr_module.Agrad.BH.detach().numpy()
C=fr_module.Aori.A.detach().numpy()
D=fr_module.Agrad.B.detach().numpy()

fz=9
fz2=6

plt.figure(figsize=(9,5))
plt.subplots_adjust(top=0.99, bottom=0.11, left=0.05, right=0.99, hspace=0.2, wspace=0.13)
plt.subplot(2,4,1)
plt.imshow(abs(np.matmul(C,A)))
plt.gca().set_ylabel('Index', size=fz,fontproperties='Times New Roman')
plt.gca().set_xlabel('(a) Learned Bases: '+r'${\bf{A}}^H{\bf{A}}$',size=fz,fontproperties='Times New Roman')
plt.tick_params(labelsize=fz2)



plt.subplot(2,4,2)
plt.imshow(abs(np.matmul(A,C)))
plt.gca().set_xlabel('(b) Learned Bases: '+r'${\bf{A}}{\bf{A}}^H$',size=fz,fontproperties='Times New Roman')
plt.tick_params(labelsize=fz2)


plt.subplot(2,4,3)
plt.imshow(abs(np.matmul(D,B)))
plt.gca().set_xlabel('(c) Learned Bases: '+r'${\bf{B}}^H{\bf{B}}$',size=fz,fontproperties='Times New Roman')
plt.tick_params(labelsize=fz2)


plt.subplot(2,4,4)
plt.imshow(abs(np.matmul(B,D)))
plt.gca().set_xlabel('(d) Learned Bases: '+r'${\bf{B}}{\bf{B}}^H$',size=fz,fontproperties='Times New Roman')
plt.tick_params(labelsize=fz2)

##############################  Fourier########################################################################
fr_size=128
sig_dim=64
C1 = torch.exp(2j * torch.pi / fr_size * torch.matmul(torch.arange(0, sig_dim)[:, None],
                                                     torch.arange(-fr_size // 2, fr_size // 2)[None, :])) / (fr_size * sig_dim)
D1 = 2j * torch.pi / fr_size * torch.arange(0, sig_dim)[:, None] * C1
A1 = torch.exp(-2j * torch.pi / fr_size * torch.matmul(torch.arange(0, sig_dim)[:, None],
                                                     torch.arange(-fr_size // 2, fr_size // 2)[None, :])) / (fr_size * sig_dim)
B1 = -2j * torch.pi / fr_size * torch.arange(0, sig_dim)[:, None] * A1
C1=C1.detach().cpu().numpy()
D1=D1.detach().cpu().numpy()
A1=A1.detach().cpu().numpy().T
B1=B1.detach().cpu().numpy().T


plt.subplot(2,4,5)
plt.imshow(abs(np.matmul(C1,A1)))
plt.gca().set_ylabel('Index', size=fz,fontproperties='Times New Roman')
plt.gca().set_xlabel('Index\n'+'(e) Fourier Bases: '+r'${\bf{A}}^H{\bf{A}}$',size=fz,fontproperties='Times New Roman')
plt.tick_params(labelsize=fz2)


plt.subplot(2,4,6)
plt.imshow(abs(np.matmul(A1,C1)))
plt.gca().set_xlabel('Index\n'+'(f) Fourier Bases: '+r'${\bf{A}}^H{\bf{A}}$',size=fz,fontproperties='Times New Roman')
plt.tick_params(labelsize=fz2)



plt.subplot(2,4,7)
plt.imshow(abs(np.matmul(D1,B1)))
plt.gca().set_xlabel('Index\n'+'(g) Fourier Bases: '+r'${\bf{B}}^H{\bf{B}}$',size=fz,fontproperties='Times New Roman')
plt.tick_params(labelsize=fz2)


plt.subplot(2,4,8)
plt.imshow(abs(np.matmul(B1,D1)))
plt.gca().set_xlabel('Index\n'+'(h) Fourier Bases: '+r'${\bf{B}}^H{\bf{B}}$',size=fz,fontproperties='Times New Roman')
plt.tick_params(labelsize=fz2)


a=1