# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import os
from complexLayers import  ComplexConv1d,ComplexReLU
import torch.fft as fft
def abs_sep(input):
    return torch.abs(input.real)+1j*torch.abs(input.imag)

class AoriH(torch.nn.Module):
    def __init__(self,A,AH):
        super(AoriH,self).__init__()
        self.A=A
        self.AH=AH

    def forward(self,y):
        x=torch.matmul(self.AH,y)
        recon_y=torch.matmul(self.A,x)
        err=y-recon_y
        return x,err

class AgridH(torch.nn.Module):
    def __init__(self,B,BH):
        super(AgridH,self).__init__()
        self.B=B
        self.BH=BH

    def forward(self,y):
        x=torch.matmul(self.BH,y)
        recon_y=torch.matmul(self.B,x)
        err=y-recon_y
        return x,err

class Aori(torch.nn.Module):
    def __init__(self,A,AH):
        super(Aori,self).__init__()
        self.A=A
        self.AH=AH

    def forward(self,x):
        y=torch.matmul(self.A,x)
        recon_x=torch.matmul(self.AH,y)
        err=x-recon_x
        return y,err

class Agrid(torch.nn.Module):
    def __init__(self,B,BH):
        super(Agrid,self).__init__()
        self.B=B
        self.BH=BH

    def forward(self,x):
        y=torch.matmul(self.B,x)
        recon_x=torch.matmul(self.BH,y)
        err=x-recon_x
        return [y,err]


class BasicBlock(torch.nn.Module):
    def __init__(self,n_filters,reslu,fr_size,device):
        super(BasicBlock, self).__init__()

 
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.reslu=reslu
        self.device=device

        self.conv_D=ComplexConv1d(1, n_filters, kernel_size=3,padding=1)
        self.conv1_forward=ComplexConv1d(n_filters, n_filters, kernel_size=3,padding=1)
        self.conv2_forward=ComplexConv1d(n_filters, n_filters, kernel_size=3,padding=1)
        self.conv1_backward =ComplexConv1d(n_filters, n_filters, kernel_size=3,padding=1)
        self.conv2_backward = ComplexConv1d(n_filters, n_filters, kernel_size=3, padding=1)
        self.conv_G=ComplexConv1d(n_filters, 1,kernel_size=3,padding=1)

 
        self.beta_soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.beta_conv_D=ComplexConv1d(1, n_filters, kernel_size=3,padding=1)
        self.beta_conv1_forward=ComplexConv1d(n_filters, n_filters, kernel_size=3,padding=1)
        self.beta_conv2_forward=ComplexConv1d(n_filters, n_filters, kernel_size=3,padding=1)
        self.beta_conv1_backward =ComplexConv1d(n_filters, n_filters, kernel_size=3,padding=1)
        self.beta_conv2_backward = ComplexConv1d(n_filters, n_filters, kernel_size=3, padding=1)
        self.beta_conv_G=ComplexConv1d(n_filters, 1,kernel_size=3,padding=1)

    
        self.attn=torch.nn.AvgPool1d(kernel_size=fr_size)
        self.fc = nn.Sequential(
            nn.Linear(n_filters, n_filters),
            nn.BatchNorm1d(n_filters),
            nn.ReLU(inplace=True),
            nn.Linear(n_filters, n_filters),
            nn.Sigmoid(),
        )


  
        self.beta_attn=torch.nn.AvgPool1d(kernel_size=fr_size)
        self.fc_beta = nn.Sequential(
            nn.Linear(n_filters, n_filters),
            nn.BatchNorm1d(n_filters),
            nn.ReLU(inplace=True),
            nn.Linear(n_filters, n_filters),
            nn.Sigmoid(),
        )

    def forward(self, x,beta,y,alpha,Aori,AoriH,Agrid,AgridH):
        ## x update
        bsz=x.shape[0]

        Aori_err=torch.zeros(x.size()).to(y.device).type(torch.complex64)
        AoriH_err=torch.zeros(y.size()).to(y.device).type(torch.complex64)
        Agrid_err = torch.zeros(x.size()).to(y.device).type(torch.complex64)
        AgridH_err=torch.zeros(y.size()).to(y.device).type(torch.complex64)
        A_Agrid_err=torch.zeros(y.size()).to(y.device).type(torch.complex64)
        y_alpha=alpha+y
        [AHy,tmp]=AoriH(y_alpha)
        AoriH_err=AoriH_err+(tmp)

        [AgridHy,tmp]=AgridH(y_alpha)
        AgridH_err=AgridH_err+(tmp)
        x_forward=AHy+self.reslu*beta.conj()*AgridHy

        #######
        y_alpha_recon=torch.matmul(Aori.A,x_forward)+torch.matmul(Agrid.B,self.reslu*beta*x_forward)
        A_Agrid_err=A_Agrid_err+(y_alpha_recon-y_alpha)
        #######

        x_input = x_forward

        x_D=self.conv_D(x_input.transpose(-1,-2))
        x=self.conv1_forward(x_D)
        x = ComplexReLU()(x)
        x_forward=self.conv2_forward(x)

        average=self.attn(x_forward.abs()).squeeze(-1)
        attn=self.fc(average)
        x_forward=x_forward-(attn*average).unsqueeze(-1)
        x = torch.mul(torch.sgn(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x=self.conv1_backward(x)
        x = ComplexReLU()(x)
        x_backward =self.conv2_backward(x)

        x_G=self.conv_G(x_backward).transpose(-1,-2)

        x = x_input + x_G

        ## beta update
        [Ax,tmp]=Aori(x)
        Aori_err=Aori_err+(tmp)
        beta_y=alpha+y-Ax
        [BHbetay,tmp]=AgridH(beta_y)
        AgridH_err=AgridH_err+(tmp)
        beta_forward =(x.conj()*self.reslu*BHbetay)

        beta_input = beta_forward

        beta_D = self.beta_conv_D(beta_input.transpose(-1,-2))
        beta = self.beta_conv1_forward(beta_D)
        beta = ComplexReLU()(beta)
        beta_forward = self.beta_conv2_forward(beta)

        average=self.beta_attn(beta_forward.real).squeeze(-1)
        attn=self.fc_beta(average)
        beta_forward=beta_forward-(attn*average).unsqueeze(-1)
        beta = torch.mul(torch.sgn(beta_forward), F.relu(torch.abs(beta_forward) - self.beta_soft_thr))

        beta = self.beta_conv1_backward(beta)
        beta = ComplexReLU()(beta)
        beta_backward = self.beta_conv2_backward(beta)

        beta_G = self.beta_conv_G(beta_backward).transpose(-1,-2)

        beta = beta_input + beta_G


        beta[torch.abs(beta) > 1 / 2] = torch.sgn(
            beta[torch.abs(beta) > 1 / 2]) * 1 / 100

        [Bxb,tmp]=Agrid(beta*self.reslu*x)
        alpha=alpha+y-(Ax+Bxb)
        Agrid_err =Agrid_err+ (tmp)
        x_err=(Aori_err+Agrid_err)/2
        y_err=(AoriH_err+AgridH_err+A_Agrid_err)/4
        return [x,beta,alpha,x_err,y_err]



class OGFreq(torch.nn.Module):
    def __init__(self, LayerNo,signal_dim,n_filters,inner,device,fr_size):
        super(OGFreq, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        self.n_filter=n_filters
        self.inner=inner
        self.device=device
        self.fr_size=fr_size
        self.signal_dim=signal_dim
        self.const=(torch.tensor(fr_size))

    
        self.A=torch.nn.Parameter(torch.randn((signal_dim,fr_size))/self.const+1j*torch.randn((signal_dim,fr_size))/self.const).type(torch.complex64)
        self.AH=torch.nn.Parameter(torch.randn((fr_size,signal_dim))/self.const+1j*torch.randn((fr_size,signal_dim))/self.const).type(torch.complex64)
        self.B=torch.nn.Parameter(torch.randn((signal_dim,fr_size))/self.const+1j*torch.randn((signal_dim,fr_size))/self.const).type(torch.complex64)
        self.BH=torch.nn.Parameter(torch.randn((fr_size,signal_dim))/self.const+1j*torch.randn((fr_size,signal_dim))/self.const).type(torch.complex64)

        self.Aori=Aori(self.A,self.AH)
        self.AoriH=AoriH(self.A,self.AH)
        self.Agrad=Agrid(self.B,self.BH)
        self.AgradH=AgridH(self.B,self.BH)


        for i in range(LayerNo):
            onelayer.append(BasicBlock(n_filters,1/(fr_size-1),fr_size,device))

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, y):

        y = (y[:, 0, :] + 1j * y[:, 1, :]).type(torch.complex64).unsqueeze(1)
        bsz=y.shape[0]
        sig_dim=y.shape[-1]
        fr_size=self.fr_size

 
        y=y.transpose(-1,-2)
        [AHy,tmp]=self.AoriH(y)
        x = AHy
        beta = torch.zeros((bsz,self.inner,1)).to(self.device).type(torch.complex64)/2
        alpha = torch.zeros((bsz, self.signal_dim,1)).to(self.device).type(torch.complex64)
        x=torch.zeros((bsz, self.inner,1)).to(self.device).type(torch.complex64)

        x_err_layers = []
        beta_err_layers = []
        for i in range(self.LayerNo):
            [x, beta,alpha,x_err,beta_err] = self.fcs[i](x,beta,y,alpha,self.Aori,self.AoriH,self.Agrad,self.AgradH)
            x_err_layers.append(x_err)
            beta_err_layers.append(beta_err)
        x_final = x.squeeze(-1)
        beta_final=beta.squeeze(-1)

        return [x_final, x_err_layers,beta_final,beta_err_layers]
