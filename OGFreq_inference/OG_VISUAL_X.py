#!/usr/bin/env python
# coding: utf-8
import h5py
import os
import numpy as np
import torch
import util

import matplotlib.pyplot as plt
from data import fr
import matlab.engine
import h5py
import matplotlib.font_manager as fm
from matplotlib import rcParams
import os
# os.environ['CUDA_VISIBLE_DEVICES']='6'
import pickle

eng = matlab.engine.start_matlab()


ogfreq="model/adapBase_soft.pth"
ogfreq_bn="model/epoch_320_attnsoft.pth"

device = torch.device('cpu')

#load models
og_module, _, _, _, _ = util.load(ogfreq, 'og', device)
og_module.cpu()
og_module.eval()

og_module_bn, _, _, _, _ = util.load(ogfreq_bn, 'og_bn', device)
og_module_bn.cpu()
og_module_bn.eval()

fr_size=128
xgrid = np.linspace(-0.5, 0.5, fr_size, endpoint=False)
reslu=xgrid[1]-xgrid[0]

sigs=['signal1.h5','signal2.h5','signal3.h5']
freq_no=['tgt_num1.h5','tgt_num2.h5','tgt_num3.h5']
real_freqs=['f1.h5','f2.h5','f3.h5']
fig = plt.figure(figsize=[9,5])
for idx in range(3):
    with torch.no_grad():
        signal=pickle.load(open(sigs[idx],'rb'))
        nfreq = pickle.load(open(freq_no[idx], 'rb'))
        f = pickle.load(open(real_freqs[idx], 'rb'))

        file = h5py.File('signal.h5', 'w')  # 创建一个h5文件，文件指针是f
        file['signal'] = signal  # 将数据写入文件的主键data下面
        file['tgt_num'] = nfreq
        file.close()

        mv = np.max(np.sqrt(pow(signal[0], 2) + pow(signal[1], 2)))
        signal[0]=signal[0]/mv
        signal[1] = signal[1] / mv

        ogsbl = eng.OGSBI(nargout=1)
        grsbl=eng.GRSBI(nargout=1)
        admm=eng.ADMM_ref(nargout=1)
        esprit=eng.ESPRIT(nargout=1)

        ogfreq,_,ogfreq_beta,_=og_module(torch.tensor(signal[None]))
        ogfreq_bn, _, ogfreq_bn_beta, _ = og_module_bn(torch.tensor(signal[None]))


        plt.tight_layout()
        plt.subplots_adjust(top=0.99,bottom=0.12,left=0.06,right=0.98,hspace=0.2,wspace=0.16)

        fontsz=8
        fontsz2 = 10

        plt.subplot(2,3,1)
        ogsbl_ret=np.zeros(fr_size)
        ogsbl_ret[np.array(ogsbl['I'])[0,:].astype('int')-1]=abs(np.array(ogsbl['mu'])[:,0])
        ogsbl_beta=np.array(ogsbl['beta'])[:,0]
        title_str = 'OGSBI'
        plt.gca().set_xlabel('(a)', size=fontsz2)
        labels=plt.gca().get_xticklabels()+plt.gca().get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        plt.gca().set_ylabel('Normalized Amp.', size=fontsz2)
        plt.grid(linestyle='-.')
        plt.tick_params(labelsize=fontsz)
        for ii in range(nfreq):
            if ii==0:
                plt.vlines(f[ii], 0, 1.1, color='red', linewidth=1,label='Ground-Truth Value')
            else:
                plt.vlines(f[ii], 0, 1.1, color='red', linewidth=1)
        plt.plot(xgrid+ogsbl_beta, (ogsbl_ret / np.max(ogsbl_ret)),c='k', linewidth=1,label=title_str)
        # plt.xlim([0.03, 0.35])
        # plt.xlim([0.1, 0.5])
        plt.ylim([-0.02,1.36])
        plt.legend(loc='upper right',fontsize=fontsz)


        plt.subplot(2,3,2)
        grsbl_ret=np.zeros(fr_size)
        grsbl_ret[np.array(grsbl['I'])[0,:].astype('int')-1]=abs(np.array(grsbl['mu'])[:,0])
        grsbl_xcorr=np.array(grsbl['grid'])[:,0]
        grsbl_beta=np.zeros(fr_size)
        grsbl_beta[np.array(grsbl['I'])[0,:].astype('int')-1]=grsbl_xcorr[np.array(grsbl['I'])[0,:].astype('int')-1]-xgrid[np.array(grsbl['I'])[0,:].astype('int')-1]
        title_str = 'GRSBI'
        plt.gca().set_xlabel('(b)', size=fontsz2)
        labels=plt.gca().get_xticklabels()+plt.gca().get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        # plt.gca().set_ylabel('Normalized Amp.', size=fontsz)
        plt.grid(linestyle='-.')
        plt.tick_params(labelsize=fontsz)
        for ii in range(nfreq):
            if ii==0:
                plt.vlines(f[ii], 0, 1.1, color='red', linewidth=1,label='Ground-Truth Value')
            else:
                plt.vlines(f[ii], 0, 1.1, color='red', linewidth=1)
        plt.plot(grsbl_xcorr, (grsbl_ret / np.max(grsbl_ret)),c='k', linewidth=1,label=title_str)
        # plt.xlim([0.03,0.35])
        plt.ylim([-0.02,1.36])
        # plt.xlim([0.1, 0.5])
        plt.legend(loc='upper right',fontsize=fontsz)

        plt.subplot(2,3,3)
        ista_ret=np.fft.fftshift(abs(np.array(admm['u'])[:, 0]))
        ista_beta=np.fft.fftshift(np.array(admm['beta'])[:,0])
        title_str = 'ADMM'
        plt.gca().set_xlabel('(c)', size=fontsz2)
        labels = plt.gca().get_xticklabels() + plt.gca().get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        # plt.gca().set_ylabel('Normalized Amp.', size=fontsz)
        plt.grid(linestyle='-.')
        plt.tick_params(labelsize=fontsz)
        for ii in range(nfreq):
            if ii==0:
                plt.vlines(f[ii], 0, 1.1, color='red', linewidth=1,label='Ground-Truth Value')
            else:
                plt.vlines(f[ii], 0, 1.1, color='red', linewidth=1)
        plt.plot(xgrid+ista_beta, ista_ret/np.max(ista_ret), c='k', linewidth=1,label=title_str)
        # plt.xlim([0.03,0.35])
        plt.ylim([-0.02,1.36])
        # plt.xlim([0.1, 0.5])
        plt.legend(loc='upper right',fontsize=fontsz)

        plt.subplot(2,3,4)
        esprit_res=np.array(esprit)
        if esprit_res.size > 1:
            esprit_res = esprit_res[0]

        title_str = 'ESPRIT'
        plt.gca().set_xlabel('Freq. / Hz\n ' +'(d)', size=fontsz2)
        labels = plt.gca().get_xticklabels() + plt.gca().get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        plt.gca().set_ylabel('Normalized Amp.', size=fontsz2)
        plt.grid(linestyle='-.')
        plt.tick_params(labelsize=fontsz)
        for ii in range(nfreq):
            if ii==0:
                plt.vlines(f[ii], 0, 1.1, color='red', linewidth=1,label='Ground-Truth Value')
            else:
                plt.vlines(f[ii], 0, 1.1, color='red', linewidth=1)
        if esprit_res.size > 1:
            for ii in range(nfreq):
                if ii==0:
                    plt.vlines(esprit_res[ii], 0, 1, color='k', linewidth=1,label=title_str)
                else:
                    plt.vlines(esprit_res[ii], 0, 1, color='k', linewidth=1)
        else:
            plt.vlines(esprit_res, 0, 1, color='k', linewidth=1, label=title_str)
        # plt.xlim([0.03,0.35])
        plt.ylim([-0.02,1.36])
        plt.xlim([-0.5, 0.5])
        plt.legend(loc='upper right',fontsize=fontsz)


        plt.subplot(2,3,5)
        ogfreq = ogfreq.squeeze().cpu().abs()
        ogfreq = ogfreq.data.numpy()
        ogfreq_beta = ogfreq_beta[0,:].real
        xgrid_cor = ogfreq_beta.squeeze().cpu() * reslu + xgrid
        title_str = 'Proposed_adapBases_woCA'
        plt.gca().set_xlabel('Freq. / Hz\n '+'(e)', size=fontsz2)
        labels=plt.gca().get_xticklabels()+plt.gca().get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        # plt.gca().set_ylabel('Normalized Amp.', size=fontsz)
        plt.grid(linestyle='-.')
        plt.tick_params(labelsize=fontsz)
        for ii in range(nfreq):
            if ii==0:
                plt.vlines(f[ ii], 0, 1.1, color='red', linewidth=1,label='Ground-Truth Value')
            else:
                plt.vlines(f[ii], 0, 1.1, color='red', linewidth=1)
        plt.plot(xgrid_cor,(ogfreq/ np.max(ogfreq)), c='k',linewidth=1,label=title_str)
        # plt.xlim([0.03,0.35])
        plt.ylim([-0.02,1.36])
        # plt.xlim([0.1, 0.5])
        plt.legend(loc='upper right',fontsize=fontsz)

        plt.subplot(2,3, 6)
        ogfreq_bn = ogfreq_bn.squeeze().cpu().abs()
        ogfreq_bn = ogfreq_bn.data.numpy()
        ogfreq_bn_beta = ogfreq_bn_beta[0, :].real
        xgrid_cor = ogfreq_bn_beta.squeeze().cpu() * reslu + xgrid
        title_str = 'Proposed_adapBases_wiCA'
        plt.gca().set_xlabel('Freq. / Hz\n ' + '(f)', size=fontsz2)
        labels = plt.gca().get_xticklabels() + plt.gca().get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        # plt.gca().set_ylabel('Normalized Amp.', size=fontsz)
        plt.grid(linestyle='-.')
        plt.tick_params(labelsize=fontsz)
        for ii in range(nfreq):
            if ii==0:
                plt.vlines(f[ii], 0, 1.1, color='red', linewidth=1,label='Ground-Truth Value')
            else:
                plt.vlines(f[ii], 0, 1.1, color='red', linewidth=1)
        plt.plot(xgrid_cor, (ogfreq_bn / np.max(ogfreq_bn)), c='k', linewidth=1,label=title_str)
        # plt.xlim([0.03,0.35])
        plt.ylim([-0.02,1.36])
        # plt.xlim([0.1, 0.5])
        plt.legend(loc='upper right',fontsize=fontsz)

        X=1
plt.show()

x=1

