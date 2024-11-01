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
import pickle
import time

os.environ['CUDA_VISIBLE_DEVICES']='0'
eng = matlab.engine.start_matlab()

ogfreq="model/adapBase_soft.pth"
ogfreq_bn="model/epoch_320_attnsoft.pth"



data_dir = 'test_dataset'
device = torch.device('cpu')

#load models
og_module, _, _, _, _ = util.load(ogfreq, 'og', device)
og_module.cpu()
og_module.eval()

og_module_bn, _, _, _, _ = util.load(ogfreq_bn, 'og_bn', device)
og_module_bn.cpu()
og_module_bn.eval()

#load data
f = np.load(os.path.join(data_dir, 'f.npy'))
r = np.load(os.path.join(data_dir, 'r.npy'))
kernel_param_0 = 0.12/ 64

nfreq =  np.sum(f >= -0.5, axis=1)
db=['-10.0dB.npy','-5.0dB.npy','0.0dB.npy','5.0dB.npy','10.0dB.npy','15.0dB.npy','20.0dB.npy','25.0dB.npy','30.0dB.npy']
ITER=1000
fr_size=128
xgrid = np.linspace(-0.5, 0.5, fr_size, endpoint=False)
fr_size2=512
xgrid2 = np.linspace(-0.5, 0.5, fr_size2, endpoint=False)
fr_size3=4096
xgrid3 = np.linspace(-0.5, 0.5, fr_size3, endpoint=False)
reslu=xgrid[1]-xgrid[0]

gt_fr,gt_beta,gt_pos=fr.freq2fr(f, xgrid,  param=kernel_param_0,r=r,nfreq=nfreq)


OGFreq=np.zeros([len(db),1])
OGFreq_time=np.zeros([len(db),1])
OGFreq_cnt=np.zeros([len(db),1])
OGSBI=np.zeros([len(db),1])
OGSBI_time=np.zeros([len(db),1])
OGSBI_cnt=np.zeros([len(db),1])
GRSBI=np.zeros([len(db),1])
GRSBI_time=np.zeros([len(db),1])
GRSBI_cnt=np.zeros([len(db),1])
ADMM=np.zeros([len(db),1])
ADMM_time=np.zeros([len(db),1])
ADMM_cnt=np.zeros([len(db),1])
ESPRIT=np.zeros([len(db),1])
ESPRIT_time=np.zeros([len(db),1])
ESPRIT_cnt=np.zeros([len(db),1])
OGFreq_fourier=np.zeros([len(db),1])
OGFreq_bn=np.zeros([len(db),1])
OGFreq_fourier_time=np.zeros([len(db),1])
OGFreq_fourier_cnt=np.zeros([len(db),1])
OGFreq_bn_time=np.zeros([len(db),1])
OGFreq_bn_cnt=np.zeros([len(db),1])

OGFreq_fourier_wnattn=np.zeros([len(db),1])
OGFreq_fourier_wnattn_time=np.zeros([len(db),1])
OGFreq_fourier_wnattn_cnt=np.zeros([len(db),1])

fig = plt.figure()
for db_iter in range(len(db)):
    signal = np.load(os.path.join(data_dir, db[db_iter]))
    signal2 = np.load(os.path.join(data_dir, db[db_iter]))
    win = np.hamming(signal.shape[2]).astype('float32')
    for idx in range(ITER):
        with torch.no_grad():
            print(db_iter,idx)
            mv = np.max(np.sqrt(pow(signal[idx][0], 2) + pow(signal[idx][1], 2)))
            signal[idx][0]=signal[idx][0]/mv
            signal[idx][1] = signal[idx][1] / mv
            file = h5py.File('signal.h5', 'w')  # 创建一个h5文件，文件指针是f
            file['signal'] = signal2[idx]  # 将数据写入文件的主键data下面
            file['tgt_num']=nfreq[idx]
            file.close()

            tic=time.time()
            ogsbl = eng.OGSBI(nargout=1)
            OGSBI_time[db_iter]+=time.time()-tic
            ogsbl_ret = np.zeros(fr_size)
            if np.array(ogsbl['I']).size==1:
                ogsbl_ret[np.array(ogsbl['I']).astype('int') - 1] = abs(np.array(ogsbl['mu']))
                ogsbl_beta = np.array(ogsbl['beta'])
            else:
                ogsbl_ret[np.array(ogsbl['I'])[0, :].astype('int') - 1] = abs(np.array(ogsbl['mu'])[:, 0])
                ogsbl_beta = np.array(ogsbl['beta'])[:, 0]

            tic = time.time()
            grsbl=eng.GRSBI(nargout=1)
            GRSBI_time[db_iter] += time.time() - tic
            grsbl_ret = np.zeros(fr_size)
            if np.array(grsbl['I']).size==1:
                grsbl_ret[np.array(grsbl['I']).astype('int') - 1] = abs(np.array(grsbl['mu']))
                grsbl_xcorr = np.array(grsbl['grid'])[:, 0]
                grsbl_beta = np.zeros(fr_size)
                grsbl_beta[np.array(grsbl['I']).astype('int') - 1] = grsbl_xcorr[np.array(grsbl['I']).astype('int') - 1] - xgrid[np.array(grsbl['I']).astype('int') - 1]
            else:
                grsbl_ret[np.array(grsbl['I'])[0, :].astype('int') - 1] = abs(np.array(grsbl['mu'])[:, 0])
                grsbl_xcorr = np.array(grsbl['grid'])[:, 0]
                grsbl_beta = np.zeros(fr_size)
                grsbl_beta[np.array(grsbl['I'])[0, :].astype('int') - 1] = grsbl_xcorr[np.array(grsbl['I'])[0, :].astype(
                    'int') - 1] - xgrid[np.array(grsbl['I'])[0, :].astype('int') - 1]

            tic = time.time()
            admm = eng.ADMM_ref(nargout=1)
            ADMM_time[db_iter] += time.time() - tic
            admm_ret = np.fft.fftshift(abs(np.array(admm['u'])[:, 0]))
            admm_beta = np.fft.fftshift(np.array(admm['beta'])[:, 0])

            tic = time.time()
            esprit = eng.ESPRIT(nargout=1)
            ESPRIT_time[db_iter] += time.time() - tic
            esprit_res = np.array(esprit)
            if esprit_res.size>1:
                esprit_res=esprit_res[0]

            tic = time.time()
            ogfreq,_,ogfreq_beta,_=og_module(torch.tensor(signal[idx][None]))
            OGFreq_time[db_iter] += time.time() - tic
            ogfreq = ogfreq.squeeze().cpu().abs()
            ogfreq = ogfreq.data.numpy()
            ogfreq_beta = ogfreq_beta[0, :].real
            xgrid_cor = ogfreq_beta.squeeze().cpu() * reslu + xgrid


            tic = time.time()
            ogfreq_bn,_,ogfreq_bn_beta,_=og_module_bn(torch.tensor(signal[idx][None]))
            OGFreq_bn_time[db_iter] += time.time() - tic
            ogfreq_bn = ogfreq_bn.squeeze().cpu().abs()
            ogfreq_bn = ogfreq_bn.data.numpy()
            ogfreq_bn_beta = ogfreq_bn_beta[0, :].real
            xgrid_bn_cor = ogfreq_bn_beta.squeeze().cpu() * reslu + xgrid



        plt.tight_layout()
        plt.subplots_adjust(top=0.96,bottom=0.1,left=0.06,right=0.96,hspace=0.1,wspace=0.15)


        pos = fr.find_freq_idx(abs(ogfreq_bn), nfreq[idx], xgrid)-1
        if len(pos) == 0:
           pos = np.argsort(-abs(ogfreq_bn))[:nfreq[idx]]
        est=xgrid[pos]+ogfreq_bn_beta.squeeze().cpu().numpy()[pos]*reslu
        gt=(f[idx][:nfreq[idx]])
        for k in range(nfreq[idx]):
           dist_i_direct=np.min(np.abs(est-gt[k]))
           dist_i_rshift = np.min(np.abs(est+1 - gt[k]))
           dist_i_lshift = np.min(np.abs(est-1 - gt[k]))
           dist_i=np.min([dist_i_rshift,dist_i_direct,dist_i_lshift])
           if dist_i<reslu*2:
               OGFreq_bn[db_iter]+=np.power(dist_i,2.0)
               OGFreq_bn_cnt[db_iter] += 1

        # OGFreq
        pos = fr.find_freq_idx(abs(ogfreq), nfreq[idx], xgrid)-1
        if len(pos) == 0:
           pos = np.argsort(-abs(ogfreq))[:nfreq[idx]]
        est=xgrid[pos]+ogfreq_beta.squeeze().cpu().numpy()[pos]*reslu
        gt=(f[idx][:nfreq[idx]])
        for k in range(nfreq[idx]):
           dist_i_direct=np.min(np.abs(est-gt[k]))
           dist_i_rshift = np.min(np.abs(est+1 - gt[k]))
           dist_i_lshift = np.min(np.abs(est-1 - gt[k]))
           dist_i=np.min([dist_i_rshift,dist_i_direct,dist_i_lshift])
           if dist_i<reslu*2:
               OGFreq[db_iter]+=np.power(dist_i,2.0)
               OGFreq_cnt[db_iter]+=1

        ## OGSBI
        pos = fr.find_freq_idx(abs(ogsbl_ret), nfreq[idx], xgrid)-1
        if len(pos) == 0:
           pos = np.argsort(-abs(ogsbl_ret))[:nfreq[idx]]
        est = xgrid[pos] + ogsbl_beta[pos]
        gt=(f[idx][:nfreq[idx]])
        for k in range(nfreq[idx]):
           dist_i_direct=np.min(np.abs(est-gt[k]))
           dist_i_rshift = np.min(np.abs(est+1 - gt[k]))
           dist_i_lshift = np.min(np.abs(est-1 - gt[k]))
           dist_i=np.min([dist_i_rshift,dist_i_direct,dist_i_lshift])
           if dist_i<2*reslu:
               OGSBI[db_iter]+=np.power(dist_i,2.0)
               OGSBI_cnt[db_iter]+=1

        ## GRSBI
        pos = fr.find_freq_idx(abs(grsbl_ret), nfreq[idx], xgrid)-1
        if len(pos) == 0:
           pos = np.argsort(-abs(grsbl_ret))[:nfreq[idx]]
        est=xgrid[pos]+grsbl_beta[pos]
        gt=(f[idx][:nfreq[idx]])
        for k in range(nfreq[idx]):
           dist_i_direct=np.min(np.abs(est-gt[k]))
           dist_i_rshift = np.min(np.abs(est+1 - gt[k]))
           dist_i_lshift = np.min(np.abs(est-1 - gt[k]))
           dist_i=np.min([dist_i_rshift,dist_i_direct,dist_i_lshift])
           if dist_i<2*reslu:
               GRSBI[db_iter]+=np.power(dist_i,2.0)
               GRSBI_cnt[db_iter]+=1

        ## ADMM
        pos = fr.find_freq_idx(abs(admm_ret), nfreq[idx], xgrid)-1
        if len(pos) == 0:
           pos = np.argsort(-abs(admm_ret))[:nfreq[idx]]
        est=xgrid[pos]+admm_beta[pos]
        gt=(f[idx][:nfreq[idx]])
        for k in range(nfreq[idx]):
           dist_i_direct=np.min(np.abs(est-gt[k]))
           dist_i_rshift = np.min(np.abs(est+1 - gt[k]))
           dist_i_lshift = np.min(np.abs(est-1 - gt[k]))
           dist_i=np.min([dist_i_rshift,dist_i_direct,dist_i_lshift])
           if dist_i<2*reslu:
               ADMM[db_iter]+=np.power(dist_i,2.0)
               ADMM_cnt[db_iter]+=1

        ## ESPRIT
        est=esprit_res
        gt=(f[idx][:nfreq[idx]])
        for k in range(nfreq[idx]):
           dist_i_direct=np.min(np.abs(est-gt[k]))
           dist_i_rshift = np.min(np.abs(est+1 - gt[k]))
           dist_i_lshift = np.min(np.abs(est-1 - gt[k]))
           dist_i=np.min([dist_i_rshift,dist_i_direct,dist_i_lshift])
           if dist_i<2*reslu:
               ESPRIT[db_iter]+=np.power(dist_i,2.0)
               ESPRIT_cnt[db_iter]+=1



pickle.dump(OGFreq_bn, open('./ACC_monto-carlo/ogfreq_bn_acc.txt', 'wb'))
pickle.dump(OGSBI, open('./ACC_monto-carlo/ogsbi_acc.txt', 'wb'))
pickle.dump(GRSBI, open('./ACC_monto-carlo/grsbi_acc.txt', 'wb'))
pickle.dump(ADMM, open('./ACC_monto-carlo/ADMM_acc.txt', 'wb'))
pickle.dump(ESPRIT, open('./ACC_monto-carlo/ESPRIT_acc.txt', 'wb'))
pickle.dump(OGFreq, open('./ACC_monto-carlo/ogfreq_acc.txt', 'wb'))


pickle.dump(OGSBI_time, open('./ACC_monto-carlo/ogsbi_time.txt', 'wb'))
pickle.dump(GRSBI_time, open('./ACC_monto-carlo/grsbi_time.txt', 'wb'))
pickle.dump(ADMM_time, open('./ACC_monto-carlo/ADMM_time.txt', 'wb'))
pickle.dump(ESPRIT_time, open('./ACC_monto-carlo/ESPRIT_time.txt', 'wb'))
pickle.dump(OGFreq_time, open('./ACC_monto-carlo/ogfreq_time.txt', 'wb'))
pickle.dump(OGFreq_bn_time, open('./ACC_monto-carlo/ogfreq_bn_time.txt', 'wb'))


pickle.dump(OGSBI_cnt, open('./ACC_monto-carlo/ogsbi_cnt.txt', 'wb'))
pickle.dump(GRSBI_cnt, open('./ACC_monto-carlo/grsbi_cnt.txt', 'wb'))
pickle.dump(ADMM_cnt, open('./ACC_monto-carlo/ADMM_cnt.txt', 'wb'))
pickle.dump(ESPRIT_cnt, open('./ACC_monto-carlo/ESPRIT_cnt.txt', 'wb'))
pickle.dump(OGFreq_cnt, open('./ACC_monto-carlo/ogfreq_cnt.txt', 'wb'))
pickle.dump(OGFreq_bn_cnt, open('./ACC_monto-carlo/ogfreq_bn_cnt.txt', 'wb'))

#######################################################################ACC################################################################
OGSBI = pickle.load(open('./ACC_monto-carlo/ogsbi_acc.txt', 'rb'))
GRSBI = pickle.load(open('./ACC_monto-carlo/grsbi_acc.txt', 'rb'))
OGFreq = pickle.load(open('./ACC_monto-carlo/ogfreq_acc.txt', 'rb'))
OGFreq_bn = pickle.load(open('./ACC_monto-carlo/ogfreq_bn_acc.txt', 'rb'))
ADMM = pickle.load(open('./ACC_monto-carlo/ADMM_acc.txt', 'rb'))
ESPRIT = pickle.load(open('./ACC_monto-carlo/ESPRIT_acc.txt', 'rb'))

OGSBI_cnt = pickle.load(open('./ACC_monto-carlo/ogsbi_cnt.txt', 'rb'))
GRSBI_cnt = pickle.load(open('./ACC_monto-carlo/grsbi_cnt.txt', 'rb'))
OGFreq_cnt = pickle.load(open('./ACC_monto-carlo/ogfreq_cnt.txt', 'rb'))
OGFreq_bn_cnt = pickle.load(open('./ACC_monto-carlo/ogfreq_bn_cnt.txt', 'rb'))
ADMM_cnt = pickle.load(open('./ACC_monto-carlo/ADMM_cnt.txt', 'rb'))
ESPRIT_cnt = pickle.load(open('./ACC_monto-carlo/ESPRIT_cnt.txt', 'rb'))

target_num=np.sum(nfreq[:ITER])
db=list(range(-10,35,5))

fig = plt.figure(figsize=(11,7))
plt.subplots_adjust(left=None,bottom=0.1,right=None,top=0.98,wspace=None,hspace=None)


ax = fig.add_subplot(111)
ax.set_xlabel('SNR / dB',size=20,fontproperties='Times New Roman')
ax.set_ylabel('MSE / Hz',size=20,fontproperties='Times New Roman')

plt.semilogy(db,(OGSBI/OGSBI_cnt),'--',marker='o',label='OGSBI',linewidth=3,markersize=10)
plt.semilogy(db,(GRSBI/GRSBI_cnt),'--',marker='o',label='GRSBI',linewidth=3,markersize=10)
plt.semilogy(db,(ADMM/ADMM_cnt),'--',marker='o',label='ADMM',linewidth=3,markersize=10)
plt.semilogy(db,(ESPRIT/ESPRIT_cnt),'--',marker='o',label='ESPRIT',linewidth=3,markersize=10)
plt.semilogy(db,(OGFreq/OGFreq_cnt),'--',marker='o',label='Proposed_adapBases_woCA',linewidth=3,markersize=10)
plt.semilogy(db,(OGFreq_bn/OGFreq_bn_cnt),'--',marker='o',label='Proposed_adapBases_wiCA',linewidth=3,markersize=10)
plt.grid(linestyle='-.')
labelss =plt.legend(frameon=True,prop={'size':16}).get_texts()
[label.set_fontname('Times New Roman') for label in labelss]
plt.tick_params(labelsize=18)
labels = plt.gca().get_xticklabels() + plt.gca().get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]


#######################################################################TIME################################################################
OGSBI = pickle.load(open('./ACC_monto-carlo/ogfreq_time.txt', 'rb'))
GRSBI = pickle.load(open('./ACC_monto-carlo/grsbi_time.txt', 'rb'))
OGFreq = pickle.load(open('./ACC_monto-carlo/ogfreq_time.txt', 'rb'))
OGFreq_bn = pickle.load(open('./ACC_monto-carlo/ogsbi_bn_time.txt', 'rb'))
ADMM = pickle.load(open('./ACC_monto-carlo/ADMM_time.txt', 'rb'))
ESPRIT = pickle.load(open('./ACC_monto-carlo/ESPRIT_time.txt', 'rb'))

target_num=np.sum(nfreq[:ITER])
db=list(range(-10,35,5))
time=[np.sum(OGSBI)/ITER,np.sum(GRSBI)/ITER,np.sum(OGFreq_bn)/ITER,np.sum(OGFreq)/ITER,np.sum(ADMM)/ITER,np.sum(ESPRIT)/ITER]
a=1









