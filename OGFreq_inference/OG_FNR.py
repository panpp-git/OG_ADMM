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
OGSBI=np.zeros([len(db),1])
GRSBI=np.zeros([len(db),1])
OGFreq_fourier=np.zeros([len(db),1])
OGFreq_bn=np.zeros([len(db),1])
ADMM=np.zeros([len(db),1])
ESPRIT=np.zeros([len(db),1])
OGFreq_fourier_wnattn=np.zeros([len(db),1])

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


            ogsbl = eng.OGSBI(nargout=1)
            ogsbl_ret = np.zeros(fr_size)
            if np.array(ogsbl['I']).size==1:
               ogsbl_ret[np.array(ogsbl['I']).astype('int') - 1] = abs(np.array(ogsbl['mu']))
               ogsbl_beta = np.array(ogsbl['beta'])
            else:
               ogsbl_ret[np.array(ogsbl['I'])[0, :].astype('int') - 1] = abs(np.array(ogsbl['mu'])[:, 0])
               ogsbl_beta = np.array(ogsbl['beta'])[:, 0]


            grsbl=eng.GRSBI(nargout=1)
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


            ogfreq,_,ogfreq_beta,_=og_module(torch.tensor(signal[idx][None]))
            ogfreq = ogfreq.squeeze().cpu().abs()
            ogfreq = ogfreq.data.numpy()
            ogfreq_beta = ogfreq_beta[0, :].real
            xgrid_cor = ogfreq_beta.squeeze().cpu() * reslu + xgrid

            ogfreq_bn,_,ogfreq_bn_beta,_=og_module_bn(torch.tensor(signal[idx][None]))
            ogfreq_bn = ogfreq_bn.squeeze().cpu().abs()
            ogfreq_bn = ogfreq_bn.data.numpy()
            ogfreq_bn_beta = ogfreq_bn_beta[0, :].real
            xgrid_bn_cor = ogfreq_bn_beta.squeeze().cpu() * reslu + xgrid


            admm = eng.ADMM_ref(nargout=1)
            admm_ret = np.fft.fftshift(abs(np.array(admm['u'])[:, 0]))
            admm_beta = np.fft.fftshift(np.array(admm['beta'])[:, 0])


            esprit = eng.ESPRIT(nargout=1)
            esprit_res = np.array(esprit)
            if esprit_res.size>1:
               esprit_res=esprit_res[0]


        pos = fr.find_freq_idx(abs(ogfreq_bn), nfreq[idx], xgrid)-1
        if len(pos) == 0:
           pos = np.argsort(-abs(ogfreq_bn))[:nfreq[idx]]
        est=xgrid[pos]+ogfreq_bn_beta.squeeze().cpu().numpy()[pos]*reslu
        gt=(f[idx][:nfreq[idx]])
        OGFreq_bn[db_iter] += fr.fnr_m(est, f[idx], 64)


        # ## OGFreq
        pos = fr.find_freq_idx(abs(ogfreq), nfreq[idx], xgrid)-1
        if len(pos) == 0:
           pos = np.argsort(-abs(ogfreq))[:nfreq[idx]]
        est=xgrid[pos]+ogfreq_beta.squeeze().cpu().numpy()[pos]*reslu
        gt=(f[idx][:nfreq[idx]])
        OGFreq[db_iter] += fr.fnr_m(est, f[idx], 64)


        ## OGSBI
        pos = fr.find_freq_idx(abs(ogsbl_ret), nfreq[idx], xgrid)-1
        if len(pos) == 0:
           pos = np.argsort(-abs(ogsbl_ret))[:nfreq[idx]]
        est = xgrid[pos] + ogsbl_beta[pos]
        OGSBI[db_iter] += fr.fnr_m(est, f[idx], 64)

        ## GRSBI
        pos = fr.find_freq_idx(abs(grsbl_ret), nfreq[idx], xgrid)-1
        if len(pos) == 0:
           pos = np.argsort(-abs(grsbl_ret))[:nfreq[idx]]
        est=xgrid[pos]+grsbl_beta[pos]
        gt=(f[idx][:nfreq[idx]])
        GRSBI[db_iter] += fr.fnr_m(est, f[idx], 64)

        ## admm
        pos = fr.find_freq_idx(abs(admm_ret), nfreq[idx], xgrid)-1
        if len(pos) == 0:
           pos = np.argsort(-abs(admm_ret))[:nfreq[idx]]
        est=xgrid[pos]+admm_beta[pos]
        gt=(f[idx][:nfreq[idx]])
        ADMM[db_iter] += fr.fnr_m(est, f[idx], 64)

        ## ESPRIT
        est=esprit_res
        gt=(f[idx][:nfreq[idx]])
        ESPRIT[db_iter] += fr.fnr_m(est, f[idx], 64)


pickle.dump(OGFreq_bn, open('./FNR_monto-carlo/ogfreq_bn_fnr.txt', 'wb'))
pickle.dump(OGSBI, open('./FNR_monto-carlo/ogsbi_fnr.txt', 'wb'))
pickle.dump(GRSBI, open('./FNR_monto-carlo/grsbi_fnr.txt', 'wb'))
pickle.dump(OGFreq, open('./FNR_monto-carlo/ogfreq_fnr.txt', 'wb'))
pickle.dump(ADMM, open('./FNR_monto-carlo/ADMM_fnr.txt', 'wb'))
pickle.dump(ESPRIT, open('./FNR_monto-carlo/ESPRIT_fnr.txt', 'wb'))

OGSBI = pickle.load(open('./FNR_monto-carlo/ogsbi_fnr.txt', 'rb'))
GRSBI = pickle.load(open('./FNR_monto-carlo/grsbi_fnr.txt', 'rb'))
OGFreq = pickle.load(open('./FNR_monto-carlo/ogfreq_fnr.txt', 'rb'))
OGFreq_bn = pickle.load(open('./FNR_monto-carlo/ogfreq_bn_fnr.txt', 'rb'))
ADMM = pickle.load(open('./FNR_monto-carlo/ADMM_fnr.txt', 'rb'))
ESPRIT=pickle.load(open('./FNR_monto-carlo/ESPRIT_fnr.txt', 'rb'))


target_num=np.sum(nfreq[:ITER])
db=list(range(-10,35,5))

fig = plt.figure(figsize=(11,7))
plt.subplots_adjust(left=None,bottom=0.1,right=None,top=0.98,wspace=None,hspace=None)


ax = fig.add_subplot(111)
ax.set_xlabel('SNR / dB',size=20,fontproperties='Times New Roman')
ax.set_ylabel('MDR / %',size=20,fontproperties='Times New Roman')

plt.semilogy(db,OGSBI/target_num*100,'--',marker='o',label='OGSBI',linewidth=3,markersize=10)
plt.semilogy(db,GRSBI/target_num*100,'--',marker='o',label='GRSBI',linewidth=3,markersize=10)
plt.semilogy(db,ADMM/target_num*100,'--',marker='o',label='ADMM',linewidth=3,markersize=10)
plt.semilogy(db,ESPRIT/target_num*100,'--',marker='o',label='ESPRIT',linewidth=3,markersize=10)
plt.semilogy(db,OGFreq/target_num*100,'--',marker='o',label='Proposed_adapBases_woCA',linewidth=3,markersize=10)
plt.semilogy(db,OGFreq_bn/target_num*100,'--',marker='o',label='Proposed_adapBases_wiCA',linewidth=3,markersize=10)

plt.grid(linestyle='-.')
labelss =plt.legend(frameon=True,prop={'size':16}).get_texts()
[label.set_fontname('Times New Roman') for label in labelss]
plt.tick_params(labelsize=18)
labels = plt.gca().get_xticklabels() + plt.gca().get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

plt.show()
a=1









