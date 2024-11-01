#!/usr/bin/env python
# coding: utf-8
import h5py
import os
import numpy as np
import torch
import util

import matplotlib.pyplot as plt
from data import fr

import h5py
import matplotlib.font_manager as fm
from matplotlib import rcParams
import os
import pickle
import time
os.environ['CUDA_VISIBLE_DEVICES']='4'


ogfreq="model/epoch_320_attnsoft.pth"
alpha_rand="model/robust_parm/AdapBase_attnsoft_noabssep_alpha_rand.pth"
alpha_randn="model/robust_parm/AdapBase_attnsoft_noabssep_alpha_randn.pth"
beta_rand="model/robust_parm/AdapBase_attnsoft_noabssep_beta_rand.pth"
beta_randn="model/robust_parm/AdapBase_attnsoft_noabssep_beta_randn.pth"
filter_rand="model/robust_parm/AdapBase_attnsoft_noabssep_filter_rand.pth"
filter_zeros="model/robust_parm/AdapBase_attnsoft_noabssep_filter_zeros.pth"

b0="model/robust_parm/AdapBase_attnsoft_noabssep_bthre_0.pth"
b1="model/robust_parm/AdapBase_attnsoft_noabssep_bthre_1.pth"
b_1="model/robust_parm/AdapBase_attnsoft_noabssep_bthre_-1.pth"
b_3="model/robust_parm/AdapBase_attnsoft_noabssep_bthre_-3.pth"
b_4="model/robust_parm/AdapBase_attnsoft_noabssep_bthre_-4.pth"

x0="model/robust_parm/AdapBase_attnsoft_noabssep_xthre_0.pth"
x1="model/robust_parm/AdapBase_attnsoft_noabssep_xthre_1.pth"
x_1="model/robust_parm/AdapBase_attnsoft_noabssep_xthre_-1.pth"
x_3="model/robust_parm/AdapBase_attnsoft_noabssep_xthre_-3.pth"
x_4="model/robust_parm/AdapBase_attnsoft_noabssep_xthre_-4.pth"


data_dir = 'robust_parm_dataset'

device = torch.device('cpu')

#load models


og_module, _, _, _, _ = util.load(ogfreq, 'og_bn', device)
og_module.cpu()
og_module.eval()
x1_module, _, _, _, _ = util.load(x1, 'og_bn', device)
x1_module.cpu()
x1_module.eval()
x0_module, _, _, _, _ = util.load(x0, 'og_bn', device)
x0_module.cpu()
x0_module.eval()
x_1_module, _, _, _, _ = util.load(x_1, 'og_bn', device)
x_1_module.cpu()
x_1_module.eval()
x_3_module, _, _, _, _ = util.load(x_3, 'og_bn', device)
x_3_module.cpu()
x_3_module.eval()
x_4_module, _, _, _, _ = util.load(x_4, 'og_bn', device)
x_4_module.cpu()
x_4_module.eval()



b1_module, _, _, _, _ = util.load(b1, 'og_bn', device)
b1_module.cpu()
b1_module.eval()
b0_module, _, _, _, _ = util.load(b0, 'og_bn', device)
b0_module.cpu()
b0_module.eval()
b_1_module, _, _, _, _ = util.load(b_1, 'og_bn', device)
b_1_module.cpu()
b_1_module.eval()
b_3_module, _, _, _, _ = util.load(b_3, 'og_bn', device)
b_3_module.cpu()
b_3_module.eval()
b_4_module, _, _, _, _ = util.load(b_4, 'og_bn', device)
b_4_module.cpu()
b_4_module.eval()




alpha_rand_module, _, _, _, _ = util.load(alpha_rand, 'og_bn', device)
alpha_rand_module.cpu()
alpha_rand_module.eval()
alpha_randn_module, _, _, _, _ = util.load(alpha_randn, 'og_bn', device)
alpha_randn_module.cpu()
alpha_randn_module.eval()
beta_rand_module, _, _, _, _ = util.load(beta_rand, 'og_bn', device)
beta_rand_module.cpu()
beta_rand_module.eval()
beta_randn_module, _, _, _, _ = util.load(beta_randn, 'og_bn', device)
beta_randn_module.cpu()
beta_randn_module.eval()
filter_rand_module, _, _, _, _ = util.load(filter_rand, 'og_bn', device)
filter_rand_module.cpu()
filter_rand_module.eval()
filter_zeros_module, _, _, _, _ = util.load(filter_zeros, 'og_bn', device)
filter_zeros_module.cpu()
filter_zeros_module.eval()





#load data
f = np.load(os.path.join(data_dir, 'f.npy'))
r = np.load(os.path.join(data_dir, 'r.npy'))
kernel_param_0 = 0.12/ 64

nfreq =  np.sum(f >= -0.5, axis=1)

db=['20.0dB.npy']
ITER=1000
fr_size=128
xgrid = np.linspace(-0.5, 0.5, fr_size, endpoint=False)
reslu=xgrid[1]-xgrid[0]


OGFreq=np.zeros([ITER,1])
x1_res=np.zeros([ITER,1])
x0_res=np.zeros([ITER,1])
x_1_res=np.zeros([ITER,1])
x_4_res=np.zeros([ITER,1])
x_3_res=np.zeros([ITER,1])
b1_res=np.zeros([ITER,1])
b0_res=np.zeros([ITER,1])
b_1_res=np.zeros([ITER,1])
b_4_res=np.zeros([ITER,1])
b_3_res=np.zeros([ITER,1])
alpha_randn_res=np.zeros([ITER,1])
alpha_rand_res=np.zeros([ITER,1])
beta_rand_res=np.zeros([ITER,1])
beta_randn_res=np.zeros([ITER,1])
filter_zeros_res=np.zeros([ITER,1])
filter_rand_res=np.zeros([ITER,1])


for db_iter in range(len(db)):
    signal = np.load(os.path.join(data_dir, db[db_iter]))
    for idx in range(ITER):
        with torch.no_grad():
            print(db_iter,idx)
            mv = np.max(np.sqrt(pow(signal[idx][0], 2) + pow(signal[idx][1], 2)))
            signal[idx][0]=signal[idx][0]/mv
            signal[idx][1] = signal[idx][1] / mv

            ogfreq,_,ogfreq_beta,_=og_module(torch.tensor(signal[idx][None]))
            ogfreq = ogfreq.squeeze().cpu().abs()
            ogfreq = ogfreq.data.numpy()
            ogfreq_beta = ogfreq_beta[0, :].real

            x1_ret,_,x1_ret_beta,_=x1_module(torch.tensor(signal[idx][None]))
            x1_ret = x1_ret.squeeze().cpu().abs()
            x1_ret = x1_ret.data.numpy()
            x1_ret_beta = x1_ret_beta[0, :].real

            x0_ret,_,x0_ret_beta,_=x0_module(torch.tensor(signal[idx][None]))
            x0_ret = x0_ret.squeeze().cpu().abs()
            x0_ret = x0_ret.data.numpy()
            x0_ret_beta = x0_ret_beta[0, :].real

            x_1_ret,_,x_1_ret_beta,_=x_1_module(torch.tensor(signal[idx][None]))
            x_1_ret = x_1_ret.squeeze().cpu().abs()
            x_1_ret = x_1_ret.data.numpy()
            x_1_ret_beta = x_1_ret_beta[0, :].real

            x_3_ret,_,x_3_ret_beta,_=x_3_module(torch.tensor(signal[idx][None]))
            x_3_ret = x_3_ret.squeeze().cpu().abs()
            x_3_ret = x_3_ret.data.numpy()
            x_3_ret_beta = x_3_ret_beta[0, :].real

            x_4_ret,_,x_4_ret_beta,_=x_4_module(torch.tensor(signal[idx][None]))
            x_4_ret = x_4_ret.squeeze().cpu().abs()
            x_4_ret = x_4_ret.data.numpy()
            x_4_ret_beta = x_4_ret_beta[0, :].real

            b1_ret,_,b1_ret_beta,_=b1_module(torch.tensor(signal[idx][None]))
            b1_ret = b1_ret.squeeze().cpu().abs()
            b1_ret = b1_ret.data.numpy()
            b1_ret_beta = b1_ret_beta[0, :].real

            b0_ret,_,b0_ret_beta,_=b0_module(torch.tensor(signal[idx][None]))
            b0_ret = b0_ret.squeeze().cpu().abs()
            b0_ret = b0_ret.data.numpy()
            b0_ret_beta = b0_ret_beta[0, :].real

            b_1_ret,_,b_1_ret_beta,_=b_1_module(torch.tensor(signal[idx][None]))
            b_1_ret = b_1_ret.squeeze().cpu().abs()
            b_1_ret = b_1_ret.data.numpy()
            b_1_ret_beta = b_1_ret_beta[0, :].real

            b_3_ret,_,b_3_ret_beta,_=x_3_module(torch.tensor(signal[idx][None]))
            b_3_ret = b_3_ret.squeeze().cpu().abs()
            b_3_ret = b_3_ret.data.numpy()
            b_3_ret_beta = b_3_ret_beta[0, :].real

            b_4_ret,_,b_4_ret_beta,_=b_4_module(torch.tensor(signal[idx][None]))
            b_4_ret = b_4_ret.squeeze().cpu().abs()
            b_4_ret = b_4_ret.data.numpy()
            b_4_ret_beta = b_4_ret_beta[0, :].real


            alpha_rand_ret,_,alpha_rand_ret_beta,_=alpha_rand_module(torch.tensor(signal[idx][None]))
            alpha_rand_ret = alpha_rand_ret.squeeze().cpu().abs()
            alpha_rand_ret = alpha_rand_ret.data.numpy()
            alpha_rand_ret_beta = alpha_rand_ret_beta[0, :].real

            alpha_randn_ret,_,alpha_randn_ret_beta,_=alpha_randn_module(torch.tensor(signal[idx][None]))
            alpha_randn_ret = alpha_randn_ret.squeeze().cpu().abs()
            alpha_randn_ret = alpha_randn_ret.data.numpy()
            alpha_randn_ret_beta = alpha_randn_ret_beta[0, :].real

            beta_rand_ret,_,beta_rand_ret_beta,_=beta_rand_module(torch.tensor(signal[idx][None]))
            beta_rand_ret = beta_rand_ret.squeeze().cpu().abs()
            beta_rand_ret = beta_rand_ret.data.numpy()
            beta_rand_ret_beta = beta_rand_ret_beta[0, :].real

            beta_randn_ret, _, beta_randn_ret_beta, _ = beta_randn_module(torch.tensor(signal[idx][None]))
            beta_randn_ret = beta_randn_ret.squeeze().cpu().abs()
            beta_randn_ret = beta_randn_ret.data.numpy()
            beta_randn_ret_beta = beta_randn_ret_beta[0, :].real

            filter_rand_ret,_,filter_rand_ret_beta,_=filter_rand_module(torch.tensor(signal[idx][None]))
            filter_rand_ret = filter_rand_ret.squeeze().cpu().abs()
            filter_rand_ret = filter_rand_ret.data.numpy()
            filter_rand_ret_beta = filter_rand_ret_beta[0, :].real

            filter_zeros_ret, _, filter_zeros_ret_beta, _ = filter_zeros_module(torch.tensor(signal[idx][None]))
            filter_zeros_ret = filter_zeros_ret.squeeze().cpu().abs()
            filter_zeros_ret = filter_zeros_ret.data.numpy()
            filter_zeros_ret_beta = filter_zeros_ret_beta[0, :].real




        pos = fr.find_freq_idx(abs(ogfreq), nfreq[idx], xgrid)-1
        if len(pos) == 0:
            pos = np.argsort(-abs(ogfreq))[:nfreq[idx]]
        est=xgrid[pos]+ogfreq_beta.squeeze().cpu().numpy()[pos]*reslu
        gt=(f[idx][:nfreq[idx]])
        OGFreq[idx] = fr.fnr_m(est, f[idx], 64)


        pos = fr.find_freq_idx(abs(x1_ret), nfreq[idx], xgrid)-1
        if len(pos) == 0:
            pos = np.argsort(-abs(x1_ret))[:nfreq[idx]]
        est=xgrid[pos]+x1_ret_beta.squeeze().cpu().numpy()[pos]*reslu
        gt=(f[idx][:nfreq[idx]])
        x1_res[idx] = fr.fnr_m(est, f[idx], 64)


        pos = fr.find_freq_idx(abs(x0_ret), nfreq[idx], xgrid)-1
        if len(pos) == 0:
            pos = np.argsort(-abs(x0_ret))[:nfreq[idx]]
        est=xgrid[pos]+x0_ret_beta.squeeze().cpu().numpy()[pos]*reslu
        gt=(f[idx][:nfreq[idx]])
        x0_res[idx] = fr.fnr_m(est, f[idx], 64)


        pos = fr.find_freq_idx(abs(x_1_ret), nfreq[idx], xgrid)-1
        if len(pos) == 0:
            pos = np.argsort(-abs(x_1_ret))[:nfreq[idx]]
        est=xgrid[pos]+x_1_ret_beta.squeeze().cpu().numpy()[pos]*reslu
        gt=(f[idx][:nfreq[idx]])
        x_1_res[idx] = fr.fnr_m(est, f[idx], 64)


        pos = fr.find_freq_idx(abs(x_3_ret), nfreq[idx], xgrid)-1
        if len(pos) == 0:
            pos = np.argsort(-abs(x_3_ret))[:nfreq[idx]]
        est=xgrid[pos]+x_3_ret_beta.squeeze().cpu().numpy()[pos]*reslu
        gt=(f[idx][:nfreq[idx]])
        x_3_res[idx] = fr.fnr_m(est, f[idx], 64)


        pos = fr.find_freq_idx(abs(x_4_ret), nfreq[idx], xgrid)-1
        if len(pos) == 0:
            pos = np.argsort(-abs(x_4_ret))[:nfreq[idx]]
        est=xgrid[pos]+x_4_ret_beta.squeeze().cpu().numpy()[pos]*reslu
        gt=(f[idx][:nfreq[idx]])
        x_4_res[idx] = fr.fnr_m(est, f[idx], 64)


        pos = fr.find_freq_idx(abs(b1_ret), nfreq[idx], xgrid) - 1
        if len(pos) == 0:
            pos = np.argsort(-abs(b1_ret))[:nfreq[idx]]
        est = xgrid[pos] + b1_ret_beta.squeeze().cpu().numpy()[pos] * reslu
        gt = (f[idx][:nfreq[idx]])
        b1_res[idx] = fr.fnr_m(est, f[idx], 64)


        pos = fr.find_freq_idx(abs(b0_ret), nfreq[idx], xgrid) - 1
        if len(pos) == 0:
            pos = np.argsort(-abs(b0_ret))[:nfreq[idx]]
        est = xgrid[pos] + b0_ret_beta.squeeze().cpu().numpy()[pos] * reslu
        gt = (f[idx][:nfreq[idx]])
        b0_res[idx] = fr.fnr_m(est, f[idx], 64)


        pos = fr.find_freq_idx(abs(b_1_ret), nfreq[idx], xgrid) - 1
        if len(pos) == 0:
            pos = np.argsort(-abs(b_1_ret))[:nfreq[idx]]
        est = xgrid[pos] + b_1_ret_beta.squeeze().cpu().numpy()[pos] * reslu
        gt = (f[idx][:nfreq[idx]])
        b_1_res[idx] = fr.fnr_m(est, f[idx], 64)


        pos = fr.find_freq_idx(abs(b_3_ret), nfreq[idx], xgrid) - 1
        if len(pos) == 0:
            pos = np.argsort(-abs(b_3_ret))[:nfreq[idx]]
        est = xgrid[pos] + b_3_ret_beta.squeeze().cpu().numpy()[pos] * reslu
        gt = (f[idx][:nfreq[idx]])
        b_3_res[idx] = fr.fnr_m(est, f[idx], 64)


        pos = fr.find_freq_idx(abs(b_4_ret), nfreq[idx], xgrid) - 1
        if len(pos) == 0:
            pos = np.argsort(-abs(b_4_ret))[:nfreq[idx]]
        est = xgrid[pos] + b_4_ret_beta.squeeze().cpu().numpy()[pos] * reslu
        gt = (f[idx][:nfreq[idx]])
        b_4_res[idx] = fr.fnr_m(est, f[idx], 64)



        pos = fr.find_freq_idx(abs(alpha_rand_ret), nfreq[idx], xgrid)-1
        if len(pos) == 0:
            pos = np.argsort(-abs(alpha_rand_ret))[:nfreq[idx]]
        est=xgrid[pos]+alpha_rand_ret_beta.squeeze().cpu().numpy()[pos]*reslu
        gt=(f[idx][:nfreq[idx]])
        alpha_rand_res[idx] = fr.fnr_m(est, f[idx], 64)


        pos = fr.find_freq_idx(abs(alpha_randn_ret), nfreq[idx], xgrid)-1
        if len(pos) == 0:
            pos = np.argsort(-abs(alpha_randn_ret))[:nfreq[idx]]
        est=xgrid[pos]+alpha_randn_ret_beta.squeeze().cpu().numpy()[pos]*reslu
        gt=(f[idx][:nfreq[idx]])
        alpha_randn_res[idx] = fr.fnr_m(est, f[idx], 64)


        pos = fr.find_freq_idx(abs(beta_randn_ret), nfreq[idx], xgrid)-1
        if len(pos) == 0:
            pos = np.argsort(-abs(beta_randn_ret))[:nfreq[idx]]
        est=xgrid[pos]+beta_randn_ret_beta.squeeze().cpu().numpy()[pos]*reslu
        gt=(f[idx][:nfreq[idx]])
        beta_randn_res[idx] = fr.fnr_m(est, f[idx], 64)


        pos = fr.find_freq_idx(abs(beta_rand_ret), nfreq[idx], xgrid)-1
        if len(pos) == 0:
            pos = np.argsort(-abs(beta_rand_ret))[:nfreq[idx]]
        est=xgrid[pos]+beta_rand_ret_beta.squeeze().cpu().numpy()[pos]*reslu
        gt=(f[idx][:nfreq[idx]])
        beta_rand_res[idx] = fr.fnr_m(est, f[idx], 64)


        pos = fr.find_freq_idx(abs(filter_rand_ret), nfreq[idx], xgrid)-1
        if len(pos) == 0:
            pos = np.argsort(-abs(filter_rand_ret))[:nfreq[idx]]
        est=xgrid[pos]+filter_rand_ret_beta.squeeze().cpu().numpy()[pos]*reslu
        gt=(f[idx][:nfreq[idx]])
        filter_rand_res[idx] = fr.fnr_m(est, f[idx], 64)


        pos = fr.find_freq_idx(abs(filter_zeros_ret), nfreq[idx], xgrid)-1
        if len(pos) == 0:
            pos = np.argsort(-abs(filter_zeros_ret))[:nfreq[idx]]
        est=xgrid[pos]+filter_zeros_ret_beta.squeeze().cpu().numpy()[pos]*reslu
        gt=(f[idx][:nfreq[idx]])
        filter_zeros_res[idx] = fr.fnr_m(est, f[idx], 64)




pickle.dump(OGFreq, open('./robust_parm_FNR/ogfreq_fnr.txt', 'wb'))

pickle.dump(x1_res, open('./robust_parm_FNR/x1_res.txt', 'wb'))
pickle.dump(x0_res, open('./robust_parm_FNR/x0_res.txt', 'wb'))
pickle.dump(x_1_res, open('./robust_parm_FNR/x_1_res.txt', 'wb'))
pickle.dump(x_3_res, open('./robust_parm_FNR/x_3_res.txt', 'wb'))
pickle.dump(x_4_res, open('./robust_parm_FNR/x_4_res.txt', 'wb'))


pickle.dump(b1_res, open('./robust_parm_FNR/b1_res.txt', 'wb'))
pickle.dump(b0_res, open('./robust_parm_FNR/b0_res.txt', 'wb'))
pickle.dump(b_1_res, open('./robust_parm_FNR/b_1_res.txt', 'wb'))
pickle.dump(b_3_res, open('./robust_parm_FNR/b_3_res.txt', 'wb'))
pickle.dump(b_4_res, open('./robust_parm_FNR/b_4_res.txt', 'wb'))

pickle.dump(alpha_rand_res, open('./robust_parm_FNR/alpha_rand_res.txt', 'wb'))
pickle.dump(alpha_randn_res, open('./robust_parm_FNR/alpha_randn_res.txt', 'wb'))
pickle.dump(beta_rand_res, open('./robust_parm_FNR/beta_rand_res.txt', 'wb'))
pickle.dump(beta_randn_res, open('./robust_parm_FNR/beta_randn_res.txt', 'wb'))
pickle.dump(filter_zeros_res, open('./robust_parm_FNR/filter_zeros_res.txt', 'wb'))
pickle.dump(filter_rand_res, open('./robust_parm_FNR/filter_rand_res.txt', 'wb'))



OGFreq = pickle.load(open('./robust_parm_FNR/ogfreq_fnr.txt', 'rb'))

x1_res = pickle.load(open('./robust_parm_FNR/x1_res.txt', 'rb'))
x0_res = pickle.load(open('./robust_parm_FNR/x0_res.txt', 'rb'))
x_1_res = pickle.load(open('./robust_parm_FNR/x_1_res.txt', 'rb'))
x_3_res = pickle.load(open('./robust_parm_FNR/x_3_res.txt', 'rb'))
x_4_res = pickle.load(open('./robust_parm_FNR/x_4_res.txt', 'rb'))

b1_res = pickle.load(open('./robust_parm_FNR/b1_res.txt', 'rb'))
b0_res = pickle.load(open('./robust_parm_FNR/b0_res.txt', 'rb'))
b_1_res = pickle.load(open('./robust_parm_FNR/b_1_res.txt', 'rb'))
b_3_res = pickle.load(open('./robust_parm_FNR/b_3_res.txt', 'rb'))
b_4_res = pickle.load(open('./robust_parm_FNR/b_4_res.txt', 'rb'))

alpha_rand_res = pickle.load(open('./robust_parm_FNR/alpha_rand_res.txt', 'rb'))
alpha_randn_res = pickle.load(open('./robust_parm_FNR/alpha_randn_res.txt', 'rb'))
beta_rand_res = pickle.load(open('./robust_parm_FNR/beta_rand_res.txt', 'rb'))
beta_randn_res = pickle.load(open('./robust_parm_FNR/beta_randn_res.txt', 'rb'))
filter_zeros_res = pickle.load(open('./robust_parm_FNR/filter_zeros_res.txt', 'rb'))
filter_rand_res = pickle.load(open('./robust_parm_FNR/filter_rand_res.txt', 'rb'))


tgt_num=8
x_mean=[np.mean(x1_res/tgt_num*100),np.mean(x0_res/tgt_num*100),np.mean(x_1_res/tgt_num*100),np.mean(OGFreq/tgt_num*100),np.mean(x_3_res/tgt_num*100),np.mean(x_4_res/tgt_num*100)]
x_var=[np.sqrt(np.var(x1_res/tgt_num*100)),np.sqrt(np.var(x0_res/tgt_num*100)),np.sqrt(np.var(x_1_res/tgt_num*100)),np.sqrt(np.var(OGFreq/tgt_num*100)),np.sqrt(np.var(x_3_res/tgt_num*100)),np.sqrt(np.var(x_4_res/tgt_num*100))]

b_mean=[np.mean(b1_res/tgt_num*100),np.mean(b0_res/tgt_num*100),np.mean(b_1_res/tgt_num*100),np.mean(OGFreq/tgt_num*100),np.mean(b_3_res/tgt_num*100),np.mean(b_4_res/tgt_num*100)]
b_var=[np.sqrt(np.var(b1_res/tgt_num*100)),np.sqrt(np.var(b0_res/tgt_num*100)),np.sqrt(np.var(b_1_res/tgt_num*100)),np.sqrt(np.var(OGFreq/tgt_num*100)),np.sqrt(np.var(b_3_res/tgt_num*100)),np.sqrt(np.var(b_4_res/tgt_num*100))]

alpha_mean=[np.mean(alpha_rand_res/tgt_num*100),np.mean(OGFreq/tgt_num*100),np.mean(alpha_randn_res/tgt_num*100)]
alpha_var=[np.sqrt(np.var(alpha_rand_res/tgt_num*100)),np.sqrt(np.var(OGFreq/tgt_num*100)),np.sqrt(np.var(alpha_randn_res/tgt_num*100))]


beta_mean=[np.mean(beta_rand_res/tgt_num*100),np.mean(OGFreq/tgt_num*100),np.mean(beta_randn_res/tgt_num*100)]
beta_var=[np.sqrt(np.var(beta_rand_res/tgt_num*100)),np.sqrt(np.var(OGFreq/tgt_num*100)),np.sqrt(np.var(beta_randn_res/tgt_num*100))]

filter_mean=[np.mean(filter_rand_res/tgt_num*100),np.mean(filter_zeros_res/tgt_num*100),np.mean(OGFreq/tgt_num*100)]
filter_var=[np.sqrt(np.var(filter_rand_res/tgt_num*100)),np.sqrt(np.var(filter_zeros_res/tgt_num*100)),np.sqrt(np.var(OGFreq/tgt_num*100))]


font2=16
font=13
fig = plt.figure(figsize=(11,5))
plt.subplots_adjust(left=0.08,bottom=0.18,right=0.98,top=0.98,wspace=0.18,hspace=0.34)
ax = fig.add_subplot(1,2,1)
ax.set_xlabel(r'Initial Values of Sparsity Threshold for $\bf{x}$'+'\n (a)',size=font2,fontproperties='Times New Roman')
ax.set_ylabel('MDR / %',size=font2,fontproperties='Times New Roman')
plt.errorbar(range(6), x_mean[::-1], x_var[::-1], fmt='ro--')
plt.arrow(2, 30, 0, -5,width=0.05,
          length_includes_head=False, head_width=0.2,
          head_length=1)
xlabel = [r'${10^{ - 4}}$', r'${10^{ - 3}}$', r'${10^{ - 2}}$', r'${10^{ - 1}}$', r'${10^{ 0}}$', r'${10^{ 1}}$']
plt.xticks(range(6), labels=xlabel)
plt.ylim([-10,50])
plt.grid(linestyle='-.')
plt.tick_params(labelsize=font)


ax = fig.add_subplot(1,2,2)
ax.set_xlabel(r'Initial Values of Sparsity Threshold for $\bf{\beta}$'+'\n (b)',size=font2,fontproperties='Times New Roman')
ax.set_ylabel('MDR / %',size=font2,fontproperties='Times New Roman')
plt.errorbar(range(6), b_mean[::-1], b_var[::-1], fmt='ro--')
plt.arrow(2, 30, 0, -5,width=0.05,
          length_includes_head=False, head_width=0.2,
          head_length=1)
xlabel = [r'${10^{ - 4}}$', r'${10^{ - 3}}$', r'${10^{ - 2}}$', r'${10^{ - 1}}$', r'${10^{ 0}}$', r'${10^{ 1}}$']
plt.xticks(list(range(6)), labels=xlabel)
plt.ylim([-10,50])
plt.grid(linestyle='-.')
plt.tick_params(labelsize=font)



#########################################################################################################
font2=16
font=13
fig = plt.figure(figsize=(11,3))
plt.subplots_adjust(left=0.08,bottom=0.28,right=0.97,top=0.98,wspace=0.29,hspace=0.34)
ax = fig.add_subplot(1,3,1)
ax.set_xlabel(r'Initialization for $\bf{\beta}$'+'\n (a)',size=font2,fontproperties='Times New Roman')
ax.set_ylabel('MDR / %',size=font2,fontproperties='Times New Roman')
plt.errorbar(range(3), beta_mean, beta_var, fmt='ro--')
plt.arrow(1, 30, 0, -5,width=0.05,
          length_includes_head=False, head_width=0.2,
          head_length=1)
xlabel = [r'Uniform', r'Zeros', r'Gaussian']
plt.xticks(range(3), labels=xlabel)
plt.ylim([-10,50])
plt.grid(linestyle='-.')
plt.tick_params(labelsize=font)


ax = fig.add_subplot(1,3,2)
ax.set_xlabel(r'Initialization for $\bf{\alpha}$'+'\n (b)',size=font2,fontproperties='Times New Roman')
ax.set_ylabel('MDR / %',size=font2,fontproperties='Times New Roman')
plt.errorbar(range(3), alpha_mean, alpha_var, fmt='ro--')
plt.arrow(1, 30, 0, -5,width=0.05,
          length_includes_head=False, head_width=0.2,
          head_length=1)
xlabel = [r'Uniform', r'Zeros', r'Gaussian']
plt.xticks(list(range(3)), labels=xlabel)
plt.ylim([-10,50])
plt.grid(linestyle='-.')
plt.tick_params(labelsize=font)

ax = fig.add_subplot(1,3,3)
ax.set_xlabel(r'Initialization for Dictionary'+'\n (c)',size=font2,fontproperties='Times New Roman')
ax.set_ylabel('MDR / %',size=font2,fontproperties='Times New Roman')
plt.errorbar(range(3), filter_mean, filter_var, fmt='ro--')
plt.arrow(2, 30, 0, -5,width=0.05,
          length_includes_head=False, head_width=0.2,
          head_length=1)
xlabel = [r'Uniform', r'Zeros', r'Gaussian']
plt.xticks(list(range(3)), labels=xlabel)
plt.ylim([-10,50])
plt.grid(linestyle='-.')
plt.tick_params(labelsize=font)

a=1









