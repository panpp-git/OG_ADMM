import numpy as np
import scipy.signal
#import matplotlib.pyplot as plt


def freq2fr(f, xgrid, kernel_type='gaussian', param=None, r=None,nfreq=None):
    """
    Convert an array of frequencies to a frequency representation discretized on xgrid.
    """
    if kernel_type == 'gaussian':
        return gaussian_kernel(f, xgrid, param, r,nfreq)
    elif kernel_type == 'triangle':
        return triangle(f, xgrid, param)

def gaussian_kernel(f, xgrid, sigma, r,nfreq):
    fr = np.zeros((f.shape[0], xgrid.shape[0]))

    reslu=xgrid[1]-xgrid[0]
    beta=np.zeros([fr.shape[0],xgrid.shape[0]], dtype='float32')
    for ii in range(fr.shape[0]):
        for i in range(f.shape[1]):
            if r[ii, i] == 0 or f[ii,i]==-10:
                continue
            l_diff=np.zeros(xgrid.shape[0]-1)
            if f[ii,i] not in xgrid:
                l_diff=f[ii,i]-xgrid[:-1]
                l_diff[l_diff<0]=2
                pos=np.argmin(l_diff)
                val=np.min(l_diff)
                if val<xgrid[pos+1]-f[ii,i]:
                    beta[ii,pos]=val/reslu
                    fr[ii,pos]=1
                else:
                    beta[ii,pos+1]=-(xgrid[pos+1]-f[ii,i])/reslu
                    fr[ii,pos+1]=1
            else:
                pos=np.argwhere(xgrid==f[ii,i])[0]
                fr[ii,pos]=1
    return fr, beta


def triangle(f, xgrid, slope):
    """
    Create a frequency representation with a triangle kernel.
    """
    fr = np.zeros((f.shape[0], xgrid.shape[0]))
    for i in range(f.shape[1]):
        dist = np.abs(xgrid[None, :] - f[:, i][:, None])
        rdist = np.abs(xgrid[None, :] - (f[:, i][:, None] + 1))
        ldist = np.abs(xgrid[None, :] - (f[:, i][:, None] - 1))
        dist = np.minimum(dist, rdist, ldist)
        fr += np.clip(1 - slope * dist, 0, 1)
    return fr


def find_freq_m(fr, nfreq, xgrid, max_freq=10):
    """
    Extract frequencies from a frequency representation by locating the highest peaks.
    """
    ff = -np.ones((1, max_freq))
    for n in range(1):
        find_peaks_out = scipy.signal.find_peaks(fr, height=(None, None))
        num_spikes = min(len(find_peaks_out[0]), nfreq)
        idx = np.argpartition(find_peaks_out[1]['peak_heights'], -num_spikes)[-num_spikes:]
        ff[n, :num_spikes] = np.sort(xgrid[find_peaks_out[0][idx]])
    return ff

def find_freq_idx(fr, nfreq, xgrid, max_freq=10):
    """
    Extract frequencies from a frequency representation by locating the highest peaks.
    """
    ff = -np.ones((1, max_freq))
    for n in range(1):
        find_peaks_out = scipy.signal.find_peaks(fr, height=(None, None))
        num_spikes = min(len(find_peaks_out[0]), nfreq)
        idx = np.argpartition(find_peaks_out[1]['peak_heights'], -num_spikes)[-num_spikes:]
        freq_idx=find_peaks_out[0][idx]
    return np.sort(freq_idx)


def find_freq(fr, nfreq, xgrid, max_freq=10):
    """
    Extract frequencies from a frequency representation by locating the highest peaks.
    """
    ff = -np.ones((nfreq.shape[0], max_freq))
    for n in range(len(nfreq)):

        if nfreq[n] < 1:  # at least one frequency
            nf = 1
        else:
            nf = nfreq[n]

        find_peaks_out = scipy.signal.find_peaks(fr[n], height=(None, None))
        num_spikes = min(len(find_peaks_out[0]), int(nf))
        idx = np.argpartition(find_peaks_out[1]['peak_heights'], -num_spikes)[-num_spikes:]
        ff[n, :num_spikes] = np.sort(xgrid[find_peaks_out[0][idx]])
    return ff


def periodogram(signal, xgrid):
    """
    Compute periodogram.
    """
    js = np.arange(signal.shape[1])
    return (np.abs(np.exp(-2.j * np.pi * xgrid[:, None] * js).dot(signal.T) / signal.shape[1]) ** 2).T

def make_hankel(signal, m):
    """
    Auxiliary function used in MUSIC.
    """
    n = len(signal)
    h = np.zeros((m, n - m + 1), dtype='complex128')
    for r in range(m):
        for c in range(n - m + 1):
            h[r, c] = signal[r + c]
    return h


def music(signal, xgrid, nfreq, m=20):
    """
    Compute frequency representation obtained with MUSIC.
    """
    music_fr = np.zeros((signal.shape[0], len(xgrid)))
    for n in range(signal.shape[0]):
        hankel = make_hankel(signal[n], m)
        _, _, V = np.linalg.svd(hankel)
        v = np.exp(-2.0j * np.pi * np.outer(xgrid[:, None], np.arange(0, signal.shape[1] - m + 1)))
        u = V[nfreq[n]:]
        fr = -np.log(np.linalg.norm(np.tensordot(u, v, axes=(1, 1)), axis=0) ** 2)
        music_fr[n] = fr
    return music_fr
