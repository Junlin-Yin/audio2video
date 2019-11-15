# Copyright (c) 2006 Carnegie Mellon University
#
# You may copy and modify this freely under the same terms as
# Sphinx-III
# -*- coding:utf8 -*-

"""Compute MFCC coefficients.

This module provides functions for computing MFCC (mel-frequency
cepstral coefficients) as used in the Sphinx speech recognition
system.
"""

__author__ = "David Huggins-Daines <dhuggins@cs.cmu.edu>"
__version__ = "$Revision: 6390 $"

import librosa
import numpy as np
import numpy.fft
import os
import math

def mel(f):
    return 2595. * np.log10(1. + f / 700.)

def melinv(m):
    return 700. * (np.power(10., m / 2595.) - 1.)

class MFCC(object):
    def __init__(self, 
                 nfilt=40,          # Mel滤波器的个数
                 ncep=13,           # 取MFCC的前多少个系数
                 lowerf=133.3333,   # Mel滤波器频率范围的下限
                 upperf=6855.4976,  # Mel滤波器频率范围的上限
                 alpha=0.97,        # 语音信号的预加重系数
                 samprate=16000,    # 音频的频率，即音点的采样频率，Hz
                 frate=100,         # 分析帧的采样频率，Hz
                 wlen=0.0256,       # 分析帧的周期，s
                 nfft=512):         # FFT中的用于离散化波形数据的采样点数，离散频率的个数为nfft/2+1
        # Store parameters
        self.lowerf = lowerf
        self.upperf = upperf
        self.nfft = nfft
        self.ncep = ncep
        self.nfilt = nfilt
        self.frate = frate
        self.samprate = samprate
        self.fshift = float(samprate) / frate   # 相邻两分析帧之间的音点个数

        # Build Hamming window
        self.wlen = int(wlen * samprate)    # 每个分析帧包含的音点数
        self.win = np.hamming(self.wlen)    # 创建汉明窗，窗的宽度就是每帧包含的音点数

        # Prior sample for pre-emphasis
        self.prior = 0
        self.alpha = alpha

        # Build mel filter matrix
        self.filters = np.zeros((nfft//2+1,nfilt), 'd') # 列数为FFT分析出的不同频率数，行数为Mel滤波器数量
        dfreq = float(samprate) / nfft                  # FFT的结果中，所有频率都是dfreq的整数倍，整数不大于nfft/2
        if upperf > samprate/2:                         # 香农定理
            raise(Exception,
                   "Upper frequency %f exceeds Nyquist %f" % (upperf, samprate/2))
        melmax = mel(upperf)
        melmin = mel(lowerf)
        dmelbw = (melmax - melmin) / (nfilt + 1)        # 各滤波器的边界在Mel尺度上是均匀分布的
        # Filter edges, in Hz
        filt_edge = melinv(melmin + dmelbw * np.arange(nfilt + 2, dtype='d'))

        for whichfilt in range(0, nfilt):
            # Filter triangles, in DFT points
            leftfr = round(filt_edge[whichfilt] / dfreq)        # 明确每个滤波器的左右边界和中心
            centerfr = round(filt_edge[whichfilt + 1] / dfreq)
            rightfr = round(filt_edge[whichfilt + 2] / dfreq)
            # For some reason this is calculated in Hz, thoug
            # it doesn't really matter
            fwidth = (rightfr - leftfr) * dfreq
            height = 2. / fwidth                                # 滤波器的顶点高度和其所跨频率宽度成反比

            if centerfr != leftfr:                              # 确定各频率在该滤波器上的滤波结果（三角形）
                leftslope = height / (centerfr - leftfr)
            else:
                leftslope = 0
            freq = int(leftfr + 1)
            while freq < centerfr:
                self.filters[freq,whichfilt] = (freq - leftfr) * leftslope
                freq = freq + 1
            if freq == centerfr: # This is always true
                self.filters[freq,whichfilt] = height
                freq = freq + 1
            if centerfr != rightfr:
                rightslope = height / (centerfr - rightfr)
            while freq < rightfr:
                self.filters[freq,whichfilt] = (freq - rightfr) * rightslope
                freq = freq + 1
#             print("Filter %d: left %d=%f center %d=%f right %d=%f width %d" %
#                   (whichfilt,
#                   leftfr, leftfr*dfreq,
#                   centerfr, centerfr*dfreq,
#                   rightfr, rightfr*dfreq,
#                   freq - leftfr))
#             print self.filters[leftfr:rightfr,whichfilt]

        # Build DCT matrix
        self.s2dct = s2dctmat(nfilt, ncep, 1./nfilt)
        self.dct = dctmat(nfilt, ncep, np.pi/nfilt)

    def sig2s2mfc(self, sig):
        '''sig: 音频信号，类型为ndarray或list
        '''
        nfr = int(len(sig) / self.fshift + 1)       # 该音频总共采得的分析帧数
        mfcc = np.zeros((nfr, self.ncep), 'd')      # mfcc.shape = nfr * ncep
        fr = 0
        while fr < nfr:
            start = round(fr * self.fshift)         # 每一分析帧的开始音点位置
            end = min(len(sig), start + self.wlen)  # 每一分析帧的结束音点位置
            frame = sig[start:end]                  # 抽取此分析帧
            if len(frame) < self.wlen:              # 将分析帧长度统一化，多余空位填零
                frame = np.resize(frame,self.wlen)
                frame[self.wlen:] = 0
            mfcc[fr] = self.frame2s2mfc(frame)      # 对每一帧计算其前若干个MFCC系数
            fr = fr + 1
        return mfcc

    def sig2s2mfc_energy(self, sig):
        nfr = int(len(sig) / self.fshift + 1)

        mfcc = np.zeros((nfr, self.ncep + 2), 'd')  # mfcc.shape = nfr * (ncep+2)
        fr = 0
        while fr < nfr:
            start = int(round(fr * self.fshift))    # 抽取出每一个分析帧并统一长度
            end = min(len(sig), start + self.wlen)
            frame = sig[start:end]
            if len(frame) < self.wlen:
                frame = np.resize(frame,self.wlen)
                frame[self.wlen:] = 0
            mfcc[fr,:-2] = self.frame2s2mfc(frame)  # 每一行前ncep个元素对应MFCC系数
            mfcc[fr, -2] = math.log(1 + np.mean(np.power(frame.astype(float), 2)))
                                                    # 倒数第二个元素刻画该分析帧的平均音量
            mid = 0.5 * (start + end - 1)
            mfcc[fr, -1] = mid / self.samprate      # 倒数第一个元素是该分析帧正中央在音频中所处的时刻

            fr = fr + 1
        return mfcc

    def sig2logspec(self, sig):
        nfr = int(len(sig) / self.fshift + 1)
        mfcc = np.zeros((nfr, self.nfilt), 'd')
        fr = 0
        while fr < nfr:
            start = round(fr * self.fshift)
            end = min(len(sig), start + self.wlen)
            frame = sig[start:end]
            if len(frame) < self.wlen:
                frame = np.resize(frame,self.wlen)
                frame[self.wlen:] = 0
            mfcc[fr] = self.frame2logspec(frame)
            fr = fr + 1
        return mfcc

    def pre_emphasis(self, frame):
        '''消除发声过程中，声带和嘴唇造成的效应，来补偿语音信号受到发音系统所压抑的高频部分
        ，并能突显高频的共振峰

        这里采用的是一阶FIR高通数字滤波器，转换公式为y(n) = x(n) - alpha*x(n-1)
        '''
        # FIXME: Do this with matrix multiplication
        outfr = np.empty(len(frame), 'd')
        outfr[0] = frame[0] - self.alpha * self.prior
        for i in range(1,len(frame)):
            outfr[i] = frame[i] - self.alpha * frame[i-1]
        self.prior = frame[-1]
        return outfr
        
    def frame2logspec(self, frame):
        frame = self.pre_emphasis(frame) * self.win         # 预强调，再与汉明窗点积
                                                            # 加汉明窗的作用是减少频谱能量泄漏，更有利于做傅里叶分析
        fft = np.fft.rfft(frame, self.nfft)                 # 分析出这一帧的频率，一共有nfft/2+1个离散的频率（香农定理）
        # Square of absolute value
        power = fft.real * fft.real + fft.imag * fft.imag   # 取平方值
        return np.log(np.dot(power, self.filters).clip(1e-5,np.inf))    # 滤波和取对数

    def frame2s2mfc(self, frame):
        logspec = self.frame2logspec(frame)                 # logspec.shape = (nfilt, )，即倒谱
        return np.dot(logspec, self.s2dct.T) / self.nfilt   # DCT过程，即倒谱分析

def s2dctmat(nfilt,ncep,freqstep):
    """Return the 'legacy' not-quite-DCT matrix used by Sphinx"""
    melcos = np.empty((ncep, nfilt), 'double')
    for i in range(0,ncep):
        freq = np.pi * float(i) / nfilt
        melcos[i] = np.cos(freq * np.arange(0.5, float(nfilt)+0.5, 1.0, 'double'))
    melcos[:,0] = melcos[:,0] * 0.5
    return melcos

def logspec2s2mfc(logspec, ncep=13):
    """Convert log-power-spectrum bins to MFCC using the 'legacy'
    Sphinx transform"""
    nframes, nfilt = logspec.shape
    melcos = s2dctmat(nfilt, ncep, 1./nfilt)
    return np.dot(logspec, melcos.T) / nfilt

def dctmat(N,K,freqstep,orthogonalize=True):
    """Return the orthogonal DCT-II/DCT-III matrix of size NxK.
    For computing or inverting MFCCs, N is the number of
    log-power-spectrum bins (nfilt) while K is the number of cepstra (ncept)."""
    cosmat = np.zeros((N, K), 'double')
    for n in range(0,N):
        for k in range(0, K):
            cosmat[n,k] = np.cos(freqstep * (n + 0.5) * k)
    if orthogonalize:
        cosmat[:,0] = cosmat[:,0] * 1./np.sqrt(2)
    return cosmat

def dct(input, K=13):
    """Convert log-power-spectrum to MFCC using the orthogonal DCT-II"""
    nframes, N = input.shape
    freqstep = np.pi / N
    cosmat = dctmat(N,K,freqstep)
    return np.dot(input, cosmat) * np.sqrt(2.0 / N)

def dct2(input, K=13):
    """Convert log-power-spectrum to MFCC using the normalized DCT-II"""
    nframes, N = input.shape
    freqstep = np.pi / N
    cosmat = dctmat(N,K,freqstep,False)
    return np.dot(input, cosmat) * (2.0 / N)

def idct(input, K=40):
    """Convert MFCC to log-power-spectrum using the orthogonal DCT-III"""
    nframes, N = input.shape
    freqstep = np.pi / K
    cosmat = dctmat(K,N,freqstep).T
    return np.dot(input, cosmat) * np.sqrt(2.0 / K)

def dct3(input, K=40):
    """Convert MFCC to log-power-spectrum using the unnormalized DCT-III"""
    nframes, N = input.shape
    freqstep = np.pi / K
    cosmat = dctmat(K,N,freqstep,False)
    cosmat[:,0] = cosmat[:,0] * 0.5
    return np.dot(input, cosmat.T)

def audio_process(base_dirs):
    for base_dir in base_dirs:
        mp3dir, audiodir = '%s/mp3' % base_dir, '%s/mfcc' % base_dir
        files = os.listdir(mp3dir)
        sr = 16000
        mfccGen = MFCC(nfilt=40, ncep=13, samprate=sr, frate=100, wlen=0.025)
        for mp3file in files:
            y, sr = librosa.load('%s/%s' % (mp3dir, mp3file), sr=None)
            audio = mfccGen.sig2s2mfc_energy(y)     # audio.shape = (nfr, 15)
            numpy.save('%s/%s.npy' % (audiodir, mp3file[:-4]), audio)
            print(mp3file, 'done')

if __name__ == '__main__':
    print('Hello, World')
