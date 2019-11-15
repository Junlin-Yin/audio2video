# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
import math
import random
import bisect
import pickle
from __init__ import ref_dir, trn_dir, raw_mfcc_dir, raw_fids_dir, inp_dir
from visual import vfps

def load_raw(vali_rate):
    inps  = {'training':[], 'validation':[]}
    outps = {'training':[], 'validation':[]}
    
    vsubdirs = os.listdir(raw_fids_dir)
    vsubdirs = [x[:-4] for x in vsubdirs]
    unqlinks, unqcnts = np.unique(vsubdirs, return_counts=True)
    
    for link, cnt in zip(unqlinks, unqcnts):
        
        # extract audio features
        afeatures = np.load(os.path.join(raw_mfcc_dir, link+'.npy'))   # (N, 15)
        audio           = afeatures[ :, :-1]                           # (N, 14)
        audio_diff      = afeatures[1:, :-1] - afeatures[:-1, :-1]     # (N-1, 14)
        audio_timestamp = afeatures[ :,  -1]
        
        vsubdir_list = [link + "}}" + str(100+i)[1:3] + '/' for i in range(cnt)]
        for vsubdir in vsubdir_list:
            print(vsubdir)
            
            # in each video fragments
            vfeatures = np.load('%s/%s/fidsCoeff.npy' % (raw_fids_dir, vsubdir))
            with open('%s/%s/startframe.txt' % (raw_fids_dir, vsubdir)) as f:
                vstartfr = int(f.read())
            with open('%s/%s/nframe.txt' % (raw_fids_dir, vsubdir)) as f:
                vnfr = int(f.read())
                
            # slice corresponding audio fragment
            astartfr = bisect.bisect_left(audio_timestamp, vstartfr/vfps)
            aendfr = bisect.bisect_right(audio_timestamp, (vstartfr+vnfr-1)/vfps)
            inp = np.hstack((audio[astartfr:aendfr], audio_diff[astartfr:aendfr]))
            
            # interpolating video fragment
            outp= np.zeros((aendfr-astartfr, vfeatures.shape[1]))
            for i in range(outp.shape[0]):
                vfr = audio_timestamp[i+astartfr] * vfps
                left_vfr = int(vfr - vstartfr)
                right_vfr = min(left_vfr+1, vfeatures.shape[0]-1)
                alpha = vfr - vstartfr - left_vfr
                outp[i] = (1-alpha) * vfeatures[left_vfr] + alpha * vfeatures[right_vfr]
        
            # deciding training or validation
            key = 'training' if random.random() > vali_rate else 'validation'
            inps[key].append(inp)
            outps[key].append(outp)      
    return inps, outps

def normalize_data(inps, outps, outp_norm=False):
    tinps, toutps = inps['training'], outps['training']
    vinps, voutps = inps['validation'], outps['validation']
    
    means, stds = [0]*2, [0]*2
    for idx, data in enumerate((tinps, toutps)):
        # idx = 0: inps
        # idx = 1: outps
        merged_data = np.vstack(data)        # merged_data.shape = (?, F)

        if idx == 1 and outp_norm == False:
            # do not normalize output data
            means[idx] = np.zeros((merged_data.shape[1],))
            stds[idx]  = np.ones((merged_data.shape[1],))
        else:
            means[idx] = np.mean(merged_data, axis=0)
            stds[idx]  = np.std(merged_data, axis=0)
        
        for i in range(len(data)):
            data[i] = (data[i] - means[idx]) / stds[idx]
    
    for idx, data in enumerate((vinps, voutps)):
        # idx = 0: inps
        # idx = 1: outps
        for i in range(len(data)):
            data[i] = (data[i] - means[idx]) / stds[idx]
    
    return inps, outps, means, stds

def load_data(pass_id, args, preprocess=False, outp_norm=False):
    '''load input and output from save_dir, or from video_dir and audio_dir
    if preprocessing is needed.
    ### Parameters
    pass_id        (str)  name of this pass, including training and testing \\
    args           (dict) argument dictionary containing vr, step_delay, seq_len, etc. \\
    preprocess     (bool) whether preprocessing is needed \\
    ### Return Values
    new_inps       (dict) processed input data \\
    new_outps      (dict) processed output data
    '''
    vali_rate  = args['vr']
    step_delay = args['step_delay']
    seq_len    = args['seq_len']

    # create work space for current pass
    if os.path.exists('%s/%s' % (trn_dir, pass_id)) == False:
        os.mkdir('%s/%s' % (trn_dir, pass_id))
    if os.path.exists('%s/ldmk/%s' % (inp_dir, pass_id)) == False:
        os.mkdir('%s/ldmk/%s' % (inp_dir, pass_id))
    if os.path.exists('%s/vis/%s' % (inp_dir, pass_id)) == False:
        os.mkdir('%s/vis/%s' % (inp_dir, pass_id))

    dpath = '%s/%s/data.pkl' % (trn_dir, pass_id)
    if preprocess or os.path.exists(dpath) == False:
        # extract raw features from video_dir and audio_dir
        inps, outps = load_raw(vali_rate)
        
        # normalize them
        inps, outps, means, stds = normalize_data(inps, outps, outp_norm=outp_norm)

        # save them to .../data.pkl  
        inout_data = {'inps':inps,      'outps':outps,
                      'imean':means[0], 'omean':means[1],
                      'istd':stds[0],   'ostd':stds[1]}
        with open(dpath, 'wb') as f:
            pickle.dump(inout_data, f)

    # extract inputs and outputs from save/passId/inout_stat
    with open(dpath, 'rb') as f:
        inout_data = pickle.load(f)
    inps, outps = inout_data['inps'], inout_data['outps']
    
    # deal with step delay
    new_inps  = {'training':[], 'validation':[]}
    new_outps = {'training':[], 'validation':[]}
    for key in new_inps.keys():
        for inp, outp in zip(inps[key], outps[key]):
            # throw away those which have less than <step_delay+seq_len> frames
            if inp.shape[0] - step_delay >= seq_len:
                new_inps[key].append(np.copy(inp[step_delay:, :]))
                new_outps[key].append(np.copy(outp[:(-step_delay if step_delay > 0 else None), :]))
    
    return new_inps, new_outps

def next_batch(inps, outps, mode, batch_pt, nbatches, args):
    '''fetch next batch of inputs and outputs and update batch pointer
    ### Parameters
    inps     (dict) input dictionary which contains two list \\
    outps    (dict) output dictionary which contains two list \\
    mode     (str)  training or validation \\
    batch_pt (dict) batch pointer dictionary which contains two modes \\
    nbatches (dict) number-of-batch dictionary containing two modes \\
    args     (dict) argument dictionary containing batch_size, seq_len, etc. \\
    ### Return Values
    x        (ndarray) input batch \\
    y        (ndarray) output batch
    '''
    batch_size = args['batch_size']
    seq_len    = args['seq_len']
    
    x, y = [], []
    for i in range(batch_size):
        idx = batch_pt[mode]
        inp, outp = inps[mode][idx], outps[mode][idx]

        # formatting each sequence randomly
        startfr = random.randint(0, inp.shape[0]-seq_len)
        x.append(np.copy(inp[startfr : startfr+seq_len]))
        y.append(np.copy(outp[startfr: startfr+seq_len]))
        
        # determine whether to move to next fragment of data
        nseq = math.ceil(inp.shape[0] / seq_len)
        if random.random() < (1 / nseq):
            # let X be the number of sequences extracted from inp,
            # then E(X) = nseq
            batch_pt[mode] = (batch_pt[mode] + 1) % nbatches[mode]
        
    x, y = np.array(x), np.array(y)
    return x, y

def save_args(pass_id, args):
    with open('%s/%s/args.pkl' % (trn_dir, pass_id), 'wb') as f:
        pickle.dump(args, f)
        
def load_args(pass_id):
    with open('%s/%s/args.pkl' % (trn_dir, pass_id), 'rb') as f:
        args = pickle.load(f)
    return args

def load_stat(pass_id):
    with open('%s/%s/data.pkl' % (trn_dir, pass_id), 'rb') as f:
        data = pickle.load(f)
    return data['imean'], data['istd'], data['omean'], data['ostd']

def restore_state(sess, pass_id):
    '''restore network states
    ### Parameters
    sess       (Session) \\
    pass_id    (str)    name of this pass, including training and testing
    ### Return Values
    startEpoch (int)    we should continue training from this epoch \\
    saver      (Saver)  tf.train.Saver
    '''
    # define checkpoint saver
    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

    # restore start epoch number and global variables
    last_ckpt = tf.train.latest_checkpoint('%s/%s' % (trn_dir, pass_id))
    if last_ckpt is None:
        # no ckpt file yet
        startEpoch = 0
        print('train from the very beginning')
    else:
        startEpoch = int(last_ckpt.split('-')[-1])
        saver.restore(sess, last_ckpt)
        print('train from epoch', startEpoch+1)
            
    return startEpoch, saver

def cvt2vfps(outp, times, delay):
    PCA_MAT = np.load('%s/PCA_MAT.npy' % ref_dir)
    delay_times = times[delay+1:]
    period = times[1] - times[0]
    # delay_times.shape == outp.shape[0]

    vnfr = int((delay_times[-1] - delay_times[0]) * vfps) 
    all_ldmks = []
    for i in range(vnfr):
        # interpolating and get the 20 landmark positions
        itime = i / vfps - delay_times[0]
        left = bisect.bisect_left(delay_times, itime)
        alpha = (itime - delay_times[left]) / period
        ldmks = outp[left,:] * (1-alpha) + outp[left+1,:] * alpha
        ldmks = np.matmul(PCA_MAT, ldmks).reshape((20, 2))
        all_ldmks.append(ldmks)

    return np.array(all_ldmks)     # (vnfr, 20, 2)

def report_batch(pass_id, e, b, nepochs, nbatches, b_during, b_savef, tloss):
    '''summary after each training batch
    ### Parameters
    pass_id    (str)    name of this pass, including training and testing \\
    e          (int)    current epoch index \\
    b          (int)    current training batch index \\
    nepochs    (int)    total number of epochs \\
    nbatches   (int)    total number of training batches per epoch \\
    b_during   (float)  time elapse per training batch \\
    b_savef    (int)    frequency to save to progress.log \\
    tloss      (float)  training loss in current training batch
    '''
    # calculate eta-time
    cur_batches = e * nbatches + b
    eta_batches = nepochs*nbatches - 1 - cur_batches
    eta_time = round(eta_batches * b_during)
    m, s = divmod(eta_time, 60)
    h, m = divmod(m, 60)
    eta_str = "%02d:%02d:%02d" % (h, m, s)
    
    # initialize log file
    if cur_batches == 0:
        with open('%s/%s/loss.log' % (trn_dir, pass_id), 'w') as f:
            f.write("0 %f %f\n" % (tloss, tloss))
    
    # print & save batch report
    report = "epoch:%03d/%03d batch:%03d/%03d tloss:%8.6f eta_time:%s" % (e+1, nepochs, b+1, nbatches, tloss, eta_str)
    if b == 0 or (b+1) % b_savef == 0 or b == nbatches - 1:
        print(report)

def report_epoch(pass_id, sess, saver, e, nepochs, e_savef, trainLoss, validLoss):
    '''summary after each epoch
    ### Parameters
    pass_id    (str)     name of this pass, including training and testing \\
    sess       (Session) tf.Session \\
    saver      (Saver)   tf.train.Saver \\
    e          (int)     current epoch index \\
    nepochs    (int)     total number of epochs \\
    e_savef    (int)     frequency to save to checkpoint file \\
    trainLoss  (float)   final training loss in this epoch \\
    validLoss  (float)   average validation loss in this epoch
    '''
    # save checkpoint
    if (e+1) % e_savef == 0 or e == nepochs - 1:
        saver.save(sess, '%s/%s/model' % (trn_dir, pass_id), global_step=e+1)
        print('checkpoint at epoch %d/%d saved' % (e+1, nepochs))
        
    # report loss
    with open('%s/%s/loss.log' % (trn_dir, pass_id), 'r') as f:
        records = f.readlines()[:e+1]
    records.append("%d %f %f\n" % (e+1, trainLoss, validLoss))
    with open('%s/%s/loss.log' % (trn_dir, pass_id), 'w') as f:
        f.write(''.join(records))
    print('loss.log at epoch %d/%d saved' % (e+1, nepochs))

def plot_loss(pass_id):
    with open('%s/%s/loss.log' % (trn_dir, pass_id), 'r') as f:
        records = f.readlines()
    xs = list(range(1, len(records)))
    ts, vs = [], []
    for record in records[1:]:
        e, t, v = record.split()
        ts.append(float(t))
        vs.append(float(v))

    plt.figure()
    plt.title(pass_id+'/loss')
    plt.plot(xs, ts, color='blue', label='training loss')
    plt.plot(xs, vs, color='red', label='validation loss')
    plt.legend() # 显示图例
    plt.xlabel('epoch')
    plt.savefig('%s/%s/loss.png' % (trn_dir, pass_id))
    plt.show()
        
if __name__ == '__main__':
    name = 'L1-h60-d20-u'
    plot_loss(name)