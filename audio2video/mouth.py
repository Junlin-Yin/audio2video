# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np 
import os
import cv2
from face import get_landmark, LandmarkIndex, facefrontal, resize
from teeth import process_teeth, process_proxy

black_lower = np.array([0, 0, 0])
black_upper = np.array([180, 255, 46])

white_lower = np.array([0, 0, 46])
white_upper = np.array([180, 50, 255])

def mask_inpaint(img, ksize=3):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, black_lower, black_upper)
    mask2 = cv2.inRange(hsv, white_lower, white_upper)
    mask = mask1 | mask2
    # mask = cv2.dilate(mask, np.ones((ksize, ksize), dtype=np.uint8))
    mimg = cv2.inpaint(img, mask, 10, cv2.INPAINT_TELEA)
    return mimg
        
def preprocess(mp4_path, save_path, rsize, padw, startfr=0, endfr=None):
    '''
    ### parameters
    mp4_path: path of mp4 file \\
    sq: Squre instance which defines the boundary of lower face texture
    rsize: width (height) of clipped texture in every target video frame

    ### retval
    savepath: path that saves landmarks and textures
    '''
    landmarks = []
    textures = []
    cap = cv2.VideoCapture(mp4_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, startfr)
    cnt = startfr
    endfr = cap.get(cv2.CAP_PROP_FRAME_COUNT) if endfr is None else endfr
    print('Start preprocessing...')
    while cap.isOpened():
        if cnt == endfr:
            break
        print("%s: %04d/%04d" % (save_path, cnt, endfr-1))
        cnt += 1
        
        ret, img_ = cap.read()
        img = facefrontal(img_)
        if img is None:
            continue
        det, ldmk = get_landmark(img, LandmarkIndex.LIP, norm=True)
        if det is None or ldmk is None:
            continue
        landmarks.append(ldmk)
        
        # resize texture and inpaint the blank part
        txtr = resize(img, det, ldmk, detw=rsize, padw=padw)[0]
        txtr = mask_inpaint(txtr)
        # cv2.imshow('txtr', txtr)
        # cv2.waitKey(0)
        textures.append(txtr)
        
    landmarks = np.array(landmarks)
    textures = np.array(textures)
    np.savez(save_path, landmarks=landmarks, textures=textures)

def optimize_sigma(L2, n, alpha):
    left, right = 0, 1
    epsilon = 1e-2
    i, maxI = 0, 20
    while True:
        if(i == maxI):
            raise Exception("Infinite loop in optimize_sigma()!")
        i += 1
        
        sigma = (left + right) / 2
        weights = np.exp(-L2**2 / (2 * sigma**2))
        indices = np.argsort(weights)[::-1] # large -> small
        weights = weights[indices]
        ratio = np.sum(weights[:n]) / np.sum(weights)
        if abs(alpha-ratio) < epsilon:
            break
        elif ratio < alpha:
            right = sigma
        else:
            left = sigma
    # ratio is highly close to alpha, meaning we find a proper sigma
    weights = weights[:n]
    weights /= np.sum(weights)              # np.sum(weights) = 1
    indices = indices[:n]
    return weights, indices        

def locate_median(weights):
    # make sure that np.sum(weights) == 1
    s = 0
    for i in range(weights.shape[0]):
        s += weights[i]
        if s >= 0.5:
            break
    return i

def weighted_median(inpS, tgtS, tgtI, n=100, alpha=0.9):
    # choose n candidates
    L2 = np.linalg.norm(tgtS-inpS, ord=2, axis=(1, 2))
    weights, indices = optimize_sigma(L2, n, alpha)
    candI = tgtI[indices, :, :, :]
    candS = tgtS[indices, :, :]
    
    # calculate weighted-average of landmarks for teeth enhancement
    outpS = np.sum([l*w for l, w in zip(candS, weights)], axis=0)
    
    # form output texture
    outpI = np.zeros(tgtI.shape[1:], dtype=np.uint8)
    for y in range(outpI.shape[0]):
        for x in range(outpI.shape[1]):
            # at the position (x, y)
            for c in range(outpI.shape[2]):
                # in each channel
                intencity = candI[:, y, x, c]
                indices = np.argsort(intencity)
                weights_sort = weights[indices]
                idx = locate_median(weights_sort)
                outpI[y, x, c] = intencity[indices[idx]]
    
    return outpI, outpS  

def sharpen(inpI, ksize=15, sigma=1.0, k=0.5):
    smooth_inpI = cv2.GaussianBlur(inpI, (ksize, ksize), sigma)
    outpI = (inpI.astype(np.float) - smooth_inpI.astype(np.float))*k + inpI.astype(np.float)
    outpI[outpI < 0]   = 0
    outpI[outpI > 255] = 255
    outpI = outpI.astype(np.uint8)
    return outpI

def lowerface(sq, ipath, tpath, tprocpath, opath, rsize=300, padw=100, preproc=False):
    # preprocess target video
    tar_id = os.path.splitext(os.path.split(tpath)[1])[0]
    if preproc or os.path.exists(tprocpath) == False:
        preprocess(tpath, tprocpath, rsize, padw)
    
    # load target data and clip them
    tgtdata = np.load(tprocpath)
    tgtS, tgtI = tgtdata['landmarks'], tgtdata['textures']
    boundary = sq.align(rsize, padw)      # (left, right, upper, lower)
    tgtI = tgtI[:, boundary[2]:boundary[3], boundary[0]:boundary[1], :]  
    # load input data and proxy data
    inpdata = np.load(ipath)
    nfr = inpdata.shape[0]
    pxyF, pxyS = process_proxy(tar_id, rsize)
    
    # create every frame and form a mp4
    outpdata = []
    print('Start to create new video...')
    for cnt, inpS in enumerate(inpdata):
        print("%s: %04d/%04d" % (opath, cnt+1, nfr))
        tmpI, tmpS = weighted_median(inpS, tgtS, tgtI)
        tmpI = process_teeth(tmpI, tmpS, pxyF, pxyS, rsize, padw, boundary)
        outpI = sharpen(tmpI)
        # cv2.imshow('outpI', outpI)
        # cv2.waitKey(0)
        outpdata.append(outpI)
    outpdata = np.array(outpdata)
    np.save(opath, outpdata)
    return outpdata
    
if __name__ == '__main__':
    from __init__ import Square
    ipath = '../input/ldmk/std_u/a015.npy'
    tprocpath = '../target/preproc/t187.npz'
    sq = Square(0.25, 0.75, 0.5, 1.1)
    rsize, padw = 300, 100
    # load target data and clip them
    tgtdata = np.load(tprocpath)
    tgtS, tgtI = tgtdata['landmarks'], tgtdata['textures']
    boundary = sq.align(rsize, padw)      # (left, right, upper, lower)
    tgtI = tgtI[:, boundary[2]:boundary[3], boundary[0]:boundary[1], :]  
    # load input data and proxy data
    inpdata = np.load(ipath)
    nfr = inpdata.shape[0]
    pxyF, pxyS = process_proxy('t187', rsize)
    cv2.imwrite('../tmp/upperpxy.png', (pxyF['upper']*255).astype(np.uint8))
    cv2.imwrite('../tmp/lowerpxy.png', (pxyF['lower']*255).astype(np.uint8))
    
    # create every frame and form a mp4
    fid = 750
    inpS = inpdata[fid]
    tmpI, tmpS = weighted_median(inpS, tgtS, tgtI)
    cv2.imwrite('../tmp/median_%04d.png' % fid, tmpI)
    tmpI2 = process_teeth(tmpI, tmpS, pxyF, pxyS, rsize, padw, boundary)
    cv2.imwrite('../tmp/teeth_%04d.png' % fid, tmpI2)
    outpI = sharpen(tmpI2)
    cv2.imwrite('../tmp/sharp_%04d.png' % fid, outpI)