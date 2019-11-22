# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np 
import os
import cv2
from __init__ import ref_dir
from face import get_landmark, LandmarkIndex, facefrontal
from teeth import local_enhancement, detect_region_1, detect_region_2, process_proxy

k = 1.8
data = np.load('%s/stat.npz' % ref_dir)
mean, std = data['mean'], data['std']
boundL = mean - k*std
boundU = mean + k*std

teeth_hsv_lower  = np.array([0, 0, 60])
teeth_hsv_upper  = np.array([180, 150, 255])

black_lower = np.array([0, 0, 0])
black_upper = np.array([180, 255, 46])

white_lower = np.array([0, 0, 46])
white_upper = np.array([180, 50, 255])

def mask_inpaint(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, black_lower, black_upper)
    mask2 = cv2.inRange(hsv, white_lower, white_upper)
    mask = mask1 | mask2
    mimg = cv2.inpaint(img, mask, 10, cv2.INPAINT_TELEA)
    return mimg
        
def preprocess(mp4_path, save_path, rsize, startfr=0, endfr=None):
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
        print("%04d/%04d" % (cnt, endfr-1))
        cnt += 1
        
        ret, img_ = cap.read()
        img = facefrontal(img_)
        if img is None:
            continue
        det, ldmk = get_landmark(img, LandmarkIndex.LIP, norm=True)
        if det is None or ldmk is None:
            continue

        # validate landmarks using statistics in the dataset
        # if np.sum(np.logical_or(ldmk < boundL, ldmk > boundU)) > 0:
        #     continue
        landmarks.append(ldmk)
        
        # resize texture into a square
        txtr = img[det.top():det.top()+det.height(), det.left():det.left()+det.width()]
        txtr = cv2.resize(txtr, (rsize, rsize))       
        # mask & inpaint for clothes region
        txtr = mask_inpaint(txtr)
        textures.append(txtr)
        
    landmarks = np.array(landmarks)
    textures = np.array(textures)
    
    # filter frames which are not locally smooth
    # approx = (landmarks[2:, :] + landmarks[:-2, :]) / 2
    # L2 = np.linalg.norm(landmarks[1:-1, :]-approx, ord=2, axis=1)
    # check = (L2 <= 0.1).nonzero()
    # landmarks = landmarks[1:-1][check].reshape((-1, 20, 2))
    # textures  = textures[1:-1][check]
    
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
        weights = np.exp(-L2 / (2 * sigma**2))
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

def sharpen(inpI, ksize=(15, 15), sigma=1.0, k=0.5):
    smooth_inpI = cv2.GaussianBlur(inpI, ksize, sigma)
    outpI = (inpI.astype(np.float) - smooth_inpI.astype(np.float))*k + inpI.astype(np.float)
    outpI[outpI < 0]   = 0
    outpI[outpI > 255] = 255
    outpI = outpI.astype(np.uint8)
    return outpI

def process_teeth(inpI, inpS, pxyF, pxyS, rsize, boundary, hsvU=teeth_hsv_upper, hsvL=teeth_hsv_lower, biasU=2, biasL=6):
    regionU, regionL = detect_region_1(inpI, inpS, hsvU, hsvL, biasU, biasL, rsize, boundary)
    # enhance upper region
    tmpI = local_enhancement(inpI, inpS, pxyF['upper'], pxyS['upper'], regionU, rsize, boundary, 'upper')
    # enhance lower region
    tmpI = local_enhancement(tmpI, inpS, pxyF['lower'], pxyS['lower'], regionL, rsize, boundary, 'lower')
    # sharpening
    outpI = sharpen(tmpI)
    return outpI

def lowerface(sq, ipath, tpath, tprocpath, opath, rsize=300, preproc=False):
    # preprocess target video
    if preproc or os.path.exists(tprocpath) == False:
        preprocess(tpath, tprocpath, rsize)
    
    # load target data and clip them
    tgtdata = np.load(tprocpath)
    tgtS, tgtI = tgtdata['landmarks'], tgtdata['textures']
    boundary = sq.align(tgtI.shape[1])      # (left, right, upper, lower)
    tgtI = tgtI[:, boundary[2]:boundary[3], boundary[0]:boundary[1], :]
    
    # load input data and proxy data
    inpdata = np.load(ipath)
    nfr = inpdata.shape[0]
    pxyF, pxyS = process_proxy(rsize)
    
    # create every frame and form a mp4
    outpdata = []
    print('Start to create new video...')
    for cnt, inpS in enumerate(inpdata):
        print("%s: %04d/%04d" % (opath, cnt+1, nfr))
        tmpI, tmpS = weighted_median(inpS, tgtS, tgtI)
        outpI = process_teeth(tmpI, tmpS, pxyF, pxyS, rsize, boundary)
        outpdata.append(outpI)
    outpdata = np.array(outpdata)
    np.save(opath, outpdata)
    return outpdata

# def test1():
# #    tar_id = "target001"
# #    savepath = preprocess(tar_dir+tar_id+'.mp4', sq, rsize=150, startfr=300)
#     savepath = 'target/target001.npz'
#     data = np.load(savepath)
#     landmarks = data['landmarks']
#     textures = data['textures']
    
#     inp_id = "test036_ldmks"
#     ldmks = np.load(inp_dir+inp_id+'.npy')
#     ldmk_test = ldmks[160, :, :]    # the 100th frame
#     outpI, outpS = weighted_median(ldmk_test, landmarks, textures, n=50)
    
#     cv2.imshow('title', outpI)
#     cv2.waitKey(0)
    
# def test2():
#     import os
#     imgfiles = os.listdir('tmp/')
#     for imgfile in imgfiles:
#         img = cv2.imread('tmp/' + imgfile)
#         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         black_lower = np.array([0, 0, 0])
#         black_upper = np.array([180, 255, 46])
#         white_lower = np.array([0, 0, 46])
#         white_upper = np.array([180, 50, 255])
#         mask1 = cv2.inRange(hsv, black_lower, black_upper)
#         mask2 = cv2.inRange(hsv, white_lower, white_upper)
#         mask = mask1 | mask2
        
#         cv2.imshow('mask', mask)
#         cv2.waitKey(0)
        
#         masked = cv2.inpaint(img, mask, 10, cv2.INPAINT_TELEA)
        
#         cv2.imshow('masked', masked)
#         cv2.waitKey(0)    
    
if __name__ == '__main__':
    pass