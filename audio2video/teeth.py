# -*- coding: utf-8 -*-

import os
import cv2
import glob
import numpy as np
from face import get_landmark, LandmarkIndex
from __init__ import Square, ref_dir

teeth_hsv_lower  = np.array([0, 0, 60])
teeth_hsv_upper  = np.array([180, 150, 255])

uppermostidx = np.array([12, 13, 14, 15, 16])
lowermostidx = np.array([12, 19, 18, 17, 16])

upperlipidx  = np.array([1, 2, 3, 4, 5, 13, 14, 15])
lowerlipidx  = np.array([7, 8, 9, 10, 11, 17, 18, 19])

def select_proxy(tar_path, mode, fr):  
    # select and generate teeth proxy frame(s)
    assert(mode == 'upper' or mode == 'lower')
    cap = cv2.VideoCapture(tar_path)
    # get face and landmarks
    cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
    ret, img = cap.read()
    pxyfile = glob.glob('%s/proxy_%s_*.png' % (ref_dir, mode))
    for f in pxyfile:
        os.remove(f)
    cv2.imwrite('%s/proxy_%s_%04d.png' % (ref_dir, mode, fr), img)

def process_proxy(rsize, ksize=(17,17), sigma=1e2, k=1):
    # process teeth proxies to get their landmarks and high-pass filters
    F, S = {}, {}
    for mode in ('upper', 'lower'):
        pxyfile, = glob.glob('%s/proxy_%s_*.png' % (ref_dir, mode))
        img = cv2.imread(pxyfile)
        det, ldmk = get_landmark(img, LandmarkIndex.LIP, norm=True)
        
        # resize texture
        txtr = img[det.top():det.top()+det.height(), det.left():det.left()+det.width()]
        txtr = cv2.resize(txtr, (rsize, rsize))
        
        # generate hgih-pass filter (only one channel)
        norm_txtr   = txtr.astype(np.float) / 255
        smooth_txtr = cv2.GaussianBlur(txtr, ksize, sigma) / 255
        filt   = (norm_txtr - smooth_txtr) * k + 0.5
        filt[filt < 0] = 0
        filt[filt > 1] = 1

        # add landmarks and filter into dict S and F respectively
        F[mode] = filt
        S[mode] = ldmk
    return F, S

def detect_region_1(inpI, inpS, thresholdU, thresholdL, biasU, biasL, rsize, boundary):
    # automatically detect upper and lower teeth region in input image
    # boundary: (left, right, upper, lower)
    upper_xp = inpS[uppermostidx][:, 0] * rsize - boundary[0]
    upper_fp = inpS[uppermostidx][:, 1] * rsize - boundary[2]
    lower_xp = inpS[lowermostidx][:, 0] * rsize - boundary[0]
    lower_fp = inpS[lowermostidx][:, 1] * rsize - boundary[2]
    upper_bd = np.ceil(np.interp(np.arange(rsize), upper_xp, upper_fp)).astype(np.int)
    lower_bd = np.ceil(np.interp(np.arange(rsize), lower_xp, lower_fp)).astype(np.int)
    
    hsv = cv2.cvtColor(inpI, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, thresholdL, thresholdU)
    xs, ys = mask.nonzero()
    region = np.array([(y, x) for (x, y) in zip(xs, ys)])
    if region.shape[0] == 0:
        # no region detected; return empty arrays
        upper_region, lower_region = np.array([]), np.array([])
        return upper_region, lower_region
        
    check  = np.logical_and(region[:, 0] > upper_bd[region[:, 1]] + biasU,
                            region[:, 0] < lower_bd[region[:, 1]] - biasL)
    region = region[check.nonzero()]
        
    if region.shape[0] == 0:
        # no region detected; return empty arrays
        upper_region, lower_region = np.array([]), np.array([])
    else:
        # divide region into upper and lower ones
        axis = (np.max(region[:, 0]) + np.min(region[:, 0])) / 2
        check = region[:, 0] <= axis
        upper_region = region[check.nonzero()]
        lower_region = region[np.logical_not(check).nonzero()]
        
    return upper_region, lower_region

def detect_region_2():
    return None, None
    
def local_enhancement(inpI, inpS, pxyF, pxyS, region, rsize, boundary, mode):
    # enhance quality of input image given teeth proxy filter and their landmarks
    # region is the part which is to be enhanced
    assert(mode == 'upper' or mode == 'lower')
    
    # calculate displacement between input image and proxy
    inpS_half = inpS[upperlipidx, :] if mode == 'upper' else inpS[lowerlipidx, :]
    pxyS_half = pxyS[upperlipidx, :] if mode == 'upper' else pxyS[lowerlipidx, :]
    dx, dy = np.round(np.mean(inpS_half - pxyS_half, axis=0) * rsize).astype(np.int)
    
    outpI = np.copy(inpI)
    x_bd, y_bd = boundary[0], boundary[2]
    for (y_pt, x_pt) in region:
        # (x_pxy, y_pxy) in proxy -> (x_pt, y_pt) in input image
        x_pxy = x_bd + x_pt - dx
        y_pxy = y_bd + y_pt - dy
        for c in range(outpI.shape[2]):
            if pxyF[y_pxy, x_pxy, c] < 0.5:
                outpI[y_pt, x_pt, c] = 2*pxyF[y_pxy, x_pxy, c] * outpI[y_pt, x_pt, c]
            else:
                outpI[y_pt, x_pt, c] = 255 - 2*(255-outpI[y_pt, x_pt, c])*(1-pxyF[y_pxy, x_pxy, c])
    return outpI

# def test1():
#     # tuning parameters in proxy processing
#     from candidate import weighted_median
#     sq = Square(0.25, 0.75, 0.6, 1.00)
#     n     = 100
#     kzs   = [15, 17, 19, 21, 23, 25]
#     sigmas= [1e1, 1e2, 2e2, 3e2, 4e2, 5e2]
#     k     = 1
#     rsize = 300
#     tgtdata = np.load('target/target001.npz')
#     tgtS, tgtI = tgtdata['landmarks'], tgtdata['textures']
#     boundary = sq.align(tgtI.shape[1])      # (left, right, upper, lower)
#     tgtI = tgtI[:, boundary[2]:boundary[3], boundary[0]:boundary[1], :]
#     inpS = np.load('input/test036_ldmks.npy')[25]
#     _, H, W, C = tgtI.shape
#     specI = np.zeros((H*len(sigmas), W*len(kzs), C))
#     for y, sigma in enumerate(sigmas):
#         for x, kz in enumerate(kzs):
#             print('%g - %g' % (sigma, kz))
#             pxyF, pxyS = process_proxy(rsize=rsize, ksize=(kz, kz), sigma=sigma, k=k)        
#             tmpI, tmpS = weighted_median(inpS, tgtS, tgtI, n)
#             outpI = process_teeth(tmpI, tmpS, pxyF, pxyS, rsize, boundary)
#             specI[y*H:y*H+H, x*W:x*W+W, :] = outpI
#             if sigma == 1e2 and kz == 17:
#                 cv2.imwrite('reference/0025_teeth.png', outpI)
#     cv2.imwrite('reference/spec_k=1.png', specI)
  
# def test2():
#     # tuning parameters in sharpening
#     kz    = 15
#     sigmas= [1, 2, 3, 4, 5]
#     ks    = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
#     inpI = cv2.imread('reference/0025_teeth.png')
#     H, W, C = inpI.shape
#     specI = np.zeros((H*len(sigmas), W*len(ks), C))
#     for y, sigma in enumerate(sigmas):
#         for x, k in enumerate(ks):  
#             print('%g - %g' % (sigma, k))
#             outpI = sharpen(inpI, ksize=(kz, kz), sigma=sigma, k=k)
#             specI[y*H:y*H+H, x*W:x*W+W, :] = outpI
#             if sigma == 2 and k == 0.5:
#                 cv2.imwrite('reference/0025_final.png', outpI)
#     cv2.imwrite('reference/spec_sharp_kz%d.png'%kz, specI)

# def test3():
#     # tuning parameters in teeth region detection
#     from candidate import weighted_median
#     indices = [10, 33, 45, 52, 111, 544, 1390]
#     biases  = [1, 2, 3, 4, 5]
#     bL      = 6
#     sq = Square(0.25, 0.75, 0.6, 1.00)
#     tgtdata = np.load('target/target001.npz')
#     tgtS, tgtI = tgtdata['landmarks'], tgtdata['textures']
#     boundary = sq.align(tgtI.shape[1])      # (left, right, upper, lower)
#     tgtI = tgtI[:, boundary[2]:boundary[3], boundary[0]:boundary[1], :]
#     inpdata = np.load('input/test036_ldmks.npy')[indices]
#     pxyF, pxyS = process_proxy(rsize=300)
#     _, H, W, C = tgtI.shape
#     specI = np.zeros((len(indices)*H, (len(biases)+2)*W, C))
#     for idx, inpS in enumerate(inpdata):
#         fr = indices[idx]
#         ref1I = cv2.imread('tmp/%04d.png' % fr)
#         ref2I = cv2.imread('tmp/%04d_t1.png' % fr)
#         lineI = [ref1I, ref2I]
#         tmpI, tmpS = weighted_median(inpS, tgtS, tgtI, n=100)
#         for bU in biases:
#             print('%04d-%d' % (fr, bU))
#             outpI = process_teeth(tmpI, tmpS, pxyF, pxyS, 300, boundary, biasU=bU, biasL=bL)   
#             lineI.append(outpI)
#         specI[idx*H:idx*H+H, :, :] = np.concatenate(lineI, axis=1)
#     cv2.imwrite('reference/spec_bUs_bL%d.png' % (bL), specI)

# def test4():
#     # select upper or lower teeth proxy
#     tar_path = 'target/target001.mp4'
#     fr = 3261
#     mode = 'upper'
#     select_proxy(tar_path, mode, fr)

if __name__ == '__main__':
    pass