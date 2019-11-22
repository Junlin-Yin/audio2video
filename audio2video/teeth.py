# -*- coding: utf-8 -*-

import os
import cv2
import glob
import numpy as np
from face import get_landmark, LandmarkIndex
from __init__ import ref_dir

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

def detect_region(inpI, inpS, rsize, padw, boundary, ksize=4):
    ctr_idx = np.array(LandmarkIndex.CONTOUR_TEETH)-LandmarkIndex.LIP[0]
    contour = inpS[ctr_idx] * rsize + (padw - boundary[0], padw - boundary[2])
    contour = contour[:, ::-1].astype(np.int)
    mask = np.zeros(inpI.shape[:2], dtype=np.uint8)
    mask = cv2.drawContours(mask, [contour], -1, 255, -1)
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    mask = cv2.erode(mask, kernel)

    xs, ys = mask.nonzero()
    region = np.array([(y, x) for (x, y) in zip(xs, ys)])
    if region.shape[0] == 0:
        # no region detected; return empty arrays
        upper_region, lower_region = np.array([]), np.array([])
    else:
        # divide region into upper and lower ones
        axis = (np.max(region[:, 0]) + np.min(region[:, 0])) / 2
        upper_region = region[region[:, 0] <= axis]
        lower_region = region[region[:, 0] >  axis]

    return upper_region, lower_region
    
def local_enhancement(inpI, inpS, pxyF, pxyS, region, rsize, padw, boundary, mode):
    # enhance quality of input image given teeth proxy filter and their landmarks
    # region is the part which is to be enhanced
    assert(mode == 'upper' or mode == 'lower')
    
    # calculate displacement between input image and proxy
    idxu = np.array(LandmarkIndex.LIPU) - LandmarkIndex.LIP[0]
    idxl = np.array(LandmarkIndex.LIPL) - LandmarkIndex.LIP[0]
    inpS_half = inpS[idxu, :] if mode == 'upper' else inpS[idxl, :]
    pxyS_half = pxyS[idxu, :] if mode == 'upper' else pxyS[idxl, :]
    dx, dy = np.round(np.mean(inpS_half - pxyS_half, axis=0) * rsize).astype(np.int)
    
    outpI = np.copy(inpI)
    x_bd, y_bd = boundary[0]-padw, boundary[2]-padw
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

def process_teeth(inpI, inpS, pxyF, pxyS, rsize, padw, boundary):
    # detect upper and lower teeth region
    regionU, regionL = detect_region(inpI, inpS, rsize, padw, boundary)
    # enhance upper region
    tmpI = local_enhancement(inpI, inpS, pxyF['upper'], pxyS['upper'], regionU, rsize, padw, boundary, 'upper')
    # enhance lower region
    outpI = local_enhancement(tmpI, inpS, pxyF['lower'], pxyS['lower'], regionL, rsize, padw, boundary, 'lower')
    return outpI

# def test1():
#     # select upper or lower teeth proxy
#     tar_path = 'target/target001.mp4'
#     fr = 3261
#     mode = 'upper'
#     select_proxy(tar_path, mode, fr)

if __name__ == '__main__':
    pass