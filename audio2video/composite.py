# -*- coding: utf-8 -*-

import cv2
from __init__ import Square
from face import facefrontal, warp_mapping, get_landmark, LandmarkIndex, fronter
from mouth import sharpen
import numpy as np

def getGaussianPyr(img, layers):
    g = img.astype(np.float64)
    pyramid = [g]
    for i in range(layers):
        g = cv2.pyrDown(g)
        pyramid.append(g)
    return pyramid

def getLaplacianPyr(Gaupyr):
    pyramid = []
    for i in range(len(Gaupyr)-2, -1, -1):
        # len(Gaupyr)-2, ..., 1, 0
        gi = Gaupyr[i]
        gi_aprx = cv2.pyrUp(Gaupyr[i+1])
        gi_aprx = cv2.resize(gi_aprx, gi.shape[:2][::-1])
        pyramid.append((gi - gi_aprx))
    return pyramid[::-1]

def reconstruct(G, Lappyr):
    for i in range(len(Lappyr)-1, -1, -1):
        # len(Gaupyr)-1, ..., 1, 0
        G = cv2.pyrUp(G)
        G = cv2.resize(G, Lappyr[i].shape[:2][::-1])
        G += Lappyr[i]
    return G.astype(np.uint8)

def pyramid_blend(img1, img2, mask_, layers=3):
    assert(img1.shape == img2.shape and img1.shape[:2] == mask_.shape)
    mask = mask_ / np.max(mask_)    # 0 ~ 1
    # construct Gaussian pyramids of input images
    Gaupyr1 = getGaussianPyr(img1, layers+1)
    Gaupyr2 = getGaussianPyr(img2, layers+1)
    Gaupyrm = getGaussianPyr(mask, layers+1)
    
    # construct Laplacian pyramids of input images
    Lappyr1 = getLaplacianPyr(Gaupyr1)
    Lappyr2 = getLaplacianPyr(Gaupyr2)
    
    # blend pyramids in every layer
    Gaupyrm1 = Gaupyrm[:-1]
    Gaupyrm2 = [1-msk for msk in Gaupyrm1]
    BLappyr1 = [lap * msk[:, :, np.newaxis] for lap, msk in zip(Lappyr1, Gaupyrm1)]
    BLappyr2 = [lap * msk[:, :, np.newaxis] for lap, msk in zip(Lappyr2, Gaupyrm2)]
    BLappyr  = [lap1 + lap2 for lap1, lap2 in zip(BLappyr1, BLappyr2)]
    initG = Gaupyr1[-1] * Gaupyrm[-1][:, :, np.newaxis] + Gaupyr2[-1] * (1-Gaupyrm[-1])[:, :, np.newaxis]
    
    # collapse pyramids and form the blended image
    img = reconstruct(initG, BLappyr)
    return img

def align2target(syntxtr, sq, spadw, fpadw, fpadh, fdetw, fWH):
    # align syn_face to ftl_face
#   |fpadw| fdetw |     |               |spadw| sdetw |     |
#   |-----|-------|----------           |-----|-------|----------
#   |                   |fpadh          |                   |spadh
#   |     ---------     ------          |     ---------     ------
#   |     |       |     |               |     |       |     |
#   |     |       |     |fdetw          |     |       |     |sdetw
#   |     |       |     |               |     |       |     |
#   |     ---------     -----           |     ---------     -----
#   |     ftl_face      |               |     syn_face      |
#   -----------------------             -----------------------

    sdetw = sq.getrsize(syntxtr.shape)
    sWH = sdetw + 2*spadw
    syn_face_ = np.zeros((sWH, sWH, syntxtr.shape[2]), dtype=np.uint8)
    left, right, upper, lower = sq.align(sdetw, spadw)
    syn_face_[upper:lower, left:right, :] = syntxtr
    
    ratio = fdetw / sdetw
    syn_face_ = cv2.resize(syn_face_, (round(sWH*ratio), round(sWH*ratio)))
    
    sWH = round(sWH*ratio)
    spadw = (sWH - fdetw) // 2
    
    syn_face = np.zeros((max(fpadh+sWH, spadw+fWH), max(fpadw+sWH, spadw+fWH), 3), dtype=np.uint8)
    syn_face[fpadh:fpadh+sWH, fpadw:fpadw+sWH] = syn_face_
    syn_face = syn_face[spadw:spadw+fWH, spadw:spadw+fWH]
    
    return syn_face

def getindices(ftl_face, sq, fpadw, fpadh, fdetw, fWH, fp2d):
    # get mask region using boundary, chin landmarks and nose landmarks
    # boundary region: left -> right, upper -> lower
    left, right, upper, lower = sq.alignh(fdetw, fpadw, fpadh)
    indices = np.array([(x, y) for x in range(left, right) for y in range(upper, lower)])
    
    # get landmarks of frontalized face
    chin_xp, chin_fp = fp2d[ 3:14, 0], fp2d[ 3:14, 1]
    chin_line = np.interp(np.arange(fWH), chin_xp, chin_fp)

    # filter the position which is out of chin line  
    return indices[indices[:, 1] < chin_line[indices[:, 0]]]

def recalc_pixel(pt, coords, pixels, thr=5, sigma=0.5):
    L2 = np.linalg.norm(coords-pt, ord=2, axis=1)
    indx  = np.where(L2 <= thr)
    weights = np.exp(-L2[indx]**2 / (2* sigma**2))
    weights /= np.sum(weights)  # np.sum(weights) == 1
    return np.matmul(weights, pixels[indx, :])

def warpback(face, tarfr, tarldmk, indices, projM, transM, scaleM, tmpshape, ksize=100):
    # get the pixels of given indices
    pixels = face[indices[:, 1], indices[:, 0], :]           # (N, 3)
    
    # get the to-be-recalculated region in the original frame
    ratio = scaleM[0, 0]
    tmpldmk = tarldmk * ratio
    tmpfr = cv2.resize(tarfr, (tmpshape[1], tmpshape[0]))
    warp_mask, env_mask, region, coords, pixels = warp_mapping(indices, pixels, tmpfr, tmpldmk, projM, transM)

    # do recalculation for every pixel in the region
    lfacefr = np.zeros(tmpshape, dtype=np.uint8)
    for pt in region:
        lfacefr[pt[1], pt[0], :] = recalc_pixel(pt, coords, pixels)

    # seperate upper face from the environment
    envfr, ufacefr = tmpfr & (~env_mask[:, :, np.newaxis]), tmpfr & env_mask[:, :, np.newaxis]
    
    # blend lower and upper face
    inp_mask = cv2.dilate(env_mask, np.ones((ksize, ksize), dtype=np.uint8)) - env_mask
    ufacefr = cv2.inpaint(ufacefr, inp_mask, 10, cv2.INPAINT_TELEA)
    inp_mask = cv2.dilate(warp_mask, np.ones((ksize, ksize), dtype=np.uint8)) - warp_mask
    lfacefr = cv2.inpaint(lfacefr, inp_mask, 10, cv2.INPAINT_TELEA) 

    lfacefr = sharpen(lfacefr)   
    facefr  = pyramid_blend(lfacefr, ufacefr, warp_mask)

    # combine face and environment
    finalfr = (facefr & env_mask[:, :, np.newaxis]) + (envfr & (~env_mask[:, :, np.newaxis]))  
    return finalfr

def syn_frame(tarfr, syntxtr, sq, spadw):
    # frontalize the target frame
    ftl_face, det, ldmk, projM, transM, scaleM, tmpshape = facefrontal(tarfr, detail=True)
    ftl_det, ftl_p2d = get_landmark(ftl_face, idx=LandmarkIndex.FULL, norm=False)

    # align lower-face to target frame
    # fpadw, fpadh, fdetw = ftl_det.left(), ftl_det.top(), ftl_det.width()
    fdet = fronter.det
    fpadw, fpadh, fdetw = fdet.left(), fdet.top(), fdet.width()
    fWH = ftl_face.shape[0]
    syn_face = align2target(syntxtr, sq, spadw, fpadw, fpadh, fdetw, fWH)

    # jaw-correction (omitted here)
   
    # get indices of pixels in ftl_face which needs to be blended into target frame
    indices = getindices(ftl_face, sq, fpadw, fpadh, fdetw, fWH, ftl_p2d)
    
    # warp the synthesized face to the original pose and blending
    return warpback(syn_face, tarfr, ldmk, indices, projM, transM, scaleM, tmpshape)
   
def test1():
    import time
    syntxtr = cv2.imread('../tmp/0750s.png')
    tarfr = cv2.imread('../tmp/0750t.png')
    sq = Square(0.3, 0.7, 0.5, 1.1)
    start = time.time()
    outpfr = syn_frame(tarfr, syntxtr, sq, 100)
    print('duration: %.2f' % (time.time()-start))
    cv2.imwrite('../tmp/0750o_50.png', outpfr)

if __name__ == '__main__':
    test1()
