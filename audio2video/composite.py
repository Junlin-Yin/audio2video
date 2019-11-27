# -*- coding: utf-8 -*-

import cv2
from __init__ import Square
from visual import vfps, size
from face import facefrontal, warp_mapping, fronter, get_landmark, LandmarkIndex
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

def pyramid_blend(img1, img2, mask_, layers=4):
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

def align2target(syntxtr, sq, spadw, fpadw=fronter.padw, fdetw=fronter.detw):
    # align lower-face to target frame
#   |padw| detw  |padw|
#   |----|-------|---------
#   |                 |fpadw
#   |    ---------    -----
#   |    |       |    |
#   |    |       |    |fdetw
#   |    |       |    |
#   |    ---------    -----
#   |     ftl_face    |fpadw
#   -----------------------
    sdetw = sq.getrsize(syntxtr.shape)
    sW = sdetw + 2*spadw
    syn_face_ = np.zeros((sW, sW, syntxtr.shape[2]), dtype=np.uint8)
    left, right, upper, lower = sq.align(sdetw, spadw)
    syn_face_[upper:lower, left:right, :] = syntxtr
    
    ratio = fdetw / sdetw
    syn_face_ = cv2.resize(syn_face_, (int(sW*ratio), int(sW*ratio)))
    spadw = int(sW*ratio - fdetw) // 2
    
    if fpadw < spadw:
        syn_face = syn_face_[spadw-fpadw:spadw+fdetw+fpadw, spadw-fpadw:spadw+fdetw+fpadw, :]
    else:
        fW = 2*fpadw + fdetw
        syn_face = np.zeros((fW, fW, syntxtr.shape[2]), dtype=np.uint8)
        syn_face[fpadw-spadw:fpadw+fdetw+spadw, fpadw-spadw:fpadw+fdetw+spadw, :] = syn_face_
    
    return syn_face

def jaw_adjust(syn_face, tarfr, det, p2d):
    return syn_face

def getindices(ftl_face, sq, fpadw=fronter.padw, fdetw=fronter.detw):
    # get mask region using boundary, chin landmarks and nose landmarks
    # boundary region: left -> right, upper -> lower
    WH = ftl_face.shape[0]
    boundary = sq.align(fdetw, fpadw)
    left, right, upper, lower = np.array(boundary)
    indices = np.array([(x, y) for x in range(left, right) for y in range(upper, lower)])
    
    # get landmarks of frontalized face
    _, ldmk = get_landmark(ftl_face, LandmarkIndex.FULL, norm=False)
    chin_xp, chin_fp = ldmk[ 3:14, 0], ldmk[ 3:14, 1]
    chin_line = np.interp(np.arange(WH), chin_xp, chin_fp)

    # filter the position which is out of chin line and nose line    
    return indices[indices[:, 1] < chin_line[indices[:, 0]]]

def recalc_pixel(pt, coords, pixels, thr=5, sigma=0.5):
    L2 = np.linalg.norm(coords-pt, ord=2, axis=1)
    indx  = np.where(L2 <= thr)
    weights = np.exp(-L2[indx]**2 / (2* sigma**2))
    weights /= np.sum(weights)  # np.sum(weights) == 1
    return np.matmul(weights, pixels[indx, :])

def warpback(face, tarfr, tarldmk, indices, projM, transM, scaleM, tmpshape):
    # get the pixels of given indices
    pixels = face[indices[:, 1], indices[:, 0], :]           # (N, 3)
    
    # get the to-be-recalculated region in the original frame
    ratio = scaleM[0, 0]
    tmpldmk = tarldmk * ratio
    warp_mask, env_mask, region, coords, pixels = warp_mapping(indices, pixels, tmpshape, tmpldmk, projM, transM)
    
    # do recalculation for every pixel in the region
    lfacefr = np.zeros(tmpshape, dtype=np.uint8)
    for pt in region:
        lfacefr[pt[1], pt[0], :] = recalc_pixel(pt, coords, pixels)
    
    # seperate upper face from the environment
    tarfr = cv2.resize(tarfr, (tmpshape[1], tmpshape[0]))
    envfr, ufacefr = tarfr & (~env_mask[:, :, np.newaxis]), tarfr & env_mask[:, :, np.newaxis]
    
    # blend lower and upper face
    ufacefr = cv2.inpaint(ufacefr, ~env_mask, 10, cv2.INPAINT_TELEA)
    lfacefr = cv2.inpaint(lfacefr, ~warp_mask, 10, cv2.INPAINT_TELEA) 
    lfacefr = sharpen(lfacefr)   
    facefr  = pyramid_blend(lfacefr, ufacefr, warp_mask)
    
    # combine face and environment
    finalfr = (facefr & env_mask[:, :, np.newaxis]) + (envfr & (~env_mask[:, :, np.newaxis]))
    
    return finalfr

def syn_frame(tarfr, syntxtr, sq, padw):
    # frontalize the target frame
    ftl_face, det, ldmk, projM, transM, scaleM, tmpshape = facefrontal(tarfr, detail=True)
    
    # align lower-face to target frame
    syn_face = align2target(syntxtr, sq, padw)

    # jaw-correction
    syn_face = jaw_adjust(syn_face, tarfr, det, ldmk)
   
    # get indices of pixels in ftl_face which needs to be blended into target frame
    indices = getindices(ftl_face, sq)
    
    # warp the synthesized face to the original pose and blending
    return warpback(syn_face, tarfr, ldmk, indices, projM, transM, scaleM, tmpshape)
   
def test1():
    syntxtr = cv2.imread('../tmp/s1490.png')
    tarfr = cv2.imread('../tmp/t1490.png')
    sq = Square(0.2, 0.8, 0.4, 1.15)
    outpfr = syn_frame(tarfr, syntxtr, sq, 100)
    cv2.imwrite('../tmp/o1490.png', outpfr)

if __name__ == '__main__':
    test1()