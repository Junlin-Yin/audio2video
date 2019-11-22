# -*- coding: utf-8 -*-

'''
This is the python code as a re-implementation of the matlab code from:
http://www.openu.ac.il/home/hassner/projects/frontalize/
Tal Hassner, Shai Harel*, Eran Paz* and Roee Enbar, Effective Face Frontalization in Unconstrained Images, IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), Boston, June 2015
The algorithm credit belongs to them. 
I implement it as I dislike reading matlab code--i started using matlab when it was around 1GB but look at it now.....
In order to make the code run you need: 
1. compile the dlib python code: http://dlib.net/
2. download the shape_predictor_68_face_landmarks.dat file from:
http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
3. install python dependencies 
Contact me if you have problem using this code: 
heng.yang@cl.cam.ac.uk 
'''

import numpy as np 
import cv2
import pickle as pkl
from scipy import ndimage
from __init__ import ref3dir, detector, predictor

class LandmarkIndex():
    CHEEK = list(range(0, 17))
    BROWL = list(range(17, 22))
    BROWR = list(range(22, 27))
    NOSE  = list(range(27, 36))
    EYEL  = list(range(36, 42))
    EYER  = list(range(42, 48))
    LIPO  = list(range(48, 60))
    LIPI  = list(range(60, 68))
    LIPU  = [49, 50, 51, 52, 53, 61, 62, 63]
    LIPL  = [55, 56, 57, 58, 59, 65, 66, 67]
    BROWS = BROWL + BROWR
    FACE  = CHEEK + BROWS
    EYES  = EYEL  + EYER
    LIP   = LIPO  + LIPI
    FULL  = FACE  + NOSE + EYES + LIP
    CONTOUR_FACE  = CHEEK + BROWS[::-1]
    CONTOUR_TEETH = LIPI

def get_landmark(img, idx, norm):
    dets = detector(img, 1)
    if len(dets) != 1:
        return None, None

    det = dets[0]
    shape = predictor(img, det)
    landmarks = np.asarray([(shape.part(n).x, shape.part(n).y) for n in range(shape.num_parts)], np.float32)
    
    if norm:
        origin = np.array([det.left(), det.top()])
        size = np.array([det.width(), det.height()])
        landmarks = (landmarks - origin) / size         # restrained in [0, 0] ~ [1, 1]

    return det, landmarks[idx]

def plot3d(p3ds):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(p3ds[:,0],p3ds[:,1], p3ds[:,2])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
def resize(img, det, p2d, detw=200, padw=60):
    # detw and imgw are both determined by the template model.
    # assume that det.width() ~ det.height()
    left, top = det.left(), det.top()
    p2d_ = np.copy(p2d)
    img_ = np.copy(img)
    H, W, C = img_.shape
    
    curdetw = (det.width() + det.height()) / 2
    ratio = detw / curdetw
    # imgw == 320 (default for face_frontal)
    
    img_2 = cv2.resize(img_, (round(W*ratio), round(H*ratio)))
    H, W, C = img_2.shape
    img_ = np.zeros((H+2*padw, W+2*padw, C), dtype=np.uint8)
    img_[padw:-padw, padw:-padw, :] = img_2
    
    p2d_ = p2d_ * ratio + padw
    left = round(left * ratio + padw)
    top  = round(top  * ratio + padw)
    
    img_ = img_[top-padw:top+detw+padw, left-padw:left+detw+padw, :]
    p2d_ -= (left-padw, top-padw)
    transM = np.array([[ratio, 0, 0], [0, ratio, 0], [2*padw-left, 2*padw-top, 1]])
    
    return img_, p2d_, transM

class frontalizer():
    def __init__(self,refname):
        # initialise the model with the 3d reference face 
        # and the camera intrinsic parameters 
        # they are stored in the ref3d.pkl  
        with open(refname, 'rb') as f:
            ref = pkl.load(f, encoding='iso-8859-1')
            self.refU  = ref['refU']
            self.A = ref['outA']
            self.refxy = ref['ref_XY']
            self.p3d = ref['p3d']
            self.refimg = ref['refimg']
    def get_headpose(self,p2d):
        assert(len(p2d) == len(self.p3d))
        p3_ = np.reshape(self.p3d,(-1,3,1)).astype(np.float)
        p2_ = np.reshape(p2d,(-1,2,1)).astype(np.float)
        # print(self.A)                 # camera intrinsic matrix
        distCoeffs = np.zeros((5,1))    # distortion coefficients
        succ,rvec,tvec = cv2.solvePnP(p3_,p2_, self.A, distCoeffs)
        # rvec.shape = (3, 1) which is a compact way to represent a rotation
        # tvec.shape = (3, 1) which is used to represent a transformation
        if not succ:
            print('There is something wrong, please check.')
            return None
        else:
            matx = cv2.Rodrigues(rvec)      # matx[0] := R matrix
            ProjM_ = self.A.dot(np.insert(matx[0],3,tvec.T,axis=1))     # intrinsic * extrinsic
            return rvec,tvec,ProjM_
        
    def frontalization(self, img_, facebb, p2d_):
        #we rescale the face region (twice as big as the detected face) before applying frontalisation
        ACC_CONST = 0
        img, p2d, TransM = resize(img_, facebb, p2d_)
       
        tem3d = np.reshape(self.refU,(-1,3),order='F')
        bgids = tem3d[:,1] < 0# excluding background 3d points 
        # plot3d(tem3d)
        # print tem3d.shape 
        ref3dface = np.insert(tem3d, 3, np.ones(len(tem3d)),axis=1).T   # homogeneous coordinates
        _, _, ProjM = self.get_headpose(p2d)
        proj3d = ProjM.dot(ref3dface)
        proj3d[0] /= proj3d[2]      # homogeneous normalization
        proj3d[1] /= proj3d[2]      # homogeneous normalization
        proj2dtmp = proj3d[0:2]
        #The 3D reference is projected to the 2D region by the estimated pose 
        #Then check the projection lies in the image or not 
        vlids = np.logical_and(np.logical_and(proj2dtmp[0] > 0, proj2dtmp[1] > 0), 
                               np.logical_and(proj2dtmp[0] < img.shape[1] - 1,  proj2dtmp[1] < img.shape[0] - 1))
        vlids = np.logical_and(vlids, bgids)
        proj2d_valid = proj2dtmp[:,vlids]       # totally vlids points can be projected into the query image

        sp_  = self.refU.shape[0:2]
        synth_front = np.zeros(sp_,np.float)    # 320 * 320
        inds = np.ravel_multi_index(np.round(proj2d_valid).astype(int),(img.shape[1], img.shape[0]),order = 'F')
        unqeles, unqinds, inverids, conts  = np.unique(inds, return_index=True, return_inverse=True, return_counts=True)
        tmp_ = synth_front.flatten()
        tmp_[vlids] = conts[inverids].astype(np.float)
        synth_front = tmp_.reshape(synth_front.shape,order='F')
        synth_front = cv2.GaussianBlur(synth_front, (17,17), 30).astype(np.float)

        # color all the valid projected 2d points according to the query image
        rawfrontal = np.zeros((self.refU.shape[0],self.refU.shape[1], 3)) 
        for k in range(3):
            intervalues = ndimage.map_coordinates(img[:,:,k].T,proj2d_valid,order=3,mode='nearest')
            tmp_  = rawfrontal[:,:,k].flatten()
            tmp_[vlids] = intervalues
            rawfrontal[:,:,k] = tmp_.reshape(self.refU.shape[0:2],order='F')

        mline = synth_front.shape[1]//2
        sumleft = np.sum(synth_front[:,0:mline])
        sumright = np.sum(synth_front[:,mline:])
        sum_diff = sumleft - sumright
        # print(sum_diff)
        if np.abs(sum_diff) > ACC_CONST:
            weights = np.zeros(sp_)
            if sum_diff > ACC_CONST:        # sumleft > sumright => face to left
                weights[:,mline:] = 1.
            else:                           # sumright > sumleft => face to right
                weights[:,0:mline] = 1.
            weights = cv2.GaussianBlur(weights, (33,33), 60.5).astype(np.float)
            synth_front /= np.max(synth_front) 
            weight_take_from_org = 1 / np.exp(1 + synth_front)
            weight_take_from_sym = 1 - weight_take_from_org
            weight_take_from_org = weight_take_from_org * np.fliplr(weights)
            weight_take_from_sym = weight_take_from_sym * np.fliplr(weights) 

            weights = np.tile(weights,(1,3)).reshape((weights.shape[0],weights.shape[1],3),order='F')
            weight_take_from_org = np.tile(weight_take_from_org,(1,3)).reshape((weight_take_from_org.shape[0],weight_take_from_org.shape[1],3),order='F')
            weight_take_from_sym = np.tile(weight_take_from_sym,(1,3)).reshape((weight_take_from_sym.shape[0],weight_take_from_sym.shape[1],3),order='F')
            
            denominator = weights + weight_take_from_org + weight_take_from_sym
            frontal_sym = (rawfrontal * weights + rawfrontal * weight_take_from_org + np.fliplr(rawfrontal) * weight_take_from_sym) / denominator
        else:
            frontal_sym = rawfrontal
        return rawfrontal, frontal_sym, ProjM, TransM

fronter  = frontalizer(ref3dir)


def facefrontal(img, detail=False):
    '''
    ### parameters
    img: original image to be frontalized \\
    ### retval
    newimg: (320, 320, 3), frontalized image
    '''
    det, p2d = get_landmark(img, LandmarkIndex.FULL, norm=False)
    if det is None or p2d is None:
        return None
        
    rawfront, symfront, projM, transM = fronter.frontalization(img, det, p2d)
    newimg = symfront.astype('uint8')
    if detail == False:
        return newimg
    else:
        return newimg, p2d, projM, transM
        
def warp_mapping(fronter, indices, pixels, tarfr, tarldmk, projM, transM, ksize=10):
    # frontal points -> original resized points
    pt3d = fronter.refU[indices[:, 1], indices[:, 0], :]        # (N, 3)
    pt3d_homo = np.insert(pt3d, 3, [1]*pt3d.shape[0], axis=1)   # (N, 4)
    pt2d_homo = np.matmul(pt3d_homo, projM.T)                   # (N, 3)
#    pt2d_homo = pt2d_homo / pt2d_homo[:, 2][:, np.newaxis]
    
    # original resized points -> true original points
    opt2d_homo = np.matmul(pt2d_homo, np.linalg.inv(transM))    # (N, 3)
    opt2d_homo = opt2d_homo / opt2d_homo[:, 2][:, np.newaxis]
    opt2d = opt2d_homo[:, :2]                                   # (N, 2)
    opt2d_grid = opt2d.astype(np.int)                           # (N, 2)
    
    # eliminate occlusion caused by rotation
    # skip this part for the moment, and complete it if necessary
    
    # define the region in the original frame field to be recalculated
    warp_mask = np.zeros(tarfr.shape[:2], dtype=np.uint8)
    warp_mask[opt2d_grid[:, 1], opt2d_grid[:, 0]] = 255
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    warp_mask = cv2.morphologyEx(warp_mask, cv2.MORPH_CLOSE, kernel)
    
    face_contour = tarldmk[[2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                            12, 13, 14, 35, 34, 33, 32, 31], :].astype(np.int)
    face_mask = np.zeros(tarfr.shape[:2], dtype=np.uint8)
    face_mask = cv2.drawContours(face_mask, [face_contour], -1, 255, -1)
    
    # eliminate region that is out of face landmarks
    warp_mask = warp_mask & face_mask
    ys, xs = warp_mask.nonzero()
    region = np.array([(x, y) for x, y in zip(xs, ys)])     # (N, 2)
    
    return warp_mask, region, opt2d, pixels

if __name__ == "__main__":
    from __init__ import test_dir
    img = cv2.imread('%s/0660.png' % test_dir)
    fimg = facefrontal(img, fronter)
    cv2.imshow('test', fimg)
    cv2.waitKey(0)