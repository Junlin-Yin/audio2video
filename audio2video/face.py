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
import cv2, dlib
import pickle as pkl
from scipy import ndimage
from __init__ import ref3dir, detector as detector_default, predictor as predictor_default

face_lower = np.array([0, 90, 145])
face_upper = np.array([40, 255, 255])

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
    FRONTAL = BROWS + EYES + NOSE

class PointState:
    m_deltaTime = 1.0
    m_accelNoiseMag = 0.005

    def __init__(self, point):
        self.m_point = point
        self.m_velocity = 0
        self.m_kalman = cv2.KalmanFilter(4, 2, 0, cv2.CV_64F)
        # initialize kalman filter parameters
        self.m_kalman.transitionMatrix = np.array([
            [1, 0, self.m_deltaTime, 0],
            [0, 1, 0, self.m_deltaTime],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float64)
        self.m_kalman.statePre  = np.array([point[0], point[1], 0, 0])  # x, y, vx, vy
        self.m_kalman.statePost = np.array(point)
        self.m_kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float64)
        self.m_kalman.processNoiseCov = np.array([
            [self.m_deltaTime**4 / 4, 0, self.m_deltaTime**3 / 2, 0],
            [0, self.m_deltaTime**4 / 4, 0, self.m_deltaTime**3 / 2],
            [self.m_deltaTime**3 / 2, 0, self.m_deltaTime**2 / 1, 0],
            [0, self.m_deltaTime**3 / 2, 0, self.m_deltaTime**2 / 1]
        ])
        self.m_kalman.processNoiseCov *= self.m_accelNoiseMag
        cv2.setIdentity(self.m_kalman.measurementNoiseCov, 0.5)
        cv2.setIdentity(self.m_kalman.errorCovPost, 0.1)

    def update(self, point):
        measurement = np.copy(point).astype(np.float64)
        self.m_kalman.correct(measurement)
        prediction = self.m_kalman.predict()
        self.m_point = prediction[:2]
        self.m_velocity = np.linalg.norm(prediction[2:], ord=2)

class LandmarkFetcher:
    def __init__(self, detector=None, predictor=None):
        self.prevGray = None
        self.trackPoints = None
        self.detector = detector_default if detector is None else detector
        self.detector_alt = dlib.get_frontal_face_detector()
        self.predictor = predictor_default if predictor is None else predictor
    
    def clear(self):
        self.prevGray = None
        self.trackPoints = None

    def get_landmark(self, currFrame, idx, norm, mean=None, minWH=150, debug=False):
        # generate all possible bboxes
        self.currGray = cv2.cvtColor(currFrame, cv2.COLOR_BGR2GRAY)
        dets = self.detector.detectMultiScale(self.currGray, minSize=(minWH, minWH)) 
        if len(dets) == 0:
            print('[Warning] Using alternative cascade classifier (face.py)...')
            dets = self.detector_alt(currFrame, 1)
            dets = [(det.left(), det.top(), det.width(), det.height()) for det in dets]
            if len(dets) == 0:
                return None, None
        if debug: print(dets)

        # select the optimal bbox
        def getkey(det):
            pt1 = np.array([det[0], det[1], det[2]])
            pt2 = np.array(mean)
            return np.linalg.norm(pt1-pt2, ord=2)
        if mean is None:
            dets = sorted(dets, key=lambda x: x[2], reverse=True)
        else:
            dets = sorted(dets, key=getkey, reverse=False)
        x, y, w, h = dets[0]

        # generate landmark coordinates
        det = dlib.rectangle(x, y, x+w, y+h)
        shape = self.predictor(currFrame, det)
        landmarks = np.asarray([(shape.part(n).x, shape.part(n).y) for n in range(shape.num_parts)], np.float32)
        
        # anti-jitter using optical-flow and kalman filter
        if self.prevGray is None or self.trackPoints is None:
            self.trackPoints = [PointState(pt) for pt in landmarks]
        else:
            for i in range(landmarks.shape[0]):
                self.trackPoints[i].update(landmarks[i])
        new_landmarks = np.array([ps.m_point for ps in self.trackPoints])
        velocities = np.array([ps.m_velocity for ps in self.trackPoints])
        alphas = 1 / 3**velocities[:, np.newaxis]
        alphas[alphas>0.7] == 1
        landmarks = landmarks*(1-alphas) + new_landmarks*alphas
        self.prevGray = self.currGray

        # normalization
        if norm:
            origin = np.array([det.left(), det.top()])
            size = np.array([det.width(), det.height()])
            landmarks = (landmarks - origin) / size         # restrained in [0, 0] ~ [1, 1]
        return det, landmarks[idx]

def get_landmark(img, idx, norm, detector=None, predictor=None, mean=None, debug=False):
    detector = detector_default if detector is None else detector
    predictor = predictor_default if predictor is None else predictor
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector.detectMultiScale(gray, minSize=(150, 150))
    if len(dets) == 0:
        print('[Warning] Using alternative cascade classifier (face.py)...')
        detector_alt = dlib.get_frontal_face_detector()
        dets = detector_alt(img, 1)
        dets = [(det.left(), det.top(), det.width(), det.height()) for det in dets]
        if len(dets) == 0:
            return None, None
    if debug: print(dets)

    def getkey(det):
        pt1 = np.array([det[0], det[1], det[2]])
        pt2 = np.array(mean)
        return np.linalg.norm(pt1-pt2, ord=2)
    if mean is None:
        dets = sorted(dets, key=lambda x: x[2], reverse=True)
    else:
        dets = sorted(dets, key=getkey, reverse=False)
    x, y, w, h = dets[0]

    det = dlib.rectangle(x, y, x+w, y+h)
    shape = predictor(img, det)
    landmarks = np.asarray([(shape.part(n).x, shape.part(n).y) for n in range(shape.num_parts)], np.float32)

    if norm:
        origin = np.array([det.left(), det.top()])
        size = np.array([det.width(), det.height()])
        landmarks = (landmarks - origin) / size         # restrained in [0, 0] ~ [1, 1]

    return det, landmarks[idx]

def test_cvdet(subdir, fr, idx):
    import glob
    from __init__ import raw_dir
    mp4f = glob.glob('%s/mp4/*_%s.mp4' % (raw_dir, subdir))[0]
    with open('%s/fids/%s}}%02d/stat.txt' % (raw_dir, subdir, idx)) as f:
        s = f.read()
        mean = [float(i) for i in s.split()]
    cap = cv2.VideoCapture(mp4f)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
    _, frame = cap.read()
    det, p2d = get_landmark(frame, LandmarkIndex.FULL, False, mean=mean, debug=True)
    ft = facefrontal(frame, mean=mean)
    print(det.left(), det.top(), det.width(), det.height())
    # if det is not None and p2d is not None:
    x0, y0, x1, y1 = det.left(), det.top(), det.right(), det.bottom()
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 3)
    for pt in p2d:
        cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)
    det, p2d = get_landmark(ft, LandmarkIndex.FULL, False)
    cv2.imshow('f', frame)
    cv2.imshow('s', ft)
    cv2.waitKey(0)

def plot3d(p3ds):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(p3ds[:,0],p3ds[:,1], p3ds[:,2])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
def resize(img, det, p2d, detw=182, padw=70, padh=57, WH=320):
    # detw and imgw are both determined by the template model.
    # assume that det.width() ~ det.height()
    left, top = det.left(), det.top()
    p2d_ = np.copy(p2d)
    img_ = np.copy(img)
    H, W, C = img_.shape
    
    curdetw = (det.width() + det.height()) / 2
    ratio = detw / curdetw
    
    img_2 = cv2.resize(img_, (round(W*ratio), round(H*ratio)))
    H, W, C = img_2.shape
    img_ = np.zeros((H+WH-padh, W+WH-padw, C), dtype=np.uint8)
    img_[padh:padh+H, padw:padw+W, :] = img_2
    
    p2d_ = p2d_ * ratio + (padw, padh)
    left = round(left * ratio + padw)
    top  = round(top  * ratio + padh)
    
    img_ = img_[top-padh:top-padh+WH, left-padw:left-padw+WH, :]
    p2d_ -= (left-padw, top-padh)
    transM = np.array([[1, 0, 0], [0, 1, 0], [2*padw-left, 2*padh-top, 1]])
    scaleM = np.array([[ratio, 0, 0], [0, ratio, 0], [0, 0, 1]])
    
    return img_, p2d_, transM, scaleM, img_2.shape

def get_projM(p3d, p2d, intrM):
    # p3_ = np.reshape(p3d, (-1, 3, 1)).astype(np.float)
    # p2_ = np.reshape(p2d, (-1, 2, 1)).astype(np.float)
    p3_ = np.reshape(p3d[LandmarkIndex.FRONTAL],(-1,3,1)).astype(np.float)
    p2_ = np.reshape(p2d[LandmarkIndex.FRONTAL],(-1,2,1)).astype(np.float)
    distCoeffs = np.zeros((5, 1))    # distortion coefficients
    succ,rvec,tvec = cv2.solvePnP(p3_, p2_, intrM, distCoeffs)

    if not succ: return None
    matx = cv2.Rodrigues(rvec)      # matx[0] := R matrix
    ProjM = intrM.dot(np.insert(matx[0], 3, tvec.T, axis=1))     # intrinsic * extrinsic
    return ProjM

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
            self.det = dlib.rectangle(70, 57, 252, 239)
    def get_headpose(self,p2d):
        assert(len(p2d) == len(self.p3d))
        ProjM_ = get_projM(self.p3d, p2d, self.A)
        return ProjM_
        
    def frontalization(self, img_, facebb, p2d_):
        #we rescale the face region (twice as big as the detected face) before applying frontalisation
        ACC_CONST = 0
        img, p2d, TransM, ScaleM, tmpshape = resize(img_, facebb, p2d_)
       
        tem3d = np.reshape(self.refU,(-1,3),order='F')
        bgids = tem3d[:,1] < 0# excluding background 3d points 
        # plot3d(tem3d)
        # print tem3d.shape 
        ref3dface = np.insert(tem3d, 3, np.ones(len(tem3d)),axis=1).T   # homogeneous coordinates
        ProjM = self.get_headpose(p2d)
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
        return rawfrontal, frontal_sym, ProjM, TransM, ScaleM, tmpshape

fronter  = frontalizer(ref3dir)


def facefrontal(img, det, p2d, detail=False):
    '''
    ### parameters
    img: original image to be frontalized \\
    ### retval
    newimg: (320, 320, 3), frontalized image
    '''
    rawfront, symfront, projM, transM, scaleM, tmpshape = fronter.frontalization(img, det, p2d)
    newimg = symfront.astype('uint8')
    if detail == False:
        return newimg
    else:
        return newimg, projM, transM, scaleM, tmpshape
        
def warp_mapping(indices, pixels, tmpfr, tmpldmk, projM, transM, ksize1=10, ksize2=50):
    # frontal points -> original resized points
    pt3d = fronter.refU[indices[:, 1], indices[:, 0], :]        # (N, 3)
    pt3d_homo = np.insert(pt3d, 3, [1]*pt3d.shape[0], axis=1)   # (N, 4)
    pt2d_homo = np.matmul(pt3d_homo, projM.T)                   # (N, 3)
    
    # original resized points -> true original points
    opt2d_homo = np.matmul(pt2d_homo, np.linalg.inv(transM))    # (N, 3)
    opt2d_homo = opt2d_homo / opt2d_homo[:, 2][:, np.newaxis]
    opt2d = opt2d_homo[:, :2]                                   # (N, 2)
    opt2d_grid = opt2d.astype(np.int)                           # (N, 2)
    
    # eliminate occlusion caused by rotation
    # skip this part for the moment, and complete it if necessary
    
    # define the region in the original frame field to be recalculated
    warp_mask = np.zeros(tmpfr.shape[:2], dtype=np.uint8)
    warp_mask[opt2d_grid[:, 1], opt2d_grid[:, 0]] = 255
    warp_mask = cv2.morphologyEx(warp_mask, cv2.MORPH_CLOSE, np.ones((ksize1, ksize1), dtype=np.uint8))
    
    # p2d_mask: face contour mask based on landmark positions
    ctr_idx = LandmarkIndex.CONTOUR_FACE
    p2d_contour = tmpldmk[ctr_idx, :].astype(np.int)
    p2d_mask = np.zeros(tmpfr.shape[:2], dtype=np.uint8)
    p2d_mask = cv2.drawContours(p2d_mask, [p2d_contour], -1, 255, -1)

    # hsv_mask: face contour mask based on hsv bounds
    hsv = cv2.cvtColor(tmpfr, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, face_lower, face_upper)
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, np.ones((ksize2, ksize2), dtype=np.uint8))
  
    # eliminate region that is out of face landmarks
    face_mask = p2d_mask & hsv_mask
    warp_mask = warp_mask & face_mask
    ys, xs = warp_mask.nonzero()
    region = np.array([(x, y) for x, y in zip(xs, ys)])     # (N, 2)
    
    return warp_mask, face_mask, region, opt2d, pixels

if __name__ == "__main__":
    pass
    # test_cvdet('6cKIPvfvxKo', 3842, 2)