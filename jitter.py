# -- coding: utf-8 --.
import audio2video
from audio2video.face import get_landmark, LandmarkIndex as LI, facefrontal, fronter, get_projM, resize, LandmarkFetcher
from audio2video.visual import vfps, size, combine_vm, visual_lipsyn
import cv2, numpy as np

gndo_path = '../tmp/jitter/frontalize/gndo.mp4'
gndn_path = '../tmp/jitter/frontalize/gnd_npca.npy'
gndnv_path = '../tmp/jitter/frontalize/gnd_npca.mp4'
gndnvt_path = '../tmp/jitter/frontalize/gnd_npca.avi'
gnd_path = '../tmp/jitter/frontalize/gnd.npy'
gndv_path = '../tmp/jitter/frontalize/gnd.mp4'
gndvt_path = '../tmp/jitter/frontalize/gnd.avi'
msc_path = '../input/mp3/a015.mp3'
syn_path = '../input/ldmk/std_u/a015.npy'
synv_path = '../tmp/jitter/frontalize/syn.mp4'
synvt_path = '../tmp/jitter/frontalize/syn.avi'
with open('../raw/fids/6jlaKyvf8WA}}00/stat.txt') as f:
    s = f.read()
mean = [float(i) for i in s.split()]
PCA_MAT = np.load('../reference/PCA_MAT.npy')

def difL2(FIDS, smo=True):
    # FIDS.shape == (nfr, 68, 2)
    avg_FIDS = (FIDS[2:] + FIDS[:-2])/2             #(nfr-2, 68, 2)
    dif_FIDS = FIDS[1:-1] - avg_FIDS
    L2 = np.linalg.norm(dif_FIDS, ord=2, axis=2)    #(nfr-2, 68)
    S = np.linalg.norm((FIDS[2:] - FIDS[:-2])/2, ord=2, axis=2)
    smooth = np.mean(L2, axis=0) / np.mean(S, axis=0)
    return L2 if not smo else L2, smooth

def test1():
    # compare ground truth, gound truth after PCA and synthesized result     
    writer = cv2.VideoWriter('../tmp/jitter/frontalize/op2d2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), vfps, size)
    cap = cv2.VideoCapture(gndo_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(nfr-5):
        ret, frame = cap.read()
        assert(ret)
        det, p2d = get_landmark(frame, LI.FULL, False, mean=mean)
        cv2.rectangle(frame, (det.left(), det.top()), (det.right(), det.bottom()), (0, 255, 0), 2)
        for pt in p2d.astype(np.int):
            cv2.circle(frame, tuple(pt), 3, (0, 0, 255), -1)
        writer.write(frame)
    writer.release()
    print('Scan original video done.')

def test2():
    # 采用ftl_face的p3d和原图的p2d得到projM
    writer = cv2.VideoWriter('../tmp/jitter/frontalize/revn!.mp4', cv2.VideoWriter_fourcc(*'mp4v'), vfps, size)
    cap = cv2.VideoCapture(gndo_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    refU = fronter.refU
    ldmk = []
    LF1 = LandmarkFetcher()
    LF2 = LandmarkFetcher()
    for i in range(nfr-5):
        if i%100==99: print(i+1)
        ret, frame = cap.read()
        assert(ret)
        det, p2d = LF1.get_landmark(frame, LI.FULL, False, mean=mean)
        frontal = facefrontal(frame, det, p2d)

        _, rs_p2d, transM, scaleM, _ = resize(frame, det, p2d)

        det_, p2d_ = LF2.get_landmark(frontal, LI.FULL, False)
        p2d_ = p2d_.astype(np.int)
        p3d = refU[p2d_[:, 1], p2d_[:, 0], :]
        projM = get_projM(p3d[LI.NOSE+LI.LIP], rs_p2d[LI.NOSE+LI.LIP], fronter.A)

        p3d_homo = np.insert(p3d, 3, [1]*p3d.shape[0], axis=1)
        p2d_homo = np.matmul(p3d_homo, projM.T)
        p2d_homo = np.matmul(p2d_homo, np.linalg.inv(transM))
        p2d_homo = np.matmul(p2d_homo, np.linalg.inv(scaleM))
        p2d_homo = p2d_homo / p2d_homo[:, 2][:, np.newaxis]
        rev_p2d = p2d_homo[:, :2].astype(np.int)
        
        for pt in p2d[LI.NOSE+LI.LIP].astype(np.int):
            cv2.circle(frame, tuple(pt), 3, (0, 0, 255), -1) 
        for pt in rev_p2d[LI.NOSE+LI.LIP].astype(np.int):
            cv2.circle(frame, tuple(pt), 3, (255, 255, 0), -1)           

        writer.write(frame)
        # for pt in p2d_.astype(np.int):
        #     cv2.circle(frontal, tuple(pt), 1, (0, 0, 255), -1)
        # writer.write(frontal)
    writer.release()

def test3():
    # 采用template的p3d和原图的p2d得到projM
    writer = cv2.VideoWriter('../tmp/jitter/frontalize/revn.mp4', cv2.VideoWriter_fourcc(*'mp4v'), vfps, size)
    cap = cv2.VideoCapture(gndo_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    refU = fronter.refU
    LF1 = LandmarkFetcher()
    LF2 = LandmarkFetcher()
    ldmk = []
    for i in range(nfr-5):
        if i%100==99: print(i+1)
        ret, frame = cap.read()
        assert(ret)
        det, p2d = LF1.get_landmark(frame, LI.FULL, False, mean=mean)
        frontal, projM, transM, scaleM, tmpshape = facefrontal(frame, det, p2d, detail=True)

        _, rs_p2d, _, _, _ = resize(frame, det, p2d)

        det_, p2d_ = LF2.get_landmark(frontal, LI.FULL, False)
        p2d_ = p2d_.astype(np.int)
        p3d = refU[p2d_[:, 1], p2d_[:, 0], :]
 
        p3d_homo = np.insert(p3d, 3, [1]*p3d.shape[0], axis=1)
        p2d_homo = np.matmul(p3d_homo, projM.T)
        p2d_homo = np.matmul(p2d_homo, np.linalg.inv(transM))
        p2d_homo = np.matmul(p2d_homo, np.linalg.inv(scaleM))
        p2d_homo = p2d_homo / p2d_homo[:, 2][:, np.newaxis]
        rev_p2d = p2d_homo[:, :2].astype(np.int)
        
        for pt in p2d[LI.NOSE+LI.LIP].astype(np.int):
            cv2.circle(frame, tuple(pt), 3, (0, 0, 255), -1) 
        for pt in rev_p2d[LI.NOSE+LI.LIP].astype(np.int):
            cv2.circle(frame, tuple(pt), 3, (255, 255, 0), -1)           

        writer.write(frame)
    writer.release()

def test4():
    from face import LandmarkFetcher
    LF = LandmarkFetcher()
    writer = cv2.VideoWriter('../tmp/jitter/frontalize/antijitter_0.mp4', cv2.VideoWriter_fourcc(*'mp4v'), vfps, size)
    cap = cv2.VideoCapture('../tmp/jitter/frontalize/gndo.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # landmarks = []
    for i in range(nfr-5):
        if i%100==99: print(i+1)
        ret, frame = cap.read()
        assert(ret)

        det, p2d = LF.get_landmark(frame, LI.FULL, False, mean=mean)
        # print('%.3f' % LF.trackPoints[51].m_velocity)
        # det, p2d = get_landmark(frame, LI.FULL, False, mean=mean)
        for pt in p2d.astype(np.int):
            cv2.circle(frame, tuple(pt), 3, (0, 0, 255), -1)
        writer.write(frame)

if __name__ == '__main__':
    test4()