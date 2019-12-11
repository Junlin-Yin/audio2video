# -- coding: utf-8 --.
import audio2video
from audio2video.face import get_landmark, LandmarkIndex as LI, facefrontal, fronter
from audio2video.visual import vfps, size, combine_vm, visual_lipsyn
import cv2, numpy as np

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
    
    writer = cv2.VideoWriter('../tmp/jitter/frontalize/op2d.mp4', cv2.VideoWriter_fourcc(*'mp4v'), vfps, size)
    cap = cv2.VideoCapture(gndo_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
#    cap.set(cv2.CAP_PROP_POS_FRAMES, 70)
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(nfr-5):
        ret, frame = cap.read()
        assert(ret)
        det, p2d = get_landmark(frame, LI.LIP, False, mean=mean)
#        print(det.left(), det.top(), det.width())
        for pt in p2d.astype(np.int):
            cv2.circle(frame, tuple(pt), 3, (0, 0, 255), -1)
#        cv2.imshow('f', frame)
#        cv2.waitKey(0)
        writer.write(frame)
    writer.release()
    print('Scan original video done.')

#    np.save(gndn_path, gnd_ldmk)
#    visual_lipsyn(gnd_ldmk, msc_path, gndnv_path, gndnvt_path)
#
#    PCA_MAT = np.load('../reference/PCA_MAT.npy')
#    gnd_pca = np.matmul(gnd_ldmk.reshape((-1, 40)), PCA_MAT)
#    gnd_ldmk = np.matmul(gnd_pca, PCA_MAT.T).reshape(gnd_ldmk.shape)
#
#    np.save(gnd_path, gnd_ldmk)
#    visual_lipsyn(gnd_ldmk, msc_path, gndv_path, gndvt_path)
    '''
    上述实验证明：
    1.使用PCA可在一定程度上提高时序平滑度，可以预测，PCA降到的维数越低，时序平滑度越高
    2.LSTM的结果大幅度地提高了时序平滑度，说明网络结构和训练过程是没有问题的
    3.脸部转正算法和正面归一化标志点坐标的计算对破坏时序平滑具有重大嫌疑
    4.下一步实验：比较计算标志点坐标的两种方法
    '''

if __name__ == '__main__':
    test1()