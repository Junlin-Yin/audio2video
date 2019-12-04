def copy_fids(srcdir):
    import os
    import shutil
    cnt = 0
    subdirs = os.listdir(srcdir)
    for subdir in subdirs:
        if os.path.exists('%s/%s/features.npy' % (srcdir, subdir)) == True:
            print(subdir)
            shutil.copy('%s/%s/features.npy' % (srcdir, subdir), '../raw/fids/%s/features.npy' % subdir)
            cnt += 1
    print('Totally copy %d files.' % cnt)

def add_missing_fids():
    from audio2video.vfids import video_process, unqeles, unqcnts
    import os
    links, cnts = [], []
    for ele, cnt in zip(unqeles, unqcnts):
        if os.path.exists('../raw/fids/%s}}00/features.npy' % ele) == False:
            print(ele)
            links.append(ele)
            cnts.append(cnt)
    print(len(links))
    video_process(links=links, cnts=cnts)
    
def test_sharp():
    import numpy as np
    import cv2
    from audio2video import mouth
    img = cv2.imread('../tmp/lface.png')
    ksizes = [3, 7, 11, 15, 19, 23]
    sigmas = [0.5, 1, 2, 5, 10, 20]
    H, W = img.shape[:2]
    spec = np.zeros((H*len(ksizes), W*len(sigmas), 3), dtype=np.uint8)
    for y, ksize in enumerate(ksizes):
        for x, sigma in enumerate(sigmas):
            print('%d-%g' % (ksize, sigma))
            tmp = mouth.sharpen(img, ksize=ksize, sigma=sigma, k=0.7)
            spec[H*y:H*y+H, W*x:W*x+W] = tmp
    cv2.imwrite('../tmp/k=0.7.png', spec)

def test_facedet1():
    import cv2, numpy as np
    from face import facefrontal
#    from audio2video import detector, predictor
    cap = cv2.VideoCapture('../target/t001.mp4')
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(nfr):
        ret, frame = cap.read()
        assert(ret)
        old_frontal = facefrontal(frame, new=False)
        new_frontal = facefrontal(frame, new=True)
        center = cv2.resize(frame, (569, 320))
        img = np.zeros((320, 1209, 3), dtype=np.uint8)
        img[:, :320, :] = old_frontal
        img[:, 320:889, :] = center
        img[:, 889:, :] = new_frontal
        cv2.imwrite('../tmp/frontal/%04d.png' % i, img)
        print('../tmp/frontal: %04d/%04d' % (i+1, nfr))
    print('Done')
    
def test_facedet2():
    import cv2, numpy as np
    from face import get_landmark, get_landmark_new, LandmarkIndex as LI
    cap = cv2.VideoCapture('../target/mp4/t001.mp4')
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    OLD, NEW = [], []
#    HW = 400
    for i in range(nfr):
        ret, frame = cap.read()
        assert(ret)
        _, oldmk = get_landmark(frame, LI.FULL, norm=True)
        _, nldmk = get_landmark_new(frame, LI.FULL, norm=True)
#        img = np.zeros((HW, HW, 3), np.uint8)
#        for pt in oldmk*HW:
#            x, y = pt.astype(np.int)
#            cv2.circle(img, (x, y), 2, (0, 0, 255))
#        for pt in nldmk*HW:
#            x, y = pt.astype(np.int)
#            cv2.circle(img, (x, y), 2, (255, 255, 0))
#        cv2.imwrite('../tmp/smooth/%04d.png' % i, img)
        OLD.append(oldmk)
        NEW.append(nldmk)
        print('../tmp/smooth: %04d/%04d' % (i+1, nfr))
    OLD, NEW = np.array(OLD), np.array(NEW)
    OLDL2, NEWL2 = difL2(OLD), difL2(NEW)
    np.savez('../tmp/smooth.npz', old=OLDL2, new=NEWL2)
    
def difL2(FIDS):
    import numpy as np
    # FIDS.shape == (nfr, 68, 2)
    avg_FIDS = (FIDS[2:] + FIDS[:-2])/2             #(nfr-2, 68, 2)
    dif_FIDS = FIDS[1:-1] - avg_FIDS
    L2 = np.linalg.norm(dif_FIDS, ord=2, axis=2)    #(nfr-2, 68)
    return L2

def test_facedet3():
    import cv2, numpy as np
    from visual import vfps, size
    from face import facefrontal, get_landmark, LandmarkIndex as LI

    cap = cv2.VideoCapture('../raw/mp4/280_i2yQJ5VCHA0.mp4')
    writer = cv2.VideoWriter('../tmp/mindet.mp4', cv2.VideoWriter_fourcc(*'mp4v'), vfps, size)
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 157)
    with open('../raw/fids/i2yQJ5VCHA0}}00/stat.txt') as f:
        s = f.read()
        mean = [float(i) for i in s.split()]
    print(mean)
    for i in range(157, 157+468):
        ret, frame = cap.read()
        assert(ret)
        det, p2d = get_landmark(frame, LI.FULL, False, mean=mean)
        if det is not None and p2d is not None:
            x0, y0, x1, y1 = det.left(), det.top(), det.right(), det.bottom()
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 3)
        writer.write(frame)
        print('../tmp/frontal: %04d/%04d' % (i+1, nfr))
    print('Done')
    
if __name__ == '__main__':
    add_missing_fids()