import os, glob
import cv2, dlib
import numpy as np
from threading import Thread
from __init__ import raw_dir, raw_fids_dir, ref_dir, log_dir
from pca import PCA
from face import get_landmark, LandmarkIndex, facefrontal

srcdir = '%s/mp4' % raw_dir
dstdir = '%s/fids' % raw_dir
subdirs = os.listdir(dstdir)
subdirs = [x[:-4] for x in subdirs]
unqeles, unqinds, inverids, unqcnts = np.unique(subdirs, return_index=True, return_inverse=True, return_counts=True)

with open(os.path.join(raw_dir, 'links.txt')) as linkf:
    files = linkf.readlines()
    files = [x[32:43] for x in files]

def check_raw(width=10, samples=50):
    vsubdirs = os.listdir(raw_fids_dir)
    vsubdirs = [x[:-4] for x in vsubdirs]
    unqlinks, unqcnts = np.unique(vsubdirs, return_counts=True)
    
    log_path = '%s/check_raw.log' % log_dir
    if os.path.exists(log_path):
        os.remove(log_path)

    for link, cnt in zip(unqlinks, unqcnts):
        vsubdir_list = [link + "}}" + str(100+i)[1:3] + '/' for i in range(cnt)]
        mp4f = glob.glob('%s/mp4/*_%s.mp4' % (raw_dir, link))[0]
        cap = cv2.VideoCapture(mp4f)
        for idx, vsubdir in enumerate(vsubdir_list):
            # in each video fragments
            with open('%s/%s/startframe.txt' % (raw_fids_dir, vsubdir)) as f:
                vstartfr = int(f.read())
            with open('%s/%s/nframe.txt' % (raw_fids_dir, vsubdir)) as f:
                vnfr = int(f.read())
            print('%03d: %s from %d to %d' % (idx, link, vstartfr, vstartfr+vnfr))

            cap.set(cv2.CAP_PROP_POS_FRAMES, vstartfr)
            for cnt in range(vstartfr, vstartfr+min(width, vnfr)):
                ret, frame = cap.read()
                assert(ret)
                det, p2d = get_landmark(frame, LandmarkIndex.FULL, False)
                if det is None or p2d is None:
                    with open(log_path, 'a') as f:
                        f.write('[ERROR] %s}}%02d: frame %d is unacceptable.\n' % (mp4f, idx, cnt))
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, vstartfr+max(0, vnfr-width))
            for cnt in range(vstartfr+max(0, vnfr-width), vstartfr+vnfr):
                ret, frame = cap.read()
                assert(ret)
                det, p2d = get_landmark(frame, LandmarkIndex.FULL, False)
                if det is None or p2d is None:
                    with open(log_path, 'a') as f:
                        f.write('[ERROR] %s}}%02d: frame %d is unacceptable.\n' % (mp4f, idx, cnt))
    
            # generate reference statistics
            indices = np.random.randint(vstartfr, vstartfr+vnfr, samples)
            data = []
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                assert(ret)
                det, p2d = get_landmark(frame, LandmarkIndex.FULL, False)
                data.append([det.left(), det.top(), det.width()])
            data = np.array(data)
            mean = np.mean(data, axis=0)
            with open('%s/%s/stat.txt' % (raw_fids_dir, vsubdir), 'w') as f:
                f.write('%f %f %f\n'%(mean[0], mean[1], mean[2]))

    print('ALL DONE')

def video_landmark(mp4_path, args, means, detector, predictor):
    '''
    ### parameters
    mp4_path: path of mp4 file \\
    args: (n, 2) fragmentation arguments. The 1st column is startfr # and the 2nd column is nfr. 
    Totally n fragments

    ### retval
    feature_list: a list whose length is # of fragments. Each element of the list is features data 
    in ndarray format of each fragment with the shape of (nfr, 40)
    '''
    feature_list = []
    frag_id = 0

    cap = cv2.VideoCapture(mp4_path)
    for arg, mean in zip(args, means):
        # initialization
        cnt = 0
        startfr, nfr = arg
        cap.set(cv2.CAP_PROP_POS_FRAMES, startfr)
        fragment_feature = None

        while cnt < nfr:
            if (startfr+cnt) % 10 == 0:
                print(mp4_path[-15:-4],'#:', startfr+cnt)
            # fetch the frame
            ret, img_ = cap.read()
            assert(ret)
            
            # frontalization
            img = facefrontal(img_, detector=detector, predictor=predictor, mean=mean)    # img.shape = None or (320, 320, 3)
            if img is None:
                with open('%s/error.log' % log_dir, 'a') as f:
                    f.write('%s - %04d: p2d not found before frontalization.\n' % (mp4_path, startfr+cnt))
                return None

            # lip landmark extraction and normalization
            det, landmarks = get_landmark(img, LandmarkIndex.FULL, True, detector, predictor)
            if det is None or landmarks is None:
                with open('%s/error.log' % log_dir, 'a') as f:
                    f.write('%s - %04d: p2d not found after frontalization.\n' % (mp4_path, startfr+cnt))
                return None

            frame_feature = landmarks.reshape((landmarks.shape[0]*landmarks.shape[1]))  # (40, )

            # add frame_feature to fragment_feature
            if fragment_feature is None:
                fragment_feature = np.copy(frame_feature)
            else:
                fragment_feature = np.vstack((fragment_feature, frame_feature))

            cnt += 1

        # fragment_feature.shape = (fr?, 40)
        feature_list.append(fragment_feature)
        frag_id += 1

    return feature_list

def vprocess(links, cnts, tid):
    '''get lip landmark coordinates from given mp4 links
    ### parameters
    links: list of link strings, indicating mp4 file names \\
    cnts: list of int, indicating fragment numbers of each mp4 file
    ### notice
    The length of links should equal to that of cnts.
    '''
    total_cnt = 0
    tname = 'Thread-' + str(100+tid)[1:3]

    from __init__ import detdir, pdctdir
    detector = cv2.CascadeClassifier(detdir)
    predictor = dlib.shape_predictor(pdctdir)

    for link, cnt in zip(links, cnts):
        idx = files.index(link)
        srcfile = '%s/%s_%s.mp4' % (srcdir, str(1001+idx)[1:4], files[idx])

        args, means = [], []
        subdir_list = ['%s}}%s' % (link, str(100+i)[1:3]) for i in range(cnt)]
        for subdir in subdir_list:
            with open('%s/%s/stat.txt' % (dstdir, subdir)) as f:
                s = f.read()
            means.append([float(i) for i in s.split()])
            with open('%s/%s/startframe.txt' % (dstdir, subdir)) as f:
                startfr = int(f.read())
            with open('%s/%s/nframe.txt' % (dstdir, subdir)) as f:
                nfr = int(f.read())
            args.append([startfr, nfr])
        args  = np.array(args)
        means = np.array(means)

        feature_list = video_landmark(srcfile, args, means, detector, predictor)
        with open('%s/%s.log' % (log_dir, tname), 'a') as logf:
            if feature_list is not None:
                for i, subdir in enumerate(subdir_list):
                    features = feature_list[i]
                    np.save('%s/%s/features.npy' % (dstdir, subdir), features)
                    logf.write('%s/features.npy saved\n' % subdir)
            else:
                # error when extracting features
                logf.write('%s}}*/features.npy failed\n' % link)
            logf.write(str(total_cnt+1)+'/'+str(len(cnts))+' done\n====================\n')
        
        total_cnt += 1

def video_process(start_batch=0, end_batch=301, links=None, cnts=None, nthreads=10):
    logs = glob.glob('%s/*.log' % log_dir)
    for log in logs:
        os.remove(log)

    if links is None and cnts is None:
        links = unqeles[start_batch:end_batch]
        cnts  = unqcnts[start_batch:end_batch]

    neles = len(links) / nthreads
    frag_links = [links[round(i*neles):round((i+1)*neles)] for i in range(nthreads)]
    frag_cnts = [cnts[round(i*neles):round((i+1)*neles)] for i in range(nthreads)]
    threads = [Thread(target=vprocess, args=(frag_links[i], frag_cnts[i], i, )) for i in range(nthreads)]

    for tid, thread in enumerate(threads):
        thread.name = 't'+str(tid)
        print('%s start!' % thread.name)
        thread.start()
    # processing...
    for thread in threads:
        thread.join()
    print('All done')

def reduce_dim(dim=20):
    feature_list = []
    dstdir = '%s/fids' % raw_dir
    subdirs = os.listdir(dstdir)

    for subdir in subdirs:
        fea = np.load('%s/%s/features.npy' % (dstdir, subdir))
        fea = fea[:, LandmarkIndex.LIP[0]*2:]
        with open('%s/%s/nframe.txt' % (dstdir, subdir)) as f:
            nfr = int(f.read())
        assert(fea.shape[0] == nfr and fea.shape[1] == 40)
        feature_list.append(fea)

    features = np.vstack(feature_list)

    eigvector, eigvalue = PCA(features)
    A = eigvector[:, :dim]        # A.shape = (40, 20)
    np.save('%s/PCA_MAT.npy' % ref_dir, A)
    print('PCA_MAT saved')

    for fea, subdir in zip(feature_list, subdirs):
        # fea.shape = (nfr, 40)
        pca_fea = np.matmul(fea, A)     #pca_fea.shape = (nfr, 20)
        np.save('%s/%s/fidsCoeff.npy' % (dstdir, subdir), pca_fea)
        print('%s/fidsCoeff.npy saved' % subdir)

if __name__ == '__main__':
    pass