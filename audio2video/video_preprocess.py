import os
import glob
import numpy as np
from threading import Thread
from .__init__ import raw_dir, ref_dir, log_dir
from .pca import PCA
from .landmark import landmark

srcdir = '%s/mp4' % raw_dir
dstdir = '%s/fids' % raw_dir
subdirs = os.listdir(dstdir)
subdirs = [x[:-4] for x in subdirs]
unqeles, unqinds, inverids, unqcnts = np.unique(subdirs, return_index=True, return_inverse=True, return_counts=True)

with open(os.path.join(ref_dir, 'links.txt')) as linkf:
    files = linkf.readlines()
    files = [x[32:43] for x in files]

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
    logs = glob.glob('%s/*.log' % log_dir)
    for log in logs:
        os.remove(log)

    for link, cnt in zip(links, cnts):
        idx = files.index(link)
        srcfile = '%s/%s_%s.mp4' % (srcdir, str(1001+idx)[1:4], files[idx])

        args = []
        subdir_list = ['%s}}%s' % (link, str(100+i)[1:3]) for i in range(cnt)]
        for subdir in subdir_list:
            with open('%s/%s/startframe.txt' % (dstdir, subdir)) as f:
                startfr = int(f.read())
            with open('%s/%s/nframe.txt' % (dstdir, subdir)) as f:
                nfr = int(f.read())
            args.append([startfr, nfr])
        args = np.array(args)

        feature_list, new_args = landmark(srcfile, args)

        with open('%s/%s.log' % (log_dir, tname), 'a') as logf:
            for i, subdir in enumerate(subdir_list):
                features = feature_list[i]
                np.save('%s/%s/features.npy' % (dstdir, subdir), features)
                logf.write('%s/features.npy saved\n' % subdir)
                
                if args[i, 0] != new_args[i, 0]:
                    with open('%s/%s/startframe.txt' % (dstdir, subdir), 'w') as f:
                        f.write(str(new_args[i, 0]))
                        logf.write('%s/startframe.txt modified: %s to %s\n' % (subdir, str(args[i, 0]), str(new_args[i, 0])))
                if args[i, 1] != new_args[i, 1]:
                    with open('%s/%s/nframe.txt' % (dstdir, subdir), 'w') as f:
                        f.write(str(new_args[i, 1]))
                        logf.write('%s/nframe.txt modified: %s to %s\n' % (subdir, str(args[i, 1]), str(new_args[i, 1])))
            logf.write(str(total_cnt+1)+'/'+str(len(cnts))+' done\n====================\n')

        total_cnt += 1

def video_process(start_batch=0, end_batch=301, links=None, cnts=None, nthreads=10):
    if links is None and cnts is None:
        links = unqeles[start_batch:end_batch]
        cnts  = unqcnts[start_batch:end_batch]

    neles = len(links) / nthreads
    frag_links = [links[round(i*neles):round((i+1)*neles)] for i in range(nthreads)]
    frag_cnts = [cnts[round(i*neles):round((i+1)*neles)] for i in range(nthreads)]
    threads = [Thread(target=vprocess, args=(frag_links[i], frag_cnts[i], i, )) for i in range(nthreads)]

    for tid, thread in enumerate(threads):
        thread.name = 't'+str(tid)
        thread.start()
    # processing...
    for thread in threads:
        thread.join()
    print('All done')

def reduce_dim():
    feature_list = []
    dstdir = '%s/fids' % raw_dir
    subdirs = os.listdir(dstdir)

    for subdir in subdirs:
        fea = np.load('%s/%s/features.npy' % (dstdir, subdir))
        with open('%s/%s/nframe.txt' % (dstdir, subdir)) as f:
            nfr = int(f.read())
        assert(fea.shape[0] == nfr and fea.shape[1] == 40)
        feature_list.append(fea)

    features = np.vstack(feature_list)
    mean = np.mean(features, axis=0).reshape((20, 2))
    std  = np.std(features, axis=0).reshape((20, 2))
    np.savez('%s/stat.npz' % ref_dir, mean=mean, std=std)
    print('ldmk stat saved')

    eigvector, eigvalue = PCA(features)
    A = eigvector[:,:20]        # A.shape = (40, 20)
    np.save('%s/PCA_MAT.npy' % ref_dir, A)
    print('PCA_MAT saved')

    for fea, subdir in zip(feature_list, subdirs):
        # fea.shape = (nfr, 40)
        pca_fea = np.matmul(fea, A)     #pca_fea.shape = (nfr, 20)
        np.save('%s/%s/fidsCoeff.npy' % (dstdir, subdir), pca_fea)
        print('%s/fidsCoeff.npy saved' % subdir)

if __name__ == '__main__':
    print('Hello, World')