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

if __name__ == '__main__':
    pass