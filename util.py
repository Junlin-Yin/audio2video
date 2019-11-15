from audio2video.video_preprocess import video_process, unqeles, unqcnts
import os
import shutil

def copy_fids(srcdir):
    cnt = 0
    subdirs = os.listdir(srcdir)
    for subdir in subdirs:
        if os.path.exists('%s/%s/features.npy' % (srcdir, subdir)) == True:
            print(subdir)
            shutil.copy('%s/%s/features.npy' % (srcdir, subdir), '../raw/fids/%s/features.npy' % subdir)
            cnt += 1
    print('Totally copy %d files.' % cnt)

def add_missing_fids():
    links, cnts = [], []
    for ele, cnt in zip(unqeles, unqcnts):
        if os.path.exists('../raw/fids/%s}}00/features.npy' % ele) == False:
            print(ele)
            links.append(ele)
            cnts.append(cnt)
    print(len(links))
    video_process(links=links, cnts=cnts)

if __name__ == '__main__':
    add_missing_fids()