# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:28:05 2019

@author: xinzhu
"""

from subprocess import call
import numpy as np
import cv2

vfps = 30
size  = (1280, 720)
rsize = 300
start = (440, 160)

def combine_vm(tpath, mpath, vpath):
    command = 'ffmpeg -i %s -i %s -c:v copy -c:a aac -strict experimental %s' % (tpath, mpath, vpath)
    call(command)

def visual_lipsyn(ldmks, mpath, vpath, tmppath):   
    writer = cv2.VideoWriter(tmppath, cv2.VideoWriter_fourcc(*'DIVX'), vfps, size)
    for ldmk in ldmks:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for pt in ldmk:
            show_pt = pt*rsize + start
            show_pt = show_pt.astype(np.int)
            frame = cv2.circle(frame, tuple(show_pt), 3, (255, 255, 255), -1)
        writer.write(frame)
    combine_vm(tmppath, mpath, vpath)

def visual_lfacesyn(txtrs, mpath, vpath, tmppath):
    writer = cv2.VideoWriter(tmppath, cv2.VideoWriter_fourcc(*'DIVX'), vfps, size)
    for cnt, img in enumerate(txtrs):
        H, W = img.shape[:2]
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        left, upper = start
        frame[upper:upper+H, left:left+W, :] = img
        writer.write(frame)
    combine_vm(tmppath, mpath, vpath)
    
def visual_retiming(L, tpath, orpath, rtpath):
    from retiming import half_warp
    # read in all frames of target video
    print('Start forming new retimed target video...')
    cap = cv2.VideoCapture(tpath)
    writer1 = cv2.VideoWriter(rtpath, cv2.VideoWriter_fourcc(*'mp4v'), vfps, size)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, L[0])
    ret, firstfr = cap.read()
    assert(ret)
    writer1.write(firstfr)

    cap.set(cv2.CAP_PROP_POS_FRAMES, L[1])
    ret, prefr = cap.read()
    assert(ret)
    for i in range(2, L.shape[0]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, L[i])
        ret, curfr = cap.read()
        assert(ret)
        
        # check and warp the duplicate frames
        if L[i-2] == L[i-1]:
            tmpfr1 = half_warp(prefr, curfr).astype(np.int)
            tmpfr2 = half_warp(curfr, prefr).astype(np.int)            
            prefr = ((tmpfr1 + tmpfr2) / 2).astype(np.uint8)
        
        writer1.write(prefr)
        prefr = curfr

    writer1.write(prefr)
    print('Done')
    
    print('Start forming new NON-retimed target video...')
    writer2 = cv2.VideoWriter(orpath, cv2.VideoWriter_fourcc(*'mp4v'), vfps, size)
    startfr = int(L[0])
    for idxfr in range(startfr, startfr+len(L)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idxfr)
        ret, curfr = cap.read()
        assert(ret)
        writer2.write(curfr)
    print('Done')
    
if __name__ == '__main__':
    pass