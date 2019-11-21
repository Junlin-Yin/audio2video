# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:28:05 2019

@author: xinzhu
"""

from subprocess import call
import numpy as np
import cv2
import os

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
    
if __name__ == '__main__':
    pass