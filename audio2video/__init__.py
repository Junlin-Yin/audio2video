import os, sys
import cv2, dlib

package_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(package_path)

os.chdir(os.path.abspath(os.path.dirname(__file__)))
inp_dir  = '../input'
log_dir  = '../log'
outp_dir = '../output'
raw_dir  = '../raw' 
ref_dir  = '../reference'
tar_dir  = '../target'
tmp_dir  = '../tmp'
trn_dir  = '../train'

raw_mfcc_dir = '%s/mfcc' % raw_dir
raw_fids_dir = '%s/fids' % raw_dir

detdir  = '%s/haarcascade_frontalface_default.xml' % ref_dir
pdctdir = '%s/shape_predictor_68_face_landmarks.dat' % ref_dir
ref3dir = '%s/ref3d.pkl' % ref_dir
detector = cv2.CascadeClassifier(detdir)
predictor = dlib.shape_predictor(pdctdir)

class Square:
    def __init__(self, l, r, u, d):
        self.left = l
        self.right = r
        self.up = u
        self.down = d
        
    def align(self, S, padw, margin=False):
        left  = round(self.left  * S + padw)
        right = round(self.right * S + padw)
        upper = round(self.up    * S + padw)
        lower = round(self.down  * S + padw)
        if margin == False: return left, right, upper, lower
        else: return left+1, right-1, upper+1, lower-1

    def alignh(self, S, padw, padh, margin=True):
        left  = round(self.left  * S + padw)
        right = round(self.right * S + padw)
        upper = round(self.up    * S + padh)
        lower = round(self.down  * S + padh)
        if margin == False: return left, right, upper, lower
        else: return left+1, right-1, upper+1, lower-1

    def getrsize(self, sp):
        rs1 = sp[1] / (self.right - self.left)
        rs2 = sp[0] / (self.down - self.up)
        return round((rs1+rs2)/2)