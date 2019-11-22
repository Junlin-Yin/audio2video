import os
import dlib

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

pdctdir = '%s/shape_predictor_68_face_landmarks.dat' % ref_dir
ref3dir = '%s/ref3d.pkl' % ref_dir
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(pdctdir)

class Square:
    def __init__(self, l, r, u, d):
        self.left = l
        self.right = r
        self.up = u
        self.down = d
        
    def align(self, S, padw):
        left  = round(self.left  * S + padw)
        right = round(self.right * S + padw)
        upper = round(self.up    * S + padw)
        lower = round(self.down  * S + padw)
        return left, right, upper, lower
        