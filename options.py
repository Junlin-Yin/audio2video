from audio2video.__init__ import inp_dir, tar_dir
import argparse
import glob
import os

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps',      type=int,   nargs='+',      help='what steps you want to run (0-4)')
    
    parser.add_argument('--inp_id',     type=str,   default=None,   help='input audio file name')
    parser.add_argument('--tar_id',     type=str,   default=None,   help='target video file name')
    parser.add_argument('--pass_id',    type=str,   default=None,     help='LSTM pass (run) id')
    
    parser.add_argument('--nthreads',   type=int,   default=10,     help='number of threads when preprocessing')
    
    parser.add_argument('--train',      type=bool,  default=True,   help='need to train the network or not')
    parser.add_argument('--predict',    type=bool,  default=True,   help='need to predict (infer) sparse mouth shape or not')
    parser.add_argument('--outp_norm',  type=bool,  default=False,  help='normalize sparse mouth shape or not')
    parser.add_argument('--preprocess', type=bool,  default=False,  help='need to re-preprocess data or not')    
    parser.add_argument('--vr',         type=float, default=0.2,    help='validation data ratio')
    parser.add_argument('--step_delay', type=int,   default=20,     help='delay step of LSTM')
    parser.add_argument('--dim_hidden', type=int,   default=60,     help='dimension of hidden state')
    parser.add_argument('--nlayers',    type=int,   default=1,      help='number of layers of LSTM')
    parser.add_argument('--keep_prob',  type=float, default=0.8,    help='keep probability')
    parser.add_argument('--seq_len',    type=int,   default=100,    help='length of every training sequence')
    parser.add_argument('--batch_size', type=int,   default=100,    help='training batch size')
    parser.add_argument('--nepochs',    type=int,   default=300,    help='number of epochs')
    parser.add_argument('--grad_clip',  type=int,   default=10,     help='parameter to clip the gradients')
    parser.add_argument('--lr',         type=float, default=1e-3,   help='learning rate')
    parser.add_argument('--dr',         type=float, default=0.99,   help='decay rate of learning rate')
    parser.add_argument('--b_savef',    type=int,   default=50,     help='batch report save frequency')
    parser.add_argument('--e_savef',    type=int,   default=5,      help='checkpoint save frequency')
    parser.add_argument('--argspath',   type=str,   default=None,   help='user-specified args file path')
    parser.add_argument('--showGraph',  type=bool,  default=False,  help='generate tensorboard report or not')
    
    parser.add_argument('--lineU',      type=int,   default=-1,     help='height of the line seperating upper & lower teeth in upper teeth proxy')
    parser.add_argument('--lineL',      type=int,   default=-1,     help='height of the line seperating upper & lower teeth in lower teeth proxy')
    
    parser.add_argument('--retiming',   type=bool,  default=True,   help='use the target video after re-timing or the original one')
    return parser

def check_input(opt):
    assert(opt.inp_id is not None)
    flag = os.path.exists('%s/mp3/%s.mp3' % (inp_dir, opt.inp_id))
    assert(flag is True)

def check_target(opt):
    assert(opt.tar_id is not None)
    flag = os.path.exists('%s/mp4/%s.mp4' % (tar_dir, opt.tar_id))
    assert(flag is True)

def check_proxy(opt):
    assert(opt.lineU > 0)
    assert(opt.lineL > 0)
    lower_f = glob.glob('%s/proxy/%s_%s*.png' % (tar_dir, opt.tar_id, 'lower'))
    upper_f = glob.glob('%s/proxy/%s_%s*.png' % (tar_dir, opt.tar_id, 'upper'))
    assert(len(lower_f) == 1 and len(upper_f) == 1)

def parse(parser):
    opt = parser.parse_args()
    opt.steps = sorted(opt.steps)
    assert(min(opt.steps)>=0 and max(opt.steps) <= 4)

    if 0 in opt.steps:
        assert(opt.nthreads > 0)

    if 1 in opt.steps:
        assert(opt.pass_id is not None)
        if opt.predict is True:
            check_input(opt)

    if 2 in opt.steps:
        assert(opt.pass_id is not None)
        check_input(opt)
        check_target(opt)
        check_proxy(opt)

    if 3 in opt.steps:
        check_input(opt)
        check_target(opt)

    if 4 in opt.steps:
        check_input(opt)
        check_target(opt)
    return opt  

def ret_opt():
    return parse(init_parser())
