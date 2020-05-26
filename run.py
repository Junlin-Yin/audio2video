import audio2video as a2v
from audio2video import amfcc, vfids
from audio2video import lipnn, loadstore
from audio2video import retiming, mouth
from audio2video import visual
from options import ret_opt
import os

raw_dir = a2v.raw_dir
inp_dir = a2v.inp_dir
tmp_dir = a2v.tmp_dir
tar_dir = a2v.tar_dir
outp_dir= a2v.outp_dir

def step0_dataset(opt):
    print('[*] ==================== Start of Step 0 ====================')
    amfcc.audio_process([raw_dir, inp_dir])
    vfids.check_raw()
    vfids.video_process(nthreads=opt.nthreads)
    vfids.reduce_dim()
    print('[*] ==================== End of Step 0 ====================')

def step1_lipsyn(opt):
    print('[*] ==================== Start of Step 1 ====================')
    # important parameters
    args = {}
    args['vr']         = opt.vr             # validation set ratio
    args['step_delay'] = opt.step_delay     # step delay for LSTM (10ms per step)
    args['dim_hidden'] = opt.dim_hidden     # dimension of hidden layer and cell state
    args['nlayers']    = opt.nlayers        # number of LSTM layers
    args['keep_prob']  = opt.keep_prob      # dropout keep probability
    args['seq_len']    = opt.seq_len        # sequence length (nframes per sequence)
    args['batch_size'] = opt.batch_size     # batch size (nseq per batch)
    args['nepochs']    = opt.nepochs        # number of epochs
    args['grad_clip']  = opt.grad_clip      # gradient clipping threshold
    args['lr']         = opt.lr             # initial learning rate
    args['dr']         = opt.dr             # learning rate's decay rate
    args['b_savef']    = opt.b_savef        # batch report save frequency
    args['e_savef']    = opt.e_savef        # epoch report save frequency

    args['pass_id']    = opt.pass_id
    args['argspath']   = opt.argspath
    args['showGraph']  = opt.showGraph
    args['preprocess'] = opt.preprocess
    args['outp_norm']  = opt.outp_norm

    a2v_cvter = lipnn.Audio2Video(args=args)
    if opt.train:
        a2v_cvter.train()
        loadstore.plot_loss(args['pass_id'])
    if opt.predict:
        pass_id = opt.pass_id
        inp_id = opt.inp_id

        apath   = '%s/mfcc/%s.npy' % (inp_dir, inp_id)                 # audio mfcc
        mpath   = '%s/mp3/%s.mp3'  % (inp_dir, inp_id)                 # music mp3
        opath   = '%s/ldmk/%s/%s.npy' % (inp_dir, pass_id, inp_id)     # output npy data
        tmppath = '%s/visual/%s/%s.avi' % (inp_dir, pass_id, inp_id)   # temporary visualized mp4
        vpath   = '%s/visual/%s/%s.mp4' % (inp_dir, pass_id, inp_id)   # final visualization
        
        if not os.path.exists(apath):
            amfcc.single_audio_process(mpath, apath)

        ldmks = a2v_cvter.test(apath=apath, opath=opath)
        visual.visual_lipsyn(ldmks, mpath, vpath, tmppath)
        print('Lip fids are synthesized at path: %s' % vpath) 
    print('[*] ==================== End of Step 1 ====================')

def step2_lfacesyn(opt, sq, rsize=300, padw=100):
    print('[*] ==================== Start of Step 2 ====================')
    pass_id = opt.pass_id
    inp_id = opt.inp_id
    tar_id = opt.tar_id

    ipath       = '%s/ldmk/%s/%s.npy' % (inp_dir, pass_id, inp_id)
    mpath       = '%s/mp3/%s.mp3' % (inp_dir, inp_id)
    tpath       = '%s/mp4/%s.mp4' % (tar_dir, tar_id)
    tprocpath   = '%s/preproc/%s.npz' % (tar_dir, tar_id)
    opath       = '%s/%s-%s/lface.npy' % (outp_dir, inp_id, tar_id)
    tmppath     = '%s/%s-%s/lface.avi' % (outp_dir, inp_id, tar_id)    # temporary visualized mp4
    vpath       = '%s/%s-%s/lface.mp4' % (outp_dir, inp_id, tar_id)    # final visualization
    
    txtrs = mouth.lowerface(sq, ipath, tpath, tprocpath, opath, line={'upper':opt.lineU, 'lower':opt.lineL}, rsize=rsize, padw=padw)
    visual.visual_lfacesyn(txtrs, mpath, vpath, tmppath)
    print('Lower face synthesized at path %s' % vpath)
    print('[*] ==================== End of Step 2 ====================')
    
def step3_retiming(opt):
    print('[*] ==================== Start of Step 3 ====================')
    inp_id = opt.inp_id
    tar_id = opt.tar_id

    mpath   = '%s/mp3/%s.mp3' % (inp_dir, inp_id)
    tpath   = '%s/mp4/%s.mp4' % (tar_dir, tar_id)
    orpath  = '%s/%s-%s/tgt_original.mp4' % (outp_dir, inp_id, tar_id)
    rtpath  = '%s/%s-%s/tgt_retiming.mp4' % (outp_dir, inp_id, tar_id)
    opath   = '%s/%s-%s/retiming.npy' % (outp_dir, inp_id, tar_id)
    
    L = retiming.timing_opt(mpath, tpath, opath)
    visual.visual_retiming(L, tpath, orpath, rtpath)
    print('Retiming frame-mapping saved at path %s' % opath)
    print('Retiming target videos saved at path %s and %s' % (orpath, rtpath))
    print('[*] ==================== End of Step 3 ====================')

def step4_composite(opt, sq, padw=100, debug=False):
    print('[*] ==================== Start of Step 4 ====================')
    inp_id = opt.inp_id
    tar_id = opt.tar_id
    retiming = opt.retiming

    ipath       = '%s/%s-%s/lface.npy' % (outp_dir, inp_id, tar_id)
    mpath       = '%s/mp3/%s.mp3' % (inp_dir, inp_id)
    tpath       = '%s/%s-%s/tgt_%s.mp4' % (outp_dir, inp_id, tar_id, 'retiming' if retiming else 'original')
    tmppath     = '%s/%s-%s/%s-%s_%s.avi' % (outp_dir, inp_id, tar_id, inp_id, tar_id, 'retiming' if retiming else 'original')
    vpath       = '%s/%s-%s/%s-%s_%s.mp4' % (outp_dir, inp_id, tar_id, inp_id, tar_id, 'retiming' if retiming else 'original')
        
    visual.visual_composite(sq, padw, mpath, ipath, tpath, vpath, tmppath, debug)
    print('Final video synthesized at path %s' % vpath)
    print('[*] ==================== End of Step 4 ====================')

if __name__ == '__main__':
    # python run.py --steps 0 1 2 3 4 --pass_id std_u --inp_id a015 --tar_id t187 --lineU 369 --lineL 369
    sq = a2v.Square(0.25, 0.75, 0.5, 1.1)
    
    opt = ret_opt()
    print(opt)

    if 0 in opt.steps:
        step0_dataset(opt)
    if 1 in opt.steps:
        step1_lipsyn(opt)
    if 2 in opt.steps:
        step2_lfacesyn(opt, sq)
    if 3 in opt.steps:
        step3_retiming(opt)
    if 4 in opt.steps:
        step4_composite(opt, sq)
