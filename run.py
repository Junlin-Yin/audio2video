import audio2video as a2v
from audio2video import amfcc, vfids
from audio2video import lipnn, loadstore
from audio2video import retiming, mouth
from audio2video import visual

raw_dir = a2v.raw_dir
inp_dir = a2v.inp_dir
tmp_dir = a2v.tmp_dir
tar_dir = a2v.tar_dir
outp_dir= a2v.outp_dir

def step0_dataset():
    amfcc.audio_process([raw_dir, inp_dir])
    vfids.check_raw()
    vfids.video_process(nthreads=10)
    vfids.reduce_dim()

def step1_lipsyn(pass_id, train, predict, inp_id=None, outp_norm=False, preprocess=False,
                 vr=0.2, step_delay=20, dim_hidden=60, nlayers=1, keep_prob=0.8,
                 seq_len=100, batch_size=100, nepochs=300, grad_clip=10,
                 lr=1e-3, dr=0.99, b_savef=50, e_savef=5, argspath=None, showGraph=True):
    # important parameters
    args = {}
    args['vr']         = vr             # validation set ratio
    args['step_delay'] = step_delay     # step delay for LSTM (10ms per step)
    args['dim_hidden'] = dim_hidden     # dimension of hidden layer and cell state
    args['nlayers']    = nlayers        # number of LSTM layers
    args['keep_prob']  = keep_prob      # dropout keep probability
    args['seq_len']    = seq_len        # sequence length (nframes per sequence)
    args['batch_size'] = batch_size     # batch size (nseq per batch)
    args['nepochs']    = nepochs        # number of epochs
    args['grad_clip']  = grad_clip      # gradient clipping threshold
    args['lr']         = lr             # initial learning rate
    args['dr']         = dr             # learning rate's decay rate
    args['b_savef']    = b_savef        # batch report save frequency
    args['e_savef']    = e_savef        # epoch report save frequency

    args['pass_id']    = pass_id
    args['argspath']   = argspath
    args['showGraph']  = showGraph
    args['preprocess'] = preprocess
    args['outp_norm']  = outp_norm

    a2v_cvter = lipnn.Audio2Video(args=args)
    if train:
        a2v_cvter.train()
        loadstore.plot_loss(args['pass_id'])
    if predict:
        apath   = '%s/mfcc/a%s.npy' % (inp_dir, inp_id)                 # audio mfcc
        mpath   = '%s/mp3/a%s.mp3'  % (inp_dir, inp_id)                 # music mp3
        opath   = '%s/ldmk/%s/a%s.npy' % (inp_dir, pass_id, inp_id)     # output npy data
        tmppath = '%s/visual/%s/a%s.avi' % (inp_dir, pass_id, inp_id)   # temporary visualized mp4
        vpath   = '%s/visual/%s/a%s.mp4' % (inp_dir, pass_id, inp_id)   # final visualization
        
        ldmks = a2v_cvter.test(apath=apath, opath=opath)
        visual.visual_lipsyn(ldmks, mpath, vpath, tmppath)
        print('Lip fids are synthesized at path: %s' % vpath) 

def step2_lfacesyn(pass_id, inp_id, tar_id, sq, rsize=300, padw=100, preproc=False):
    ipath       = '%s/ldmk/%s/a%s.npy' % (inp_dir, pass_id, inp_id)
    mpath       = '%s/mp3/a%s.mp3' % (inp_dir, inp_id)
    tpath       = '%s/mp4/t%s.mp4' % (tar_dir, tar_id)
    tprocpath   = '%s/preproc/t%s.npz' % (tar_dir, tar_id)
    opath       = '%s/a%st%s/lface.npy' % (outp_dir, inp_id, tar_id)
    tmppath     = '%s/a%st%s/lface.avi' % (outp_dir, inp_id, tar_id)    # temporary visualized mp4
    vpath       = '%s/a%st%s/lface.mp4' % (outp_dir, inp_id, tar_id)    # final visualization
    
    txtrs = mouth.lowerface(sq, ipath, tpath, tprocpath, opath, rsize=rsize, padw=padw, preproc=preproc)
    visual.visual_lfacesyn(txtrs, mpath, vpath, tmppath)
    print('Lower face synthesized at path %s' % vpath)
    
def step3_retiming(inp_id, tar_id):
    mpath   = '%s/mp3/a%s.mp3' % (inp_dir, inp_id)
    tpath   = '%s/mp4/t%s.mp4' % (tar_dir, tar_id)
    orpath  = '%s/a%st%s/tgt_original.mp4' % (outp_dir, inp_id, tar_id)
    rtpath  = '%s/a%st%s/tgt_retiming.mp4' % (outp_dir, inp_id, tar_id)
    opath   = '%s/a%st%s/retiming.npy' % (outp_dir, inp_id, tar_id)
    
    L = retiming.timing_opt(mpath, tpath, opath)
    visual.visual_retiming(L, tpath, orpath, rtpath)
    print('Retiming frame-mapping saved at path %s' % opath)
    print('Retiming target videos saved at path %s and %s' % (orpath, rtpath))

def step4_composite(inp_id, tar_id, sq, padw=100, retiming=True, debug=False):
    ipath       = '%s/a%st%s/lface.npy' % (outp_dir, inp_id, tar_id)
    mpath       = '%s/mp3/a%s.mp3' % (inp_dir, inp_id)
    tpath       = '%s/a%st%s/tgt_%s.mp4' % (outp_dir, inp_id, tar_id, 'retiming' if retiming else 'original')
    tmppath     = '%s/a%st%s/a%st%s_%s.avi' % (outp_dir, inp_id, tar_id, inp_id, tar_id, 'retiming' if retiming else 'original')
    vpath       = '%s/a%st%s/a%st%s_%s.mp4' % (outp_dir, inp_id, tar_id, inp_id, tar_id, 'retiming' if retiming else 'original')
        
    visual.visual_composite(sq, padw, mpath, ipath, tpath, vpath, tmppath, debug)
    print('Final video synthesized at path %s' % vpath)

if __name__ == '__main__':
    pass_id = "std_u"
    inp_id  = "015"
    tar_id  = "187"
    sq = a2v.Square(0.25, 0.75, 0.5, 1.1)
    step0_dataset()
    step1_lipsyn(pass_id=pass_id, train=True, predict=True, inp_id=inp_id)
    step2_lfacesyn(pass_id, inp_id, tar_id, sq)
    step3_retiming(inp_id, tar_id)
    step4_composite(inp_id, tar_id, sq)
    print('Done.')