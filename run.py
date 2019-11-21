import audio2video as a2v
from audio2video import amfcc, vfids, lipnn, loadstore, visual, mouth

raw_dir = a2v.raw_dir
inp_dir = a2v.inp_dir
tmp_dir = a2v.tmp_dir
tar_dir = a2v.tar_dir
outp_dir= a2v.outp_dir

def step0_dataset():
    amfcc.audio_process([raw_dir, inp_dir])
    vfids.video_process(nthreads=10)
    vfids.reduce_dim()

def step1_lipsyn(pass_id, train, predict, inp_id, outp_norm=False, preprocess=False,
                 vr=0.2, step_delay=20, dim_hidden=60, nlayers=1, keep_prob=0.8,
                 seq_len=100, batch_size=100, nepochs=300, grad_clip=10,
                 lr=1e-3, dr=0.99, b_savef=50, e_savef=5, argspath=None, showGraph=False):
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

    apath   = '%s/mfcc/a%s.npy' % (inp_dir, inp_id)               # audio mfcc
    mpath   = '%s/mp3/a%s.mp3'  % (inp_dir, inp_id)               # music mp3
    opath   = '%s/ldmk/%s/a%s.npy' % (inp_dir, pass_id, inp_id)   # output npy data
    tmppath = '%s/vis/%s/a%s.avi' % (inp_dir, pass_id, inp_id)    # temporary visualized mp4
    vpath   = '%s/vis/%s/a%s.mp4' % (inp_dir, pass_id, inp_id)    # final visualization

    a2v_cvter = lipnn.Audio2Video(args=args)
    if train:
        a2v_cvter.train()
        loadstore.plot_loss(args['pass_id'])
    if predict:
        ldmks = a2v_cvter.test(apath=apath, opath=opath)
        visual.visual_lipsyn(ldmks, mpath, vpath, tmppath)
        print('Lip fids are synthesized at path: %s' % vpath) 

def step2_lfacesyn(pass_id, inp_id, tar_id, sq, rsize=300, preproc=False):
    ipath       = '%s/ldmk/%s/a%s.npy' % (inp_dir, pass_id, inp_id)
    mpath       = '%s/mp3/a%s.mp3' % (inp_dir, inp_id)
    tpath       = '%s/mp4/t%s.mp4' % (tar_dir, tar_id)
    tprocpath   = '%s/preproc/t%s.npz' % (tar_dir, tar_id)
    opath       = '%s/a%st%s/lface.npy' % (outp_dir, inp_id, tar_id)
    tmppath     = '%s/a%st%s/lface.avi' % (outp_dir, inp_id, tar_id)    # temporary visualized mp4
    vpath       = '%s/a%st%s/lface.mp4' % (outp_dir, inp_id, tar_id)    # final visualization
    txtrs = mouth.lowerface(sq, ipath, tpath, tprocpath, opath, rsize=rsize, preproc=preproc)
    visual.visual_lfacesyn(txtrs, mpath, vpath, tmppath)
    print('Lower face synthesized at path %s' % vpath)

if __name__ == '__main__':
    pass_id = "std_u"
    inp_id  = "036"
    tar_id  = "001"
    sq = a2v.Square(0.2, 0.8, 0.4, 1.00)
    # step0_dataset()
    # step1_lipsyn(pass_id=pass_id, train=False, predict=True, inp_id=inp_id, outp_norm=False)
    step2_lfacesyn(pass_id, inp_id, tar_id, sq)