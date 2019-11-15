from audio2video.__init__ import raw_dir, inp_dir, tmp_dir
from audio2video.amfcc import audio_process
from audio2video.vfids import video_process, reduce_dim
from audio2video.lipnn import Audio2Video
from audio2video.loadstore import plot_loss
from audio2video.visual import visual_lipsync

def step0_dataset():
    audio_process([raw_dir, inp_dir])
    video_process(nthreads=10)
    reduce_dim()

def step1_lipsync(pass_id, train, predict, inp_id, outp_norm=False, preprocess=False,
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

    apath = '%s/mfcc/a%s.npy' % (inp_dir, inp_id)
    mpath = '%s/mp3/a%s.mp3'  % (inp_dir, inp_id)
    opath = '%s/ldmk/%s/a%s.npy' % (inp_dir, pass_id, inp_id)
    tpath = '%s/ldmk/%s/a%s#1.mp4' % (inp_dir, pass_id, inp_id)
    vpath = '%s/vis/%s/a%s.mp4'  % (inp_dir, pass_id, inp_id)

    a2v_cvter = Audio2Video(args=args)
    if train:
        a2v_cvter.train()
        plot_loss(args['pass_id'])
    if predict:
        ldmks = a2v_cvter.test(apath=apath, opath=opath)
        visual_lipsync(ldmks, tpath, mpath, vpath)
        print('Final results are successfully saved to path: %s' % vpath) 

if __name__ == '__main__':
    step1_lipsync(pass_id='std_u', train=True, predict=True, inp_id='036', outp_norm=False)
