from audio2video.__init__ import raw_dir, inp_dir
from audio2video.audio_preprocess import audio_process
from audio2video.video_preprocess import video_process, reduce_dim

def step1():
    # audio_process([raw_dir, inp_dir])
    video_process(nthreads=10)
    reduce_dim()

if __name__ == '__main__':
    step1()