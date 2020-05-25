# audio2video-v1
My Undergraduate Graduation Project, the whole pipeline included, version 1 

Given an input audio *A* and a speaker's target video *V*, this project generates an output video where speaker in *V* is saying contents in *A*.

This project is BASICALLY re-implemented from the paper ["Synthesizing Obama: Learning Lip Sync from Audio"](http://grail.cs.washington.edu/projects/AudioToObama/) by [Supasorn Suwajanakorn](https://homes.cs.washington.edu/~supasorn/) et al. They published the LSTM part of codes but the whole pipeline cannot be found.

Here are some frames in my demo output video.
![avatar](misc/demo-frames.png)

## Architecture & Pipeline
![avatar](misc/architecture.png)
1. Apply MFCC analysis to the input audio, and get the 13-D MFCC coefficients. Combine this 13-D features and log mean energy (volume) as well as their first temporal derivatives to form 28-D vectors as final audio features.
2. An LSTM with delay is used to train and infer the mapping from audio features to sparse mouth shapes which consist of 20 landmark coordinates around speaker's mouth.
3. Given a target video, automatically select candidate frames from it according to similarity to the synthesized sparse mouth shape, and then use pixel-wise weighted median algorithm to generate mouth texture.
4. Manually select proxy frame from target video, and use teeth proxy algorithm to recover details in the teeth region.
5. Re-timing: re-mapping from synthesized mouth & teeth texture sequence to the target frame sequence, in order to make the final speaker's head movement match the input audio and seem natual.
6. For each pair of matched mouth & teeth texture and target frame, transfer and Laplacian pyramid blend synthesized texture onto the target frame to form a final frame. Combine final frame sequence and input audio to get the final speaker's video.

## Dataset
In this project, I use US former president Obama's weekly address as the audio and video dataset. Raw data and preprocessed data can be found in raw/*.

I use preprocessed audio and video data mentioned above to train the delayed LSTM. For demonstration, I choose audio from this speech: ["Improving Economic Security by Strengthening and Modernizing the Unemployment Insurance System"](https://www.youtube.com/watch?v=6jlaKyvf8WA) as input (saved at input/mp3/a015.mp3) and video from this speech: ["Congress Must Move Forward, Not Back On Wall Street Reform"](https://www.youtube.com/watch?v=6qylcsQjLTA) (saved at target/mp4/t187.mp4) as target, to generate the final output video (saved at output/a015t187/final.mp4). Of course you can try other data as inputs and targets, even resources other than Obama's weekly address.

## Directory
```
- audio2video/      source codes of this project
- raw/
    - links.txt     Obama's weekly address links
    - mfcc/         features of raw audios
    - fids/         sparse mouth shapes of raw videos
- input/
    - mp3/          input audios
    - mfcc/         features of input audios
    - ldmk/         synthesized sparse mouth shapes of input audio
    - visual/       visualize data in ldmk/*
- target/
    - mp4/          target videos
    - proxy/        proxy frames selected for teeth enhancement
- reference/        some auxiliary data that will be used in the program
- train/            delayed LSTM training outputs
- log/              set for tensorboard
- output/           temporary and final outputs
```