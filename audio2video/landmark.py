import numpy as np 
import cv2
import dlib
from .__init__ import detector, predictor, fronter
from .facefrontal import facefrontal

def landmark(mp4_path, args):
    '''
    ### parameters
    mp4_path: path of mp4 file \\
    args: (n, 2) fragmentation arguments. The 1st column is startfr # and the 2nd column is nfr. 
    Totally n fragments

    ### retval
    feature_list: a list whose length is # of fragments. Each element of the list is features data 
    in ndarray format of each fragment with the shape of (nfr, 40)
    new_args: (n, 2) corrected fragmentation arguments. Some frames in the beginning or at the end 
    of one fragment cannot used to detect a face, so we drop them out.
    '''
    feature_list = []
    new_args = np.copy(args)
    frag_id = 0

    cap = cv2.VideoCapture(mp4_path)
    for startfr, nfr in args:
        # initialization
        cnt = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, startfr)
        fragment_feature = None

        while cap.isOpened() and cnt < nfr:
            print(mp4_path[-15:-4],'#:', startfr + cnt)
            # fetch the frame
            ret, img_ = cap.read()
            # cv2.imshow('img_', img_)
            # cv2.waitKey(0)
            
            # frontalization
            img = facefrontal(img_, detector, predictor, fronter)    # img.shape = None or (320, 320, 3)
            
            # exception handling
            if img is None:
                if cnt < nfr/2:     # beginning
                    new_args[frag_id, 0] += 1
                    new_args[frag_id, 1] -= 1
                    cnt += 1
                    continue
                else:               # ending
                    new_args[frag_id, 1] = startfr + cnt - new_args[frag_id, 0]
                    break

            # lip landmark extraction
            dets = detector(img, 1)
            if len(dets) == 0:
                if cnt < nfr/2:     # beginning
                    new_args[frag_id, 0] += 1
                    new_args[frag_id, 1] -= 1
                    cnt += 1
                    continue
                else:               # ending
                    new_args[frag_id, 1] = startfr + cnt - new_args[frag_id, 0]
                    break

            det = dets[0]
            shape = predictor(img, det)
            landmarks = np.asarray([(shape.part(n).x, shape.part(n).y) for n in range(48, shape.num_parts)], np.float32)
        
            # normalization according to det.shape & reshape into 40-D features
            origin = np.array([det.left(), det.top()])
            size = np.array([det.width(), det.height()])
            landmarks = (landmarks - origin) / size         # restrained in [0, 0] ~ [1, 1]
            frame_feature = landmarks.reshape((landmarks.shape[0]*landmarks.shape[1]))  # (40, )

            # add frame_feature to fragment_feature
            if fragment_feature is None:
                fragment_feature = np.copy(frame_feature)
            else:
                fragment_feature = np.vstack((fragment_feature, frame_feature))

            # plot the landmarks
            # plt.figure(figsize=(10, 10))
            # plt.xlim(0, 1)
            # plt.ylim(0, 1)
            # plt.scatter(landmarks[:,0], 1-landmarks[:,1])
            # plt.show()

            cnt += 1

        # fragment_feature.shape = (fr?, 40)
        feature_list.append(fragment_feature)
        frag_id += 1
    
    # length of feature_list = args.shape[0]
    return feature_list, new_args

if __name__ == "__main__":
    print('Hello, World')