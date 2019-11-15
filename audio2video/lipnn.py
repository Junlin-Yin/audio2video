# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
import numpy as np
import glob
import math
import time
import os

from loadstore import load_data, restore_state, next_batch, report_batch, report_epoch
from loadstore import save_args, load_stat, cvt2vfps
from __init__ import trn_dir, log_dir

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

class Audio2Video:
    def __init__(self, args):
        self.args = args
        argspath = args['argspath']
        if argspath is not None and os.path.exists(argspath):
            data = pd.read_csv(argspath, delimiter='\t')
            for key, value in zip(data['key'], data['value']):
                self.args[key] = value
                
        self.pass_id = args['pass_id']
        self.init_data(args['preprocess'], args['outp_norm'])
                
    def init_data(self, preprocess=False, outp_norm=False):
        '''Load data for training and do some more initialization
        '''
        self.inps, self.outps = load_data(pass_id      = self.pass_id,
                                          args         = self.args,
                                          preprocess   = preprocess,
                                          outp_norm    = outp_norm)
        # In this network, self.dimin = 28, self.dimout = 20
        self.dimin, self.dimout = self.inps['training'][0].shape[1], self.outps['training'][0].shape[1]
        
        # initialize batch information
        self.nbatches, self.batch_pt = {}, {}
        for key in self.inps.keys():
            # count the total number of sequences we can get from the data
            nseq = sum([math.ceil(inp.shape[0] / self.args['seq_len']) for inp in self.inps[key]])
            # every batch_size sequences consist of one batch
            self.nbatches[key] = nseq // self.args['batch_size']
            # batch pointers
            self.batch_pt[key] = 0
        self.total_batches = self.nbatches['training'] * self.args['nepochs']
        
    def LSTM_model(self, predict=False):
        '''Construct LSTM network for lip synthesizing
        ### Parameters
        mode          training or validation \\
        '''
        # check predict
        if not predict:
            seq_len    = self.args['seq_len']
            batch_size = self.args['batch_size']
        else: # predict
            seq_len    = 1  # for we don't know how long the series is
            batch_size = 1  # for we only predict one serial for one time
        
        # add this statement to avoid restart-error and duplicate graphs in tensorboard
        tf.compat.v1.reset_default_graph()
        
        # prepare placeholders
        with tf.name_scope('inputs'):
            self.input_data = tf.compat.v1.placeholder(tf.float32, [None, seq_len, self.dimin], name='audio')
            self.output_data= tf.compat.v1.placeholder(tf.float32, [None, seq_len, self.dimout], name='video')
        
        # add dropout wrapper & define multilayer network
        network = multi_LSTM(self.args['dim_hidden'], self.args['nlayers'], self.args['keep_prob'], predict) 
        
        # init_state.shape = (nlayers, 2, batch_size, dimhidden)
        with tf.name_scope('init_state'):
            self.init_state = network.zero_state(batch_size, tf.float32)    

        inputs = tf.split(self.input_data, seq_len, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
        hiddens, state = tf.contrib.legacy_seq2seq.rnn_decoder(inputs, self.init_state, network, loop_function=None)     
        self.final_state = state

        # add final weight matrix and final bias vector as said in the paper
        with tf.compat.v1.variable_scope('final_wb'):
            final_w = tf.compat.v1.get_variable('w', [self.args['dim_hidden'], self.dimout])
            final_b = tf.compat.v1.get_variable('b', [self.dimout])        
        
        with tf.name_scope('outputs'):
            # tmp.shape = [batch_size, dim_hidden*seq_len]
            tmp = tf.concat(hiddens, axis=1)
            # hidden_flat.shape = [batch_size*seq_len, dim_hidden]
            hidden_flat = tf.reshape(tmp, [-1, self.args['dim_hidden']])
            # output_hat_flat.shape = [batch_size*seq_len, dimout]
            output_hat_flat = tf.compat.v1.nn.xw_plus_b(hidden_flat, final_w, final_b, name='output_hat')
            
            # self.output.shape = [batch_size, seq_len, dimout]
            self.output = tf.reshape(output_hat_flat, [-1, seq_len, self.dimout])
        
        # define loss function (L2-norm error)
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.math.squared_difference(self.output, self.output_data))
      
        # deal with gradient and optimization
        with tf.name_scope('train'):
            self.lr = tf.Variable(0., trainable=False)
            tvars = tf.compat.v1.trainable_variables()            # all trainable variables in this model
            grads = tf.gradients(self.loss, tvars)      # partial{loss}/partial{tvars}
            # clip gradients to avoid gradient explosion
            grads, _ = tf.clip_by_global_norm(grads, self.args['grad_clip'])
            # optimizer
            optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        
    def train(self):
        feed_dict = {}
        init_lr, dr = self.args['lr'], self.args['dr']
        
        save_args(self.pass_id, self.args)
        
        self.LSTM_model(predict=False)
        
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            
            # restore model states
            startEpoch, saver = restore_state(sess, self.pass_id)
            
            if self.args['showGraph']:
                # clear log dir
                logs = glob.glob('%s/*.log' % log_dir)
                for log in logs:
                    os.remove('%s/%s' % (log_dir, log))
                    
                # create tensorboard file
                writer = tf.summary.FileWriter(log_dir, sess.graph)
                writer.close()
                
            endEpoch = self.args['nepochs']
            for e in range(startEpoch, endEpoch):
                # in each epoch (totally <nbatches> batches per epoch)
                                
                # define learning rate of this epoch
                sess.run(tf.assign(self.lr, init_lr * (dr)**e))
                
                # feed initial cell state and hidden layer of every batch
                # when training each batch, always feed the initial state
                # because there's no relationships between two batches (just clear the memory)
                state_val = [(c.eval(), v.eval()) for (c, v) in self.init_state]
                for i, (c, v) in enumerate(self.init_state):
                    feed_dict[c], feed_dict[v] = state_val[i]
                    
                # reset batch pointers
                self.batch_pt['training'] = self.batch_pt['validation'] = 0
                
                trainLoss = 0
                    
                # define fetch list
                fetches = [self.loss, self.output, self.train_op]          
                for b in range(self.nbatches['training']):
                    begin = time.time()
                    # in each batch (totally <nseq> sequences per epoch)
                    x, y = next_batch(inps       = self.inps, 
                                      outps      = self.outps, 
                                      mode       = 'training', 
                                      batch_pt   = self.batch_pt, 
                                      nbatches   = self.nbatches, 
                                      args       = self.args)
                    feed_dict[self.input_data], feed_dict[self.output_data] = x, y
                    
                    # do training in this step, get the loss
                    tloss, y_hat, _ = sess.run(fetches, feed_dict)
                    trainLoss += tloss
                    report_batch(self.pass_id, e, b, self.args['nepochs'], self.nbatches['training'], time.time()-begin, self.args['b_savef'], tloss)
                    
                # at the end of each epoch, do validation
                trainLoss /= self.nbatches['training']       # final training loss at each epoch
                validLoss = 0
             
                # define fetch list 
                # note that self.train_op is thrown away because we don't optimize
                # the model during validation
                fetches = [self.loss]
                for b in range(self.nbatches['validation']):
                    x, y = next_batch(inps       = self.inps, 
                                      outps      = self.outps, 
                                      mode       = 'validation', 
                                      batch_pt   = self.batch_pt, 
                                      nbatches   = self.nbatches, 
                                      args       = self.args)
                    feed_dict[self.input_data], feed_dict[self.output_data] = x, y                  
                    vloss, = sess.run(fetches, feed_dict)
                    validLoss += vloss
                validLoss /= self.nbatches['validation']
                report_epoch(self.pass_id, sess, saver, e, self.args['nepochs'], self.args['e_savef'], trainLoss, validLoss)
        
    def test(self, apath, opath):
        # get audio features
        afeatures = np.load(apath)
        audio           = afeatures[ :, :-1]                        # (N, 14)
        audio_diff      = afeatures[1:, :-1] - afeatures[:-1, :-1]  # (N-1, 14)
        audio_timestamp = afeatures[ :,  -1]                        # (N-1, )
        inp             = np.hstack((audio[:-1, :], audio_diff))      # (N-1, 28)
        
        # normalization using pretrained statistics
        imean, istd, omean, ostd = load_stat(self.pass_id)
        inp  = (inp - imean) / istd
        
        step_delay = self.args['step_delay']
        outp_list = [0]*(inp.shape[0]-step_delay)
        
        # construct predict network
        self.LSTM_model(predict=True)
        with tf.compat.v1.Session() as sess:
            # load trainable variables
            saver = tf.compat.v1.train.Saver()
            ckpt = tf.train.get_checkpoint_state('%s/%s' % (trn_dir, self.pass_id))
            saver.restore(sess, ckpt.model_checkpoint_path)
            
            # state: [((batch_size, dimhidden), (batch_size, dimhidden))]*nlayers
            #          -----------c-----------  -----------v-----------
            state = [(c.eval(), v.eval()) for (c, v) in self.init_state]
            
            # fetch state and output
            fetches = [self.output, self.final_state]
            feed_dict = {}
            for i in range(inp.shape[0]):
                # feed input and state
                feed_dict[self.input_data] = np.array([[inp[i]]])    # [1, 1, dimin]
                
                for l, (c, v) in enumerate(self.init_state):
                    # i: layer id
                    feed_dict[c], feed_dict[v] = state[l]
                            
                # pass the network
                # output: (1, 1, dimout)
                # newstate: (nlayers, 2, batch_size, dimhidden)
                output, newstate = sess.run(fetches, feed_dict)
                
                # reshape newstate and output
                state = [(c, v) for (c, v) in newstate]
                output = np.reshape(output, (self.dimout,))
                
                # further precess to output
                outp_list[max(0, i-step_delay)] = output
        
        outp = np.vstack(outp_list)     # outp.shape = (N, 20)
        outp = outp * ostd + omean
        ldmks = cvt2vfps(outp, audio_timestamp, step_delay)
        np.save(opath, ldmks)
        print('Predict results saved into path: %s' % opath)
        return ldmks

def multi_LSTM(dim_hidden, nlayers, kp, predict=False):                      
    if not predict and kp < 1:
        cell_list = []
        for i in range(nlayers):
            cell = tf.nn.rnn_cell.LSTMCell(dim_hidden)
            ikp, okp = kp, (1.0 if i < nlayers-1 else kp)
            cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=ikp, output_keep_prob=okp)
            cell_list.append(cell)
    else:
        cell_list = [tf.nn.rnn_cell.LSTMCell(dim_hidden) for i in range(nlayers)]
    
    network = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cell_list)
    return network
                    
if __name__ == "__main__":
    print('Hello, World')
    