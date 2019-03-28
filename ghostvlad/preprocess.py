from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import numpy as np
import librosa

import toolkits
import utils as ut

import pdb
# ===========================================
#        Parse the argument
# ===========================================
import argparse
parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--resume', default=r'pretrained/weights.h5', type=str)
parser.add_argument('--data_path', default='4persons', type=str)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=8, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
parser.add_argument('--test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)

global args
args = parser.parse_args()

def similar(matrix):  # calc speaker-embeddings similarity in pretty format output.
    ids = matrix.shape[0]
    for i in range(ids):
        for j in range(ids):
            dist = matrix[i,:]*matrix[j,:]
            dist = np.linalg.norm(matrix[i,:] - matrix[j,:])
            print('%.2f  ' % dist, end='')
            if((j+1)%3==0 and j!=0):
                print("| ", end='')
        if((i+1)%3==0 and i!=0):
            print('\n')
            print("*"*80, end='')
        print("\n")

# ===============================================
#       code from Arsha for loading data.
# ===============================================
def load_wav(vid_path, sr):
    wav, sr_ret = librosa.load(vid_path, sr=sr)
    assert sr_ret == sr

    intervals = librosa.effects.split(wav, top_db=20)
    wav_output = []
    for sliced in intervals:
      wav_output.extend(wav[sliced[0]:sliced[1]])
    wav_output = np.array(wav_output)
    return wav_output

def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
    return linear.T

def load_data(path, split=False, win_length=400, sr=16000, hop_length=160, n_fft=512, min_slice=720):
    wav = load_wav(path, sr=sr)
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    spec_mag = mag_T

    utterances_spec = []

    if(split):
        minSpec = min_slice//(1000//(sr//hop_length)) # The minimum timestep of each slice in spectrum
        randStarts = np.random.randint(0,time, 10)   # generate 10 slices at most.
        for start in randStarts:
            if(time-start<=minSpec):
                continue
            randDuration = np.random.randint(minSpec, time-start)
            spec_mag = mag_T[:, start:start+randDuration]

            # preprocessing, subtract mean, divided by time-wise var
            mu = np.mean(spec_mag, 0, keepdims=True)
            std = np.std(spec_mag, 0, keepdims=True)
            spec_mag = (spec_mag - mu) / (std + 1e-5)
            utterances_spec.append(spec_mag)

    else:
        # preprocessing, subtract mean, divided by time-wise var
        mu = np.mean(spec_mag, 0, keepdims=True)
        std = np.std(spec_mag, 0, keepdims=True)
        spec_mag = (spec_mag - mu) / (std + 1e-5)
        utterances_spec.append(spec_mag)

    return utterances_spec

def main():

    # gpu configuration
    toolkits.initialize_GPU(args)

    import model
    # ==================================
    #       Get Train/Val.
    # ==================================
    
    total_list = [os.path.join(args.data_path, file) for file in os.listdir(args.data_path)]
    unique_list = np.unique(total_list)

    # ==================================
    #       Get Model
    # ==================================
    # construct the data generator.
    params = {'dim': (257, None, 1),
              'nfft': 512,
              'min_slice': 720,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True,
              }

    network_eval = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                num_class=params['n_classes'],
                                                mode='eval', args=args)

    # ==> load pre-trained model ???
    if args.resume:
        # ==> get real_model from arguments input,
        # load the model if the imag_model == real_model.
        if os.path.isfile(args.resume):
            network_eval.load_weights(os.path.join(args.resume), by_name=True)
            print('==> successfully loading model {}.'.format(args.resume))
        else:
            raise IOError("==> no checkpoint found at '{}'".format(args.resume))
    else:
        raise IOError('==> please type in the model to load')

    print('==> start testing.')

    # The feature extraction process has to be done sample-by-sample,
    # because each sample is of different lengths.

    train_cluster_id = []
    train_sequence = []
    SRC_PATH = r'/data/dataset/SpkWav120'

    wavDir = os.listdir(SRC_PATH)
    wavDir.sort()
    for i,spkDir in enumerate(wavDir):   # Each speaker's directory
        spk = spkDir    # speaker name
        wavPath = os.path.join(SRC_PATH, spkDir, 'audio')
        print('Processing speaker({}) : {}'.format(i, spk))

        for wav in os.listdir(wavPath): # wavfile

            utter_path = os.path.join(wavPath, wav)
            feats = []
            specs = load_data(utter_path, split=True, win_length=params['win_length'], sr=params['sampling_rate'],
                                 hop_length=params['hop_length'], n_fft=params['nfft'],
                                 min_slice=params['min_slice'])
            if(len(specs)<1):
                continue
            for spec in specs:
                spec = np.expand_dims(np.expand_dims(spec, 0), -1)
                v = network_eval.predict(spec)
                feats += [v]

            feats = np.array(feats)[:,0,:]  # [splits, embedding dim]

            train_cluster_id.append([spk]*feats.shape[0])
            train_sequence.append(feats)

    np.savez('training_data', train_sequence=train_sequence, train_cluster_id=train_cluster_id)


if __name__ == "__main__":
    main()
