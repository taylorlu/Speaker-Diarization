
"""A demo script showing how to DIARIZATION ON WAV USING UIS-RNN."""

import numpy as np
import uisrnn
import librosa
import sys
sys.path.append('ghostvlad')
import toolkits
import model as spkModel
import os

# ===========================================
#        Parse the argument
# ===========================================
import argparse
parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--resume', default=r'model/weights.h5', type=str)
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


SAVED_MODEL_NAME = 'pretrained/saved_model.uisrnn_benchmark'

def append2dict(speakerSlice, spk_period):
    key = list(spk_period.keys())[0]
    value = list(spk_period.values())[0]
    if(key in speakerSlice):
        speakerSlice[key].append(value)
    else:
        speakerSlice[key] = [value]

    return speakerSlice

def arrangeResult(labels, time_spec_rate): # {'1': [(10, 20), (30, 40)], '2': [(90, 100)]}
    lastLabel = labels[0]
    speakerSlice = {}
    j = 0
    for i,label in enumerate(labels):
        if(label==lastLabel):
            continue
        speakerSlice = append2dict(speakerSlice, {lastLabel: (time_spec_rate*j,time_spec_rate*i)})
        j = i
        lastLabel = label

    speakerSlice = append2dict(speakerSlice, {lastLabel: (time_spec_rate*j,time_spec_rate*(len(labels)))})
    return speakerSlice

def genMap(intervals):  # interval slices to maptable
    slicelen = [sliced[1]-sliced[0] for sliced in intervals.tolist()]
    mapTable = {}  # vad erased time to origin time, only split points
    idx = 0
    for i, sliced in enumerate(intervals.tolist()):
        mapTable[idx] = sliced[0]
        idx += slicelen[i]
    mapTable[sum(slicelen)] = intervals[-1,-1]

    keys = [k for k,_ in mapTable.items()]
    keys.sort()
    return mapTable, keys

def fmtTime(timeInMillisecond):
    millisecond = timeInMillisecond%1000
    minute = timeInMillisecond//1000//60
    second = (timeInMillisecond-minute*60*1000)//1000
    time = '{}:{:02d}.{}'.format(minute, second, millisecond)
    return time

def load_wav(vid_path, sr):
    wav, _ = librosa.load(vid_path, sr=sr)
    intervals = librosa.effects.split(wav, top_db=20)
    wav_output = []
    for sliced in intervals:
      wav_output.extend(wav[sliced[0]:sliced[1]])
    return np.array(wav_output), (intervals/sr*1000).astype(int)

def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
    return linear.T

def load_data(path, win_length=400, sr=16000, hop_length=160, n_fft=512, embedding_per_second=0.5):
    wav, intervals = load_wav(path, sr=sr)
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    spec_mag = mag_T
    print('freq = {}, time = {}'.format(freq, time))

    spec_len = int(sr//hop_length//embedding_per_second)
    spec_hop_len = spec_len//4

    cur_slide = 0
    utterances_spec = []

    while(True):  # slide window.
        if(cur_slide + spec_len > time):
            break
        spec_mag = mag_T[:, cur_slide : cur_slide+spec_len]
        
        # preprocessing, subtract mean, divided by time-wise var
        mu = np.mean(spec_mag, 0, keepdims=True)
        std = np.std(spec_mag, 0, keepdims=True)
        spec_mag = (spec_mag - mu) / (std + 1e-5)
        utterances_spec.append(spec_mag)

        cur_slide += spec_hop_len

    return utterances_spec, intervals

def similar(matrix):  # calc d-vectors similarity in pretty format output.
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


def main():

    # gpu configuration
    toolkits.initialize_GPU(args)

    params = {'dim': (257, None, 1),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True,
              }

    network_eval = spkModel.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                num_class=params['n_classes'],
                                                mode='eval', args=args)
    network_eval.load_weights(os.path.join(r'ghostvlad/pretrained/weights.h5'), by_name=True)


    model_args, _, inference_args = uisrnn.parse_arguments()
    model_args.observation_dim = 512
    uisrnnModel = uisrnn.UISRNN(model_args)
    uisrnnModel.load(SAVED_MODEL_NAME)


    specs, intervals = load_data(r'E:\PPDownload\rmdmy_diarization.wav')
    # specs, intervals = load_data(r'E:\PPDownload\videoplayback.wav')
    # specs, intervals = load_data(r'D:\PythonSpace\Speaker-Diarization\ghostvlad\4persons\yyy.wav')
    mapTable, keys = genMap(intervals)
    print(mapTable)
    print(keys)

    feats = []
    for spec in specs:
        spec = np.expand_dims(np.expand_dims(spec, 0), -1)
        v = network_eval.predict(spec)
        feats += [v]

    feats = np.array(feats)[:,0,:].astype(float)  # [splits, embedding dim]
    # print(feats.shape)
    # similar(feats)
    predicted_label = uisrnnModel.predict(feats, inference_args)
    print(predicted_label)

    time_spec_rate = 500 # speaker embedding every 500ms
    speakerSlice = arrangeResult(predicted_label, time_spec_rate)

    for spk,periods in speakerSlice.items():
        for tid, period in enumerate(periods):
            s = 0
            e = 0
            for i,key in enumerate(keys):
                if(s!=0 and e!=0):
                    break
                if(s==0 and key>period[0]):
                    offset = period[0] - keys[i-1]
                    s = mapTable[keys[i-1]] + offset
                if(e==0 and key>period[1]):
                    offset = period[1] - keys[i-1]
                    e = mapTable[keys[i-1]] + offset

            speakerSlice[spk][tid] = (s,e)

    print(speakerSlice)

    for spk,periods in speakerSlice.items():
        print('========= ' + str(spk) + ' =========')
        for tid, period in enumerate(periods):
            s, e = speakerSlice[spk][tid]
            s = fmtTime(s+1000)
            e = fmtTime(e+1000)
            print(s+' ==> '+e)

    # data_path = r'D:\PythonSpace\Speaker-Diarization\ghostvlad\4persons'
    # total_list = [os.path.join(data_path, file) for file in os.listdir(data_path)]
    # unique_list = np.unique(total_list)
    # feats = []
    # for ID in unique_list:
    #     specs = load_data(ID)[0]
    #     specs = np.expand_dims(np.expand_dims(specs, 0), -1)
    
    #     v = network_eval.predict(specs)
    #     feats += [v]
    
    # feats = np.array(feats)[:,0,:]
    # similar(feats)

if __name__ == '__main__':
  main()
