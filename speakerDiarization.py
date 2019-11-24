
"""A demo script showing how to DIARIZATION ON WAV USING UIS-RNN."""

import os
import time
import sys

import numpy as np
import uisrnn
import librosa

# sys.path.append('ghostvlad')
# sys.path.append('visualization')
# import toolkits
from ghostvlad import toolkits
from ghostvlad import model as spkModel

# from viewer import PlotDiar
from visualization.viewer import PlotDiar

BASE_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
SAVED_MODEL_NAME = os.path.join(BASE_DIR, 'pretrained/saved_model.uisrnn_benchmark')

def append2dict(speakerSlice, spk_period):
    key = list(spk_period.keys())[0]
    value = list(spk_period.values())[0]
    timeDict = {}
    timeDict['start'] = int(value[0]+0.5)
    timeDict['stop'] = int(value[1]+0.5)
    if(key in speakerSlice):
        speakerSlice[key].append(timeDict)
    else:
        speakerSlice[key] = [timeDict]

    return speakerSlice

def arrangeResult(labels, time_spec_rate): # {'1': [{'start':10, 'stop':20}, {'start':30, 'stop':40}], '2': [{'start':90, 'stop':100}]}
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


# 0s        1s        2s                  4s                  6s
# |-------------------|-------------------|-------------------|
# |-------------------|
#           |-------------------|
#                     |-------------------|
#                               |-------------------|
def load_data(path, win_length=400, sr=16000, hop_length=160, n_fft=512, embedding_per_second=0.5, overlap_rate=0.5, network_eval=None):
    wav, intervals = load_wav(path, sr=sr)
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    spec_mag = mag_T

    spec_len = sr/hop_length/embedding_per_second
    spec_hop_len = spec_len*(1-overlap_rate)

    cur_slide = 0.0
    utterances_spec = []
    feats = []

    while(True):  # slide window.
        if(cur_slide + spec_len > time):
            break
        spec_mag = mag_T[:, int(cur_slide+0.5) : int(cur_slide+spec_len+0.5)]
        
        # preprocessing, subtract mean, divided by time-wise var
        mu = np.mean(spec_mag, 0, keepdims=True)
        std = np.std(spec_mag, 0, keepdims=True)
        spec_mag = (spec_mag - mu) / (std + 1e-5)
        utterances_spec.append(spec_mag)

        cur_slide += spec_hop_len
        
        if network_eval is not None:
            spec = np.expand_dims(np.expand_dims(spec_mag, 0), -1)
            v = network_eval.predict(spec)
            feats += [v]

    return utterances_spec, intervals, feats

class Args:
    pass
    
def process(wav_path, embedding_per_second=1.0, overlap_rate=0.5, after_shift=0, output_seg=False, show=False, segment_fn='output.seg', args=None):

    if args is None:
        args = Args()
        args.gpu = ''
        args.resume = os.path.join(BASE_DIR, 'ghostvlad/pretrained/weights.h5')
        args.data_path = '4persons'
        # set up network configuration.
        args.net = 'resnet34s' #, choices=['resnet34s', 'resnet34l'], type=str)
        args.ghost_cluster = 2
        args.vlad_cluster = 8
        args.bottleneck_dim = 512
        args.aggregation_mode = 'gvlad' #, choices=['avg', 'vlad', 'gvlad'], type=str)
        # set up learning rate, training loss and optimizer.
        args.loss = 'softmax' #, choices=['softmax', 'amsoftmax'], type=str)
        args.test_type = 'normal' #, choices=['normal', 'hard', 'extend'], type=str)

    # gpu configuration
    toolkits.initialize_GPU(args)

    params = {
        'dim': (257, None, 1),
        'nfft': 512,
        'spec_len': 250,
        'win_length': 400,
        'hop_length': 160,
        'n_classes': 5994,
        'sampling_rate': 16000,
        'normalize': True,
    }
    t0 = time.time()
    network_eval = spkModel.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                num_class=params['n_classes'],
                                                mode='eval', args=args)
    network_eval.load_weights(args.resume, by_name=True)

    model_args = Args()
    model_args.observation_dim = 512
    model_args.rnn_hidden_size = 512
    model_args.rnn_depth = 1
    model_args.rnn_dropout = 0.2
    model_args.transition_bias = None
    model_args.crp_alpha = 1.0
    model_args.sigma2 = None
    model_args.verbosity = 2
    
    inference_args = Args()
    inference_args.beam_size = 10
    inference_args.look_ahead = 1
    inference_args.test_iteration = 2

    # model_args, _, inference_args = uisrnn.parse_arguments()
    # model_args.observation_dim = 512
    uisrnnModel = uisrnn.UISRNN(model_args)
    uisrnnModel.load(SAVED_MODEL_NAME)
    td = time.time() - t0
    print('Load model time:', td)

    print('Loading data...')
    t0 = time.time()
    # specs, intervals = load_data(wav_path, embedding_per_second=embedding_per_second, overlap_rate=overlap_rate)
    specs, intervals, feats = load_data(wav_path, embedding_per_second=embedding_per_second, overlap_rate=overlap_rate, network_eval=network_eval)
    mapTable, keys = genMap(intervals)
    td = time.time() - t0
    print('Load data time:', td)

    print('Generating feats...')
    t0 = time.time()
    # feats = []
    # for spec in specs:
        # spec = np.expand_dims(np.expand_dims(spec, 0), -1)
        # v = network_eval.predict(spec)
        # feats += [v]
    feats = np.array(feats)[:,0,:].astype(float)  # [splits, embedding dim]
    td = time.time() - t0
    print('Load feat time:', td)

    print('inference_args:', inference_args)
    print('running uisrnn.predict...')
    t0 = time.time()
    predicted_label = uisrnnModel.predict(feats, inference_args)
    td = time.time() - t0
    print('Load uisrnn.predict time:', td)

    t0 = time.time()
    time_spec_rate = 1000*(1.0/embedding_per_second)*(1.0-overlap_rate) # speaker embedding every ?ms
    center_duration = int(1000*(1.0/embedding_per_second)//2)
    speakerSlice = arrangeResult(predicted_label, time_spec_rate)
    td = time.time() - t0
    print('Load arrangeResult time:', td)

    t0 = time.time()
    for spk,timeDicts in speakerSlice.items():    # time map to orgin wav(contains mute)
        for tid,timeDict in enumerate(timeDicts):
            s = 0
            e = 0
            for i,key in enumerate(keys):
                if(s!=0 and e!=0):
                    break
                if(s==0 and key>timeDict['start']):
                    offset = timeDict['start'] - keys[i-1]
                    s = mapTable[keys[i-1]] + offset
                if(e==0 and key>timeDict['stop']):
                    offset = timeDict['stop'] - keys[i-1]
                    e = mapTable[keys[i-1]] + offset

            speakerSlice[spk][tid]['start'] = s
            speakerSlice[spk][tid]['stop'] = e
    print('Load speakerSlicing time:', td)

    audacity_segments = []
    for spk, timeDicts in speakerSlice.items():
        for timeDict in timeDicts:
            s = timeDict['start']
            e = timeDict['stop']
            s = s * 1/1000.
            e = e * 1/1000.
            s += after_shift
            e += after_shift
            audacity_segments.append((s, e, spk))

    if output_seg:
        with open(segment_fn, 'w') as fout:
            for s, e, l in audacity_segments:
                fout.write('%s\t%s\t%s\n' % (round(s, 6), round(e, 6), spk))

    if show:
        p = PlotDiar(map=speakerSlice, wav=wav_path, gui=True, size=(25, 6))
        p.draw()
        p.plot.show()

    return audacity_segments

if __name__ == '__main__':

    # ===========================================
    #        Parse the argument
    # ===========================================
    import argparse
    parser = argparse.ArgumentParser()
    # set up training configuration.
    parser.add_argument('--gpu', default='', type=str)
    parser.add_argument('--resume', default=r'ghostvlad/pretrained/weights.h5', type=str)
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

    args = parser.parse_args()

    # fn = r'wavs/rmdmy.wav'
    # fn = 'wavs/sample1.wav'
    fn = 'wavs/Babylon_Bee_Ep19_nonsub.wav'
    process(fn, embedding_per_second=1.2, overlap_rate=0.4, output_seg=0, show=0, args=args)
