# coding: utf-8

import os
import mne
from scipy import fftpack
import numpy as np

filedir = 'D:/BeidaShuju/rawdata/ZYF'


def para_setting(train=True, filedir=filedir):
    if train:
        fname_list = list(os.path.join(
            filedir, 'MultiTraining_%d_raw_tsss.fif' % j)
            for j in range(1, 6))
        ortids = [2, 6, 9, 14, 17, 33]
        event_ids = dict(ort015=2,  ort045=6,  ort075=9,
                         ort105=14, ort135=17, ort165=33)
        tmin, t0, tmax = -0.2, 0, 0.8
    else:
        fname_list = list(os.path.join(
            filedir, 'MultiTest_%d_raw_tsss.fif' % j)
            for j in range(1, 9))
        ortids = [8, 16, 32, 64]
        event_ids = dict(ort45a=8, ort135a=16,
                         ort45b=32, ort135b=64)
        tmin, t0, tmax = -0.4, -0.2, 0.8
    return fname_list, ortids, event_ids, tmin, t0, tmax


def analysis_hilbert(x):
    # Calculate hilbert analytical signal
    y = fftpack.hilbert(x)
    return np.sqrt(x**2 + y**2)


def cal_envlop(data, picks):
    # Calculate envlop of high freq signal
    # Using hilbert analytical signal
    # picks: only transform signals, not markers
    for j in picks:
        # Calculate envlop of each sensor
        data[j] = analysis_hilbert(data[j])
    return data


prefix_default = 'MEG'
suffix_default = ['1', '2', '3']
sensor_idx_default = [172, 164, 163, 184, 183, 224, 223, 244, 243, 252,
                      171, 173, 194, 191, 201, 202, 231, 232, 251, 253,
                      174, 193, 192, 204, 203, 234, 233, 254,
                      214, 212, 211, 213, 153, 263]


def good_sensors(ch_names,
                 sensor_idx=sensor_idx_default,
                 prefix=prefix_default,
                 suffix=suffix_default):
    sensors = []
    for i in sensor_idx:
        for s in suffix:
            sensors.append(prefix+str(i)+s)
    picks = []
    for e in sensors:
        picks.append(ch_names.index(e))
    return sensors, picks


def get_epochs(fname, event_id,
               tmin, t0, tmax,
               freq_l=1, freq_h=10,
               use_good_sensors=True,
               get_envlop=False):
    # Make defaults
    baseline = (tmin, t0)
    reject = dict(mag=5e-12, grad=4000e-13)
    decim = 1

    # Prepare rawobject
    raw = mne.io.read_raw_fif(fname, preload=True)
    raw.filter(freq_l, freq_h, fir_design='firwin')
    picks = mne.pick_types(raw.info, meg=True, eeg=False,
                           eog=False, stim=False, exclude='bads')

    if use_good_sensors:
        sensors, picks = good_sensors(raw.ch_names)

    if get_envlop:
        raw = mne.io.RawArray(cal_envlop(raw.get_data(), picks), raw.info)
    else:
        raw = mne.io.RawArray(raw.get_data(), raw.info)

    events = mne.find_events(raw)

    # Get epochs
    epochs = mne.Epochs(raw, event_id=event_id, events=events,
                        decim=decim, tmin=tmin, tmax=tmax,
                        picks=picks, baseline=baseline,
                        reject=reject, preload=True)
    return epochs, raw
