
# coding: utf-8

import os
import mne
import numpy as np
from pick_good_sensors import good_sensors
from scipy import fftpack

smooth_kernel = 1/200+np.array(range(200))*0


def smooth(x, picks, y=smooth_kernel):
    return x
    for j in picks:
        x[j] = np.convolve(x[j], y, 'same')
    return x


# Prepare filename QYJ, ZYF
filedir = 'D:/BeidaShuju/rawdata/QYJ'
fname_training_list = list(os.path.join(
    filedir, 'MultiTraining_%d_raw_tsss.fif' % j)
    for j in range(1, 6))
fname_testing_list = list(os.path.join(
    filedir, 'MultiTest_%d_raw_tsss.fif' % j)
    for j in range(1, 9))
train = True
fname_list = fname_training_list
fname = fname_list[3]


def analysis_hilbert(x):
    # Calculate hilbert analytical signal
    y = fftpack.hilbert(x)
    return np.sqrt(x**2 + y**2)


def get_envlop(data, picks):
    # Calculate envlop of high freq signal
    # Using hilbert analytical signal
    # picks: only transform signals, not markers
    for j in picks:
        # Calculate envlop of each sensor
        data[j] = analysis_hilbert(data[j])
    return data


def get_epochs(fname, train, envlop=False):
    # Make defaults
    if train:
        event_id = dict(ort015=2,  ort045=6,  ort075=9,
                        ort105=14, ort135=17, ort165=33)
        tmin, t0, tmax = -0.25, 0, 1.25
    else:
        event_id = dict(ort45a=8, ort135a=16,
                        ort45b=32, ort135b=64)
        tmin, t0, tmax = -0.4, -0.2, 1.25

    freq_l, freq_h = 0.1, 5  # 2  # 1, 15
    baseline = (tmin, t0)
    reject = dict(mag=5e-12, grad=4000e-13)
    decim = 1

    # Prepare rawobject
    raw = mne.io.read_raw_fif(fname, preload=True)
    picks = mne.pick_types(raw.info, meg=True, eeg=False,
                           eog=False, stim=False, exclude='bads')
    sensors, picks = good_sensors(raw.ch_names)
    raw.filter(freq_l, freq_h, fir_design='firwin')
    if envlop:
        raw = mne.io.RawArray(get_envlop(
            smooth(raw.get_data(), picks), picks), raw.info)
    else:
        raw = mne.io.RawArray(smooth(raw.get_data(), picks), raw.info)
    events = mne.find_events(raw)

    # Get epochs
    epochs = mne.Epochs(raw, event_id=event_id, events=events,
                        decim=decim, tmin=tmin, tmax=tmax,
                        picks=picks, baseline=baseline,
                        reject=reject, preload=True)
    return epochs


'''
epochs = get_epochs(fname=fname, train=train)
epochs.plot(show=False)
plt.show()
'''
