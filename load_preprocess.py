# coding: utf-8

import os
import mne
import numpy as np
# import matplotlib.pyplot as plt
from pick_good_sensors import good_sensors

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


def get_epochs(fname, event_id, tmin, t0, tmax, use_good_sensors=True):
    # Make defaults
    freq_l, freq_h = 0, 2  # 1, 15
    baseline = (tmin, t0)
    reject = dict(mag=5e-12, grad=4000e-13)
    decim = 1

    # Prepare rawobject
    raw = mne.io.read_raw_fif(fname, preload=True)
    picks = mne.pick_types(raw.info, meg=True, eeg=False,
                           eog=False, stim=False, exclude='bads')
    if use_good_sensors:
        sensors, picks = good_sensors(raw.ch_names)
    raw = mne.io.RawArray(smooth(raw.get_data(), picks), raw.info)
    raw.filter(freq_l, freq_h, fir_design='firwin')
    events = mne.find_events(raw)

    # Get epochs
    epochs = mne.Epochs(raw, event_id=event_id, events=events,
                        decim=decim, tmin=tmin, tmax=tmax,
                        picks=picks, baseline=baseline,
                        reject=reject, preload=True)
    return epochs, raw


'''
epochs = get_epochs(fname=fname, train=train)
epochs.plot(show=False)
plt.show()
'''
