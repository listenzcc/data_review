# coding: utf-8

import sys
import numpy as np
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_morlet

import itertools

from tools_loaddata import para_setting, get_epochs

sys.path.append('C:\\Users\\liste\\Documents\\Python Scripts\\clock_tools')
from simple_timer import simple_timer

freq_h = 200
decim = 1
use_good_sensors = False

filedir = 'D:/BeidaShuju/rawdata/%s' % 'ZYF'
# initial running timer
st = simple_timer()

# parameters setting
fname_list, ortids, event_ids, tmin, t0, tmax = para_setting(
    train=True, filedir=filedir)

# define frequencies of interest (log-spaced)
freqs = np.linspace(10, 80, num=20)
n_cycles = freqs / 2.  # different number of cycle per frequency

# load raw data and epochs
epochs_run = []
for fname in fname_list:
    print(fname)
    epochs, raw = get_epochs(fname=fname, event_id=event_ids,
                             tmin=tmin, t0=t0, tmax=tmax,
                             freq_l=1, freq_h=freq_h,
                             decim=decim,
                             use_good_sensors=use_good_sensors,
                             get_envlop=False)

    event_ids = epochs.event_id

    power_orts = dict()
    for ort_ in event_ids.keys():
        print(ort_)
        power_orts[ort_] = np.vstack(np.expand_dims(
            tfr_morlet(epochs[ort_][j], freqs=freqs,
                       n_cycles=n_cycles,
                       use_fft=True, return_itc=False,
                       decim=1, n_jobs=12).data, 0)
            for j in range(epochs[ort_].events.shape[0]))

    st.click()
    break
