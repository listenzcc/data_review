# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import mne
import os
import sys
from load_preprocess_view import get_epochs
sys.path.append('C:\\Users\\liste\\Documents\\Python Scripts\\clock_tools')
from simple_timer import simple_timer

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


st = simple_timer()
train = True
fname_list, ortids, event_ids, tmin, t0, tmax = para_setting(train=train)

t = np.triu(np.ones([1001, 1001])/1001, 0)

for fname in fname_list:
    print(fname)
    basename = os.path.basename(fname[0:-4])
    epochs, raw = get_epochs(fname=fname, event_id=event_ids,
                             tmin=tmin, t0=t0, tmax=tmax,
                             use_good_sensors=False)

    evoked = epochs.average()
    evoked.plot_topo(show=False)

    data = np.dot(evoked.data, t)
    plt.figure()
    plt.plot(data.transpose())

    evoked.data = data
    evoked.plot_topo(show=False)

    st.click()

    break

plt.show()
