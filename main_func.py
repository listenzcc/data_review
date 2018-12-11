# coding: utf-8

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from load_preprocess_view import get_epochs
sys.path.append('C:\\Users\\liste\\Documents\\Python Scripts\\clock_tools')
from simple_timer import simple_timer

# Prepare filename QYJ, ZYF
filedir = 'D:/BeidaShuju/rawdata/ZYF'


def para_setting(train=True, filedir=filedir):
    if train:
        fname_list = list(os.path.join(
            filedir, 'MultiTraining_%d_raw_tsss.fif' % j)
            for j in range(1, 6))
        ortids = [2, 6, 9, 14, 17, 33]
        event_ids = dict(ort015=2,  ort045=6,  ort075=9,
                         ort105=14, ort135=17, ort165=33)
        tmin, t0, tmax = -0.25, 0, 1.25
    else:
        fname_list = list(os.path.join(
            filedir, 'MultiTest_%d_raw_tsss.fif' % j)
            for j in range(1, 9))
        ortids = [8, 16, 32, 64]
        event_ids = dict(ort45a=8, ort135a=16,
                         ort45b=32, ort135b=64)
        tmin, t0, tmax = -0.4, -0.2, 1.25
    return fname_list, ortids, event_ids, tmin, t0, tmax


def plot_evoked(evoked2plot, axes=None, title='noname'):
    fig = evoked2plot.plot(spatial_colors=True, gfp=True,
                           axes=axes,  # where to plot
                           show=False,  # we will show them togegher
                           time_unit='s', window_title=title)
    axes = fig.get_axes()
    for j in [0, 1]:
        ylim = axes[j].get_ylim()
        axes[j].plot([0, 0], ylim)
    return fig, title


def plot_tasks_in_timeline(raw, epochs, duration=1,
                           axes=None):
    ntimes = raw.n_times
    sfreq = raw.info['sfreq']
    timeline = np.zeros(ntimes)
    timeline1 = timeline.copy()
    for e in epochs.events:
        timeline[e[0]:e[0]+int(duration*sfreq)] = e[2]
        timeline1[e[0]:e[0]+int(duration*sfreq)] = 1
    sp = np.fft.fft(timeline1)
    freq = np.fft.fftfreq(timeline1.shape[-1], d=1/sfreq)
    if axes is None:
        plt.figure
        plt.plot(timeline)
        plt.plot(timeline1)
    else:
        axes[0].plot(timeline)
        axes[0].plot(timeline1)
        axes[1].plot(freq, sp.real, freq, sp.imag)
        axes[2].plot(freq, sp.real, freq, sp.imag)
        axes[2].set_xlim([-5, 5])
    return


st = simple_timer()
train = True
fname_list, ortids, event_ids, tmin, t0, tmax = para_setting(train=train)
# show pretty evoked topo
# fname = fname_list[0]
for fname in fname_list:
    print(fname)
    basename = os.path.basename(fname[0:-4])
    epochs, raw = get_epochs(fname=fname, event_id=event_ids,
                             tmin=tmin, t0=t0, tmax=tmax,
                             use_good_sensors=False)

    fig, axes = plt.subplots(3, 1)
    plot_tasks_in_timeline(raw, epochs, axes=axes)
    fig.savefig(os.path.join('pics', basename+'_task.png'), dpi=600)

    evoked = epochs.average()
    fig, axes = plt.subplots(2, 1)
    fig, title = plot_evoked(
        evoked, axes=axes[0:2],
        title=basename)
    fig.savefig(os.path.join('pics', title+'ts_.png'), dpi=600)

    st.click()

# plt.show()
plt.close('all')
