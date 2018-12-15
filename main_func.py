# coding: utf-8

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mne.stats.regression import linear_regression_raw
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


def save_epochs_as_txt(epochs, cvdir):
    num_ = epochs.events.shape[0]
    if not os.path.isdir(cvdir):
        os.mkdir(cvdir)
    path_events = os.path.join(cvdir, 'events.txt')
    np.savetxt(path_events, epochs.events)
    data = epochs.get_data()
    for j in range(num_):
        path_data_ = os.path.join(cvdir, 'data_%d.txt' % j)
        np.savetxt(path_data_, data[j])


st = simple_timer()
train = True
fname_list, ortids, event_ids, tmin, t0, tmax = para_setting(train=train)
# show pretty evoked topo
# fname = fname_list[0]
if not os.path.isdir('pics'):
    os.mkdir('pics')

for fname in fname_list:
    print(fname)
    basename = os.path.basename(fname[0:-4])
    epochs, raw = get_epochs(fname=fname, event_id=event_ids,
                             tmin=tmin, t0=t0, tmax=tmax,
                             use_good_sensors=False)
    #  save_epochs_as_txt(epochs, os.path.join('pics', basename))

    fig, axes = plt.subplots(1, 1)
    raw.plot_sensors(show_names=True, axes=axes, show=False)
    fig.savefig(os.path.join('pics', basename+'_sensor.png'),
                dpi=300)

    fig, axes = plt.subplots(3, 1)
    plot_tasks_in_timeline(raw, epochs, axes=axes)
    fig.savefig(os.path.join('pics', basename+'_task.png'), dpi=600)

    evoked = epochs.average()
    evokeds = linear_regression_raw(raw,
                                    events=epochs.events,
                                    event_id=epochs.event_id,
                                    reject=None,
                                    tmin=tmin,
                                    tmax=tmax)
    for e in epochs.event_id.keys():
        print(e)
        fig, axes = plt.subplots(2, 1)
        fig, title = plot_evoked(
            evokeds[e], axes=axes[0:2],
            title=basename)
        fig.savefig(os.path.join('pics', title+e+'_ts_.png'),
                    dpi=300)
    fig, axes = plt.subplots(2, 1)
    fig, title = plot_evoked(evoked, axes=axes[0:2],
                             title=basename)
    fig.savefig(os.path.join('pics', title+'_ts_.png'),
                dpi=300)

    times = np.arange(0, 0.5, 0.05)
    fig, axes = plt.subplots(6, times.shape[0])
    idx = 0
    for e in epochs.event_id.keys():
        print(e)
        evokeds[e].plot_topomap(times,
                                ch_type='mag', average=0.05,
                                axes=axes[idx, :], show=False,
                                time_unit='s')
        idx += 1
        # evoked.animate_topomap(ch_type='mag', times=times, frame_rate=10)
    fig.savefig(os.path.join('pics', basename+'topo.png'),
                dpi=300)

    plt.close('all')
    st.click()

# plt.show()
plt.close('all')
