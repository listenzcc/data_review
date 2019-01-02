# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt


def plot_scores(scores, times, axes=None):
    # plot scores,
    # left: diag values, right: across time resolution
    # Plotting layout
    if axes is None:
        fig, axes = plt.subplots(1, 2)
    # Plot Decoding over time
    ax = axes[0]
    im = ax.plot(times, np.diag(scores), label='score')
    ax.axhline(.5, color='k', linestyle='--', label='chance')
    ax.set_xlabel('Times')
    ax.set_ylabel('ACC')
    # ax.legend()
    ax.axvline(.0, color='k', linestyle='-')
    ax.set_title('Decoding MEG sensors over time')
    # Plot Generalization over time
    ax = axes[1]
    im = ax.matshow(scores, vmin=0, vmax=1., cmap='RdBu_r',
                    origin='lower',
                    extent=times[[0, -1, 0, -1]])
    ax.axhline(0., color='k')
    ax.axvline(0., color='k')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    ax.set_title('Generalization across time and condition')
    plt.colorbar(im, ax=ax)


def shrink_to_scores(mat_4d):
    # this function is to squeeze 1st and 2nd dim by mean values
    # it is designed to mean different ort conditions
    tmp = np.mean(mat_4d, 0)
    return np.mean(tmp, 0)


def plot_confuse_mat(confuse_mat, times):
    # this is used to plot confume_mat in a full manner
    # compare each two orts
    fig, axes = plt.subplots(6, 6)
    for j in range(6):
        for k in range(6):
            if j >= k:
                continue
            scores = shrink_to_scores(confuse_mat[j][k])
            ax = axes[j][k]
            plot_scores(scores, times, axes=[axes[j][k], axes[k][j]])
            continue
            im = ax.matshow(scores, vmin=0, vmax=1., cmap='RdBu_r',
                            origin='lower',
                            extent=times[[0, -1, 0, -1]])
            ax.axhline(0., color='k')
            ax.axvline(0., color='k')
            ax.xaxis.set_ticks_position('bottom')
            ax.set_xlabel('Testing Time (s)')
            ax.set_ylabel('Training Time (s)')
            ax.set_title('Generalization across time and condition')
            plt.colorbar(im, ax=ax)
    return fig


mat_fname = 'confuse_mat_cross_QYJ_lr_5hz'
confuse_mat = np.load('pics/%s.npy' % mat_fname)
# confuse_mat's shape is 6 x 6 x 5 x 100 x 101 x 101
# 6 orts x 6 orts x 5 cross x 100 repeats x 101 times x 101 times
confuse_mat = np.transpose(confuse_mat, [0, 1, 4, 5, 2, 3])


times = np.linspace(-0.2, 0.8, 101)

# plot every confuse_mat in non-diag positions
fig = plot_confuse_mat(confuse_mat, times)
fig.set_figwidth(10)
fig.set_figheight(10)
fig.savefig('pics/%s_0.png' % mat_fname, dpi=300)

ort_combine = dict()
ort_combine[0] = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
                  (1, 2), (1, 3), (1, 4), (1, 5),
                  (2, 3), (2, 4), (2, 5),
                  (3, 4), (3, 5),
                  (4, 5)]
ort_combine[30] = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
ort_combine[60] = [(0, 2), (1, 3), (2, 4), (3, 5)]
ort_combine[90] = [(0, 3), (1, 4), (2, 5)]

scores_d = dict()
for dort_ in ort_combine.keys():
    confuse_mat_ = np.vstack(
        confuse_mat[e[0], e[1]] for e in ort_combine[dort_])
    scores_d[dort_] = shrink_to_scores(confuse_mat_)

# plot confuse_mat in mean manner, mean on 4 combines
fig, axes = plt.subplots(4, 2)
for dort_ in ort_combine.keys():
    plot_scores(scores_d[dort_], times, axes=axes[int(dort_/30)])
fig.set_figwidth(10)
fig.set_figheight(10)
fig.savefig('pics/%s_1.png' % mat_fname, dpi=300)

scores_dd = dict()
for j in range(6):
    other_orts = list(range(6))
    other_orts.pop(j)
    confuse_mat_ = np.vstack(
        confuse_mat[min(j, e), max(j, e)] for e in other_orts)
    scores_dd[j] = shrink_to_scores(confuse_mat_)

# plot confuse_mat in mean manner, mean on 6 orts with others
fig, axes = plt.subplots(6, 2)
for j in range(6):
    plot_scores(scores_dd[j], times, axes=axes[j])
fig.set_figwidth(10)
fig.set_figheight(10)
fig.savefig('pics/%s_2.png' % mat_fname, dpi=300)

plt.show()
