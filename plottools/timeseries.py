import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# matplotlib inline
from . import presets
from itertools import cycle, islice
# import seaborn as sns
# sns.set_style('white', {"xtick.major.size": 2, "ytick.major.size": 2})
# flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71","#f4cae4"]
# sns.set_palette(sns.color_palette(flatui, 7))


def plot_series(dates, data_list, labels, colors=None, linewidths=None, linestyles=None,
                subplots=False, xlabel='Time [s]', ylabel='Amplitude', logx=False, logy=False):

    ticks_font = presets.plotconfig(ctype='time', lbsize=16, lgsize=14)

    values = np.array(data_list).T
    df = pd.DataFrame(values, dates, columns=labels)

    if subplots:
        fig, axes = plt.subplots(len(data_list), 1, figsize=(7.2, 4.45))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(7.2, 4.45), sharex=True)

    if colors is None:
        colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(df)))
    if linewidths is None:
        linewidths = np.ones(len(df))
    if linestyles is None:
        linestyles = ['-' for i in range(len(df))]

    for col, style, lw, clr in zip(df.columns, linestyles, linewidths, colors):
        df[col].plot(subplots=subplots, style=style, lw=lw, color=clr, logx=logx, logy=logy)
    # df[labels].plot(subplots=subplots, ax=axes, color=colors, linewidth=linewidths)

    if subplots:
        # for subplots we must add features by subplot axis
        for ax, col in zip(axes, labels):
            # ax.axvspan(recs2k_bgn, recs2k_end, color=sns.xkcd_rgb['grey'], alpha=0.5)
            # ax.axvspan(recs2k8_bgn, recs2k8_end, color=sns.xkcd_rgb['grey'], alpha=0.5)
            # lets add horizontal zero lines
            # ax.axhline(0, color='k', linestyle='-', linewidth=1)

            # add titles
            # ax.set_title('Monthly ' + col + ' \nRecessions Shaded Gray')

            # add axis labels
            ax.set_ylabel(xlabel)
            ax.set_xlabel(ylabel)

            # add cool legend
            ax.legend(loc='upper left', fontsize=11, frameon=True).get_frame().set_edgecolor('black')

            # add more ticks
            ax.minorticks_on()
    else:
        # add axis labels
        axes.set_ylabel(ylabel)
        axes.set_xlabel(xlabel)

        # add cool legend
        axes.legend(loc='upper left', fontsize=11, frameon=True).get_frame().set_edgecolor('black')
        axes.minorticks_on()
        # now to use tight layout
    plt.tight_layout()

    return fig, axes

