import matplotlib as mpl
# import numpy as np
# import matplotlib.pyplot as plt


def plotconfig(lbsize=17, lgsize=14, autolayout=True, figsize=[8, 6],
               ticklabelsize=16):

    ticks_font = mpl.font_manager.FontProperties(family='serif',
                                                 style='normal',
                                                 weight='normal',
                                                 stretch='normal',
                                                 size=lbsize)

    mpl.rcParams['xtick.labelsize'] = ticklabelsize
    mpl.rcParams['ytick.labelsize'] = ticklabelsize
    mpl.rcParams['font.size'] = 15
    mpl.rcParams['figure.autolayout'] = autolayout
    # mpl.rcParams['figure.figsize'] = 7.2, 4.45
    mpl.rcParams['figure.figsize'] = figsize[0], figsize[1]
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['axes.labelsize'] = lbsize
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['lines.markersize'] = 6
    mpl.rcParams['legend.fontsize'] = lgsize
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'STIXGeneral'

    return ticks_font
