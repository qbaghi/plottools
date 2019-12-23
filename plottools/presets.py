import matplotlib as mpl
# import numpy as np
# import matplotlib.pyplot as plt


def plotconfig(ctype='frequency', lbsize=16, lgsize=14):

    ticks_font = mpl.font_manager.FontProperties(family='serif',
                                                 style='normal',
                                                 weight='normal',
                                                 stretch='normal',
                                                 size=lbsize)
    if ctype == 'frequency':
        mpl.rcdefaults()
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['xtick.major.size'] = 6
        mpl.rcParams['ytick.major.size'] = 6
        mpl.rcParams['xtick.minor.size'] = 3
        mpl.rcParams['ytick.minor.size'] = 3
        mpl.rcParams['xtick.major.width'] = 1
        mpl.rcParams['ytick.major.width'] = 1
        mpl.rcParams['xtick.minor.width'] = 1
        mpl.rcParams['ytick.minor.width'] = 1
        mpl.rcParams['lines.markeredgewidth'] = 1
        mpl.rcParams['legend.handletextpad'] = 0.3
        mpl.rcParams['legend.fontsize'] = lgsize
        mpl.rcParams['figure.figsize'] = 8, 6
        # mpl.rcParams['text.usetex'] = True
        mpl.rcParams['xtick.labelsize'] = lbsize
        mpl.rcParams['ytick.labelsize'] = lbsize
        # mpl.rcParams['figure.autolayout'] = True

    elif ctype == 'time':

        mpl.rcParams['xtick.labelsize'] = 16
        mpl.rcParams['ytick.labelsize'] = 16
        mpl.rcParams['font.size'] = 15
        mpl.rcParams['figure.autolayout'] = True
        # mpl.rcParams['figure.figsize'] = 7.2, 4.45
        mpl.rcParams['figure.figsize'] = 8, 6
        mpl.rcParams['axes.titlesize'] = 16
        mpl.rcParams['axes.labelsize'] = 17
        mpl.rcParams['lines.linewidth'] = 2
        mpl.rcParams['lines.markersize'] = 6
        mpl.rcParams['legend.fontsize'] = 13
        mpl.rcParams['mathtext.fontset'] = 'stix'
        mpl.rcParams['font.family'] = 'STIXGeneral'

    return ticks_font