# ==============================================================================
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import os
GRAPH_N = 0
from . import presets


class fplot:

    def __init__(self, plotconf='time'):

        self.title=None
        self.ftsize = 19
        self.lbsize=16
        self.lgsize=14
        self.fontfamily='serif'
        self.xlims=[]
        self.ylims=[]
        if plotconf == 'frequency':
            self.xlabel = 'Frequency [Hz]'
            self.ylabel = 'Relative frequency'
        elif plotconf == 'time':
            self.xlabel = 'Time [s]'
            self.ylabel = 'Relative frequency'
        else:
            self.xlabel = 'x'
            self.ylabel = 'y'
            
        self.ras = False
        self.legendloc = 'upper left'
        self.xscale = 'linear'
        self.yscale = 'linear'
        self.colors = ['k','b','r','g','m','0.75']
        self.linewidths = np.ones(2)
        self.plotconf = plotconf
        self.draw_frame = False
        self.subplotnum = 111

        self.ticks_font = presets.plotconfig(ctype=plotconf, lbsize=self.lbsize, lgsize=self.lgsize)


    def plot(self, X, Y, colors, linewidths, labels, linestyles=None, zorders=None, fig = None, plot_type='lines'):

        if fig is None:
            fig = plt.figure(0)
        if linestyles is None:
            linestyles = [None for x in X]

        ax1 = fig.add_subplot(self.subplotnum)
        #fig0.subplots_adjust(wspace=.5)
        S = np.shape(Y)

        # Ordering of the plots
        if zorders is None:
            zorders = np.arange(1,S[0]+1)

        for i in range(S[0]):

            if plot_type=='lines':
                ax1.plot(X[i],Y[i],colors[i],linewidth=linewidths[i],
                linestyle = linestyles[i],label = labels[i],rasterized=self.ras,
                zorder=zorders[i])

            elif plot_type=='bars':
                ax1.bar(X[i], Y[i], color=colors[i], width=linewidths[i], label = labels[i], zorder=zorders[i])


        if self.plotconf == 'time':
            # Font properties of axe ticks
            for label in ax1.get_xticklabels():
                label.set_fontproperties(self.ticks_font)

            for label in ax1.get_yticklabels():
                label.set_fontproperties(self.ticks_font)
            ax1.minorticks_on()

        ax1.set_xscale(self.xscale )
        ax1.set_yscale(self.yscale )


        if self.xlims != []:
            ax1.set_xlim(self.xlims)
        if self.ylims != []:
            ax1.set_ylim(self.ylims)

        #mpl.rc('font',family='monospace')
        leg1 = ax1.legend(fancybox=True,
        loc = self.legendloc,
        fontsize= self.lgsize)
        leg1.draw_frame(self.draw_frame)



        if self.ylabel == 'PSD':
            ylabel = r'$\sqrt{\rm PSD}$ [$\rm{ms}^{-2}/\sqrt{\rmHz}$]'
        elif self.ylabel == 'Amplitude':
            ylabel = r'Amplitude [$\rm{ms}^{-2}$]'
        else :
            ylabel = self.ylabel

        ax1.set_xlabel(self.xlabel,
        family = self.fontfamily,
        fontsize= self.ftsize)
        ax1.set_ylabel(ylabel,
        family = self.fontfamily,
        fontsize= self.ftsize)

        if self.title != None :
            fig.suptitle(self.title)

        plt.draw()

        # Adjust figure size to content
        # fig.tight_layout()

        return fig, ax1


def myhistogram(data,xlabel,ylabel,colors,labels,n_bins='auto',ax = None,figsize=(8,4)):
    """
    Customize the histogram plot
    source: randaolson.com

    Parameters
    ----------
    data : list of array_like
        list of data vectors
    colors : list of strings
        list of plot colors

    """
    mpl.rcdefaults()
    mpl.rcParams['font.family'] = 'serif'
    plt.rcParams['text.usetex'] = True
    plt.rc('text', usetex=True)
    # ticks_font = mpl.font_manager.FontProperties(family='serif', style='normal',
    # usetex = True,
    # weight='normal',
    # stretch='normal')


    # You typically want your plot to be ~1.33x wider than tall.
    # Common sizes: (10, 7.5) and (12, 9)
    plt.figure(figsize=figsize)

    # Remove the plot frame lines. They are unnecessary chartjunk.
    if ax is None:
        ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Ensure that the axis ticks only show up on the bottom and left of the plot.
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()



    # Make sure your axis ticks are large enough to be easily read.
    # You don't want your viewers squinting to read your plot.
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Along the same vein, make sure your axis labels are large
    # enough to be easily read as well. Make them slightly larger
    # than your axis tick labels so they stand out.
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)

    # Plot the histogram. Note that all I'm passing here is a list of numbers.
    # matplotlib automatically counts and bins the frequencies for us.
    # "#3F5D7D" is the nice dark blue color.
    # Make sure the data is sorted into enough bins so you can see the distribution.
    for j in range(len(data)):
        n,bins,patches=plt.hist(data[j],color=colors[j], bins=n_bins, alpha = 0.5, label = labels[j])
        plt.draw()
    # Always include your data source(s) and copyright notice! And for your
    # data sources, tell your viewers exactly where the data came from,
    # preferably with a direct link to the data. Just telling your viewers
    # that you used data from the "U.S. Census Bureau" is completely useless:
    # the U.S. Census Bureau provides all kinds of data, so how are your
    # viewers supposed to know which data set you used?
    #plt.text((np.max(data)-np.min(data))*0.33, -np.max(n)*0.15, datasource,
    #fontsize=10)
    plt.legend(loc='upper right')

    # Finally, save the figure as a PNG.
    # You can also save it as a PDF, JPEG, etc.
    # Just change the file extension in this call.
    # bbox_inches="tight" removes all the extra whitespace on the edges of your plot.
    #plt.savefig("chess-elo-rating-distribution.png", bbox_inches="tight");
    ax.minorticks_on()

    # plt.tight_layout()

    return ax,n,bins,patches



def confidenceIntervals(axe,X,ci_low,ci_up,color,a):

   axe.fill_between(X, ci_low,ci_up, facecolor = color, alpha = a)


def stemPlot(X,Y,color):

   mpl.rcdefaults()
   mpl.rcParams['font.family'] = 'monospace'
   ticks_font = mpl.font_manager.FontProperties(family='serif', style='normal',
                                  weight='normal',
                                  stretch='normal')
   mpl.rcParams['xtick.major.size'] = 6
   mpl.rcParams['ytick.major.size'] = 6
   mpl.rcParams['xtick.minor.size'] = 3
   mpl.rcParams['ytick.minor.size'] = 3
   mpl.rcParams['mathtext.cal'] = 'monospace'
   mpl.rcParams['mathtext.rm'] = 'monospace'
   mpl.rcParams['xtick.major.width']=1
   mpl.rcParams['ytick.major.width']=1
   mpl.rcParams['xtick.minor.width']=1
   mpl.rcParams['ytick.minor.width'] =1
   mpl.rcParams['lines.markeredgewidth']=1
   mpl.rcParams['legend.handletextpad']=0.3
   mpl.rcParams['legend.fontsize']= 'medium'
   mpl.rcParams['figure.figsize'] = 8,6

   fig = plt.figure()
   ax1 = fig.add_subplot(111)
   #fig0.subplots_adjust(wspace=.5)
   ax1.stem(X, Y)
   ax1.setp(markerline, 'markerfacecolor', color)

   return ax1





class Pgf:

   def __init__(z, xlabel='', ylabel='', dataprefix = ''):
      """Initialize and provide axis labels."""
      z.buf = []
      z.options = []
      z.opt('xlabel={{{0}}}'.format(xlabel))
      z.opt('ylabel={{{0}}}'.format(ylabel))
      z.legend = []
      z.dataf = []
      z.datacount = 0
      z.dataprefix = dataprefix

   def opt(z, *args):
      """Write arguments to the AXIS environment."""
      for arg in args:
         z.options.append(arg)
   def plot(z, x, y, legend=None, *args):
      """Plot the data contained in the vectors x and y.

      Options to the \addplot command can be provided in *args.
      """
      coor = ''.join(['({0}, {1})'. format(u, v) for u, v in zip(x,y)])
      z.buf.append('\\addplot{0} coordinates {{{1}}};\n'.format(
               ('[' + ', '.join(args) + ']') if len(args) else '' ,coor))
      if legend is not None:
         z.legend.append(legend)

   def plot_file(z, x, y, error = None, legend=None, *args):
      """Plot the data contained in the vectors x and y.
      Options to the \addplot command can be provided in *args.
      """
      if error == None:
         z.dataf.append( np.hstack( (np.array([x]).T,np.array([y]).T)) )
         coor = 'table[x=X,y=Y] {'+z.dataprefix+'data'+str(z.datacount)+'.dat}'
      else:
         z.dataf.append(  np.hstack((np.array([x]).T,np.array([y]).T,np.array([error]).T)) )
         coor = 'table[x=X,y=Y,y error=Y_error] {'+z.dataprefix+'data'+str(z.datacount)+'.dat}'

      if len(args):
         arguments = '[' + ', '.join(args) + ']'
      else:
         arguments = ''

      z.buf.append('\\addplot'+arguments+'  '+coor+';\n')
      if legend is not None:
         z.legend.append(legend)

      z.datacount = z.datacount + 1

   def plot_fill(z,line,*args):
      """Fill an area described by line with some arguments
      """
      if len(args):
         arguments = '[' + ', '.join(args) + ']'
      else:
         arguments = ''
      z.buf.append('\\fill'+arguments+'  '+line+';\n')

   def plot_fillbetween(z,options,*args):

      if len(args):
         arguments = '[' + ', '.join(args) + ']'
      else:
         arguments = ''
      if len(options):
         opt = '['+options+']'
      else:
         opt = ''

      z.buf.append('\\addplot'+arguments+'  fill between'+opt+';\n')

   def plot_line(z,x1,x2,legend=None, *args):

      if len(args):
         arguments = '[' + ', '.join(args) + ']'
      else:
         arguments = ''

      # x1 contains the coordinates of the line bottom
      # x2 contains the coordinates of the line second edge
      s = np.shape(x1)

      if len(s) == 1 :
         z.buf.append('\\addplot'+arguments+' coordinates {('+str(x1[0])+','+str(x1[1])+')'+'('+str(x2[0])+','+str(x2[1])+')'+'};\n')
      #elif len(s)>1 :
         #z.buf.append('\\addplot'+arguments+' coordinates {')
         #for i in range(s[0]):
            #z.buf.append('('+str(x1[i,0])+','+str(x1[i,1])+')'+'('+str(x2[i,0])+','+str(x2[i,1])+')')
         #z.buf.append('};\n')


      if legend is not None:
         z.legend.append(legend)

   def plot_text(z,x,y,text,*args):

      if len(args):
         arguments = '[' + ', '.join(args) + ']'
      else:
         arguments = ''

      z.buf.append("\\node"+arguments+" at (axis cs:"+str(x)+","+str(y)+"){\small{"+text+"}};")

   def plot_arrow(z,x1,y1,x2,y2,text,*args):

      if len(args):
         arguments = '[' + ', '.join(args) + ']'
      else:
         arguments = ''

      for j in range(len(x1)):

         z.buf.append("\\node"+arguments+" (source) at (axis cs:"+str(x1[j])+","+str(y1[j])+"){};\n")
         z.buf.append("\\node"+arguments+" (destination) at (axis cs:"+str(x2[j])+","+str(y2[j])+"){};\n")
         z.buf.append("\\draw[->](source)--(destination);\n")

   def save(z, file_name,axis_type = 'axis'):
      """Generate graph.

      If graph_n is None or a number, the graph in a file beginning with
      zzz.  This file is meant to be temporary.  If graph_n is a string,
      that string is used as the file name.
      """
      if z.dataf != []:
         #file_name = 'zzz{0}'.format(graph_n)
         head, tail = os.path.split(file_name)
         for j in range(z.datacount):
            datafile_path = head + '/'+z.dataprefix+'data'+str(j)+'.dat'
            if np.shape(z.dataf[j])[1] == 2 :
               np.savetxt(datafile_path,z.dataf[j], header = 'X Y',comments='')
            elif np.shape(z.dataf[j])[1] == 3 :
               np.savetxt(datafile_path,z.dataf[j], header = 'X Y Y_error', comments='')

      #if type(graph_n) is str:
         #file_name = graph_n
      #else:
         #if graph_n is None:
            #global GRAPH_N
            #graph_n = GRAPH_N
            #GRAPH_N += 1
         #elif type(graph_n) is not int:
            #raise Error('graph_n should be a string or an integer')
         #file_name = 'zzz{0}'.format(graph_n)
      with open(file_name + '.tex', 'w') as f:
         b = []
         b.append('\\documentclass{standalone}\n')
         b.append('\\usepackage{pgfplots}\n')
         b.append('\\usepgfplotslibrary{fillbetween}\n')
         b.append('\\begin{document}\n')
         b.append('\\begin{tikzpicture}')
         b.append('\\begin{'+axis_type+'}[\n')
         b.append('{0}]'.format(',\n'.join(z.options)))
         b.extend(z.buf)
         if z.legend:
            b.append('\\legend{{' + '}, {'.join(z.legend) + '}}\n')
         b.append('\\end{'+axis_type+'}\n')
         b.append('\\end{tikzpicture}\n')
         b.append('\\end{document}')
         f.writelines(b)
      print(''.join(b))
      #os.system('pdflatex {0}.tex'.format(file_name))
      #os.remove(file_name + '.aux')
      #os.remove(file_name + '.log')
      #subprocess.Popen(['xpdf',  '{0}.pdf'.format(file_name)])


def time_series_plot(t,y,file_path,step=1,y_min=None,y_max=None,label = None,
                     xlabel='Time [s]', ylabel='Acceleration [ms$^{-2}$]'):


    head, tail = os.path.split(file_path)

    if y_min is None:
        y_min = np.min(y)
    if y_max is None:
        y_max = np.max(y)

    N = len(y)
    p = Pgf(xlabel, ylabel, tail+'_')
    p.opt('xmin='+str(t[0]))
    p.opt('xmax='+str(t[N-1]))
    p.opt('ymin='+str(y_min))
    p.opt('ymax='+str(y_max))
    p.opt('legend pos= north west')
    p.opt('legend cell align=left')
    p.opt('minor y tick num={2}')
    p.opt('minor x tick num={2}')
    p.opt('x tick style={color=black}')
    p.opt('y tick style={color=black}')
    #p.opt('legend style={font=\scriptsize}')
    # First observed data span

    localinds = np.arange(0,N,step)
    p.plot_file(t[localinds], y[localinds], None,label,'black')
    p.save(file_path,axis_type = 'axis')

def time_series_missing_plot(t_select,y_select,Nds,Nfs,file_path,y_min=-6e-9,y_max=7e-9, scale=1.0):


   p = Pgf('Time [s]', 'Acceleration [ms$^{-2}$]')
   p.opt('xmin='+str(t_select[0]))
   p.opt('xmax='+str(t_select[Ns-1]))
   p.opt('ymin='+str(y_min))
   p.opt('ymax='+str(y_max))
   p.opt('legend pos= north west')
   p.opt('legend cell align=left')
   p.opt('minor y tick num={2}')
   p.opt('minor x tick num={2}')
   p.opt('x tick style={color=black}')
   p.opt('y tick style={color=black}')
   #p.opt('legend style={font=\scriptsize}')
   l = len(Nds)
   step = 8
   # First observed data span
   localinds = np.arange(0,Nds[0],step)
   p.plot_file(t_select[localinds], y_select[localinds]*scale, None,'Observed data','black')
   # Last observed data span
   localinds = np.arange(Nfs[l-1],Ns,step)
   p.plot_file(t_select[localinds], y_select[localinds]*scale, None,None,'black','forget plot')


   if l>1:
      for i in range(l-1):
         localinds = np.arange(Nfs[i],Nds[i+1],step)
         p.plot_file(t_select[localinds], y_select[localinds]*scale, None,None,'black','forget plot')

   letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N']


   for i in range(len(Nds)):

      p.plot_fill('(axis cs:'+str(t_select[Nds[i]-1])+','+str(y_min+0.02e-9)+') rectangle (axis cs:'+str(t_select[Nfs[i]])+','+str(y_max-0.02e-9)+')','gray!20')

      localinds = np.arange(Nds[i],Nfs[i],step)
      if i == 0 :
         p.plot_file(t_select[localinds], y_select[localinds]*scale, None,'Missing data', 'gray')
      else:
         p.plot_file(t_select[localinds], y_select[localinds]*scale, None,None, 'gray','forget plot')
      #p.plot_file(t_select[Nds[i]:Nfs[i]], cit_low_select[Nds[i]:Nfs[i]]*scale, None, None,'gray!50','name path='+letters[i])
      #p.plot_file(t_select[Nds[i]:Nfs[i]], cit_up_select[Nds[i]:Nfs[i]]*scale, None, None,'gray!50','name path='+letters[i+1])
      #p.plot_fillbetween('of='+letters[i]+' and '+letters[i+1]+',soft clip={domain='+str(t_select[Nds[i]])+':'+str(t_select[Nfs[i]-1])+'}','gray','fill opacity=0.2')
      #p.plot_fillbetween('of='+letters[i]+' and '+letters[i+1],'gray','fill opacity=0.2')

   p.save(file_path,axis_type = 'axis')
