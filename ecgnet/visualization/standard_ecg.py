import numpy as np
from matplotlib import pylab as plt
import os

LAYOUT = {'3x4_1': [[0, 3, 6, 9],
                    [1, 4, 7, 10],
                    [2, 5, 8, 11],
                    [1]],
          '3x4': [[0, 3, 6, 9],
                  [1, 4, 7, 10],
                  [2, 5, 8, 11]],
          '6x2': [[0, 6],
                  [1, 7],
                  [3, 8],
                  [4, 9],
                  [5, 10],
                  [6, 11]],
          '12x1': [[0],
                   [1],
                   [2],
                   [3],
                   [4],
                   [5],
                   [6],
                   [7],
                   [8],
                   [9],
                   [10],
                   [11]]}


class StandardECG(object):
    """The class representing the ECG object
    """

    paper_w, paper_h = 200, 90

    # Dimensions in mm of plot area
    width = 200
    height = 90
    margin_left = margin_right = (paper_w - width) // 2
    margin_bottom = 5

    # Normalized in [0, 1]
    left = margin_left / paper_w
    right = left + width / paper_w
    bottom = margin_bottom / paper_h
    top = 1

    def __init__(self, waveform, sampling_freq: int, gain: float = 1, 
                 name: str = "Anonymous", test_dict: dict = {}):
        """The ECG class constructor.

        waveform: numpy matrix with ECG waveform
            and shape n_channels x n_samples
        sampling_freq (int): the sampling frequency of the ECG
        gain (float): the gain of the ECG to convert to millivolts
            (if applicable)
        name (str): name to show on the ECG
        test_dict: dictionary with additional test characteristics:
            'TestID', 'Name', 'PseudoID', 'PatientAge', 'Gender',
            'QT Interval', 'QTc Interval', 'RR Interval',
            'QRS Duration', 'QRS Axis', 'T Axis', 'P Axis',
            'PR Interval', 'Sampling freq', 'mm/mv'
        """
        if waveform.shape[0] == 8:
            self.signals = self._to12lead(waveform)
        else:
            self.signals = waveform

        self.signals *= gain

        self.test_dict = test_dict
        self.channels_no = self.signals.shape[0]
        self.samples = self.signals.shape[1]
        self.sampling_frequency = sampling_freq
        self.duration = self.samples / self.sampling_frequency
        self.mm_s = self.width / self.duration
        self.mm_mv = 10
        self.fig, self.axis = self.create_figure()
        self.name = name

    def __del__(self):
        """
        Figures created through the pyplot interface
        (`matplotlib.pyplot.figure`) are retained until explicitly
        closed and may co nsume too much memory.
        """

        plt.cla()
        plt.clf()
        plt.close()

    def _to12lead(self, waveform):
        out = np.zeros((12,waveform.shape[1]))
        out[0:2,:] = waveform[0:2,:] # I and II
        out[2,:] = waveform[1,:] - waveform[0,:] # III = II - I
        out[3,:] = -(waveform[0,:] + waveform[1,:])/2 # aVR = -(I + II)/2
        out[4,:] = waveform[0,:] - (waveform[1,:]/2) # aVL = I - II/2
        out[5,:] = waveform[1,:] - (waveform[0,:]/2) # aVF = II - I/2
        out[6:12,:] = waveform[2:8,:] # V1 to V6
        return out
    
    def create_figure(self):
        """Prepare figure and axes"""

        # Init figure and axes
        fig = plt.figure(tight_layout=False)
        axes = fig.add_subplot(1, 1, 1)

        fig.subplots_adjust(left=self.left, right=self.right, top=self.top,
                            bottom=self.bottom)

        axes.set_ylim([0, self.height])

        # We want to plot N points, where N=number of samples
        axes.set_xlim([0, self.samples - 1])
        return fig, axes

    def draw_grid(self, minor_axis):
        """Draw the grid in the ecg plotting area."""

        if minor_axis:
            self.axis.xaxis.set_minor_locator(
                plt.LinearLocator(self.width + 1)
            )
            self.axis.yaxis.set_minor_locator(
                plt.LinearLocator(self.height + 1)
            )

        self.axis.xaxis.set_major_locator(
            plt.LinearLocator(self.width // 5 + 1)
        )
        self.axis.yaxis.set_major_locator(
            plt.LinearLocator(self.height // 5 + 1)
        )

        color = {'minor': '#ff5333', 'major': '#d43d1a'}
        linewidth = {'minor': .1, 'major': .2}

        for axe in 'x', 'y':
            for which in 'major', 'minor':
                self.axis.grid(
                    which=which,
                    axis=axe,
                    linestyle='-',
                    linewidth=linewidth[which],
                    color=color[which]
                )

                self.axis.tick_params(
                    which=which,
                    axis=axe,
                    color=color[which],
                    bottom=False,
                    top=False,
                    left=False,
                    right=False
                )

        self.axis.set_xticklabels([])
        self.axis.set_yticklabels([])

    def legend(self):
        """A string containing the legend.

        Auxiliary function for the print_info method.
        """
        ecgdata = self.test_dict
        
        bpm = ecgdata.get('VentricularRate', '')
        ret_str = "Ventr. freq.: %.1f BPM\n" % (float(bpm))

        ret_str_tmpl = "%s: %s ms\n%s: %s ms\n%s: %s/%s ms\n%s: %s %s %s"
        ret_str += ret_str_tmpl % (
            "PR Interval",
            ecgdata.get('PRInterval', ''),
            "QRS Duration",
            ecgdata.get('QRSDuration', ''),
            "QT/QTc Duration",
            ecgdata.get('QTInterval', ''),
            ecgdata.get('QTCorrected', ''),
            "P-R-T Axis",
            ecgdata.get('PAxis', ''),
            ecgdata.get('RAxis', ''),
            ecgdata.get('TAxis', '')
        )

        return ret_str

    def print_info(self):
        """Print info about the patient and about the ecg signals.
        """
        if self.test_dict.get('PseudoID'):
            pat_name = self.test_dict.get('PatientLastName','Anonymous')
            pseudo_id = str(self.test_dict.get('PseudoID',''))
            test_id = str(self.test_dict.get('TestID',''))
            pat_age = str(self.test_dict.get('Age',0))
            pat_sex = self.test_dict.get('Gender','')

            info = "%s\n%s: %s\n%s: %s\n%s: %s\n%s: %s %s" % (
                pat_name,
                'PseudoID',
                pseudo_id,
                'TestID',
                test_id,
                'Gender',
                pat_sex,
                'Age',
                pat_age,
                'years old',
            )

            plt.figtext(0.08, 0.87, info, fontsize=8)

        if self.test_dict.get('VentricularRate'):
            plt.figtext(0.30, 0.87, self.legend(), fontsize=8)

        info = "%s: %s s %s: %s Hz" % (
            "total time", self.duration,
            "sample_freq",
            self.sampling_frequency
        )

        plt.figtext(0.08, 0.025, info, fontsize=8)

        info = self.test_dict.get('SiteName', 'Unknown institution')

        plt.figtext(0.48, 0.025, info, fontsize=8)

        info = "%s mm/s %s mm/mV" % (self.mm_s, self.mm_mv)
        plt.figtext(0.81, 0.025, info, fontsize=8)

    def save(self, outputfolder=None, outformat=None):
        """Save the plot result either on a file or on a output buffer,
        depending on the input params.

        @param outputfile: the output filename.
        @param outformat: the ouput file format.
        """

        plt.savefig(
            os.path.join(outputfolder, 
                         str(self.name) + '.' + outformat),
            dpi=300, format=outformat,
            papertype='a4', orientation='landscape'
        )

    def plot(self, layoutid):
        """Plot the ecg signals inside the plotting area.
        Possible layout choice are:
        * 12x1 (one signal per line)
        * 6x2 (6 rows 2 columns)
        * 3x4 (4 signal chunk per line)
        * 3x4_1 (4 signal chunk per line. on the last line
        is drawn a complete signal)
        * ... and much much more

        The general rule is that the layout list is formed
        by as much lists as the number of lines we want to draw into the
        plotting area, each one containing the number of the signal chunk
        we want to plot in that line.

        @param layoutid: the desired layout
        @type layoutid: C{list} of C{list}
        """
        layout = LAYOUT[layoutid]
        rows = len(layout)
        channel_names = ['I','II','III','aVR','aVL','aVF',
                         'V1','V2','V3','V4','V5','V6']

        for numrow, row in enumerate(layout):

            columns = len(row)
            row_height = self.height / rows

            # Horizontal shift for lead labels and separators
            h_delta = self.samples / columns

            # Vertical shift of the origin
            v_delta = round(
                self.height * (1.0 - 1.0 / (rows * 2)) -
                numrow * row_height
            )

            # Let's shift the origin on a multiple of 5 mm
            v_delta = (v_delta + 2.5) - (v_delta + 2.5) % 5

            # Lenght of a signal chunk
            chunk_size = int(self.samples / len(row))
            for numcol, signum in enumerate(row):
                left = numcol * chunk_size
                right = (1 + numcol) * chunk_size

                # The signal chunk, vertical shifted and
                # scaled by mm/mV factor
                signal = v_delta + self.mm_mv * self.signals[signum][left:right]
                self.axis.plot(
                    list(range(left, right)),
                    signal,
                    clip_on=False,
                    linewidth=1,
                    color='black',
                    zorder=10)

                meaning = channel_names[signum]

                h = h_delta * numcol
                v = v_delta + row_height / 2.6
                plt.plot(
                    [h, h],
                    [v - 3, v],
                    lw=1,
                    color='black',
                    zorder=50
                )

                self.axis.text(
                    h + 40,
                    v_delta + row_height / 3,
                    meaning,
                    zorder=50,
                    fontsize=10
                )

        # A4 size in inches
        self.fig.set_size_inches(15, 8.27)

    def draw(self, layoutid, minor_axis=True, print_info=True):
        """Draw grid, info and signals"""

        self.draw_grid(minor_axis)
        if print_info:
            self.print_info()
        self.plot(layoutid)

