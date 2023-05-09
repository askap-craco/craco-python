#!/usr/bin/env python
# function to plot difference in gain solution

from craco import calibration
import numpy as np
import matplotlib.pyplot as plt

def _chanwidth(freq1):
    """
    work out the channel width based on a series of frequencies
    """
    return np.mean(freq1[1:] - freq1[:-1])

def _find_freq_overlap(freq1, freq2):
    chanwidth1 = _chanwidth(freq1)
    chanwidth2 = _chanwidth(freq2)
    chanwidth = np.min([chanwidth1, chanwidth2])
    
    # get overlap between 
    freq_s = np.max([freq1[0], freq2[1]])
    freq_e = np.min([freq1[-1], freq2[-1]])
    
    return np.arange(freq_s, freq_e + chanwidth/2, chanwidth)  

def calculate_gain_diff(sol1, sol2):
    """
    calculate gain difference based on the two solutions

    Params
    ----------
    sol1, sol2: str
        Path to the calibration solution file

    Returns
    ----------
    gain_diff: numpy.array
        gain differance calculated by gain1 / gain2
    """
    gain1, freq1 = calibration.load_gains(sol1)
    gain2, freq2 = calibration.load_gains(sol2)

    freqs = _find_freq_overlap(freq1, freq2)

    gain_interp1 = calibration.interpolate_gains(
        gain1, freq1, freqs
    )
    gain_interp2 = calibration.interpolate_gains(
        gain2, freq2, freqs
    )

    return gain_interp1 / gain_interp2

def plot_gain_diff(gain_diff):
    """
    plot gain differance (gain1 / gain2)
    """
    fig = plt.figure(figsize=(24, 24))

    for ia in range(36):
        axidx = 12*(ia // 6) + (ia % 6) + 1
        ax = fig.add_subplot(12, 6, axidx)
        ax.plot(np.abs(gain_diff[ia, :, 0]), color="black")
        ax.plot(np.abs(gain_diff[ia, :, 1]), color="red")
        ax.set_ylim(0, 2)
        ax.set_title("ak{:0>2}".format(ia))

        axidx = 12*(ia // 6) + (ia % 6) + 7
        ax = fig.add_subplot(12, 6, axidx)
        ax.plot(np.angle(gain_diff[ia, :, 0], deg=True), color="black")
        ax.plot(np.angle(gain_diff[ia, :, 1], deg=True), color="red")
        ax.set_ylim(-200, 200)

    plt.subplots_adjust(hspace=0.5)
    
    return fig

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Plot gain solution difference', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-sol1", type=str, help="path to the gain solution1", default=None)
    parser.add_argument("-sol2", type=str, help="path to the gain solution2", default=None)
    parser.add_argument("-dir", type=str, help="path to save the image directory", default="./")

    values = parser.parse_args()
    gain_diff = calculate_gain_diff(values.sol1, values.sol2)
    fig = plot_gain_diff(gain_diff)
    fig.savefig(f"{values.dir}/gain_diff.png")


if __name__ == "__main__":
    _main()