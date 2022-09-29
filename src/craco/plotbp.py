#!/usr/bin/env python

import numpy as np
#from casacore.tables import *
import matplotlib.pyplot as plt
import sys
import os
import glob

class Bandpass:
    def __init__(self):
        """Initialises parameters for reading a bandpass table
        """
        self.nsol = None
        self.nant = None
        self.npol = None
        self.nchan = None
        self.bandpass = None

    def invert(self, matrix):
        temp = matrix.reshape((2,2))
        return np.linalg.inv(temp).reshape((4))

    def singular(self, matrix):
        temp = np.array([matrix[0] * np.conj(matrix[0]) + matrix[1] * np.conj(matrix[1]),
            matrix[0] * np.conj(matrix[2]) + matrix[1] * np.conj(matrix[3]),
            matrix[2] * np.conj(matrix[0]) + matrix[3] * np.conj(matrix[1]),
            matrix[2] * np.conj(matrix[2]) + matrix[3] * np.conj(matrix[3])])
        # Use quadratic formula, with a=1.
        b = -temp[0].real - temp[3].real
        c = temp[0].real*temp[3].real - (temp[1]*temp[2]).real
        d = b*b - (4.0*1.0)*c
        sqrtd = np.sqrt(d)

        e1 = np.sqrt((-b + sqrtd) * 0.5);
        e2 = np.sqrt((-b - sqrtd) * 0.5);
        return e1, e2

    def printSolutionInv(self, ant, chan):
        matrix = self.bandpass[ant, chan]
        imatrix = self.invert(matrix)
        s1, s2 = self.singular(imatrix)
        phases = np.angle(imatrix)
        matrix = imatrix
        print("(%9.6f%+9.6fj %9.6f%+9.6fj) amplitudes=(%9.6f %9.6f) phases=(%+9.6f %+9.6f) SV=%9.6f" %(matrix[0].real, matrix[0].imag, matrix[1].real, matrix[1].imag, np.abs(matrix[0]), np.abs(matrix[1]), phases[0], phases[1], s1))
        print("(%9.6f%+9.6fj %9.6f%+9.6fj)            (%9.6f %9.6f)        (%+9.6f %+9.6f)    %9.6f" %(matrix[2].real, matrix[2].imag, matrix[3].real, matrix[3].imag, np.abs(matrix[2]), np.abs(matrix[3]), phases[2], phases[3], s2))

    def printSolution(self, ant, chan):
        matrix = self.bandpass[ant, chan]
        s1, s2 = self.singular(matrix)
        phases = np.angle(matrix)
        print("(%9.6f%+9.6fj %9.6f%+9.6fj) amplitudes=(%9.6f %9.6f) phases=(%+9.6f %+9.6f) SV=%9.6f" %(matrix[0].real, matrix[0].imag, matrix[1].real, matrix[1].imag, np.abs(matrix[0]), np.abs(matrix[1]), phases[0], phases[1], s1))
        print("(%9.6f%+9.6fj %9.6f%+9.6fj)            (%9.6f %9.6f)        (%+9.6f %+9.6f)    %9.6f" %(matrix[2].real, matrix[2].imag, matrix[3].real, matrix[3].imag, np.abs(matrix[2]), np.abs(matrix[3]), phases[2], phases[3], s2))

    @classmethod
    def load(cls, filename):
        self = Bandpass()
        dt = np.dtype('<i4')
        fp = open(filename,'r')
        header = np.fromfile(fp, dtype=dt, count=2)
        headerValues = np.fromfile(fp, dtype=dt, count=10)
        self.nsol = headerValues[2]
        self.nant = headerValues[3]
        self.nchan = headerValues[4]
        self.npol = headerValues[5]
        self.bandpass = np.zeros((self.nsol, self.nant, self.nchan, self.npol), dtype=np.complex)
        dtc = np.dtype('<c16')
        self.bandpass = np.array(np.fromfile(fp, dtype=dtc, count=self.nsol * self.nant * self.nchan * self.npol))
        self.bandpass = self.bandpass.reshape((self.nsol, self.nant, self.nchan, self.npol))
        self.bandpass = np.sqrt(2.0) / self.bandpass
        fp.close()
        print("Read bandpass: %d solutions, %d antennas, %d channels, %d polarisations" %(self.nsol, self.nant, self.nchan, self.npol))
        return self

    def plotGains(self, sol, ref_ant = 0, out_file = None):
        fig = plt.figure(figsize=(14, 14))
        ant = 0
        amplitudes = np.abs(self.bandpass[sol])
#        amplitudes[np.where(amplitudes>2.0)] = 0.0
        self.bandpass[sol] = self.bandpass[sol] / self.bandpass[sol,ref_ant,:,:]
        phases = np.angle(self.bandpass[sol], deg=True)
#        phases[np.where(amplitudes==0.0)] = 0.0
        channels = np.array(range(self.nchan))
        NY = 6
        NX = 6
        max_val = 20.0
        for y in range(NY):
            for x in range(NX):
                amps_xx = amplitudes[ant,:,0]
                amps_yy = amplitudes[ant,:,3]
                good_xx = amps_xx[np.where(amps_xx<max_val)]
                good_yy = amps_yy[np.where(amps_yy<max_val)]
                if len(good_xx) > 0 and len(good_yy) > 0:
                    plt.subplot(NY*2, NX, y * 2 * NX + x + 1)
                    plt.title("ak%02d" %(ant+1), fontsize=8)
                    plt.plot(channels, amplitudes[ant,:,0], marker=None, color="black")
                    plt.plot(channels, amplitudes[ant,:,3], marker=None, color="red")
                    plt.ylabel("Amp")
                    max_xx = np.max(good_xx)
                    max_yy = np.max(good_yy)
                    max_amp = 1.2*np.max([max_xx, max_yy])
#                    plt.ylim(0.0, 2)
                    # Plot phase
                    plt.subplot(NY*2, NX, y * 2 * NX + x + NX + 1) # maybe NY+1 instead of NX+1
                    plt.plot(channels, phases[ant,:,0], marker=None, color="black")
                    plt.plot(channels, phases[ant,:,3], marker=None, color="red")
                    plt.ylabel("Phases")
                    plt.ylim(-200.0, 200.0)
                ant += 1

        plt.tight_layout()
        if out_file == None:
            plt.show()
        else:
            plt.savefig(out_file)
        plt.close()


if __name__ == '__main__':
    # Beam is provided as a parameter
    ref_ant = 2
    flist = glob.glob("*.bin")
    flist.sort()
    for cal_file in flist:
        out_file = cal_file.replace(".bin", ".png")
        print("Processing %s -> %s" %(cal_file, out_file))
        bp = Bandpass()
        bp.load(cal_file)
        bp.plotGains(0, ref_ant, out_file)
