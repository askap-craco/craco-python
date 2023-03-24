#!/usr/bin/env python
"""
Parses a metadata file capture by saving the ASKAP metadata as a gzipped json

Copyright (C) CSIRO 2022
"""
import numpy as np
from scipy.interpolate import interp1d
import os
import sys
import logging
import gzip
import json
from astropy.time import Time
from collections import OrderedDict
from astropy.coordinates import SkyCoord
from astropy import units as u
import pylab

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def get_uvw(ants):
    '''
    Returns a numpy array shape [nants, nbeams, 3] of UVW values
    units - probably meters
    '''
    nants = len(ants.keys())
    antnames = sorted(ants.keys())
    uvw = np.zeros((nants, 36, 3)) # 36 beams - 3 axes
    for iant,antname in enumerate(antnames):
        u = np.array(ants[antname]['uvw']).reshape(36,3)    
        uvw[iant,:,:] = u
        
    return uvw

def ts2time(ts):
    return Time(ts/1e6/3600/24, format='mjd', scale='tai')


class MetadataFile:
    def __init__(self, fname):
        self.fname = fname
        with gzip.open(fname, 'rt') as f:
            try:
                self.data = json.load(f)
            except json.JSONDecodeError: # known problem where I forgot to put it in a list with commas
                f.seek(0)
                s = f.read()
                s = '[' + s.replace('}{', '},{') + ']'
                self.data = json.loads(s)

        # the final packet contains a timestamp of zero, which we want to get rid of
        if self.data[-1]['timestamp'] == 0:
            self.data = self.data[:-1]

        d0 = self.data[0]
        self.d0 = d0
        antnames = sorted(d0['antennas'])
        # timestamps according to Max Voronkov, are
        # microseconds since MJD in TAI frame, at the start of the correlator integration cycle.
        self.time_floats = np.array( [d['timestamp']/1e6/3600/24 for d in self.data])
        self.all_uvw = np.array([get_uvw(d['antennas']) for d in self.data])
        self.antflags = np.array([[d['antennas'][a]['flagged'] for a in antnames] for d in self.data])
        self.times = Time(self.time_floats, format='mjd', scale='tai')
        self.uvw_interp = interp1d(self.times.value, self.all_uvw,  kind='linear', axis=0, bounds_error=True, copy=False)
        self.flag_interp = interp1d(self.times.value, self.antflags, kind='previous', axis=0, bounds_error=True, copy=False)
        self.index_interp = interp1d(self.times.value, np.arange(len(self.data)), kind='previous', bounds_error=True, copy=False)
        self.antnames = antnames
        self._sources_b0 = self.sources(0) # just used for sourcenames

        # Keys for eeach etnry are:
        #dict_keys(['antenna_targets', 'antennas', 'beams_direction', 'beams_offsets', 'cycle_period', 'flagged', 'phase_direction', 'polangle', 'polmode', 'sbid', 'scan_id', 'schedulingblock_id', 'sky_frequency', 'target_direction', 'target_name', 'timestamp'])

    def source_name_at_time(self, time : float):
        srcname = self.data_at_time(time)['target_name']
        return srcname

    def source_index_at_time(self, time : float):
        srcname = self.source_name_at_time(time)
        # Looup index by stupid loop:
        for i, k in enumerate(self._sources_b0.keys()):
            if k == srcname:
                break
        
        sourceidx = i
        return sourceidx

    def data_index_at_time(self, time : float):
        idx = int(self.index_interp(time))
        return idx

    def data_at_time(self, time : float):
        return self.data[self.data_index_at_time(time)]

    def flags_at_time(self, time : float):
        '''
        Returns the flags applicable at the time
        :returns: np array of length nant with True if flagged
        '''
        flags = self.flag_interp(time) == 1.0
        return flags

    def uvw_at_time(self, time : float):
        '''
        Returns UVW for each antenna in meters. Interpolated to the given time 

        :time: MJD time (float)
        :returns: np.array shape [NANT, NBEAM, 3] type float
        '''
        return self.uvw_interp(time)

    def sources(self, beam):
        '''
        Returns an ordered dictionary of sources
        Each source is a key of the source name,
        The value is a dictionary containing the following.
        keys are 
        ra: degrees
        dec: degrees
        epoch: 'J2000'
        name: string
        skycoord: Astropy SkyCoord with that stuff filled in

        Assumes that the first time it sees anew source name, teh value in the beam direciton
        is the phase center

        :beam: Beam number for source directions
        
        '''

        assert 0<= beam < 36, f'Invalid beam {beam}'
        sources = OrderedDict()
        last_source_name = None
        for d in self.data:
            name = d['target_name']
            time = ts2time(d['timestamp'])
            if name not in sources.keys():
                data = {'name':name}
                beamdirs = d['beams_direction']
                data['ra'] = beamdirs[beam][0]
                data['dec'] = beamdirs[beam][1]
                data['epoch'] = beamdirs[beam][2]
                data['skycoord'] = SkyCoord(ra=beamdirs[beam][0],
                                            dec=beamdirs[beam][1],
                                            unit='deg',
                                            equinox=beamdirs[beam][2],
                                            frame='icrs')
                data['scan_times'] = []
                sources[name] = data

            if last_source_name == name:
                data['scan_times'][-1][1] = time
            else:
                data['scan_times'].append([time, None])

            last_source_name = name

        return sources
                

    @property
    def sbid(self):
        '''
        Returns an integer of the schedblock id
        '''
        return self.d0['sbid']

    def __str__(self):
        s = f'''Metadata for {self.fname} is SB{self.sbid} contains {len(self.data)} packets from {self.times[0].iso} to {self.times[-1].iso} = {self.times[0].mjd}-{self.times[-1].mjd} duration={(self.times[-1] - self.times[0]).datetime} d:m:s for {len(self.antnames)} antennas {self.antnames} and has sources {self.sources(0)}'''
        return s
        

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('--plotuv', help='Plot UVW for given beam', type=int)
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    for f in values.files:
        mf = MetadataFile(f)
        print(mf)
        for b in range(36):
            for name, data in mf.sources(b).items():
                print('beam', b, name, data['skycoord'].to_string('hmsdms'), data['scan_times'][0][0].iso, data['scan_times'][0][1].iso)

        fig, axs = pylab.subplots(4,1, sharex=True)
        if values.plotuv is not None:
            beam = values.plotuv
            mjds = mf.time_floats
            uvws = mf.uvw_at_time(mjds) # (time, nant, nbeam, 3)
            nant = uvws.shape[1]
            bluvws = []

            for iant1 in range(nant):
                for iant2 in range(iant1+1,nant):
                    bluvws.append(uvws[:,iant1,:,:] - uvws[:,iant2,:,:])

            bluvws = np.array(bluvws)
            uvws.shape
            for i,lbl in enumerate(('U','V','W')):
                axs[i].plot(uvws[:,:,beam, i])
                axs[i].set_ylabel(lbl)

            flags = mf.flags_at_time(mjds)
            axs[3].plot(flags)
            axs[3].set_ylabel('Flagged == 1')
            axs[3].set_xlabel('Metadata dump')
            

            print(bluvws.shape)
            fig2,axs = pylab.subplots(1,2)
            for ibl in range(bluvws.shape[0]):
                axs[0].plot(bluvws[ibl,:,beam,0], bluvws[ibl,:,beam,1],'o')
                for i in range(3):
                    axs[1].plot(bluvws[ibl,:,beam,i])

            axs[0].set_xlabel('UU')
            axs[0].set_ylabel('VV')
            axs[1].set_ylabel('UU,VV,WW')
            axs[1].set_xlabel('Sample')

    
            pylab.show()
            
            
    

if __name__ == '__main__':
    _main()
