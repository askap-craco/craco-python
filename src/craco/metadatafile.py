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
from scipy import constants
from craft.craco import ant2bl, baseline_iter, to_uvw, uvw_to_array

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

def get_azel(ants):
    nants = len(ants.keys())
    antnames = sorted(ants.keys())
    azel = np.zeros((nants, 2)) # nants, 2 az,el
    for iant,antname in enumerate(antnames):
        u = np.array(ants[antname]['actual_azel'])[0:2]
        azel[iant,:] = u
        
    return azel

def get_par_angle(ants):
    nants = len(ants.keys())
    antnames = sorted(ants.keys())
    azel = np.zeros((nants)) # nants, 2 az,el
    for iant,antname in enumerate(antnames):
        u = ants[antname]['par_angle']
        azel[iant] = u
        
    return azel


def ts2time(ts):
    return Time(ts/1e6/3600/24, format='mjd', scale='tai')

def load_robust(f):
    '''
    If there's GZIP correuption you can use gzrecover to recover the file partially, then use some dumb assumptions to see if you can get JSON to parse it
    Assumes the input is not gzipped
    '''
    f.seek(0)
    nblocks = 0
    try:
        lines = []
        for line in f:
            if line.strip().replace(' ','') == '}{':
                lines.append('}')
                s = ''.join(lines)
                block = json.loads(s)
                yield block
                lines = ['{']
                nblocks += 1
            else:
                lines.append(line)
    except UnicodeDecodeError: # junk data - thats the end
        pass
    except json.JSONDecodeError: # never had this but it might happen and signal the end
        pass
            

def open_gz_or_plain(fname, mode='rt'):
    if fname.endswith('.gz'):
        fout = gzip.open(fname, mode)
    else:
        fout = open(fname, mode)

    return fout

class MetadataDummy:
    '''
    Dummy metadata object implemnets same interface as metadata file but isn't backed by a file. Just returns fixed values or nothing for UVWs
    '''
    def __init__(self, src_name='UNKNOWN', skycoord=None):
        
        self.nant = 36
        self.nbeam = 36
        if skycoord is None:
            skycoord = SkyCoord(ra=0,dec=0,unit='deg',
                                equinox='J2000',
                                frame='icrs')
        name = src_name
        data = {'name':name}
        data['ra'] = skycoord.ra.deg
        data['dec'] = skycoord.dec.deg
        data['epoch'] = 'J2000'
        data['skycoord'] = skycoord
        data['scan_times'] = []

        self.__source = data
        s = OrderedDict()
        s[name] = data
        self.__sources = s


    def sources(self, beamid):
        return self.__sources

    def source_index_at_time(self, mjd):
        return 0

    def source_at_time(self, beamid, mjd):
        return self.__source

    def uvw_at_time(self, mjd):
        return np.zeros((self.nant, self.nbeam, 3))

    def flags_at_time(self, mjd):
        return np.zeros(self.nant, dtype=bool)
    

class MetadataFile:
    def __init__(self, fname):
        self.fname = fname
        with open_gz_or_plain(fname, 'rt') as f:
            try:
                self.data = json.load(f)
            except json.JSONDecodeError: # known problem where I forgot to put it in a list with commas
                f.seek(0)
                s = f.read()
                s = '[' + s.replace('}{', '},{') + ']'
                self.data = json.loads(s)
            except UnicodeDecodeError:
                print('Trying robust load')
                self.data = list(load_robust(f))

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
        self.all_azel = np.array([get_azel(d['antennas']) for d in self.data])
        self.all_parangle = np.array([get_par_angle(d['antennas']) for d in self.data])
        
        self.mainflag = np.array([d['flagged'] for d in self.data]) # top level flag. probably useless
        self.antflags = np.array([[d['antennas'][a]['flagged'] for a in antnames] for d in self.data]) # antenna based flags
        self.ant_onsrc = np.array([[d['antennas'][a]['on_source'] for a in antnames] for d in self.data]) # whether antenna is on source

        # need to OR the flags together to get a correct total flag. See CRACO-132
        self.anyflag = self.mainflag[:,np.newaxis] | self.antflags | ~self.ant_onsrc
        
        self.times = Time(self.time_floats, format='mjd', scale='tai')
        
        self.uvw_interp = interp1d(self.times.value, self.all_uvw,  kind='linear', axis=0, bounds_error=True, copy=False)
        self.flag_interp = interp1d(self.times.value, self.anyflag, kind='previous', axis=0, bounds_error=True, copy=False)
        self.index_interp = interp1d(self.times.value, np.arange(len(self.data)), kind='previous', bounds_error=True, copy=False)

        
        self.antnames = antnames
        self._sources_b0 = self.sources(0) # just used for sourcenames

        # Keys for eeach etnry are:
        #dict_keys(['antenna_targets', 'antennas', 'beams_direction', 'beams_offsets', 'cycle_period', 'flagged', 'phase_direction', 'polangle', 'polmode', 'sbid', 'scan_id', 'schedulingblock_id', 'sky_frequency', 'target_direction', 'target_name', 'timestamp'])

    def source_name_at_time(self, time : Time):
        srcname = self.data_at_time(time)['target_name']
        return srcname

    def source_index_at_time(self, time : Time):
        srcname = self.source_name_at_time(time)
        # Looup index by stupid loop:
        for i, k in enumerate(self._sources_b0.keys()):
            if k == srcname:
                break
        
        sourceidx = i
        return sourceidx

    def source_at_time(self, beam: int,  time : Time):
        '''
        Return the source dictionary being tracked at the given time and beam
        '''
        sourcename = self.source_name_at_time(time)
        source = self.sources(beam)[sourcename]
        
        return source

    def data_index_at_time(self, time : Time):
        idx = int(self.index_interp(time.tai.mjd))
        return idx

    def data_at_time(self, time : Time):
        return self.data[self.data_index_at_time(time)]

    def flags_at_time(self, time : Time):
        '''
        Returns the flags applicable at the time
        :returns: np array of length nant with True if flagged
        '''
        flags = self.flag_interp(time.tai.mjd) == 1.0
        return flags

    def uvw_at_time(self, time : Time):
        '''
        Returns UVW for each antenna in meters. Interpolated to the given time 

        :time: MJD time 
        :returns: np.array shape [NANT, NBEAM, 3] type float
        '''
        return self.uvw_interp(time.tai.mjd)


    def baselines_at_time(self, time : Time, valid_ants_0based, beamid):
        # UVW is a np array shape [nant, 3]
        '''
        Returns a diictionary of ['UU','VV','WW'] np.voids
        key is baselineid (float)
        Units are ... seconds?
        'UU','VV','WW'
        Units are
        '''

        uvw = self.uvw_at_time(time)[valid_ants_0based, beamid, :] / constants.c
        bluvws = {}
 
        for blinfo in baseline_iter(valid_ants_0based):
            bluvw = uvw[blinfo.ia1, :] - uvw[blinfo.ia2, :]
            assert np.all(bluvw != 0), f'UVWs were zero for {blinfo}={bluvw}'
            d = np.array(bluvw, dtype=uvw_dtype)[0]
            bluvws[float(blinfo.blid)] = d

        return bluvws

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
    parser.add_argument('--plot-beam', help='Plot UVW for given beam', type=int)
    parser.add_argument('--xtype', help='Type of data for x data of plot', choices=('mjd','utc','idx'), default='mjd')
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

        fig, axs = pylab.subplots(8,1, sharex=True)
        fig.suptitle(f)
        if values.plot_beam is not None:
            beam = values.plot_beam
            mjds_times = mf.times
            mjds = mjds_times.tai.value
            mainflag = mf.mainflag

            uvws = mf.uvw_at_time(mjds_times) # (time, nant, nbeam, 3)
            nant = uvws.shape[1]
            bluvws = []

            for iant1 in range(nant):
                for iant2 in range(iant1+1,nant):
                    bluvws.append(uvws[:,iant1,:,:] - uvws[:,iant2,:,:])

            bluvws = np.array(bluvws)
            xtype = values.xtype
            if xtype == 'mjd':
                x = (mjds - mjds[0])*24*60*60
                xlbl = f'seconds after MJD={mjds[0]:0.5f}'
            elif xtype == 'utc':
                x = mf.times.utc.datetime
                xlbl = 'UTC'
            else:
                x = np.arange(len(mjds))
                xlbl = 'Metadata dump number'
            uvws.shape
            for i,lbl in enumerate(('U','V','W')):
                axs[i].plot(x, uvws[:,:,beam, i])
                axs[i].set_ylabel(lbl)

            flags = mf.antflags
            axs[3].plot(x, flags)
            axs[3].plot(x, mainflag, label='Main array flag')

            axs[3].set_ylabel('Flagged == 1')
            axs[3].legend()

            axs[4].plot(x, mf.ant_onsrc)
            axs[4].set_ylabel('onsrc')

            axs[5].plot(x, mf.all_azel[...,0])
            axs[5].set_ylabel('Aximuth')

            axs[6].plot(x, mf.all_azel[...,1])
            axs[6].set_ylabel('Elevation')

            axs[7].plot(x, mf.all_parangle)
            axs[7].set_xlabel(xlbl)
            axs[7].set_ylabel('Parang')


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
