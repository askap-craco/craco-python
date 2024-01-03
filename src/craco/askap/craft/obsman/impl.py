# Copyright (c) 2016 CSIRO
# Australia Telescope National Facility (ATNF)
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# PO Box 76, Epping NSW 1710, Australia
# atnf-enquiries@csiro.au
#
# This file is part of the ASKAP software distribution.
#
# The ASKAP software distribution is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the License
# or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA.
#

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

"""Implementation of the Ice Interface"""

import os
import re
import subprocess
import time
from collections import namedtuple, OrderedDict
import askap.logging as logging
from . import logger

from askap.slice import CraftService, CommonTypes
from askap.interfaces import Direction
from askap.interfaces import CoordSys
from askap.interfaces.craft import ICraftObsService
import askap.interfaces as iceint
from askap.opl.services.schedblock import SchedulingBlock as SchedulingBlockSvc
from askap.epics.subsystems import AdeBmf
from askap.parset import ParameterSet, sub_parset, slice_parset
import askap.parset
from askap.craft.footprint_class import Footprint
from askap.craft import crafthdr
from askap.craft.freqconfig import FreqConfig
import askap.craft.summarise_scans as summarise_scans
from askap.iceutils import get_service_object
import askap.craft.leapseconds as leapseconds
import askap.opl.data
import numpy as np
import atexit
import datetime
import shutil
import sys
import itertools
from astropy.coordinates import SkyCoord
from .version import ASKAP_VERSION


DATA_DIR = os.environ.get('CRAFT_DATA_DIR', '/data/ban115/craft/')

# sample rate of 1 MHz channels, oversampled by 32/27
FSAMP = 1e6*32./27. 
        
ant_data = namedtuple('Antdata', 'flagged on_source actual_pol actual_azel actual_radec par_angle')

# can use OPL.common.current.askap.opl.services.beams.y/schedblock.py
# use askap.opl.common
#import askap.opl.common.setup as setup
#see setup.init_procedure()

ALL_CARDS = np.arange(1, 8)

#ALL_ANTENNAS = [2,3,4,5,6,10,12,13,14,19,16,24,26,27,28,30]
ALL_ANTENNAS = [2,3,4,5,6,10,12,13,14,19,24,26,27,28,30]
ALL_ANTENNAS = [2,4,5,6,10,12,13,14,19,24,26,27 ,28]
ALL_ANTENNAS = [a for a in range(1, 37)]

def get_antno(antname):
    if isinstance(antname, int):
        return antname
    else:
        return int(antname.replace('ant','').replace('ak','').replace('co',''))


def floatnone(s):
    try:
        v = float(s)
    except:
        v  = None

    return v

def get_parset_array(s):
    bits = s.replace('[','').replace(']','').split(',')
    bits = [b.strip() for b in bits]
    return bits
    
class CraftAntennaManager(object):
    def __init__(self, antname, mgr, iant):
        self.mgr = mgr
        self.array = self.mgr.site
        #self.antname = '{}{:02d}'.format(self.array, self.antno)
        self.antname = antname
        self.antno = get_antno(antname)
        assert 1 <= self.antno <= 36
        self.num_bf = len(ALL_CARDS)
        self.freqs = None
        self.freqconfig = None
        self.iant = iant
        self.capturing = False
        self.flagged = True
        self.capture_no = -1
        self.capture_cmd = None
        self.hdrfile_name = None

        #self.bf = AdeBmf(prefix=self.antname, config=None, parset=None)

    def update_freqmap(self):
        '''Go to the beamformer IOC for this antenna and update the freqency map'''
        #self.freqconfig = self.mgr.load_freqconfig(self.antname)
        self.freqconfig = FreqConfig.load_from_pv(self.antname, ALL_CARDS)
        return self.freqconfig

    def open_scan(self, scanid, timestamp, data):
        '''Open a scan.
        - Creates the target directory
        - Calls setupCraft on the IOC
       '''
        self.target = self.mgr.get_scan_srcname(scanid, self.antno)
        self.field_name = self.mgr.get_target_parameter(self.target, 'field_name')
        self.field_direction = self.mgr.get_target_parameter(self.target, 'field_direction')
        self.pol_axis = self.mgr.get_target_parameter(self.target, 'pol_axis')
        self.scandir = self.mgr._mkdir(self.mgr.scandir, self.antname)
        self.intent = self.mgr.get_scan_intent(self.target)
        self.capturing = False
        self.flagged = True
        self.scan_is_open = False
        
    def close_scan(self, timestamp, data):
        logging.info('Closing scan')
        self.capture_this_scan = False
        self.terminate_capture(timestamp, data)
    
    def update_scan(self, timestamp, data):
        allflagged = data['flagged'].value
        
        # extract antenna-specific data from the data dirtionary
        adat = [data['%s.%s'% (self.antname, f)].value for f in ant_data._fields]
        d = ant_data._make(adat)
        was_flagged = self.flagged
        is_flagged = allflagged or d.flagged or not d.on_source

        logging.debug('%s d.flagged %s d.on_source %s allflagged %s was_flagged %s is_flagged %s',
                      self.antname, d.flagged, d.on_source, allflagged, was_flagged, is_flagged)

        if self.intent is None or self.intent == 'UNKNOWN':
            logging.warning('Not starting scan as scan intent not specified or unknown %s', self.intent)
            is_flagged = True

        if is_flagged:
            if self.capturing:
                logging.info('%s terminating capture as flagged = True. was_flagged %s is_flagged %s allflagged %s d.flagged = %s on_source = %s', self.antname, was_flagged, is_flagged, allflagged, d.flagged, d.on_source)
                self.terminate_capture(timestamp, data)
        else:  # not currently flagged
            if was_flagged: # was flagged before - now we're open. Yay!
                self.open_capture(timestamp, data, d)
            else: # not flagged now, not flagged before. Just keep on keeping on.
                self.continue_capture(timestamp, data, d)

        self.flagged = is_flagged

    def open_capture(self, timestamp, data, antdata):
        capturing = self._check_capture_alive()
        if capturing:
            self._log('Postponing open capture')
            return


        self.capture_no += 1
        # Set capture_no to 0 for backwards compatibility
        self.capture_name = 'C%03d' % self.capture_no

        hdr = self.get_header(timestamp, data, antdata)
        self.hdr = hdr
        if hdr is None: # could not make header. Probably failed freqconfig
            return

        self.capture_dir = self.mgr._mkdir(self.scandir, self.capture_name)
        # write header

        hdrfile_name = '%s_SB%05d_SC%04d_C%03d.hdr' % (self.antname, self.mgr.sbid, self.mgr.scanid, self.capture_no)
        hdrfile_name = os.path.join(self.capture_dir, hdrfile_name)
        self.hdrfile_name = hdrfile_name
        with open(hdrfile_name, 'w') as hfile:
            hfile.write(str(hdr))

        # kickoff subprocess for data capture
        # TODO: find a capture command that isn't hard coded
        askap_root = os.environ.get('ASKAP_ROOT', '.')
        cmd = os.path.join(askap_root, 'Code/Components/craft/akcapture/current/apps/craft_capture.sh %s ' % (os.path.basename(hdrfile_name)))

        env = {}
        for k, v in list(hdr.items()):
            env[k] = str(v[0])
            
        env.update(os.environ)

        self.capturing = True
        assert self.capture_cmd is None
        self.capture_cmd = subprocess.Popen(cmd.split(), cwd=self.capture_dir, env=env)
        self._log('Capture command %s started in %s with pid %d' % (cmd, self.capture_dir, self.capture_cmd.pid))
        # wait a few milliseconds and see if it's broken
        time.sleep(0.01)
        
        self._check_capture_alive()


    def _check_capture_alive(self):
        if self.capture_cmd is None:
            self.capturing = False
        else:
            if self.capture_cmd.poll() is None:
                self.capturing = True
            else:
                self._log('Capture command %s died with return code %d' % \
                          (self.capture_cmd, self.capture_cmd.returncode))
                self.save_capture_metadata()
                self.capturing = False
                self.capture_cmd = None

        return self.capturing

    def terminate_capture(self, timestamp, data):
        if self.capture_cmd is not None:
            self._log('Terminating capture')
            try:
                self.capture_cmd.terminate()
            except:
                pass

            self._log('Capture command terimated with return code %s' % self.capture_cmd.returncode)
            self.flagged = True
            self._check_capture_alive()


    def wait_for_end(self):
        if self.capture_cmd is not None:
            self.capture_cmd.wait()
            self.terminate_capture(None,None)

    def save_capture_metadata(self, scanfile=None):
        if scanfile is None:
            metadata_dir = self.mgr._mkdir(DATA_DIR, 'metadata')
            scanfile_name = os.path.join(metadata_dir, 'SB{:04d}-{}-{}-C{}.json'.format(self.mgr.sbid, self.mgr.scanname,
                                                                                       self.antname, self.capture_no))

        hdrpath = self.hdrfile_name
        if hdrpath is not None:
            _id, summary = summarise_scans.summarise_scan(hdrpath)
            summarise_scans.write_meta_files(summary)
                
            try:
                summarise_scans.index_summary(_id, summary, url=os.getenv('CRAFT_ELASTIC_URL'))
                #logging.exception('CRAFT Elastic data base is down')
            except:
                logging.exception('Couldnt directly index metadata')
                scanfile = open(scanfile_name, 'w')
                summarise_scans.write_json(_id, summary, scanfile)
                scanfile.close()
                
        
    def continue_capture(self, timestamp, data, antdata):
        # Disable for now: - Ryan doesn't want to have more than one capture in a scan
        return self._check_capture_alive()
        
    def get_bfaddr(self,num_bf=7):
        '''Returns a list of beamformer IP addreses for this antenna'''
        assert num_bf > 0

        addr = ['10.2.%d.%d' % (self.antno, cardno+1) for cardno in range(num_bf)]

        return addr
            
    def _log(self, s):
        self.mgr._log('AntMgr %s: %s' % (self.antname, s))


    _debug = _log

    def get_freqconfig(self):
        if self.freqconfig is None:
            self.update_freqmap()

        return self.freqconfig

    def get_header(self, timestamp, data, antdata):

        # Get current footprint
        ant_ra = antdata.actual_radec.coord1 # Degrees?
        ant_dec = antdata.actual_radec.coord2 # Degrees?

        # TODO: assert antdata.actual_radec.coordsys = J2000
        pol_type, pol_angle = self.pol_axis
        pol_angle = float(pol_angle)
        field_rastr, field_decstr, field_epochstr = self.field_direction



        beams_direction = data['beams_direction'].value
        beams_dir_ra = [b.coord1 for b in beams_direction]
        beams_dir_dec = [b.coord2 for b in beams_direction]
        

 
        #assert field_epochstr == 'J2000', 'Unknown field epoch: {}'.format(field_epochstr)
        
        if field_epochstr == 'J2000':
            print('Using FIELD_POS for now')
            field_pos = SkyCoord(field_rastr + ' ' + field_decstr, unit=('hourangle','deg'))
        else:
            print('IGNORING FIELD_POS for now')
            field_pos= SkyCoord(ant_ra, ant_dec, unit=('deg','deg'))
        
        ant_pos = SkyCoord(ant_ra, ant_dec, unit=('deg','deg'))
        #sep_arcmin =0
        sep_arcmin = ant_pos.separation(field_pos).arcmin
        # RMS:   Changed this to a large value to allow for offset scans
        
        if sep_arcmin > 3000:
            self._log('WARNING WARNING Abandoning scan. Flags OK but separation between requested position {} and actual position {} is large: {} arcmin'.format(field_pos, ant_pos, sep_arcmin))
            return None
                             
        if pol_type == 'pa_fixed':
            # if pol_type is pa_fixed, then the angle is the position angle
            position_angle = pol_angle
            fp = self.mgr.get_footprint(self.target, (field_pos.ra.deg, field_pos.dec.deg), position_angle)
            fp_pitch, fp_shape, fp_pa = self.mgr.get_footprint_pitch_shape(self.target)
            ras =[np.degrees(p.ra) for p in fp.positions]
            decs = [np.degrees(p.dec) for p in fp.positions]
            nbeam = len(ras)

            # write NaN values to headers it beams aren't specified
            ras.extend([-99]*(36 - nbeam))
            decs.extend([-99]*(36 - nbeam))
            nbeam = len(ras)
            assert nbeam == 36
            ras = np.hstack([ras, ras]).flatten()
            decs = np.hstack([decs, decs]).flatten()
        else:
            ras = []
            decs = []
            fp = None
            fp_pitch = -1
            fp_shape = 'INVALID_POL_AXIS_NOT_PA_FIXED'
            fp_pa = 0
            nbeam = 0

        # assign dada ringbuffer key
        # Adjacent dada keys seems to delete each other by accident, through some mechanism I don't understand
        # Multiplying doing dada_key_num = 0xa000 + self.antno*4 seems to work reliably.
        dada_key_num = 0xa000 + self.antno*4 
        dada_key = '%04x' % dada_key_num

        # Wre goign to assume here that the beam polarisation goes: XX YY XX YY XX YY
        # And that a 'beam' is a 'dual pol' entity. Ie..g NPOL=2 always and NBEAM <=36
        # TODO: SEe if we can work out the polarisation info from the weights.
        pols = ['XX','YY']
        beam_pol = pols*(nbeam)
        beam_id = np.arange(nbeam)
        npol = len(pols)
        assert nbeam >=1 and nbeam <= 36

        # Here we assume we're not doign incoherent sum. 
        nant = len(self.mgr.antenna_managers)
        
        #obid = '%06d%03d%03d' % (self.mgr.sbid, self.mgr.scanid, self.capture_no)
        obsid = self.capture_no + self.mgr.scanid*1000 + self.mgr.sbid*1000000

        bfaddr = self.get_bfaddr()
        try:
            freq_config = self.get_freqconfig()
        except:
            self._log('Unable to load freqconfig for antenna {}'.format(self.antname))
            raise
            
        # resolution is the number of bytes in a single 'chunk' - dada_dbdisk uses it
        # here we define 'chunk' as a complete set of all channels, all integrations in
        # a craft packet. Each data element is a 4 byte float
        nbit = 32
        resolution = freq_config.nchan*self.mgr.get_int_cycles(self.target) * nbit/8
        
        # bytes_per_second - also wanted by dada_dbdisk
        bytes_per_second = nbit/8*freq_config.nchan/self.mgr.get_tsamp(self.target)
        
        # block_cycles=21*4 magic number makes dbsplit write data in 4096 byte chunks = page size
        block_cycles = 21*4

        # receive port
        iant = (ALL_ANTENNAS.index(self.antno))
        rx_port = 5000 + int(self.antno)

        # assign antenna to core -
        # according to numactl -H
        # CPU0-5 on numa-node 0
        # CPU6-11 18-23 are on numa-node 1 and 
        # eth2 is on link L1 Net L#11 which on numa node 1, socket 1 see hwloc-info -v
        # I don't know whether its bound to a particular core on numa node 1 or not
        # interrupts are going to CPU 6 -- see /proc/interrupts and assign_antenna_flows.sh
        cpuid = ((iant * 2) + 1) % 16
        #cpuid = (self.antno % 4) + 18
        intent = self.mgr.get_scan_intent(self.target)
        phasedir = data['phase_direction'].value
        targdir = data['target_direction'].value

        hdr = crafthdr.DadaHeader(add_comment=False)
        hdr += ('HDR_VERSION', 1.0, 'Version of the header')
        hdr += ('HDR_SIZE', 16384, 'Size of header')
        hdr += ('TELESCOPE', 'ASKAP', 'Telescope name')
        hdr += ('INSTRUMENT', 'CRAFT', 'Name of instrument')
        hdr += ('SOURCE', self.target, 'Source name')
        hdr += ('TARGET', self.field_name, 'Target')
        hdr += ('FIELD_NAME', self.field_name, 'Field name from parset')
        hdr += ('FIELD_DIRECTION',str(self.field_direction), 'Field direction from parset')
        hdr += ('OBS_ID', '%12d'%obsid, 'Observation ID = captureno + scanid*1000 + sbid*1000000')
        hdr += ('SBID', self.mgr.sbid, 'ASKAP Schedblock ID')
        hdr += ('SB_ALIAS', self.mgr.sbinfo.alias, 'Schedblock alias')
        hdr += ('SB_TEMPLATE', self.mgr.sbinfo.templateName, 'Schedblock template name')
        hdr += ('SB_TEMPLATE_VERSION', self.mgr.sbinfo.templateVersion, 'Schedblock template name version')
        hdr += ('SB_START_TIME', self.mgr.sbinfo.startTime, 'SB start time')
        hdr += ('SB_OWNER', self.mgr.sbinfo.owner, 'SB owner')
        hdr += ('FOOTPRINT_NAME', fp_shape, 'Footprint name')
        hdr += ('FOOTPRINT_PITCH', fp_pitch, 'Footprint pitch (deg)')
        hdr += ('FOOTPRINT_ROTATION', fp_pa, 'Footprint rotation (deg)')
        hdr += ('SCANID', self.mgr.scanid, 'ASKAP scan ID')
        hdr += ('SCANNAME', self.mgr.scanname, 'Scan Name')
        hdr += ('SCAN_INTENT', intent, 'Scan intent')
        hdr += ('SUBARRAYS', self.mgr.all_subarrays_str, 'All subarrays. Antennas within a sbarray sare separted by \'-\'. Subarrays separated by spaced')
        hdr += ('THIS_SUBARRAY', self.mgr.subarrays_by_src_str[self.target], 'Subarray that this antenna is in')
        hdr += ('CAPTUREID', self.capture_no, 'Capture ID - within a scan')
        hdr += ('CAPTURE_NAME', self.capture_name, 'Capture Name')
        hdr += ('FREQ',freq_config.freq, 'Observing frequency of first channel (MHz)')
        hdr += ('BW', freq_config.bw, 'Channel bandwidth (MHz)')
        hdr += ('NPOL', npol, 'Number of polarisations')
        hdr += ('NBEAM', nbeam, 'Number of beams')
        hdr += ('NCHAN', freq_config.nchan_span, 'Number of channels')
        hdr += ('NBIT', nbit, 'Number of bits')
        hdr += ('DTYPE', '<f4', 'Data type (see numpy dtype for meaning)')
        hdr += ('DORDER', 'TFBP', 'Data ordering. Last is fastest. T=Time, B=Beam, P=Polarsation, F=Freq')
        hdr += ('TSAMP', self.mgr.get_tsamp(self.target), 'Sampling time')
        hdr += ('RA', field_pos.ra.deg, 'Field Right Ascension of pointing (J2000 - degrees)')
        hdr += ('DEC', field_pos.dec.deg, 'Field Declination of pointing (J2000 - degrees)')
        hdr += ('FIELD_RA', field_pos.ra.deg, 'Field Right Ascension of pointing (J2000 - degrees)')
        hdr += ('FIELD_DEC', field_pos.dec.deg, 'Field Declination of pointing (J2000 - degrees)')
        hdr += ('FIELD_EPOCH', field_epochstr, 'Field Declination of pointing (J2000 - degrees)')
        hdr += ('ANT_RA', ant_ra, 'Antenna Right Ascension of pointing (J2000 - degrees)')
        hdr += ('ANT_DEC', ant_dec, 'AntennaDeclination of pointing (J2000 - degrees)')
        hdr += ('ANT_POL', antdata.actual_pol, 'Antenna polarisation angle (degrees)')
        hdr += ('POINT_SEP', sep_arcmin, 'Separation between acutal and requested positions at thebeginning of the scan')
        hdr += ('TARGET_POL_ANGLE', pol_angle, 'Pol angle of target - from parset')
        hdr += ('TARGET_POL_TYPE', pol_type, 'Pol angle tracking type - from parset')
        hdr += ('PAR_ANGLE', antdata.par_angle, 'Antenna paralactic angle (degrees)')
        hdr += ('AZ_START', antdata.actual_azel.coord1, 'Azimuth a when header written (degrees)')
        hdr += ('EL_START', antdata.actual_azel.coord2, 'Elevation when header written (degrees)')
        hdr += ('BEAM_RA', ','.join(map(str, ras)), 'Right Ascensions of all beams (J2000 - degrees)')
        hdr += ('BEAM_DEC', ','.join(map(str, decs)), 'Declinations of all beams (J2000 - degrees)')
        hdr += ('BEAM_POL', ','.join(map(str, beam_pol)), 'Polarisation mnemonic for beams. e.g. XX or YY')
        hdr += ('BEAM_ID', ','.join(map(str, beam_id)), 'Beam numbers. 0 based')
        hdr += ('BEAMS_DIR_RA', ','.join(map(str, beams_dir_ra)), 'Beam RA from metadata')
        hdr += ('BEAMS_DIR_DEC', ','.join(map(str, beams_dir_dec)), 'Beam Dec from metadata')
        hdr += ('ARRAY', self.array, 'Array name. either ak or co')
        hdr += ('ANTNO', self.antno, 'Antenna number')
        hdr += ('ANTNAME', self.antname, 'Antenna name')
        hdr += ('INT_CYCLES', self.mgr.get_int_cycles(self.target), 'Number of integrations per beamformer packet')
        hdr += ('INT_TIME', self.mgr.get_int_time(self.target), 'Number of samples per integration')
        hdr += ('CRAFT_MODE', self.mgr.get_mode(self.target), 'CRAFT mode')
        hdr += ('RECORD_MODE', self.mgr.get_record_mode(self.target), 'craft recording mode')
        hdr += ('DOWNLOAD_BEAMS', self.mgr.get_download_beams(self.target), 'specify which beams to download')
        hdr += ('DOWNLOAD_CARDS', self.mgr.get_download_cards(self.target), 'specify which cards to download')
        hdr += ('OBS_OVERLAP', 0, 'The amount by which neighbouring fiels overlap - See PSRDADA')
        hdr += ('OBS_OFFSET', 0, 'The number of byte from the start of the observation - See PSRDADA')
        hdr += ('RESOLUTION', resolution, 'The number of bytes in a single packet/chunk? - see dada_dbdisk.c???')
        hdr += ('BYTES_PER_SECOND', bytes_per_second, 'Number of bytes per second (useful for dada_dbdisk)')
        hdr += ('FILE_SIZE', 4*16*1024*1024*1024, 'Number bytes to write to a file before opening another one')
        hdr += ('RESCALE_INTERVAL_SAMPS', 4096, 'Number of samples between rescale updates')
        hdr += ('BLOCK_CYCLES', block_cycles, 'Number of intergration  cycles per DADA block')
        hdr += ('BLOCK_SIZE', block_cycles*nbeam*npol*freq_config.nchan_span*4, 'Number of bytes per DADA block')
        hdr += ('OBSMAN_VERSION', ASKAP_VERSION, 'Version of obsman that generated this file')
        hdr += ('CPUID', cpuid, 'CPU to bind to')
        hdr += ('DADA_KEY', dada_key, 'DADA key on output of udpdb')
        hdr += ('FREQ_CONFIG', str(freq_config), 'Description of frequency configuration')
        hdr += ('RX_PORT', str(rx_port), 'UDP Port number to bind to')
        hdr += ('HDR_UTC', datetime.datetime.utcnow().isoformat(), 'UT when header was created (ISO format)')
        hdr += ('SCAN_OPEN_UTC', self.mgr.scan_open_time, 'Time when scan was opened')
        hdr += ('DTAI_UTC', int(self.mgr.dTAI_UTC.total_seconds()), 'Number of seconds that must be added to UTC to compute TAI=+37 on since 1 Jan 2017')
        hdr += ('CORRMODE', data['corrmode'].value, 'Metadata corr mode')
        #hdr += ('CYCLE_PERIOD', data['cycle_period'], 'Metadata cycle period')
        hdr += ('PHASE_DIRECTION_COORD1', phasedir.coord1, 'Metadata phase direction coord 1')
        hdr += ('PHASE_DIRECTION_COORD2', phasedir.coord2, 'Metadata phase direction coord 2')
        hdr += ('PHASE_DIRECTION_SYS', phasedir.sys, 'Metadata phase direction coord sys')
        hdr += ('SKY_FREQUENCY', data['sky_frequency'].value, 'Metadata sky frequency')
        hdr += ('TARGET_NAME', data['target_name'].value, 'Metadata target name')
        hdr += ('TARGET_DIRECTION_COORD1', targdir.coord1, 'Metadata target direction coord 1')
        hdr += ('TARGET_DIRECTION_COORD2', targdir.coord2, 'Metadata target direction coord 2')
        hdr += ('TARGET_DIRECTION_SYS', targdir.sys, 'Metadata target direction coord sys')
        hdr += ('NUM_BEAMFORMERS', len(bfaddr), 'Number of beamformers')

        #chanmap = np.arange(nchan)
        #freqmap = chanmap*bw + fstart + nchan*bw*ibf
        
        for ibf, (bf, chanmap, freqmap) in enumerate(zip(bfaddr, freq_config.chanmaps, freq_config.freqmaps)):
            hdr += ('BEAMFORMER%d_ADDR'%ibf, bf, 'Address of beamformer %d' % ibf)
            hdr += ('BEAMFORMER%d_CHANMAP'%ibf, ','.join(map(str, chanmap)), 'Channel map for beamformer %d' % ibf)
            hdr += ('BEAMFORMER%d_FREQMAP'%ibf, ','.join(map(str, freqmap)), 'Frequency map for beamformer %d' % ibf)

        #hdr.reset_hdr_size()

        return hdr


# noinspection PyMethodMayBeStatic
class CraftManagerImpl(ICraftObsService):
    """The implementation of the :class:`ICraftObsService` interface.

    .. literalinclude:: ../../../../Interfaces/slice/current/craft.ice
        :language: cpp
        :linenos:
        :lines: 28-70

    """
    def __init__(self, comm, values):
        self.values = values
        self._comm = comm
        if values.extra_parset is None:
            self._extra_parset = ParameterSet()
        else:
            assert os.path.isfile(values.extra_parset)
            self._extra_parset = ParameterSet(values.extra_parset)

        self.sbid = -99
        self.scanid = -99
        self.skyfreq = 949.
        self.capture_this_scan = False
        self.capturing = False
        self.antenna_managers = OrderedDict()
        self.rootdir = DATA_DIR
        self._mkdir(self.rootdir)
        #atexit.register(self.abortObs)
        self.config_dir = './config/'
        self._setup_sbservice()
        self.sbdir = None
        self._scan_open_process = None
        self._scan_open_process_return_code = None
        if values.search_antennas is not None:
            self._log('Restricting antennas to {}'.format(values.search_antennas))

    def _setup_sbservice(self):
        self._sb = get_service_object(self._comm,
                    "SchedulingBlockService@DataServiceAdapter",
                    iceint.schedblock.ISchedulingBlockServicePrx)

    def _mkdir(self, *paths):
        assert len(paths) > 0

        if len(paths) == 1:
            path = paths[0]
        else:
            path = os.path.join(*paths)
        
        if not os.path.exists(path):
            self._log('Making directory %s' % path)
            os.makedirs(path)

        return path

    def _log(self, s):
        print(s)

    _debug = _log

    def waitForEnd(self):
        proc = self._scan_open_process
        if proc is not None:
            try:
                proc.wait()
            except:
                logging.exception('Failture waiting for scan open process to finish')

        for amg in list(self.antenna_managers.values()):
            try:
                amg.wait_for_end()
            except:
                logging.exception('Failure waiting fr antenna process to finish')

        self.check_scan_is_closed()
                

    def startObs(self, sbid, current=None):
        '''
        Starts obs with this parset
        
        :sbid: long SB ID
        :throws: NoSuchSchedulingBlockException,
        AlreadyRunningException, PipelineStartException;
        :returns: None
        '''


        self._log('startObs sbid=%d self.sbid=%d' % (sbid, self.sbid))

        if self.sbid > 0: # Hmm, got 2 startObs in a row
            self.abortObs()

        # Save SBID
        self.sbid = sbid

        # get schedblock parameters as dictionary from schedblock service
        pars = self._sb.getObsParameters(sbid)
        self.sbinfo = self._sb.getMany([sbid])[0]


        # save as parset - merge with extra parset if required
        self.parameters = askap.parset.merge(ParameterSet(pars), self._extra_parset)

        # get antennas


        #self._debug('Got {} antennas: {}'.format(len(self.antennas), self.antennas))
        self.sbdir = self._mkdir(self.rootdir, 'SB%05d'% self.sbid)
        self.write_lockfile('SCHEDBLOCK_OPEN', 'SB_OPENED_UTC')
        self.clear_lockfile('SCHEDBLOCK_CLOSED')
        self.clear_lockfile('ARCHIVED') # in case the copy thought we were finished
        self._log('startObs end self.sbid=%d'%self.sbid)
        

    def abortObs(self, current=None):
        '''
        Aborts current observation
        
        :returns: None
        '''
        self._finish_schedblock()
        self.waitForEnd()

        
    def waitObs(self, timeout, current=None):
        '''
        Blocks until the observation in progress is completed.
        
        :timeout: Maximum time to wait in milliseconds
        :returns: True if observation is complete. False otherwise
        '''
        self._finish_schedblock()
        return True

    def write_lockfile(self, filename, datecard, rootdir=None):
        if rootdir is None:
            rootdir = self.sbdir
            
        lockfile = os.path.join(rootdir, filename) 
        with open(lockfile, 'w') as fout:
            fout.write('{} {}\n'.format(filename, datetime.datetime.utcnow().isoformat()))

        return lockfile

    def clear_lockfile(self, filename, rootdir=None):
        if rootdir is None:
            rootdir = self.sbdir

        try:
            lfile = os.path.join(rootdir, filename)
            os.remove(lfile)
        except:
            pass


    def _finish_schedblock(self):
        ''' Closes everything down cleanly'''
        self._log('Finish_schedblock')
        if self.sbid == -99:
            return
                
        self._finish_scan()
        self.write_lockfile('SCHEDBLOCK_CLOSED', 'SB_CLOSED_UTC')
        self.clear_lockfile('SCHEDBLOCK_OPEN')

        self.sbid = -99
        self.skyfreq = -999


    def close_capture(self, timestamp, data):
        '''Sends kill signals but does not wait for the end'''
        logging.info('FINISH_CAPTURE: closing all capture first')
        self.capturing = False

        for amgr in list(self.antenna_managers.values()):
            try:
                amgr.terminate_capture(timestamp, data)
            except:
                logging.exception('Failure closing scan. Contining with other antennas')

        self._kill_scan_process()
        return self.check_scan_is_closed()

    def _finish_scan(self, timestamp=None, data=None):
        '''Closes the scan for all antennas'''
        if self.scanid != -99:
            self.close_capture(timestamp, data)

    def check_scan_is_closed(self):
        capturing = [amgr._check_capture_alive() for amgr in list(self.antenna_managers.values())]
        scan_open_proc = self._poll_scan_open_process
        logging.debug('Scan %d is closed?any capturing=%s scanopen=%s', self.scanid, any(capturing), scan_open_proc)
        if any(capturing) or scan_open_proc:
            scan_is_closed = False
            logging.debug('Scan %d is still open', self.scanid)
        else:
            scan_is_closed = True
            logging.info('Scan %d is closed. nantenna_managers=%d', self.scanid, len(self.antenna_managers))
            if self.scanid != -99 and len(self.antenna_managers) > 0:
                self.scanid = -99
                self.clear_lockfile('SCAN_OPEN', rootdir=self.scandir)
                self.write_lockfile('SCAN_CLOSED', 'SCAN_CLOSED_UTC', rootdir=self.scandir)
                self.exec_hook('scan_closed', self.scandir, amgr=list(self.antenna_managers.values())[0])
                self.antenna_managers.clear()

        
        return scan_is_closed

    def _open_scan(self, timestamp, data):
        '''Opens a scan'''

        scanid = data['scan_id'].value
        assert self.sbid > 0

        self.antennas, self.site = self.get_antennas(data)


        #if len(self.antennas) == 0:
        #    logging.info('Delaying new scanid {} as no valid antennas'.format(scanid))
        #    return

        if len(self.antennas) < 10:
            logging.info('Delaying new scanid {} as {} valid antennas'.format(scanid,len(self.antennas)))
            return


        intents = self.get_scan_intents_for_scanid(scanid, self.antennas)
        if all([i == 'UNKNOWN' for i in intents]):
            logging.info('No antennas have a usable scan intent for scanid %d. Ignoring', scanid)
            return

        self.nbeams = self.get_scan_nbeams(scanid)
        if self.nbeams != 36:
            logging.info('Ignoring scan as it doesnt have all 36 beams. It has %d beams', self.nbeams)
            return
        
        
        assert len(self.antenna_managers) == 0

        for iant, ant in enumerate(self.antennas):
            self.antenna_managers[ant] = CraftAntennaManager(ant, self, iant)

        self.scanid = scanid
        self._log('Opening new scan: {} with {} antennas'.format( scanid, len(self.antenna_managers)))
        skyfreq = data['sky_frequency']
        self.scan_open_time = datetime.datetime.utcnow()
        nowstr = self.scan_open_time.strftime('%Y%m%d%H%M%S')
        self.scandir = self._mkdir(self.sbdir, nowstr)
        self.scanname = nowstr
        self.set_leap_seconds()
        self.write_lockfile('SCAN_OPEN', 'SCAN_OPEN_UTC', rootdir=self.scandir)
        self._find_subarrays()
                
        # actually open the scans
        for ia, amgr in enumerate(self.antenna_managers.values()):
            amgr.open_scan(self.scanid, timestamp, data)

        self.capture_this_scan = True
        self.capturing = True

    def set_leap_seconds(self):
        ''' Sets the leap seconds attribute - assumes scant_start_time has been set'''
        self.dTAI_UTC = leapseconds.dTAI_UTC_from_utc(self.scan_open_time)
        return self.dTAI_UTC


    def _find_subarrays(self):
        # work out subarrays - a subarray is a set of antennas that has the same srcname
        curr_antennas = sorted(self.antenna_managers.keys())
        self.all_subarrays = []
        self.subarrays_by_src = {}
        # Get sourcenames for each antenna
        sourcenames = [self.get_scan_srcname(self.scanid, get_antno(ant)) for ant in curr_antennas]

        # need to sort for groupby to work
        srcant = sorted(zip(sourcenames, curr_antennas))
        for srcname, ants in itertools.groupby(srcant, lambda x:x[0]):
            antlist = [ant[1] for ant in ants]
            self.all_subarrays.append(antlist)
            self.subarrays_by_src[srcname] = antlist

        self.subarrays_by_src_str = {src:'-'.join([a for a in subarr]) for src, subarr in list(self.subarrays_by_src.items())}
        self.all_subarrays_str = ' '.join(sorted(self.subarrays_by_src_str.values()))


    def exec_hook(self, hookname, cwd, env=None, amgr=None):

        proc = None
        hook_env_names = ['SCAN_INTENT','SBID','SCANNAME','SUBARRAYS','THIS_SUBARRAY','SOURCE','FIELD_NAME', 'TARGET_NAME', 'SB_ALIAS','CAPTUREID','INT_TIME','INT_CYCLES', 'CRAFT_MODE', 'DOWNLOAD_BEAMS', 'DOWNLOAD_CARDS', 'ARRAY', 'HDR_SIZE', 'CAPTURE_NAME', 'SKY_FREQUENCY', 'FOOTPRINT_NAME','FOOTPRINT_PITCH','FOOTPRINT_ROTATION']
        hook_env = {}
        if env is not None:
            hook_env.update(env)
            
        if amgr is not None and hasattr(amgr, 'hdr') and amgr.hdr is not None:
            hook_env.update({e:str(amgr.hdr.get_value(e)) for e in hook_env_names})

        hook_env.update(os.environ)


        try:
            logging.info('Running hook %s cwd=%s', hookname, cwd)
            proc = subprocess.Popen(hookname, cwd=cwd, env=hook_env)
        except:
            logging.exception('Error running hook %s in cwd %s', hookname, cwd)

        return proc
        
    def _update_scan(self, timestamp, data):
        if not self.capture_this_scan:
            self._log('Ignoring metadata. Not capturing this can')
            return

        scanid = data['scan_id'].value
        amgr = None
        for antname, amgr in list(self.antenna_managers.items()):
            amgr.update_scan(timestamp, data)

        capture_flags = [amgr.capturing for amgr in list(self.antenna_managers.values())]
        all_antennas_capturing = all(capture_flags) and len(capture_flags) > 0
        if self._scan_open_process is None and self.capturing:
            if all_antennas_capturing: # run scan open hook
                dada_keys = ' '.join([amgr.hdr['DADA_KEY'][0] for amgr in list(self.antenna_managers.values())])
                env = {'DADA_KEYS':dada_keys}
                self._scan_open_process_return_code = None
                assert self._scan_open_process is None, 'AAAARGH - scanopen processs still running %s'%self._scan_open_process
                self._scan_open_process = self.exec_hook('scan_open', self.scandir, env=env, amgr=amgr)
                logging.info('Created scan open process PID={}'.format(self._scan_open_process.pid))
                self.scan_start_time = datetime.datetime.utcnow()
            else:
                pass # still waiting for all antennas to start

        else: # scan process is already running
            now = datetime.datetime.utcnow()
            time_delta = now - self.scan_start_time
            if all_antennas_capturing: # antennas are still running. Keep on running.
                # check scan process is running
                if self._poll_scan_open_process: # if the the scan open process has died
                    return_code = self._scan_open_process_return_code
                    if return_code is None:
                        pass
                    elif return_code == 0: # dont do restart
                        pass
                    else: # this is the signal to restart everything
                        self.close_capture(timestamp, data)
                        # TODO: Work out how to start the next capture
                        # it should restart on the next valid metadata
            else: # an antenna has stopped. close capture
                logging.warning('An antenna has stopped. Closign capture')
                self.close_capture(timestamp, data)

            if time_delta.total_seconds() > 60*30:
                logging.warning('Restarting scan as it lasted long enough')
                self.close_capture(timestamp, data)

    def _kill_scan_process(self):
        proc = self._scan_open_process
        logging.info('Killing scan process {}'.format(proc))

        if proc is not None:
            try:
                logging.info('Terminating SCANOPEN process - retcode {} pid={} '.format(proc.returncode, proc.pid))
                proc.terminate()
                logging.info('SCANOPEN process terminated with return code {} pid={}'.format(proc.returncode, proc.pid))
            except OSError:
                # gets thrown if the process is already dead
                logging.exception("Error killing scan open process - probabaly already dead")
            
        return self._scan_open_process_return_code

    @property
    def _poll_scan_open_process(self):
        p = self._scan_open_process
        running = False
        if p is None:
            retcode = None
            running = False
        else:
            retcode = p.poll()
            if retcode is None:
                running = True
            else: # its died
                self._scan_open_process = None
                self._scan_open_process_return_code = retcode
                running = False
                logging.info('SCANOPEN process died with retcode {}'.format(retcode))

                # Assume that if scanopen dies, then we should close the capture
                logging.info('Closing capture as SCANOPEN died unhappliy')
                self.close_capture(None,None)

            
        return running


    def on_metadata(self, timestamp, data):
        '''

        :data: is a directionary whose contents is defined here: https://jira.csiro.au/browse/ASKAPTOS-3320
        '''

        scanid = data['scan_id'].value

        #self._log('METADATA scanid={} self.scanid = {} SBID={}'.format(scanid, self.scanid, self.sbid))
        
        if self.sbid < 0: # nothing happening
            self._log('Err - metadata with unknown sbid. scanid={}. Ignoring.'.format(scanid))
        elif scanid == -2: # end of data
            self._log('scanid==-2. Will ignore this, as I occasionally get them at the beginning of an SB')
            #self._finish_schedblock()
            self._finish_scan(timestamp, data)
        elif scanid == -1: # invalid scan - happens when warming up an SB before the first scan starts
            #self._finish_schedblock()
            self._log('Waiting for SBID to start')
            self._finish_scan(timestamp, data)
        elif scanid != self.scanid: # 
            self._log('New scanid {}. Old scanid {} Reopening'.format(scanid, self.scanid))
            if self.scanid != -99:
                self._finish_scan(timestamp, data)
                
            if self.check_scan_is_closed():
                self._open_scan(timestamp, data)
                self._update_scan(timestamp, data)
        else: # continuation of existing scan
            if not self.check_scan_is_closed():
                self._update_scan(timestamp, data)

    def getServiceVersion(self, current=None):
        return ASKAP_VERSION

    def get_target_parameter(self, target, key, default=None, namespace='common'):
        """Return the value for a key in the `target` parset
        @see observaiton.py (ripped off from there) """
        # Get rid if annoying typedValueString stuff
        if hasattr(target, 'value'):
            target = target.value
            
        pfx = ".".join((namespace, "target"))
        fkey = ".".join((pfx, target, key))

        if fkey in list(self.parameters.keys()):
            value= self.parameters[fkey]
        else:
            rx = re.compile("\d+")
            subst = rx.sub("%d", target)
            if target != subst:
                fkey = ".".join((pfx, subst, key))

            value = self.parameters.get_value(fkey, default)

        return value

    def get_scan_srcname(self, scanid, antno):
        key = 'schedblock.scan{:03d}.target.ant{:d}'.format(scanid, antno)
        tmap = self._sb.getObsVariables(self.sbid, key)
        srcname = tmap[key]
        return srcname


    def get_scan_nbeams(self, scanid):
        #schedblock.scan000.footprint.n_beams = 1
        key = 'schedblock.scan{:03d}.footprint.n_beams'.format(scanid)
        tmap = self._sb.getObsVariables(self.sbid, key)
        nbeams = int(tmap[key])
        
        return nbeams

    def get_scan_intents_for_scanid(self, scanid, antennas):
        '''
        Returns the scan intents for the given antennas
        Antennas is a list of antennas naems (not numbers)
        '''
        antnos = [get_antno(a) for a in antennas]
        targets = [self.get_scan_srcname(scanid,antno) for antno in antnos]
        intents = [self.get_scan_intent(t) for t in targets]

        return intents

    def get_schedblock_antennas(self):
        pset = ParameterSet(self._sb.getObsVariables(self.sbid, 'schedblock.antennas'))
        
        ants = pset['schedblock.antennas']
        return ants

    def get_antennas(self, data):
        ''' Gets scan antennas, schedblock antennas  and cmdline antennas, and that have on_source=True, and flagged=False
        Returns antennas that are at the intersection of all 5
        antenna names are strings prependded by 'co' or 'ak'
        Also returns whether its 'co' or 'ak'
        '''

        # start with scan antennas - this has everything that is enabled in the FCM
        scan_antenna_names  = data['antennas'].value
        site = scan_antenna_names[0][0:2]
        assert site == 'ak' or site == 'co', 'Unknown site: {} {}'.format(site, scan_antenna_names)
        mainflag = data['flagged'].value

        antennas = set(scan_antenna_names)

        # only antennas that have flagged=False and on_source=True are valid at this point
        unflagged_antennas = [a for a in antennas if data['{}.flagged'.format(a)].value == False]
        on_source_antennas = [a for a in antennas if data['{}.on_source'.format(a)].value == True]

        # gosh this is ugly - strip out antennas that don't have flagged=False, and on_source=True
        antennas = antennas.intersection(set(unflagged_antennas))
        antennas = antennas.intersection(set(on_source_antennas))

        # you can specify a subset of antennas in the schedblock. If specified
        # then take the intersection
        # sb specifies antenna names as 'ant1','ant2', but we want to make them
        # all start with 'ak' or 'co' - and double digits
        sb_antenna_names = ['{}{:02d}'.format(site, get_antno(a)) for a in self.get_schedblock_antennas()]
        if sb_antenna_names is not None and len(sb_antenna_names) > 0:
            antennas = antennas.intersection(set(sb_antenna_names))
        
        searchants = self.values.search_antennas
        if searchants is not None:
            searchants = set(self.values.search_antennas)
            searchants = ['{}{:02d}'.format(site, ant) for ant in searchants]
            antennas = antennas.intersection(set(searchants))
        else:
            searchants = []

        antennas = sorted(antennas)

        fraction_valid = float(len(antennas)) / float(len(scan_antenna_names))

        logging.info('MainFlag = {} Got {} antennas in metadata, {} unflagged and {} on source. Command line has {} antennas. Sb has {} antennas. Final antenna list is: {}. Fraction valid: {}'.format(mainflag, len(scan_antenna_names), len(unflagged_antennas), len(on_source_antennas), len(searchants), len(sb_antenna_names), antennas, fraction_valid))


        if fraction_valid < 0.5:
            logging.info('Fraction of valid antennas = %d/%d = %f < 0.5. Not starting scan', len(antennas), len(scan_antenna_names), fraction_valid)
            antennnas = []

        # If mainflag is false then no antennas are valid
        if mainflag:
            antennas = []

        return antennas, site

    def get_scan_intent(self, target):
        intent = self.get_target_parameter(target, 'scan_intent', '', namespace='craft')
        if intent == '' or intent is None:
            intent = 'UNKNOWN'
            sbtemplate = self.sbinfo.templateName
            field_name = self.get_target_parameter(target, 'field_name')
            if sbtemplate.lower() == 'flyseye':
                if field_name.startswith('PSR'):
                    intent = 'PSR_CHECK'
                else:
                    intent = 'FRB_FLYSEYE'
            elif sbtemplate.lower() == 'standard':
                if field_name.startswith('PSR') and 'beam' in field_name:
                    intent = 'PSR_CAL'

        return intent

    def get_int_time(self, target):
        # Default is 1023 - = 1024 samples per integration. - works fine

        return self.get_target_parameter(target, 'int_time', 2047, namespace='craft')

    def get_int_cycles(self, target):
        return self.get_target_parameter(target, 'int_cycles', 7,  namespace='craft')

    def get_tsamp(self, target):
        # number of 1 MHz samples per integration
        int_nsamp = self.get_int_time(target)

        tsamp = float(int_nsamp + 1.)/FSAMP # seconds
        
        return tsamp

    def get_mode(self, target):
        return self.get_target_parameter(target, 'mode', 0, namespace='craft')

    def get_beam_id(self, target):
        return self.get_target_parameter(target, 'beamid', 0, namespace='craft')

    def get_record_mode(self, target):
        return self.get_target_parameter(target, 'record_mode', 'uint8', namespace='craft')
    
    def get_download_beams(self,target):
        return self.get_target_parameter(target, 'download_beams', '0,1', namespace='craft')

    def get_download_cards(self,target):
        return self.get_target_parameter(target, 'download_cards', '1-7', namespace='craft')

    def get_footprint_pitch_shape(self, target):
        # TODO: Pickup pitch and shape from schedblock service
        pitch = None
        shape = None
        pa = None
        shape = self.get_target_parameter(target, 'footprint.name', shape)
        pitch = floatnone(self.get_target_parameter(target, 'footprint.pitch', pitch))
        pa = floatnone(self.get_target_parameter(target, 'footprint.rotation',pa))

        if shape is None or pitch is None or pa is None:
            raise ValueError('No valid footprint for target %s SBID=%s' % (self.target, self.sbid))

        assert pitch is not None
        assert shape is not None
        assert pa is not None

        return (pitch, shape, pa)


    def get_footprint(self, target, pos_deg, pol_deg):
        '''Creates a footprint

        The pitch and shape of the footprint are taken from the target parameters
        :pos_deg: A tuple containing the RA/Dec of the footprint center in degrees
        :pol_deg: float - pol angle in degress
        '''
        pitch, shape, pa = self.get_footprint_pitch_shape(target)
        full_pa_deg = pol_deg + pa
        fp = Footprint.named(shape, np.radians(pitch), np.radians(full_pa_deg))
        pos_rad = list(map(np.radians, pos_deg))
        fp.setRefpos(pos_rad)
        return fp

    def load_freqconfig(self, antname):
        antname='ak10'
        fname = os.path.join(self.config_dir, 'freqconfig', 
                             '{}_f1280v10.freqs'.format(antname))
        hdr = crafthdr.DadaHeader.fromfile(fname)
        fconfig = FreqConfig.load_from_dada_header(hdr, freqname='PVFREQS')

        return fconfig


    


