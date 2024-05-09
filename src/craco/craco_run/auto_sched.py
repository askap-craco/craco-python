# module and functions for automatically scheduling
try:
    from aces.askapdata.schedblock import SB, SchedulingBlock
    from askap.parset import ParameterSet
except:
    print("cannot load aces package...")

from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy import units

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from craco.datadirs import DataDirs, SchedDir, ScanDir, RunDir, CalDir
from craco import plotbp

from configparser import ConfigParser
import subprocess
import time
import glob
import re
import os
import pdb


from slack_sdk import WebClient

from .metaflag import MetaManager

import logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_config(config=None, section="postgresql"):
    parser = ConfigParser()

    ### check if config file exists - otherwise use the filepath in environment variable
    if config is None:
        config = os.environ.get("CRACO_DATABASE_CONFIG_FILE")
        if config is None: config = "database.ini"
    parser.read(config)

    if not parser.has_section(section):
        raise ValueError(f"Section {section} not found in {config}")
    params = parser.items(section)
    return {k:v for k, v in params}

### load sql
import psycopg2
def get_psql_connect():
    config = load_config()
    return psycopg2.connect(**config)

from sqlalchemy import create_engine 
def get_psql_engine():
    c = load_config()
    engine_str = "postgresql+psycopg2://"
    engine_str += f"""{c["user"]}:{c["password"]}@{c["host"]}:{c["port"]}/{c["database"]}"""
    return create_engine(engine_str)

class InvalidSBIDError(Exception):
    def __init__(self, sbid):
        super().__init__(f"{sbid} is not a valid sbid")

class CracoSchedBlock:
    
    def __init__(self, sbid):
        self.sbid = sbid
        try: self.askap_schedblock = SchedulingBlock(self.sbid)
        except: self.askap_schedblock = None

        ### craco data structure related
        self.scheddir = SchedDir(sbid=sbid)
        
        ### get obsparams and obsvar
        if self.askap_schedblock is not None:
            self.obsparams = self.askap_schedblock.get_parameters()
            self.obsvar = self.askap_schedblock.get_variables()
        
        try: self.get_avail_ant()
        except:
            self.antennas = None
            self.flagants = ["None"]

    # get information from craco data
    @property
    def craco_scans(self):
        return self.scheddir.scans
    
    @property
    def craco_exists(self):
        if len(self.craco_scans) == 0: return False
        return True
    
    @property
    def craco_sched_uvfits_size(self):
        """get uvfits size in total"""
        size = 0 # in the unit of GB
        for scan in self.craco_scans:
            try:
                scandir = ScanDir(sbid=self.sbid, scan=scan)
                uvfits_exists = scandir.uvfits_paths_exists
            except:
                continue # no rank file found... aborted
            if len(uvfits_exists) == 0: continue
            size += os.path.getsize(uvfits_exists[0]) / 1024 / 1024 / 1024
        return size

    # get various information from aces
    def get_avail_ant(self):
        """get antennas that are available"""
        antennas = self.obsvar["schedblock.antennas"]
        self.antennas = re.findall("(ant\d+)", self.obsvar["schedblock.antennas"])
        ### calculate the antenna that are flagged
        if self.scheddir.flagant is not None:
            log.info("loading antenna flags from metadata...")
            self.flagants = [str(ant) for ant in self.scheddir.flagant]
        else:
            self.flagants = [str(i) for i in range(1, 37) if f"ant{i}" not in self.antennas]

    # get source field direction
    def _get_field_direction(self, src="src1"):
        ### first try common.target.src?.field_direction in obsparams
        if f"common.target.{src}.field_direction" in self.obsparams:
            field_direction_str = self.obsparams[f"common.target.{src}.field_direction"]
        ### then try schedblock.src16.field_direction in obsvar
        elif f"schedblock.{src}.field_direction" in self.obsvar:
            field_direction_str = self.obsvar[f"schedblock.{src}.field_direction"]
        return self.__parse_field_direction(field_direction_str)
            
    def __parse_field_direction(self, field_direction_str):
        """parse field_direction_str"""
        pattern = """\[(.*),(.*),.*\]"""
        matched = re.findall(pattern, field_direction_str)
        assert len(matched) == 1, f"find none or more matched pattern in {field_direction_str}"
        ### then further parse ra and dec value
        ra_str, dec_str = matched[0]
        ra_str = ra_str.replace("'", "").replace('"', "") # replace any possible " or '
        dec_str = dec_str.replace("'", "").replace('"', "")
        
        if (":" in ra_str) and (":" in dec_str):
            field_coord = SkyCoord(ra_str, dec_str, unit=(units.hourangle, units.degree))
        else:
            field_coord = SkyCoord(ra_str, dec_str, unit=(units.degree, units.degree))
            
        return field_coord.ra.value, field_coord.dec.value
    
    def get_scan_source(self):
        """
        retrieve scan and source pair based on the schedulingblock
        """
        refant = self.antennas[0]
        scan_src_match = {}
        sources = []
        for scan in range(100): # assume maximum scan number is 99
            scanstr = f"{scan:0>3}"
            scanantkey = f"schedblock.scan{scanstr}.target.{refant}"
            if scanantkey in self.obsvar: 
                src = self._find_scan_source(scan)
                scan_src_match[scan] = src
                if src not in sources: sources.append(src)
            else: break
        self.scan_src_match = scan_src_match
        self.sources = sources
            
    def _find_scan_source(self, scan):
        # in self.obsvar under schedblock.scan000.target.ant1
        scanstr = f"{scan:0>3}"
        allsrc = [self.obsvar[f"schedblock.scan{scanstr}.target.{ant}"].strip() for ant in self.antennas]
        unisrc = list(set(allsrc))
        assert len(unisrc) == 1, "cannot handle fly's eye mode..."
        return unisrc[0]
        
    def get_sources_coord(self, ):
        """
        get source and direction pair
        """
        self.source_coord = {src:self._get_field_direction(src) for src in self.sources}

    @property
    def corrmode(self):
        """corrlator mode"""
        return self.obsparams["common.target.src%d.corrmode"]        
        
    @property
    def template(self, ):
        return self.askap_schedblock.template
      
    @property
    def spw(self, ):
        try:
            if self.template in ["OdcWeights", "Beamform"]:
                return eval(self.obsvar["schedblock.spectral_windows"])[0]
            return eval(self.obsvar["weights.spectral_windows"])[0]
        except: return [-1, -1]
        # note - schedblock.spectral_windows is the actual hardware measurement sets spw
        # i.e., for zoom mode observation, schedblock.spectral_windows one is narrower
    
    @property
    def central_freq(self, ):
        try: return eval(self.obsparams["common.target.src%d.sky_frequency"])
        except: return -1
        
    @property
    def footprint(self, ):
        return self.askap_schedblock.get_footprint_name()
    
    @property
    def status(self,):
        return self.askap_schedblock._service.getState(self.sbid)
        # return sbstatus.value, sbstatus.name
    
    @property
    def alias(self, ):
        try: return self.askap_schedblock.alias
        except: return ""
    
    @property
    def start_time(self, ):
        try: return Time(self.obsvar["executive.start_time"]).mjd # in mjd
        except: return 0

    @property
    def weight_sched(self, ):
        try: return int(self.obsvar["weights.schedulingblock"])
        except: return -1
    
    @property
    def duration(self, ):
        if self.status.value <= 3: return -1 # before execution
        try: return eval(self.obsvar["executive.duration"])
        except: return -1

    @property
    def weight_reset(self, ):
        if self.template in ["OdcWeights", "Beamform"]: return True
        if self.template == "Bandpass":
            if "dcal" in self.alias: return True
        return False
    
    def rank_calibration(self, field_direction=None, altaz=None):
        """
        get rank for calibration
        0 - cannot be used for calibration
            1) odc or beamform scan 
            2) dcal scan 
            3) without RACS catalogue - Dec > 40 or Dec < -80?
            4) scan that is too short (here we use 120 seconds) 
            5) zoom mode
        1 - Usuable but not ideal
            1) Galactic field - |b| < 5
            2) elevation angle is less than 30d
        2 - Good for calibration
            1) Extragalactic field with RACS catalogue
        3 - Perfect for calibration
            1) bandpass but not dcal scan
        """
        if self.template in ["OdcWeights", "Beamform"]: return 0
        if self.duration <= 120: return 0
        if self.template == "Bandpass":
            if "dcal" in self.alias: return 0
            return 3
        if "zoom" in self.corrmode: return 0
        
        
        ### for other cases, you need to consider field_direction
        if field_direction is None: return -1
        
        coord = SkyCoord(*field_direction, unit=units.degree)
        if coord.dec.value > 40 or coord.dec.value < -80:
            log.info(f"footprint is outside of RACS catalogue... decl - {coord.dec.value}")
            return 0 # source outside RACS catalogue

        ## if the elevation angle is less than 30 degree, return 0
        if altaz is None: return 0
        altazcoord = coord.transform_to(altaz)
        if altazcoord.alt.value < 30: 
            log.info(f"not ideal to use this scan for calibration... elevation in the middle of the scan is {altazcoord.alt.value:.2f}")
            return 1
        ### this will make long scan worse... comment out atm we now using middle scan
        
        coord = coord.galactic
        if abs(coord.b.value) <= 5: 
            log.info(f"the scan is close to the Galactic Plane... b = {coord.b.value}")
            return 1
        return 2
    
    def get_sbid_calib_rank(self, ):
        rank = self.rank_calibration()
        if rank != -1: return rank
        
        self.get_scan_source()
        self.get_sources_coord()
        
        ranks = []
        ### get altaz
        if self.start_time > 0 and self.duration > 0:
            log.info("working out the time in the middle of the observation...")
            midmjd = self.start_time + self.duration / 86400 / 2
            askaploc = AltAz(
                obstime=Time(midmjd, format="mjd"),
                location=EarthLocation.of_site("ASKAP")
            )
        else: askaploc=None
        for src in self.source_coord:
            ranks.append(
                self.rank_calibration(self.source_coord[src], askaploc)
            )
        
        if len(ranks) == 0: return 0
        return min(ranks)
        
    
    def format_sbid_dict(self, ):
        ### format sbid dictionary
        d = dict(sbid=self.sbid)
        d["alias"] = self.alias
        d["corr_mode"] = self.corrmode
        d["craco_record"] = self.craco_exists
        ### spw
        try: spw = self.spw
        except: spw = [-1, -1] # so that you can post to the database
        d["start_freq"], d["end_freq"] = spw
        d["central_freq"] = self.central_freq
        
        d["footprint"] = self.footprint
        d["template"] = self.template
        d["start_time"] = self.start_time
        d["duration"] = self.duration
        
        d["flagant"] = ",".join(self.flagants)
        d["status"] = self.status.value # note - anything larger than 3 is usuable
        
        try: d["calib_rank"] = self.get_sbid_calib_rank()
        except Exception as error: 
            log.warning(f"cannot get the calibration rank for {self.sbid}... Error message is as followed: {error}")
            d["calib_rank"] = -1
            
        d["craco_size"] = self.craco_sched_uvfits_size
        d["weight_sched"] = self.weight_sched
        d["weight_reset"] = self.weight_reset
            
        return d

### functions to interact with database

############# FOR CALIBRATION ##################

def push_sbid_calibration(
        sbid, prepare=True, plot=True, updateobs=False,
        conn = None, cur = None,
    ):
    """
    add calibration information to the database
    """
    if int(sbid) == 0: return
    log.info(f"loading calibration for {sbid}")
    
    if conn is None: conn = get_psql_connect()
    if cur is None: cur = conn.cursor()
    
    calcls = CracoCalSol(sbid)
    solnum = calcls.solnum
    if prepare:
        status = 1 # this means running
        valid = False
        goodant, goodbeam = -1, -1
        solnum = -1
    else:
        status = 0
        try:
            valid, goodant, goodbeam = calcls.rank_calsol(plot=plot)
        except Exception as err:
            log.info(f"failed to get calibration quality... - {sbid}")
            log.info(f"error message - {err}")
            valid, goodant, goodbeam = False, 0, 0
            status = 2 # something goes run
    
    if updateobs:
        ### this means we also update the observation table
        try:
            ### check whether we need to update
            update = False
            if status == 0 and valid == False:
                update = True; calib_rank = -2 # this means bad calibration
            if status == 2:
                update = True; calib_rank = -3 # we cannot load quality

            if update:
                log.info(f"updating observation table - {sbid}")
                update_table_single_entry(
                    int(sbid), "calib_rank", 
                    calib_rank, "observation",
                    conn=conn, cur=cur,
                )
        except Exception as error:
            log.warning(f"cannot update observation table - error - {error}")
    
    cur.execute(f"SELECT * FROM calibration WHERE SBID={sbid}")
    res = cur.fetchall()
    if len(res) == 0: # insert
        insert_sql = f"""INSERT INTO calibration (
    sbid, valid, solnum, goodant, goodbeam, status
)
VALUES (
    {sbid}, {valid}, {solnum}, {goodant}, {goodbeam}, {status}
)
"""
        cur.execute(insert_sql)
        conn.commit()
        
    else:
        update_sql = f"""UPDATE calibration
SET valid={valid}, solnum={solnum}, 
goodant={goodant}, goodbeam={goodbeam}, status={status}
WHERE sbid={sbid}
"""
        cur.execute(update_sql)
        conn.commit()


################### THIS IS THE TEST CLASS FOR DEBUGGING ###################

"""
def get_random_num():
    return np.random.uniform()

class CracoCalSol:
    def __init__(self, sbid):
        self.sbid  = sbid

        ### get solnum
        rand = get_random_num()
        if rand < 0.95: self.solnum = 36
        else: self.solnum
    
    def rank_calsol(self, plot=True):
        if self.solnum == 35:
            return False, 26, 35
        rand = get_random_num()
        if rand < 0.8: return True, 30, 36
        return False, 28, 35
"""

############################################################################

class CracoCalSol:
    def __init__(
        self, sbid, flagfile="/home/craftop/share/fixed_freq_flags.txt"
    ):
        self.sbid = sbid
        self.caldir = CalDir(sbid)
        self.scheddir = SchedDir(sbid)

        ### load flagfreqs
        self.flagfreqs = np.loadtxt(flagfile)

    @property
    def solnum(self):
        npyfiles = glob.glob(f"{self.caldir.cal_head_dir}/??/b??.aver.4pol.smooth.npy")
        return len(npyfiles)

    @property
    def flag_ants(self):
        """
        load from metadata file - instead of database
        """
        flagant = self.scheddir.flagant
        if flagant is not None:
            return flagant
        
        ### else use database
        log.info(f"loading flagant of {self.sbid} from database...")

        engine = get_psql_engine()
        sql_df = pd.read_sql(f"SELECT flagant FROM observation WHERE sbid={sbid}", engine)

        assert len(sql_df) == 1, "no sbid found in observation table..."
        flagant = sql_df["flagant"][0]
        if flagant == "None": 
            raise ValueError("no flag antenna information found... not suitable for calibration...")
        if flagant == "": flagant = []
        else: flagant = [int(i) for i in flagant.split(",")]

        return flagant #both of them are 1-indexed

    @property
    def good_ant(self):
        """
        0-indexed good antenna - note flagant is zero indexed
        """
        flagant = self.flag_ants
        return [i-1 for i in range(1, 31) if i not in flagant]

    def _load_flagfile_chan(self, solfreqs, flag=True):
        arr_lst = [(solfreqs / 1e6 <= freqs[1]) & (solfreqs / 1e6 >= freqs[0]) for freqs in self.flagfreqs]
        freqflag_bool = np.sum(arr_lst, axis=0).astype(bool)
        if flag: return np.where(freqflag_bool)[0]
        return np.where(~freqflag_bool)[0]

    ### calculate the score
    def rank_calsol(
        self, phase_difference_threshold=30, plot=True,
        good_frac_threshold=0.6, bad_frac_threshold=0.4,
    ):
        beams = []; beam_phase_diff = []
        for ibeam in range(36):
            try:
                beamcalsol = CalSolBeam(self.sbid, ibeam)
                unflagchans = self._load_flagfile_chan(beamcalsol.freqs, flag=False)

                phdif = beamcalsol.extract_phase_diff(self.good_ant, unflagchans)
                beam_phase_diff.append(phdif)
                beams.append(ibeam)
            except Exception as error:
                log.info(f"cannot load solution from beam {ibeam} for {self.sbid}...")
                log.info(f"error message - {error}")
                continue

        ### concatnate things
        sbid_phase_diff = np.concatenate(
            [i[None, ...] for i in beam_phase_diff], axis=0
        ) # it should be in a shape of nbeam, nant, nchan
        self.sbid_phase_diff = sbid_phase_diff

        nbeam, nant, nchan = sbid_phase_diff.shape
        beams = np.array(beams)
        if plot: ### plot phase differencec image for all beams
            log.info(f"plotting calibration solution quality control plot for SB{self.sbid}")
            fig = plt.figure(figsize=(12, 8), facecolor="white", dpi=75)
            for i in range(36):
                ax = fig.add_subplot(6, 6, i+1)
                ax.set_title(f"beam{i:0>2}")
                try: index = np.where(beams==i)[0][0]
                except: 
                    log.info(f"no solution loaded from beam{i}... will not plot it")
                    continue
                ax.imshow(
                    sbid_phase_diff[index], vmin=0, vmax=90, 
                    aspect="auto", interpolation="none"
                )
            fig.tight_layout()
            fig.savefig(f"{self.caldir.cal_head_dir}/calsol_qc.png", bbox_inches="tight")
            plt.close()

        ### work out statistics
        sbid_phase_good = sbid_phase_diff < phase_difference_threshold
        sbid_good_frac = sbid_phase_good.mean(axis=-1) # take the mean of frequency

        ### decide the number of the good beam
        good_beam_count = (sbid_good_frac.mean(axis=1) > bad_frac_threshold).sum()
        good_ant_count = (sbid_good_frac.mean(axis=0) > bad_frac_threshold).sum()
        valid_calsol = np.all(sbid_good_frac > good_frac_threshold)

        ### if there is one beam missing, valid should be false
        if nbeam < 36:
            log.warning(f"only {nbeam} calibration solution found... - {beams}")
            valid_calsol = False

        return valid_calsol, good_ant_count, good_beam_count
        
class CalSolBeam:
    def __init__(self, sbid, beam,):
        self.sbid = sbid
        self.caldir = CalDir(sbid)

        ### all files
        self.binfile = self.caldir.beam_cal_binfile(beam)
        self.freqfile = self.caldir.beam_cal_freqfile(beam)
        self.smoothfile = self.caldir.beam_cal_smoothfile(beam)
        self.__load_bandpass()

    @property
    def freqs(self,):
        try:
            return np.load(self.freqfile)
        except:
            log.warning(f"cannot load frequency file - {self.freqfile}")
            return None

    def __load_bandpass(self,):
        ### load bin bandpass
        log.warning("only XX calibration solution is loaded...")
        bpcls = plotbp.Bandpass.load(self.binfile)
        self.binbp = bpcls.bandpass.copy()[0, ..., 0]
        nant, nchan = self.binbp.shape
        ### load smooth bandpass
        self.smobp = np.load(self.smoothfile)[0, ..., 0]

        ### find reference antenna based on self.binbp
        valid_data_arr = np.array([self.__count_nan(self.binbp[ia]) for ia in range(nant)])
        ira = valid_data_arr.argmin() # reference antenna, 0-indexed
        assert valid_data_arr[ira] != 1., f"no reference antenna found for {self.binfile}"
        log.info(f"use {ira} (0-indexed) as the reference antenna")

        ### workout phase
        self.binph = np.angle(self.binbp / self.binbp[ira], deg=True)
        self.smoph = np.angle(self.smobp / self.smobp[ira], deg=True)

        self.phdif = np.min([
            (self.smoph - self.binph) % 360, (self.binph - self.smoph) % 360
        ], axis=0)

    def __count_nan(self, arr, isnan=True, fraction=True):
        total = np.size(arr)
        nancount = (np.isnan(arr) | np.isinf(arr)).sum()
        if fraction:
            if isnan: return nancount / total
            return 1 - nancount / total
        if isnan: return nancount
        return total - nancount

    def extract_phase_diff(self, goodant=None, unflagchan=None):
        if goodant is None: goodant = np.arange(30)
        ### only select data known to be good
        phdif = self.phdif[goodant]
        if unflagchan is not None:  phdif = phdif[:, unflagchan]
        return phdif

########### FOR execution ###############

def push_sbid_execution(
        sbid, runname="results", calsbid=None, reset=False, newstatus=0,
        conn=None, cur=None,
    ):
    scheddir = SchedDir(sbid)

    ### get calibration sbid
    if calsbid is None:
        try: calsbid = scheddir.cal_sbid
        except: calsbid = -1

        if isinstance(calsbid, str):
            calsbid = int(calsbid[2:])

    scans = len(scheddir.scans)
    rawfile_count = 0
    clusfile_count = 0
    for scan in scheddir.scans:
        try:
            rundir = RunDir(scheddir.sbid, scan=scan, run=runname)
            rawfile_count += len(rundir.raw_candidate_paths())
            clusfile_count += len(rundir.clust_candidate_paths())
        except Exception as error:
            log.info(f"error in loading run directory - {scheddir.sbid}, {scan}, {runname}")
            continue
            
    ### start to update database
    if conn is None: conn = get_psql_connect()
    if cur is None: cur = conn.cursor()
    cur.execute(f"SELECT sbid, status FROM execution WHERE SBID={sbid} AND RUNNAME='{runname}'")
    
    res = cur.fetchall()
    if len(res) == 0: # insert
        status = 0 # set the original status to 0
        insert_sql = f"""INSERT INTO execution (
    sbid, calsbid, status, scans, rawfiles, clustfiles, runname
)
VALUES (
    {sbid}, {calsbid}, {status}, {scans}, {rawfile_count}, {clusfile_count}, '{runname}'
)
"""
        cur.execute(insert_sql)
        conn.commit()
        
    else:
        ### if reset is True, you need to set status to 0
        if reset: status = 0
        else:
            previous_status = res[0][1]
            status = previous_status + newstatus
        
        update_sql = f"""UPDATE execution
SET calsbid={calsbid}, status={status}, scans={scans}, 
rawfiles={rawfile_count}, clustfiles={clusfile_count}
WHERE sbid={sbid} AND runname='{runname}'
"""
        cur.execute(update_sql)
        conn.commit()


#### function to pick up correct calibration
def _flagant_str_to_lst(flagantstr):
    if flagantstr.lower() == "none": return None
    if flagantstr == "": return []  
    return [int(i) for i in flagantstr.split(",")]

def check_calib_flagant(calflagant, obsflagant):
    """
    check whether the calibration is suitable for calibration
    this assume that both `calflagant` and `obsflagant` are string

    if there is any calflagant *NOT* in obsflagant, then return False
    otherwise True
    """
    calflagant = _flagant_str_to_lst(calflagant)
    obsflagant = _flagant_str_to_lst(obsflagant)

    # the easiest assertion, if calflag is more than obsflag, return False
    if calflagant is None: 
        log.warning("no flagantenna information found for calibration sbid...")
        return False
    
    assert obsflagant is not None, "flagant info for observation sbid is missing..."
    calflagant = set(calflagant); obsflagant = set(obsflagant)

    if len(calflagant - obsflagant) == 0: return True
    return False

class CalFinder:
    def __init__(self, sbid):
        self.sbid = sbid

        self.conn = get_psql_connect()
        self.cur = self.conn.cursor()
        self.engine = get_psql_engine()

        self.__get_sbid_property()

    def __get_sbid_property(self):
        self.cur.execute(f"""select central_freq,footprint,weightsched,start_time,flagant from observation
where sbid={self.sbid}""")
        res = self.cur.fetchall()
        assert len(res) == 1, f"found {len(res)} records in observation database for {self.sbid}..."
        self.freq, self.footprint, self.weight_sched, self.start_time, self.flagant = res[0]

    def query_calib_table(self, timethreshold=1.5):
        """
        query calibration table to find the most appropriate sbid

        it will return calibration sbid, and calibration status (just in case something is running)
        """
        joinsql = f"""SELECT o.sbid,o.flagant,c.status
FROM calibration c JOIN observation o ON c.sbid=o.sbid
WHERE o.weightsched={self.weight_sched} AND o.central_freq={self.freq} AND o.footprint='{self.footprint}' 
AND ((c.valid=True AND c.solnum=36 AND c.status=0) OR c.status=1)
AND o.start_time>={self.start_time-timethreshold} AND o.start_time<={self.start_time+timethreshold}
ORDER BY o.sbid DESC
"""
        self.cur.execute(joinsql)
        query_result = self.cur.fetchall()
        log.info(f"{len(query_result)} calibration records found in the database...")

        ### check whether flags are useful
        for calsbid, calflagant, calstatus in query_result:
            flagcheck = check_calib_flagant(calflagant, self.flagant)
            if flagcheck: return calsbid, calstatus
        return None, None
    
    def query_observe_table(self, timethreshold=1.5):
        """
        this is used to find potential calibration - need to run
        """
        querysql = f"""SELECT sbid,flagant FROM observation
WHERE weightsched={self.weight_sched} AND central_freq={self.freq} AND footprint='{self.footprint}'
AND start_time>={self.start_time-timethreshold} AND start_time<={self.start_time+timethreshold}
AND calib_rank>=1 AND delete=False AND craco_size>0
ORDER BY calib_rank DESC, SBID DESC
"""

        self.cur.execute(querysql)
        query_result = self.cur.fetchall()
        log.info(f"{len(query_result)} potential calibration sbid found in the database...")

        for calsbid, calflagant in query_result:
            flagcheck = check_calib_flagant(calflagant, self.flagant)
            if flagcheck: return calsbid
        return None


######## initial update to the observation database #######
def _update_craco_sched_status(craco_sched_info, conn=None, cur=None):
    if conn is None: conn = get_psql_connect()
    if cur is None: cur = conn.cursor()

    sbid = craco_sched_info["sbid"]
    # find whether this sbid exists already
    cur.execute(f"SELECT * FROM observation WHERE SBID = {sbid}")
    res = cur.fetchall()
    if len(res) == 0: # insert
        d = craco_sched_info
        insert_sql = f"""INSERT INTO observation (
    sbid, alias, corr_mode, start_freq, end_freq, 
    central_freq, footprint, template, start_time, 
    duration, flagant, status, calib_rank, craco_record, 
    craco_size, weight_reset, weightsched
)
VALUES (
    {d["sbid"]}, '{d["alias"]}', '{d["corr_mode"]}', {d["start_freq"]}, {d["end_freq"]},
    {d["central_freq"]}, '{d["footprint"]}', '{d["template"]}', {d["start_time"]},
    {d["duration"]}, '{d["flagant"]}', {d["status"]}, {d["calib_rank"]}, {d["craco_record"]}, 
    {d["craco_size"]}, {d["weight_reset"]}, {d["weight_sched"]}
);
"""
        cur.execute(insert_sql)
        conn.commit()
        
    else: # update
        d = craco_sched_info
        update_sql = f"""UPDATE observation 
SET alias='{d["alias"]}', corr_mode='{d["corr_mode"]}', start_freq={d["start_freq"]},
end_freq={d["end_freq"]}, central_freq={d["central_freq"]}, footprint='{d["footprint"]}', 
template='{d["template"]}', start_time={d["start_time"]}, duration={d["duration"]}, 
flagant='{d["flagant"]}', status={d["status"]}, calib_rank={d["calib_rank"]}, 
craco_record={d["craco_record"]}, craco_size={d["craco_size"]}, 
weight_reset={d["weight_reset"]}, weightsched={d["weight_sched"]}
WHERE sbid={sbid}
"""
        cur.execute(update_sql)
        conn.commit()


def push_sbid_observation(sbid, conn=None, cur=None):
    log.info(f"updating observation database for {sbid}...")
    try:
        cracosched = CracoSchedBlock(sbid)
        d = cracosched.format_sbid_dict()
    except Exception as error:
        log.warning(f"failed to get status for SB{sbid}... use Unknown for this sbid... error - {error}")
        d = dict(
            sbid=sbid, alias="Unknown", corr_mode="Unknown",
            craco_record=False, start_freq=-1, end_freq=-1,
            central_freq=-1, footprint="Unknown", template="Unknown",
            start_time=-1, duration=-1, flagant="none",
            status=-1, calib_rank=-1, craco_size=-1,
            weight_sched=-1, weight_reset=False
        )

    try:
        _update_craco_sched_status(craco_sched_info=d, conn=conn, cur=cur)
    except Exception as error:
        log.critical(f"failed to push schedblock status for {sbid}... please check... \n error - {error}")

######### function to update observation #######
def get_db_max_sbid(conn=None, cur=None):
    if conn is None: conn = get_psql_connect()
    if cur is None: cur = conn.cursor()

    cur.execute("SELECT MAX(sbid) FROM observation")
    maxsbid = cur.fetchone()[0]

    return maxsbid

# for real time, we need to rewrite the following function without metaflag
def run_observation_update(
    latestsbid, defaultsbid=None, waittime=60, 
    maxtry=3
):
    """
    based on the latest sbid, update the observation table
    this can be used in sbrunner
    """
    # get maximum sbid in the database first
    conn = get_psql_connect()
    cur = conn.cursor()

    maxsbid = get_db_max_sbid(conn=conn, cur=cur)

    if maxsbid is None:
        log.error("no sbid registered in observation database...")
        if defaultsbid is None:
            raise ValueError("no sbid found in observation database...")
        maxsbid = defaultsbid
        log.info(f"will use {defaultsbid} to as maximum sbid to update database...")
    log.info(f"previous maximum sbid found in database is - {maxsbid}")

    for sbid in range(maxsbid+1, latestsbid+1):
        ### run meta data loader here
        log.info(f"updating sbid - {sbid}")
        success = False
        for i in range(maxtry):
            try:
                metamanager = MetaManager(sbid)
                metamanager.run(skadi=False)
                push_sbid_observation(sbid, conn=conn, cur=cur)
                success = True
                break
            except EOFError:
                log.info(f"copy metadata unsuccessfully... deleting and rerun - tried {i+1}")
                os.system(f"rm {metamanager.workdir}/{metamanager.metaname}")
                time.sleep(waittime)
            except Exception as error:
                log.info(f"something goes wrong for this metadata... deleting and rerun - tried {i+1}")
                os.system(f"rm {metamanager.workdir}/{metamanager.metaname}")
                time.sleep(waittime)
        if not success:
            log.error(f"cannot load metadata from tethys for {sbid}... use skadi one instead...")
            try:
                metamanager = MetaManager(sbid)
                metamanager.run(skadi=True)
            except Exception as error:
                log.info("cannot load metadata from skadi... push the database anyway...")
            push_sbid_observation(sbid, conn=conn, cur=cur)


### auto scheduling related - how to schedule all different stuff...
class PipeSched:
    def __init__(self, sleeptime=60, dryrun=True, test=False):
        self.sleeptime = sleeptime
        self.conn = get_psql_connect()
        self.cur = self.conn.cursor()
        self.engine = get_psql_engine()

        self.dryrun = dryrun

        self.slackbot = SlackPostManager(test=test)

    def _query_nonrun_sbid(self):
        """
        query sbid need to be queued...
        """
        sql = f"""SELECT sbid FROM observation
WHERE tsp=false AND delete=false AND weight_reset=false
AND craco_record=true AND status > 3
ORDER BY sbid ASC
""" # status need to be double checked!
        
        self.cur.execute(sql)
        res = self.cur.fetchall()

        if len(res) == 0: return []
        return [i[0] for i in res]
    

    def _sbid_run(self, sbid, timethreshold=1.5, post=False):
        """
        run a given sbid - either run prepare_skadi, or run run_calib or wait
        """
        calfinder = CalFinder(sbid)
        #pdb.set_trace()
        calsbid, calstatus = calfinder.query_calib_table(timethreshold=timethreshold)
        if calsbid is None:
            #pdb.set_trace()
            log.info(f"cannot find existing sbid for calibration for {sbid}... will create a new one")
            calsbid = calfinder.query_observe_table(timethreshold=timethreshold)
            if calsbid is None:
                log.warning(f"no calibration found for {sbid}... will wait for further observation...")
                if post:
                    self.slackbot.post_message(
                        f"*[SCHEDULER]* cannot find calibration solution for {sbid}",
                        mention_team=True,
                    )
                return # i.e., do nothing
            ### now it is time to schedule calibration run...
            log.info("scheduling calibration...")
            self._run_calib(calsbid=calsbid)
            return
            
        if calstatus == 1: # calibration is running now
            log.info(f"calibration for {sbid} - {calsbid} is still running... wait for it to finish...")
            return
        log.info(f"running pipeline run for {sbid} with {calsbid}...")
        self._run_piperun(
            obssbid=sbid, calsbid=calsbid, nqueues=2
        )

    def _subprocess_execute(self, cmds, envs, post=False,):
        if isinstance(cmds, str): cmds = [cmds]

        if self.dryrun: 
            log.info(f"dryrun - going to run `{cmds}`")
            if post:
                self.slackbot.post_message(f"*<SHELL><DRYRUN>* running {cmds}")
            return

        with subprocess.Popen(
            cmds, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True,
            env=envs, text=True, shell=True
        ) as p:
            for outputline in p.stdout:
                print(outputline, end="")
        if p.returncode != 0:
            log.warning(f"error in running the following command - {cmds}")
            if post:
                self.slackbot.post_message(f"*<SHELL>* error in running the following shell script - {cmds}")
        return p.returncode
        
    def _run_calib(self, calsbid, post=True):
        envs = os.environ.copy()
        calibcmd = f"./run_calib.py -cal {calsbid}"
        log.info(f"run the following command for calibration - {calibcmd}")
        if post:
            self.slackbot.post_message(
                f"*[SCHEDULER]* submit calibration process for {calsbid}"
            )
        # subprocess.run([calibcmd], shell=True, capture_output=True, text=True, env=envs)
        p = self._subprocess_execute(calibcmd, envs=envs, post=True)
        if p != 0: # calibration goes wrong
            update_table_single_entry(int(calsbid), "calib_rank", -3, "observation")

    def _run_piperun(self, obssbid, calsbid, nqueues=2, post=True):
        # TODO - add injection here
        envs = os.environ.copy()
        runcmd = f"./prepare_skadi.py -cal {calsbid} -obs {obssbid} --nqueues {nqueues}"
        log.info(f"running pipeline run for {obssbid} with calibration {calsbid}")
        if post:
            self.slackbot.post_message(
                f"*[SCHEDULER]* submit pipeline run for {obssbid} with calibration {calsbid}"
            )
        # subprocess.run([runcmd], shell=True, capture_output=True, text=True, env=envs)
        self._subprocess_execute(runcmd, envs=envs, post=True)

    def run(self, timethreshold=1.):
        self.slackbot.post_message(
            "*[SCHEDULER]* automatic scheduler has been enabled"
        )
        try:
            while True: #
                try:
                    sbid_to_run = self._query_nonrun_sbid()
                    log.info(f"found {len(sbid_to_run)} schedule blocks to be processed...")
                    for sbid in sbid_to_run:
                        self._sbid_run(sbid=sbid, timethreshold=timethreshold)
                        time.sleep(1.5) # sleep for 1 second so that slack to display everything
                    time.sleep(self.sleeptime)
                except Exception as error:
                    self.slackbot.post_message(
                        f"*[SCHEDULER]* exception raised - {error}"
                    )
                    time.sleep(self.sleeptime)
        except KeyboardInterrupt:
            self.slackbot.post_message(
                "*[SCHEDULER]* automatic scheduler has been disabled"
            )
            return 

### slack posting
class SlackPostManager:
    def __init__(self, test=True, ):
        self.__load_webclient(test=test)

    def __load_webclient(self, test):
        slack_config = load_config(section="slack")
        self.client = WebClient(slack_config["slacktoken"])
        if test: self.channel = slack_config["testchannel"]
        else: self.channel = slack_config["channel"]

        _oplst = load_config(section="slack_notification")
        oplst = [v for _, v in _oplst.items()] # this is for notification

        ### make a mention block
        self.mention_msg = ", ".join([f"<@{user}>" for user in oplst])
        self.mention_block = self._format_text_block(self.mention_msg)

    def post_message(self, msg_blocks, thread_ts=None, mention_team=False):
        if isinstance(msg_blocks, str):
            msg_blocks = [self._format_text_block(msg_blocks)]
        if isinstance(msg_blocks, dict):
            msg_blocks = [msg_blocks]

        if mention_team: msg_blocks.append(self.mention_block)

        try:
            if thread_ts is None:
                postresponse = self.client.chat_postMessage(
                    channel=self.channel,
                    blocks=msg_blocks,
                )
            else:
                postresponse = self.client.chat_postMessage(
                    channel=self.channel,
                    thread_ts=thread_ts,
                    blocks=msg_blocks,
                )
            return postresponse
        except Exception as error:
            log.error(f"failed to post message to {self.channel}... \n error - {error}")
            return None

    def upload_file(self, files, comment="", thread_ts=None, mention_team=False):
        if mention_team: comment += f" {self.mention_msg}"
        try:
            if thread_ts is None: # thread_ts should be a string!!!
                postresponse = self.client.files_upload(
                    channels=self.channel,
                    initial_comment=comment,
                    file=files
                )
            else:
                postresponse = self.client.files_upload(
                    channels=self.channel,
                    initial_comment=comment,
                    thread_ts=thread_ts,
                    file=files
                )
        except Exception as error:
            log.error(f"error uploading file - {error}")
            return None
        return postresponse
    
    def _format_text_block(self, msg):
        return {
            "type": "section",
            "text": dict(type="mrkdwn", text=msg)
        }
    
    def _format_text_twocols(self, msgs):
        return {
            "type": "section",
            "fields": [
                dict(type="mrkdwn", text=msg)
                for msg in msgs
            ]
        }

    def get_thread_ts_from_response(self, response):
        response_dict = response.data
        if "files" in response_dict:
            try:
                thread_files = response_dict["file"]["shares"]["private"][self.channel]
                if len(thread_files) != 1:
                    log.warning(f"{len(thread_files)} found for this response... will only choose the first one")
                return thread_files[0]["ts"]
            except:
                log.error(f"cannot load thread_ts from the response... it is in a file format...")
                return None
        try: return response_dict["ts"]
        except:
            log.info("cannot load thread_ts from the response... it is in a thread format...")
            return None


######## several other functions to use for quick scheduling #######
### update observation table
def update_table_single_entry(sbid, column, value, table, conn=None, cur=None,):
    if conn is None:
        conn = get_psql_connect()
    if cur is None:
        cur = conn.cursor()

    if isinstance(value, str):
        value = f"""'{value}'"""
    ### update part not if there is nothing if we update nothing
    updatesql = f"""UPDATE {table}
SET {column}={value} WHERE sbid={sbid}
"""
    cur.execute(updatesql)
    conn.commit()    

def query_table_single_column(sbid, column, table, conn=None, cur=None):
    if conn is None:
        conn = get_psql_connect()
    if cur is None:
        cur = conn.cursor()

    sql = f"""SELECT {column} FROM {table} WHERE sbid={sbid}"""
    cur.execute(sql)
    res = cur.fetchone()

    if res is None: return None
    return res[0]

def reject_calibration(sbid, reject_run=True):
    pass


### for loading service ###
try:
    import Ice
except:
    print("cannot load Ice package...")
import os


def _get_ice_comm():
    host = 'icehost-mro.atnf.csiro.au'
    port = 4061
    timeout_ms = 5000
    default_loc = "IceGrid/Locator:tcp -h " + host + " -p " + str(port) + " -t " + str(timeout_ms)

    init = Ice.InitializationData()
    init.properties = Ice.createProperties()
    if "ICE_CONFIG" not in os.environ:
        loc = default_loc
    else:
        ice_cfg_file = os.environ['ICE_CONFIG']
        ice_parset = ParameterSet(ice_cfg_file)
        loc = ice_parset.get_value('Ice.Default.Locator', default_loc)

    init.properties.setProperty('Ice.Default.Locator', loc)
    return Ice.initialize(init)
    
def _get_ice_service(comm=None):
    if comm is None: comm = _get_ice_comm()
    return SB.ISchedulingBlockServicePrx.checkedCast(
       comm.stringToProxy("SchedulingBlockService@DataServiceAdapter")
    )

def _get_meta_max_sbid():
    metafolder = "/CRACO/DATA_00/craco/metadata"
    recentmeta = sorted(glob.glob(f"{metafolder}/SB*.json.gz"))[-1]
    log.debug(f"extract latest sbid based on metadata... {recentmeta}")
    metafname = recentmeta.split("/")[-1]
    try:
        return int(metafname[2:7])
    except Exception as error:
        log.warning(f"cannot get sbid from latest metafile - {metafname}")
        log.warning(f"error message - {error}")
        return None

def sbid_observation_finish(sbid, service=None):
    if service is None: service = _get_ice_service()
    state = service.getState(sbid).value
    log.debug(f"observation state for {sbid} is {state}")
    if state <= 3: return False
    return True

def get_recent_finish_sbid(service=None):
    maxsbid = _get_meta_max_sbid()
    if maxsbid is None: return None
    try:
        while not sbid_observation_finish(maxsbid, service=service):
            maxsbid -= 1
        return maxsbid
    except Exception as error:
        log.error(f"cannot load maxsbid - error message {error}")
        return None
