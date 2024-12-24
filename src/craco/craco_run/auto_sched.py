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

# from .metaflag import MetaManager
from craco.craco_run.metaflag import MetaManager
# from metaflag import MetaManager

import logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.CRITICAL)

def load_config(config=None, section="dbwriter"):
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
def get_psql_connect(section="dbwriter"):
    config = load_config(section=section)
    return psycopg2.connect(**config)

from sqlalchemy import create_engine 
def get_psql_engine(section="dbwriter"):
    c = load_config(section=section)
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

    # @property
    # def weight_reset(self, ):
    #     if self.template in ["OdcWeights", "Beamform"]: return True
    #     if self.template == "Bandpass":
    #         if "dcal" in self.alias: return True
    #     return False
    @property
    def fcm_version(self, ):
        try: return eval(self.obsvar["fcm.version"])
        except: return -1
    
    def format_sbid_dict(self, ):
        ### format sbid dictionary
        d = dict(sbid=self.sbid)
        d["alias"] = self.alias
        d["corr_mode"] = self.corrmode
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
        
        d["weight_sbid"] = self.weight_sched
        d["fcm_version"] = self.fcm_version
            
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
        note = "running"
        badant = ""
    else:
        status = 0
        try:
            valid, goodant, goodbeam, note = calcls.rank_calsol(plot=plot)
            badant = ",".join([str(i) for i in calcls.calbadant])
        except Exception as err:
            log.info(f"failed to get calibration quality... - {sbid}")
            log.info(f"error message - {err}")
            valid, goodant, goodbeam = False, 0, 0
            status = 2 # something goes run
            note = "rank failed"
            badant = ""
    
    # in the real time we don't need to update the observation table...
    
    cur.execute(f"SELECT * FROM calibration WHERE SBID={sbid}")
    res = cur.fetchall()
    if len(res) == 0: # insert
        insert_sql = f"""INSERT INTO calibration (
    sbid, valid, solnum, goodant, goodbeam, status, note, badant
)
VALUES (
    {sbid}, {valid}, {solnum}, {goodant}, {goodbeam}, {status}, '{note}', '{badant}'
)
"""
        cur.execute(insert_sql)
        conn.commit()
        
    else:
        update_sql = f"""UPDATE calibration
SET valid={valid}, solnum={solnum}, goodant={goodant}, 
goodbeam={goodbeam}, status={status}, note='{note}',
badant='{badant}'
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
        self, sbid, flagfile="/home/craftop/share/fixed_freq_flags_calib.txt"
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
        sql_df = pd.read_sql(f"SELECT flagant FROM observation WHERE sbid={self.sbid}", engine)

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
        # flagant = self.flag_ants
        # return [i-1 for i in range(1, 31) if i not in flagant]
        return self.get_goodant_from_badlst(self.flag_ants)
    
    def get_goodant_from_badlst(self, badant, nant=30):
        """
        get 0-indexed good antenna based on a list of bad antennas...
        """
        return [i-1 for i in range(1, nant+1) if i not in badant]

    def _load_flagfile_chan(self, solfreqs, flag=True):
        arr_lst = [(solfreqs / 1e6 <= freqs[1]) & (solfreqs / 1e6 >= freqs[0]) for freqs in self.flagfreqs]
        freqflag_bool = np.sum(arr_lst, axis=0).astype(bool)
        if flag: return np.where(freqflag_bool)[0]
        return np.where(~freqflag_bool)[0]

    ### sometimes all antennas will be bad if one or two beams are bad!
    def _find_bad_ants_from_badant_list(self, badants, badant_thres=0.8, maxnant=30):
        """
        if the antenna is bad in 80% beams, it will be the real bad antenna
        """
        badants = np.array(badants)
        real_badants = []
        for ia in range(1, maxnant + 1):
            ant_badbeam_count = (badants == ia).sum()
            if ant_badbeam_count > badant_thres * 36:
                real_badants.append(ia)
        return real_badants

    ### calculate the score
    def rank_calsol(
        self, phase_difference_threshold=30, plot=True, saveplot=True,
        good_frac_threshold=0.8, bad_frac_threshold=0.6, # threshold to determine the good beams/antennas
        nant_threshold=12, # if there are 12 good antennas, then valid...
        nbeam_threshold = 34, # if there are 34 valid beams, then valid...
    ):
        beams = []; beam_phase_diff = []; badants = []
        for ibeam in range(36):
            try:
                beamcalsol = CalSolBeam(self.sbid, ibeam)
                unflagchans = self._load_flagfile_chan(beamcalsol.freqs, flag=False)
                badants.extend(beamcalsol.badant)

                # phdif = beamcalsol.extract_phase_diff(self.good_ant, unflagchans)
                phdif = beamcalsol.extract_phase_diff(None, unflagchans) # remove the dependent on good ant...
                beam_phase_diff.append(phdif)
                beams.append(ibeam)
            except Exception as error:
                log.info(f"cannot load solution from beam {ibeam} for {self.sbid}...")
                log.info(f"error message - {error}")
                continue
        note = ""
        self.calbadant = self._find_bad_ants_from_badant_list(badants)

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
            if saveplot:
                fig.savefig(f"{self.caldir.cal_head_dir}/calsol_qc.png", bbox_inches="tight")
                plt.close()

        ### save some value for checking...
        self.sbid_phase_diff = sbid_phase_diff # in a shape of nbeam, nant, 

        ### work out statistics
        good_ant_index = self.get_goodant_from_badlst(self.calbadant)
        sbid_phase_diff = sbid_phase_diff[:, good_ant_index, :]
        sbid_phase_good = sbid_phase_diff < phase_difference_threshold
        sbid_good_frac = sbid_phase_good.mean(axis=-1) # take the mean of frequency

        ### decide the number of the good beam
        good_beam_count = (sbid_good_frac.mean(axis=1) > bad_frac_threshold).sum()
        good_ant_count = (sbid_good_frac.mean(axis=0) > bad_frac_threshold).sum()
        valid_calsol = True
        if good_beam_count < nbeam_threshold:
            note += f"{good_beam_count} good beams found..."
            valid_calsol = False
        if good_ant_count < nant_threshold:
            note += f"{good_ant_count} good antennas found..."
            valid_calsol = False
        # valid_calsol = np.all(sbid_good_frac > good_frac_threshold)
        # if not valid_calsol: note = f"good frac"

        ### if there is one beam missing, valid should be false
        if nbeam < 36:
            log.warning(f"only {nbeam} calibration solution found... - {beams}")
            valid_calsol = False
            note += f"{nbeam} beams calibration solution..."
        # if len(self.calbadant) > nant_threshold:
        #     valid_calsol = False
        #     note = f"{len(self.calbadant)} bad antennas"

        return valid_calsol, good_ant_count, good_beam_count, note
        
class CalSolBeam:
    def __init__(self, sbid, beam, maxant=30):
        self.sbid = sbid
        self.caldir = CalDir(sbid)
        self.maxant = maxant

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

        ### work out the bad antenna... as a list
        nanval = np.isnan(self.binbp) | np.isinf(self.binbp)
        nanfrac_per_ant = nanval.mean(axis=1)
        frac_thres = 0.95 # if more than 95% data is bad for this antenna, this is a bad antenna
        badant = sorted(np.arange(1, nant+1)[nanfrac_per_ant > frac_thres])
        self.badant = [i for i in badant if i <= self.maxant]

    def __count_nan(self, arr, isnan=True, fraction=True):
        total = np.size(arr)
        nancount = (np.isnan(arr) | np.isinf(arr)).sum()
        if fraction:
            if isnan: return nancount / total
            return 1 - nancount / total
        if isnan: return nancount
        return total - nancount

    def extract_phase_diff(self, goodant=None, unflagchan=None):
        if goodant is None: goodant = np.arange(self.maxant)
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

        self.conn = get_psql_connect(section="dbreader")
        self.cur = self.conn.cursor()
        self.engine = get_psql_engine(section="dbreader")

        self.__update_database()
        self.__get_sbid_property()

    def __update_database(self):
        self.cur.execute(f"""select sbid from observation where sbid={self.sbid}""")
        res = self.cur.fetchall()
        if len(res) == 0:
            log.info(f"no sbid information found for sbid{self.sbid}... will query the aces survey to update...")
            push_sbid_observation(self.sbid)

    def __get_sbid_property(self):
        self.cur.execute(f"""select central_freq,footprint,weight_sbid,start_time,flagant,fcm_version from observation
where sbid={self.sbid}""")
        res = self.cur.fetchall()
        assert len(res) == 1, f"found {len(res)} records in observation database for {self.sbid}..."
        self.freq, self.footprint, self.weight_sched, self.start_time, self.flagant, self.fcm_version = res[0]

    def query_calib_table(self, timethreshold=365):
        """
        query calibration table to find the most appropriate sbid

        it will return calibration sbid, and calibration status (just in case something is running)
        """
        joinsql = f"""SELECT o.sbid,o.flagant,c.status
FROM calibration c JOIN observation o ON c.sbid=o.sbid
WHERE o.weight_sbid={self.weight_sched} AND o.central_freq={self.freq} AND o.footprint='{self.footprint}' 
AND ((c.valid=True AND c.solnum=36 AND c.status=0) OR c.status=1)
AND o.start_time>={self.start_time-timethreshold} AND o.start_time<={self.start_time+timethreshold}
AND o.fcm_version={self.fcm_version}
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
    
    def query_observe_table(self, timethreshold=365):
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

    def get_cal_path(self, timethreshold=365):
        calsbid, calstatus = self.query_calib_table(timethreshold=timethreshold)
        calpath = None
        if calsbid is not None and calstatus == 0: #status 0 means there is calibration
            # calsbid is a 5 digit sbid
            calpath = f"/CRACO/DATA_00/craco/calibration/SB{calsbid:0>6}"
        return calpath

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
    duration, flagant, status, weight_sbid, fcm_version
)
VALUES (
    {d["sbid"]}, '{d["alias"]}', '{d["corr_mode"]}', {d["start_freq"]}, {d["end_freq"]},
    {d["central_freq"]}, '{d["footprint"]}', '{d["template"]}', {d["start_time"]},
    {d["duration"]}, '{d["flagant"]}', {d["status"]}, {d["weight_sbid"]}, {d["fcm_version"]}
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
flagant='{d["flagant"]}', status={d["status"]}, weight_sbid={d["weight_sbid"]}, 
fcm_version={d["fcm_version"]}
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
            sbid=sbid, alias="Unknown", corr_mode="Unknown", start_freq=-1, end_freq=-1,
            central_freq=-1, footprint="Unknown", template="Unknown",
            start_time=-1, duration=-1, flagant="none",
            status=-1, weight_sbid=-1, fcm_version=-1,
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
    

    def _sbid_run(self, sbid, timethreshold=365, post=False):
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
    def __init__(self, test=True, channel=None):
        self.__load_webclient(test=test, channel=channel)

    def __load_webclient(self, test, channel=None):
        slack_config = load_config(section="slack")
        self.client = WebClient(slack_config["slacktoken"])
        if channel is not None: self.channel = channel
        elif test: self.channel = slack_config["testchannel"]
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


### for realtime operation
class CalJob:

    CAL_TS_ONFINISH = "/CRACO/SOFTWARE/craco/craftop/softwares/craco_run/ts_calibration_call.py"
    CAL_RUN_TS_SOCKET = "/data/craco/craco/tmpdir/queues/cal"
    TMPDIR = "/data/craco/craco/tmpdir"

    def __init__(self, scandir):
        # database related
        self.conn = get_psql_connect()
        self.cur = self.conn.cursor()
        self.engine = get_psql_engine()

        self._parse_scan(scandir)
        self.scandir = scandir

    def _parse_scan(self, scandir):
        # this is an example of scandir
        # /data/craco/craco/SB062220/scans/00/20240506160118/
        scanlst = scandir.strip("/").split("/")
        self.sbid = int(scanlst[3][2:]) # 62220
        self.scan = "/".join(scanlst[-2:]) # 00/20240506160118

    @property
    def _calib_is_running(self):
        """
        query the database to check if there is a calibration run running
        """
        self.cur.execute(f"""select sbid, status from calibration where sbid={self.sbid}""")
        query_result = self.cur.fetchall()

        if len(query_result) == 0: 
            running = False # nothing in the database... nothing has been ran...
        else:
            _, status = query_result[0]
            if status == 1: running = True
            else: running = False
        return running

    @property
    def _calib_is_done(self):
        """
        query the database to check if the calibration has finished...
        theoretically, we will not use this one
        """
        self.cur.execute(f"""select sbid, status from calibration where sbid={self.sbid}""")
        query_result = self.cur.fetchall()

        if len(query_result) == 0: 
            done = False # nothing in the database... nothing has been ran...
        else:
            _, status = query_result[0]
            if status == 0: done = True
            else: done = False
        return done

    def tsp_calib(self):
        log.info(f"queuing up calibration solution for {self.scandir}")
        environment = {
            "TS_SOCKET": self.CAL_RUN_TS_SOCKET,
            "TS_ONFINISH": self.CAL_TS_ONFINISH,
            "TMPDIR": self.TMPDIR,
        }
        ecopy = os.environ.copy()
        ecopy.update(environment)

        cmd = f"""mpi_run_beam.sh {self.scandir} `which mpi_do_calibrate_rt.sh`"""

        subprocess.run(
            [f"tsp {cmd}"], shell=True, capture_output=True,
            text=True, env=ecopy,
        )

    def copy_sol(self):
        log.info(f"copying calibration to head node...")
        environment = {
            "TS_SOCKET": self.CAL_RUN_TS_SOCKET,
            "TS_ONFINISH": self.CAL_TS_ONFINISH,
            "TMPDIR": self.TMPDIR,
        }
        ecopy = os.environ.copy()
        ecopy.update(environment)

        copycal_path = "/CRACO/SOFTWARE/craco/craftop/softwares/craco_run/copycal.py"
        cpcmd = f"{copycal_path} -cal {self.sbid}"

        subprocess.run(
            [f"tsp {cpcmd}"], shell=True, capture_output=True,
            text=True, env=ecopy,
        )

    def run(self):
        """
        main function to queue everything
        """
        ### continue to do something even if some calibration is running, so I commented it out
        # if self._calib_is_running:
        #     log.info("there is currently a calibration process running for this sbid...")
        #     return
        
        if self._calib_is_done:
            log.info("there is already a good calibration for this sbid...")
            return

        # in this case, we need to create a calibration
        self.tsp_calib()
        self.copy_sol()

# note - for calibration
# mpi_run_beam.sh /data/craco/craco/SB062220/scans/00/20240506172535 /CRACO/SOFTWARE/craco/craftop/softwares/miniconda3/envs/craco/bin/mpi_do_calibrate.sh --start-mjd 0
# next step have a new mpi_do_calibrate_rt.sh

def queue_calibration(scandir):
    """
    queue the calibration as a tsp job
    the whole structure is similar to what we have in run_calib.py previously
    """
    caljob = CalJob(scandir)
    caljob.run()


##### for posting candidate to slack...
def run_post_cand_with_tsp():
    CAND_POST_TS_SOCKET = "/data/craco/craco/tmpdir/queues/cands"
    TMPDIR = "/data/craco/craco/tmpdir"

    environment = {
        "TS_SOCKET": CAND_POST_TS_SOCKET,
        "TMPDIR": TMPDIR,
    }
    ecopy = os.environ.copy()
    ecopy.update(environment)

    try:
        scan_dir = os.environ['SCAN_DIR']
    except Exception as KE:
        log.critical(f"Could not fetch the scan directory from environment variables!!")
        log.critical(KE)
        return
    else:
        cmd = f"post_scan_cands_image.py -outdir '{scan_dir}'"
        subprocess.run(
            [f"tsp {cmd}"], shell=True, capture_output=True,
            text=True, env=ecopy,
        )
        log.info(f"Queued posting candidate job - with command - {cmd}")