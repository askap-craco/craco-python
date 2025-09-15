#!/usr/bin/env python
"""
CRACO data directory information

Copyright (C) CSIRO 2022
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys
import glob
import json
import pandas as pd

from astropy.io import fits

import logging
log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def get_dir_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size
    
def format_sbid(sbid, padding=True, prefix=True):
    """
    format sbid into a desired format
    """
    if isinstance(sbid, str): sbid = int(
        "".join([i for i in sbid if i.isdigit()])
    )

    sbidstr = ""
    if prefix: sbidstr += "SB"
    if padding: sbidstr += f"{sbid:0>6}"
    else: sbidstr += f"{sbid}"

    return sbidstr

def check_path(path):
    return True if os.path.exists(path) else False

class DataDirs:
    def __init__(self):
        try: self.cracodata = os.environ["CRACO_DATA"]
        except: self.cracodata = None
        if self.cracodata is None: 
            self.cracodata = "/data/craco/craco/"
            log.info("no CRACO_DATA environ var found... use `/data/craco/craco/`")

    def node_dir(self, nidx=0):
        return f"/CRACO/DATA_{nidx:0>2}/craco/" # by default head node

    @property
    def data_nodes(self, ):
        for nidx in range(1, 19):
            yield self.node_dir(nidx)

    def sched_dir(self, sbid, nidx=0):
        node_dir = self.node_dir(nidx = nidx)
        return os.path.join(node_dir, format_sbid(sbid))

    def scan_dir(self, sbid, scan=None, nidx=0):
        sched_dir = self.sched_dir(sbid, nidx=nidx)
        if scan is None: # work out the first scan 
            scans = sorted(glob.glob(f"{sched_dir}/scans/??/??????????????"))
            if len(scans) == 0: return None
            return scans[0]
        return os.path.join(sched_dir, f"scans/{scan}")

    def run_dir(self, sbid, scan=None, run="results", nidx=0):
        scan_dir = self.scan_dir(sbid, scan=scan, nidx=nidx)
        return os.path.join(scan_dir, f"{run}/")

    ### function to extract information from the path
    def _re_match_pat(self, pat, text):
        matched = re.findall(pat, text)
        if len(matched) == 0: return None
        return matched[0]

    def path_to_scan(self, path):
        return self._re_match_pat("(\d{2}/\d{14})", path)

    def path_to_sbid(self, path):
        return self._re_match_pat("(SB\d{6})", path)

    def path_to_node(self, path):
        return self._re_match_pat("DATA_(\d{2})", path)

    def path_to_runname(self, path):
        return self._re_match_pat("\d{14}/(.*?)/", path)

    ### TODO - add deletor here as well?

class SchedDir:
    def __init__(self, sbid, datadirs=None):
        self.datadirs = DataDirs() if datadirs is None else datadirs
        self.sbid = sbid # can be string, int, with or without SB

        ### get basic information from flagfile
        self._load_flagfile()

    @property
    def sched_head_dir(self):
        return self.datadirs.sched_dir(self.sbid, )

    @property
    def scans(self):
        """
        use head node to list all available scans
        """
        allscanpaths = glob.glob(os.path.join(self.sched_head_dir, "scans/??/??????????????/"))
        ### we need to exclude the scan with -1
        allscans = [self.datadirs.path_to_scan(path) for path in allscanpaths]
        return sorted([scan for scan in allscans if scan is not None])

    @property
    def metafile(self):
        metapath = "{}/{}.json.gz".format(
            self.sched_head_dir, format_sbid(self.sbid, padding=False)
        )
        backup_metapath = "{}/metadata/{}.json.gz".format(
            self.datadirs.node_dir, format_sbid(self.sbid, padding=False)
        )
        
        final_metapath = None
        if check_path(metapath):
            final_metapath = metapath
        elif check_path(backup_metapath):
            final_metapath = backup_metapath
        
        return final_metapath

    @property
    def flagfile(self):
        flagpath = "{}/{}.antflag.json".format(
            self.sched_head_dir, format_sbid(self.sbid)
        )
        return flagpath if check_path(flagpath) else None

    @property
    def run_dir(self):
        return f"{self.sched_head_dir}/runscript"

    @property
    def sched_data_dirs(self, ):
        for data_node in self.datadirs.data_nodes:
            yield f"{data_node}/{format_sbid(self.sbid)}"

    @property
    def cal_dir(self):
        return f"{self.sched_head_dir}/cal"

    @property
    def cal_sbid(self):
        cal_dir_path = os.readlink(self.cal_dir)
        return self.datadirs.path_to_sbid(cal_dir_path)

    def beam_cal_path(self, beam):
        beam = f"{int(beam):0>2}"
        return f"{self.cal_dir}/{beam}/b{beam}.aver.4pol.smooth.npy"

    def _load_flagfile(self):
        if self.flagfile is None: 
            self.start_mjd = None; self.flagant = None
            return
        with open(self.flagfile) as fp:
            metaf = json.load(fp)

        try: self.start_mjd = eval(metaf["trange"])[0]
        except: self.start_mjd = metaf["startmjd"]
        self.flagant = eval(metaf["flagants"])

    def get_size(self):
        """
        get the space used for this schedule block
        """
        self.sched_sizes = {
            f"DATA_{self.datadirs.path_to_node(data_dir)}": 
            get_dir_size(data_dir) / 1024 / 1024 / 1024
            for data_dir in self.sched_data_dirs
        }

        return self.sched_sizes

class ScanDir:
    def __init__(self, sbid, scan=None):
        self.scheddir = SchedDir(sbid)
        self.datadirs = self.scheddir.datadirs

        self.scan = scan if scan is not None else self.scheddir.scans[0]

        if not check_path(self.scan_head_dir): 
            raise ValueError(f"no scan {scan} found under {sbid}")

        self._get_beam_node_dict()

    @property
    def scan_data_dirs(self):
        for data_node in self.datadirs.data_nodes:
            yield f"{data_node}/{format_sbid(self.scheddir.sbid)}/scans/{self.scan}"
    
    @property
    def scan_head_dir(self):
        return self.datadirs.scan_dir(self.scheddir.sbid, scan=self.scan)

    @property
    def scan_rank_file(self):
        rank_file = f"{self.scan_head_dir}/beam_only.rank"
        if check_path(rank_file): return rank_file
        return f"{self.scan_head_dir}/mpipipeline.rank"

    @property
    def runs(self):
        runpaths = glob.glob(f"{self.scan_head_dir}/*/")
        return sorted([
            self.datadirs.path_to_runname(runpath) for runpath in runpaths
        ])

    def _get_beam_node_dict(self):
        """ beam (int) - node (str) mapping """
        if not check_path(self.scan_rank_file):
            raise NotImplementedError("get beam node mapping without rank file has not been implemented")
        # load it from rank file
        with open(self.scan_rank_file) as fp:
            rank_file_text = fp.read()
        node_beam_map = re.findall(
            "=skadi-(\d{2}).*# Beam .* (\d{1,2}).*xrtdevid",
            rank_file_text
        )

        if len(node_beam_map) == 0:
            _node_beam_map = re.findall(
                "=skadi-(\d{2}).*# Beam (\d{1,2})( processor)? xrtdevid",
                rank_file_text
            )
            ### cuz here we extract three element
            node_beam_map = [(node, beam) for node, beam, _ in _node_beam_map]
        
        self.beam_node_dict = {int(beam): node for node, beam in node_beam_map}

        if len(self.beam_node_dict) != 36:
            log.info("not all beams recorded in rankfile... %s", self.beam_node_dict)

    def beam_scandir_path(self, beam):
        scan_dir = self.datadirs.scan_dir(
            self.scheddir.sbid, self.scan, 
            nidx = self.beam_node_dict[int(beam)]
        )
        return scan_dir    

    def beam_uvfits_path(self, beam):
        scan_dir = self.beam_scandir_path(beam)
        uvfits_path = f"{scan_dir}/b{int(beam):0>2}.uvfits"
        return uvfits_path if check_path(uvfits_path) else None
    
    def beam_pcb_path(self, beam):
        scan_dir = self.beam_scandir_path(beam)
        pcb_path = f"{scan_dir}/pcbb{int(beam):0>2}.fil"
        return pcb_path if check_path(pcb_path) else None

    def beam_plan0_path(self, beam):
        scan_dir = self.beam_scandir_path(beam)
        plan_path =  f"{scan_dir}/beam{int(beam):0>2}/plans/plan_iblk0.pkl"
        return plan_path if check_path(plan_path) else None
    
    def beam_rfi_stats_path(self, beam):
        scan_dir = self.beam_scandir_path(beam)
        rfi_stats_path = f"{scan_dir}/flagging_stats_log_b{int(beam):0>2}.csv"
        return rfi_stats_path if check_path(rfi_stats_path) else None

    @property
    def uvfits_paths(self):
        return [self.beam_uvfits_path(beam) for beam in range(0, 36)]

    @property
    def pcb_paths(self):
        return [self.beam_pcb_path(beam) for beam in range(0, 36)]


    @property
    def uvfits_paths_exists(self):
        """return uvfits that exsits only"""
        return [path for path in self.uvfits_paths if check_path(path)]

    @property
    def uvfits_count(self):
        count = 0
        for path in self.uvfits_paths:
            if check_path(path): count += 1
        return count

    def beam_ics_path(self, beam):
        scan_dir = self.datadirs.scan_dir(
            self.scheddir.sbid, self.scan, 
            nidx = self.beam_node_dict[int(beam)]
        )
        return f"{scan_dir}/ics_b{int(beam):0>2}.fil"

    def beam_cas_path(self, beam):
        # note - cas_b??.fil file is rubbish
        return None

    def get_size(self):
        self.scan_sizes = {
            f"DATA_{self.datadirs.path_to_node(data_dir)}":
            get_dir_size(data_dir) / 1024 / 1024 / 1024
            for data_dir in self.scan_data_dirs
        }

        return self.scan_sizes

### for candidate snippet directory
class CandDir:
    def __init__(self, sbid, beam, iblk, scan=None):
        self.rundir = RunDir(sbid, scan=scan, run="results")
        self.scandir = self.rundir.scandir
        self.scheddir = self.scandir.scheddir
        self.datadirs = self.scandir.datadirs

        self.beam_node_dict = self.scandir.beam_node_dict
        self.beam = beam
        self.iblk = iblk

    @property
    def cand_snippet_dir(self):
        beam_scan_dir = self.scandir.beam_scandir_path(self.beam)
        return f"{beam_scan_dir}/beam{self.beam:0>2}/candidates/iblk{self.iblk}"

    @property
    def cand_snippet_uvfits_path(self):
        uvfitspath = self.scandir.beam_uvfits_path(self.beam)
        if isinstance(uvfitspath, str): return uvfitspath
        return f"{self.cand_snippet_dir}/candidate.uvfits"

    @property
    def cand_info(self):
        return f"{self.cand_snippet_dir}/candidate.txt"

class RunDir:
    def __init__(self, sbid, scan=None, run="results"):
        self.scandir = ScanDir(sbid, scan=scan)
        self.scheddir = self.scandir.scheddir
        self.datadirs = self.scheddir.datadirs

        self.run = run
        self.beam_node_dict = self.scandir.beam_node_dict

    @property
    def run_data_dirs(self):
        for data_node in self.datadirs.data_nodes:
            yield f"{data_node}/{format_sbid(self.scheddir.sbid)}/scans/{self.scandir.scan}/{self.run}/"

    @property
    def run_head_dir(self):
        return os.path.join(self.scandir.scan_head_dir, f"{self.run}/")

    @property
    def run_file(self):
        """
        get the latest run file
        """
        ### get all files
        scannum, scantime = self.scandir.scan.split("/")
        scantime = scantime[-6:]
        runfiles = sorted(
            glob.glob(f"{self.scheddir.run_dir}/run.{format_sbid(self.scheddir.sbid)}.{scannum}.{scantime}.{self.run}.*.sh")
        )
        ### get latest run file
        if len(runfiles) == 0: return None
        return runfiles[-1]

    def _extract_bash_param(self, par, text):
        matched = re.findall(par, text)
        if len(matched) == 0: return None
        return matched[0]

    def get_run_params(self):
        with open(self.run_file) as fp:
            run_file_content = fp.read()
        ### startmjd
        self.startmjd = self._extract_bash_param("startmjd=(.*)\n", run_file_content)
        self.flagant = self._extract_bash_param("flagant=(.*)\n", run_file_content)
        ### note - add more attributes here if more is needed

    ### candidate related
    def beam_folder(self, beam):
        nidx = self.scandir.beam_node_dict[int(beam)]
        return self.datadirs.run_dir(
            sbid=self.scheddir.sbid,
            scan=self.scandir.scan,
            run=self.run, nidx=nidx,
        )

    def beam_candidate(self, beam):
        return os.path.join(self.beam_folder(beam), f"candidates.b{beam:0>2}.txt")

    def beam_pcb(self, beam):
        pcbpath = os.path.join(self.beam_folder(beam), f"pcb{beam:0>2}.fil")
        if not check_path(pcbpath):
            pcbpath = self.scandir.beam_pcb_path(beam)
        return pcbpath

    def beam_rfimask(self, beam):
        return os.path.join(self.beam_folder(beam), f"RFI_tfmask.b{beam:0>2}.fil")

    def beam_unique_cand(self, beam):
        uniq_cand_path = os.path.join(self.beam_folder(beam), f"clustering_output/candidates.b{beam:0>2}.txt.uniq.csv")
        if not check_path(uniq_cand_path):
            uniq_cand_path = os.path.join(self.beam_folder(beam), f"clustering_output/candidates.b{beam:0>2}.uniq.csv")
        return uniq_cand_path

    def raw_candidate_paths(self):
        all_candidates = [self.beam_candidate(beam) for beam in range(0, 36)]
        return [path for path in all_candidates if check_path(path)]

    def clust_candidate_paths(self):
        all_candidates = [self.beam_unique_cand(beam) for beam in range(0, 36)]
        return [path for path in all_candidates if check_path(path)]

    # note - for any additional files, add it here

class CalDir:
    """module to manage calibrations"""
    def __init__(self, sbid):
        self.sbid = sbid

    @property
    def cal_head_dir(self):
        return f"/CRACO/DATA_00/craco/calibration/{format_sbid(self.sbid)}"

    def beam_cal_dir(self, beam):
        return f"{self.cal_head_dir}/{beam:0>2}"
    
    def beam_cal_binfile(self, beam):
        return f"{self.beam_cal_dir(beam)}/b{beam:0>2}.aver.4pol.bin"
    
    def beam_cal_freqfile(self, beam):
        return f"{self.beam_cal_dir(beam)}/b{beam:0>2}.aver.4pol.freq.npy"

    def beam_cal_smoothfile(self, beam):
        return f"{self.beam_cal_dir(beam)}/b{beam:0>2}.aver.4pol.smooth.npy"

class UvfitsDir:
    """
    this is prepared for casda...
    """
    def __init__(self, uvfitspath):
        self.uvfitspath = os.path.abspath(uvfitspath)
        self.hdulist = fits.open(self.uvfitspath)

        self.snippet = True if "candidate" in self.uvfitspath else False

        self._load_freq_config()
        self._load_observation_config()
        self._load_pointing_config()
        self._load_owner_config()

    # TODO - sbid, scan, tstart, beam, owner
    def _load_freq_config(self):
        # this is loaded from the first header
        hdr = self.hdulist[0].header
        fch1 = hdr["CRVAL4"]
        foff = hdr["CDELT4"]
        ch1 = hdr["CRPIX4"]
        nchan = hdr["NAXIS4"]
        ### get actual frequency from above values
        self.freqs = (np.arange(nchan, dtype=float) - ch1 + 1) * foff + fch1 # in the unit of Hz

    def _load_pointing_config(self):
        # this is loaded from the last hdu
        data = self.hdulist[3].data[0]
        self.field = data["SOURCE"]
        self.ra = data["RAEPO"]
        self.dec = data["DECEPO"]

    def _load_observation_config(self):
        # parse data from uvfitspath
        # currently, we will try to prase it from HISTORY header
        hdrcmt = self.hdulist[0].header["HISTORY"].__str__().replace("\n", "")
        # find it out through history header
        outdirs = re.findall("--out (.*?) --fcm", hdrcmt)
        pathpart = self.uvfitspath.split("/")
        if outdirs: 
            outdir = outdirs[0]
            # this is what you will get - /data/craco/craco/SB076041/scans/00/20250819124127
            outdirpart = outdir.split("/")
            self.sbid = int(outdirpart[-4][2:])
            self.scan = outdirpart[-2]
            self.tstart = outdirpart[-1]
            self.beam = pathpart[-1][1:3]
        elif not self.snippet:
            self.sbid = int(pathpart[-5][2:])
            self.scan = pathpart[-3]
            self.tstart = pathpart[-2]
            self.beam = pathpart[-1][1:3]
        else:
            raise ValueError("snippet uvfitsdir not implemented!")
        
    def _load_owner_config(self,):
        schedblock = SchedulingBlock(self.sbid)
        self.owner = schedblock._service.getOwner(self.sbid)

    def format_json(self,):
        return dict(
            sbid = self.sbid,
            scan = self.scan,
            tstart = self.tstart,
            beam = self.beam,
            owner = self.owner,
            field = self.field,
            ra = self.ra,
            dec = self.dec,
            fmin = np.min(self.freqs),
            fmax = np.max(self.freqs),
            nchan = len(self.freqs),
        )



def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    dir = ScanDir(values.files[0])
    print(len(dir.beam_node_dict), dir.beam_node_dict)
    
    

if __name__ == "__main__":
    _main()