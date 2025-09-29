#!/usr/bin/env python
#### this script is used for casda archiving
'''
several things should be covered is
(1) renaming files and make soft link at a given directory
(2) functions to call rclone scripts
(3) get metadata for a given uvfits
'''

import os
import re

from astropy.io import fits
from astropy.time import Time
from craft import uvfits

import numpy as np

from aces.askapdata.schedblock import SB, SchedulingBlock

from craco.fixuvfits import fix
from craco.datadirs import SchedDir, ScanDir

import logging
logger = logging.getLogger(__name__)

def metadata_dict2xml(metadata, indent=2):
    items = [" " * indent + f"<{k}>{v}</{k}>" for k, v in metadata.items()]
    return "<metadata>\n" + "\n".join(items) + "\n</metadata>"

def execute_fixuvfits(uvfitspath):
    try: fix(uvfitspath)
    except Exception as error:
        logger.info(f"cannot fix {uvfitspath} due to - {error}")

class UvfitsCasdaMetadata:
    def __init__(self, uvfitspath):
        self.uvfitspath = os.path.abspath(uvfitspath)
        self.uvsource = uvfits.open(uvfitspath)
        self.hdulist = fits.open(uvfitspath)

        ### load basic info
        self._load_pointing_config()
        self._load_observation_config()
        self._load_owner_config()
        self._load_freq_config()

        ### load calibration files
        self._load_calfile()

    def _format_isotime(self, time, fmt="%Y-%m-%dT%H:%M:%S"):
        assert isinstance(time, Time), f"wrong time type - {type(time)}"
        time_dt = time.datetime
        return time_dt.strftime(fmt)

    def _load_observation_config(self):
        # parse data from uvfitspath
        # currently, we will try to prase it from HISTORY header
        hdrcmt = self.hdulist[0].header["HISTORY"].__str__().replace("\n", "")
        # find it out through history header
        outdirs = re.findall("--out (.*?) --fcm", hdrcmt)
        outdirs += re.findall("--outdir (.*?) --fcm", hdrcmt)
        pathpart = self.uvfitspath.split("/")
        if outdirs: 
            outdir = outdirs[0]
            # this is what you will get - /data/craco/craco/SB076041/scans/00/20250819124127
            outdirpart = outdir.split("/")
            self.sbid = int(outdirpart[-4][2:])
            self.askapscan = outdirpart[-2]
            self.cracoscan = outdirpart[-1]
            self.beam = pathpart[-1][1:3]
        else:
            self.sbid = int(pathpart[-5][2:])
            self.scan = pathpart[-3]
            self.tstart = pathpart[-2]
            self.beam = pathpart[-1][1:3]

    def _load_pointing_config(self):
        # this is loaded from the last hdu
        data = self.hdulist[3].data[0]
        self.field = data["SOURCE"]
        self.ra = data["RAEPO"]
        self.dec = data["DECEPO"]

    def _load_owner_config(self,):
        schedblock = SchedulingBlock(self.sbid)
        self.owner = schedblock._service.getOwner(self.sbid)

    def _load_freq_config(self):
        # this is loaded from the first header
        hdr = self.hdulist[0].header
        fch1 = hdr["CRVAL4"]
        foff = hdr["CDELT4"]
        ch1 = hdr["CRPIX4"]
        nchan = hdr["NAXIS4"]
        ### get actual frequency from above values
        self.freqs = (np.arange(nchan, dtype=float) - ch1 + 1) * foff + fch1 # in the unit of Hz
        self.cfreq = np.mean(self.freqs)
        self.nchan = nchan
        self.chanwidth = foff

    @property
    def scanstart(self, ):
        starttime = Time(self.uvsource.start_date, format="jd")
        return self._format_isotime(starttime)
    
    @property
    def scanend(self, ):
        endtime = Time(self.uvsource.end_date, format="jd")
        return self._format_isotime(endtime)

    @property
    def timesteps(self,):
        return self.uvsource.nsamps

    @property
    def casdafname(self,):
        return f"cracoData.{self.field}.SB{self.sbid}.beam{self.beam}.{self.cracoscan}.uvfits"
    
    @property
    def archivefolder(self,):
        return f"/data/craco/craco/archive/SB{self.sbid}"
    
    @property
    def tsamp(self,):
        return self.uvsource.tsamp.to("s").value

    def _load_calfile(self,):
        calfolder = f"/CRACO/DATA_00/craco/SB{self.sbid:0>6}/cal/{self.beam:0>2}"
        self.calnpy = f"{calfolder}/b{self.beam:0>2}.aver.4pol.smooth.npy"
        self.freqnpy = f"{calfolder}/b{self.beam:0>2}.aver.4pol.freq.npy"

    def casda_metadata(self,):
        return dict(
            filename = self.casdafname,
            project = self.owner,
            sbid = self.sbid,
            beam = self.beam,
            scanid = self.cracoscan,
            scanstart = self.scanstart,
            scanend = self.scanend,
            ra = np.deg2rad(self.ra), 
            dec = np.deg2rad(self.dec),
            coordsystem = "J2000",
            fieldname = self.field,
            polarisations = "XX",
            numchan = self.nchan,
            centrefreq = self.cfreq,
            chanwidth = self.chanwidth,
            timeSteps = self.timesteps,
            inttime = self.tsamp
        )

    def dump_casda_metadata(self, folder):
        casdameta = self.casda_metadata()
        casdametaxml = metadata_dict2xml(casdameta)
        xmlfname = self.casdafname.replace(".uvfits", ".xml")
        logger.info(f"dumping metadata info to {xmlfname}")
        with open(f"{folder}/{xmlfname}", "w") as fp:
            fp.write(casdametaxml)

    def prepare_casda_upload(self):
        os.makedirs(self.archivefolder, exist_ok=True)
        ### first of all, uvfits itself
        scanfolder = f"{self.archivefolder}/{self.cracoscan}"
        os.makedirs(scanfolder, exist_ok=True)
        cmd = f"ln -s {self.uvfitspath} {scanfolder}/{self.casdafname}"
        logger.info(f"making soft link - {cmd}")
        os.system(cmd)
        self.dump_casda_metadata(folder=scanfolder)
        ### second, calibration
        calfolder = f"{self.archivefolder}/cal"
        logger.info(f"copying calibration files...")
        os.makedirs(calfolder, exist_ok=True)
        os.system(f"cp {self.calnpy} {calfolder}")
        os.system(f"cp {self.freqnpy} {calfolder}")

class ScanCasdaMetadata:
    def __init__(self, sbid, scan, tstart):
        """
        here we use craco argument definition;
        sbid contains SB0; scan is two digit value;
        tstart is literally the scan
        """
        logger.info(f"running casda preparation for {sbid} {scan}/{tstart}")
        self.scheddir = SchedDir(sbid=sbid)
        self.scandir = ScanDir(sbid=sbid, scan=f"{scan}/{tstart}")

    def run_scan_casda_prepare(self):
        for uvfitspath in self.scandir.uvfits_paths:
            logging.info(f"looking into {uvfitspath}")
            ucm = UvfitsCasdaMetadata(uvfitspath=uvfitspath)
            ucm.prepare_casda_upload()

    


if __name__ == "__main__":
    # uvfitspath = "/CRACO/DATA_01/craco/SB076946/scans/00/20250916164012/b18.uvfits"
    # UCM = UvfitsCasdaMetadata(uvfitspath=uvfitspath)
    # UCM.prepare_casda_upload()
    sbid = "SB076946"
    scan = "00"
    tstart = "20250916164012"

    scm = ScanCasdaMetadata(sbid=sbid, scan=scan, tstart=tstart)
    scm.run_scan_casda_prepare()