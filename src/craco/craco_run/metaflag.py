#!/usr/bin/env python
### functions to get 1) bad antennas; 2) get starting and end time

from scipy.interpolate import interp1d

import numpy as np
import logging
import json
import re
import os

from craco.metadatafile import MetadataFile 
from craco.datadirs import SchedDir, ScanDir

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def _format_sbid(sbid, padding=True):
    "perform formatting for the sbid"
    if isinstance(sbid, int): sbid = str(sbid)
    if sbid.isdigit(): # if sbid are digit
        if padding: return "SB{:0>6}".format(sbid)
        return f"SB{sbid}"
    return sbid

def get_mjd_start_from_uvfits_header(fname, hdr_size = 16384):
    """
    load PZERO4 from fits header directly
    """
    with open(fname, 'rb') as o:
        raw_h = o.read(hdr_size)
    pattern = "PZERO4\s*=\s*(\d+.\d+)\s*"
    matches = re.findall(pattern, str(raw_h))
    if len(matches) !=1:
        raise RuntimeError(f"Something went wrong when looking for mjd start in the header, I used this regex - {pattern}")

    jd_start = float(matches[0])
    mjd_start = jd_start - 2400000.5
    if 50000 < mjd_start < 80000:
        return mjd_start
    else:
        raise RuntimeError(f"I found a really unexpected value of MJD start from the UVfits header - {mjd_start}")

def find_true_range(bool_array):
    """
    get all start and end indices for consecutive True values
    """
    # Identify the indices where the array changes from False to True or True to False
    change_indices = np.where(np.diff(np.concatenate(([False], bool_array, [False]))))[0]

    # Pair the start and end indices
    ranges = [(start, end) for start, end in zip(change_indices[::2], change_indices[1::2])]

    return ranges

class MetaManager:
    """
    class to manage all metadata files
    """
    def __init__(self, obssbid, frac=0.2):
        self.obssbid = _format_sbid(obssbid, padding=True)
        ### get head node folder for this sbid
        self.workdir = f"/CRACO/DATA_00/craco/{self.obssbid}"
        self.metaname = f"{_format_sbid(obssbid, padding=False)}.json.gz"
        self.badfrac = frac # determine the fraction of bad antenna

    ### get meta data and save it to correct place
    def _get_skadi_metadata(self, overwrite=False):
        if not overwrite:
            if os.path.exists(f"{self.workdir}/{self.metaname}"):
                log.info("metadata exists... stop copying...")
                return
        else:
            log.warning("overwriting existing metadata...")
        
        ### get meta data from tethys
        # TODO - get meta from skadi directly
        scpcmd = f"""cp /CRACO/DATA_00/craco/metadata/{self.metaname} {self.workdir}"""
        # scpcmd = f'''scp "tethys:/data/TETHYS_1/craftop/metadata_save/{self.metaname}" {self.workdir}'''
        log.info(f"copying metadata {self.metaname} from head node")
        os.system(scpcmd)

    def _get_tethys_metadata(self, overwrite=False):
        if not overwrite:
            if os.path.exists(f"{self.workdir}/{self.metaname}"):
                log.info("metadata exists... stop copying...")
                return
        else:
            log.warning("overwriting existing metadata...")
        
        scpcmd = f'''scp "tethys:/data/TETHYS_1/craftop/metadata_save/{self.metaname}" {self.workdir}'''
        log.info(f"copying metadata {self.metaname} from tethys")
        os.system(scpcmd)

    def _get_flagger_info(self, ):
        self.metaantflag = MetaAntFlagger(
            f"{self.workdir}/{self.metaname}", fraction=self.badfrac,
        )

        dumpfname = f"{self.workdir}/{self.obssbid}.antflag.json"
        # self.metaantflag.run(dumpfname)
        self.metaantflag._run(self.obssbid, dumpfname)

        ### note there are information useful in this self.metaantflag

    def run(self, skadi=True):
        if skadi: 
            self._get_skadi_metadata(overwrite=False)
        else:
            self._get_tethys_metadata(overwrite=False)
        self._get_flagger_info()

class MetaAntFlagger:
    
    def __init__(self, metafile, sbid=None, fraction=0.2):
        log.info(f"loading metadata file from {metafile}")
        self.meta = MetadataFile(metafile)
        self.antflags = self.meta.antflags
        
        ### get basic information
        self._get_info()
        self.badants = self._find_bad_ant(fraction=fraction)
        ### make it to a list
        self.badants = list(self.badants + 1)
        log.info(f"finding {len(self.badants)} bad antennas...")
        self._find_good_ranges()

        ### get flag interp
        self._get_flag_interp()
        
        
    def _get_info(self):
        nt, na = self.antflags.shape
        self.nt = nt
        self.na = na
    
    def _find_bad_ant(self, fraction=0.2):
        """
        the antenna will be flagged if 80% of the time, it is bad
        """
        flagantsum = self.antflags.sum(axis=0)
        # work out threshold automatically
        flagmed = np.median(flagantsum)
        return np.where(flagantsum >= flagmed + fraction * self.nt)[0]
    
    def _find_good_ranges(self):
        flagtimesum = self.antflags.sum(axis=1) - len(self.badants)
        good_bool = flagtimesum <= 0
        self.good_bool = good_bool
        self.good_ranges = find_true_range(good_bool)

    # @property
    def _get_flag_interp(self):
        """
        self.meta.times.value - the times in the metadata file
        self.good_bool - corresponding flag values
        """
        self.flag_interp = interp1d(
            self.meta.times.value, self.good_bool,
            kind="previous", axis=0, bounds_error=True, copy=False,
        )

    def _check_good(self, mjd):
        """check if the data is flagged with a given mjd"""
        try:
            return self.flag_interp(mjd) == 1.
        except:
            log.info(f"error raised when interpolating data... mjd = {mjd}")
            return False

    def find_startmjd(self, mjd):
        """
        work out the start mjd with a given mjd
        """
        if self._check_good(mjd): return 0 # 0 should be fine, or any small number

        log.info(f"mjd - {mjd} is not a good startmjd... finding the good one")
        times = self.meta.times.value
        bools = self.good_bool
        for i in np.where(times > mjd)[0]:
            if bools[i]: return times[i]
        return None
        
    ### get range of time
    def get_start_end_time(self):
        if len(self.good_ranges) == 0:
            log.warning("no good range found for this schedule block...")
            return None, None
        ### get the best ranges
        ranges_time = np.array([t[1] - t[0] for t in self.good_ranges])
        best_start, best_end = self.good_ranges[np.argmax(ranges_time)]
        log.info(f"the best range found... {best_start} ~ {best_end}, it lasted for {best_end - best_start} hardware samples...")
        
        return [self.meta.times[best_start].value, self.meta.times[best_end-1].value]

    # def get_flag_ant(self):
    #     return list(self.badants + 1)

    def get_stats(self):
        self.trange = self.get_start_end_time()
        # self.badants = self.get_flag_ant()


    def run(self, dumpfname):
        """
        note - we won't use it recently
        """
        self.get_stats()

        metainfo = dict(
            trange = self.trange.__str__(),
            flagants = self.badants.__str__()
        )
        log.info(f"dumping metadata information to {dumpfname}")
        with open(dumpfname, "w") as fp:
            json.dump(metainfo, fp, indent=4)

    def _run(self, sbid, dumpfname=None):
        ### this _run function is used for scan specified tstart
        # self.get_stats()

        startmjd = {}
        scheddir = SchedDir(sbid)
        for scan in scheddir.scans:
            scandir = ScanDir(sbid=scheddir.sbid, scan=scan)
            uvfitspath = scandir.uvfits_paths[0]
            if not os.path.exists(uvfitspath):
                log.info(f"{uvfitspath} not found... use None to continue...")
                startmjd[scan] = str(None)
                continue
            
            uvfitsmjd = get_mjd_start_from_uvfits_header(uvfitspath)
            startmjd[scan] = str(self.find_startmjd(uvfitsmjd))

        self.startmjds = startmjd # store startmjd values for prepare skadi

        metainfo = dict(
            startmjd = startmjd,
            flagants = self.badants.__str__()
        )

        if dumpfname is not None:
            log.info(f"dumping metadata information to {dumpfname}")
            with open(dumpfname, "w") as fp:
                json.dump(metainfo, fp, indent=4)

        return metainfo
            
def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        description="dump information needs to be used in the pipeline", 
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-meta", "--meta", type=str, help="Path to the meta data file (.json.gz)", default=None)
    parser.add_argument("-dump", "--dump", type=str, help="Path to save the information", default="./metainfo.json")
    parser.add_argument("-frac", "--frac", type=float, help="Fractional of bad interval to be considered as a bad antenna", default=0.8)

    values = parser.parse_args()

    metaflag = MetaAntFlagger(
        metafile = values.meta,
        fraction = values.frac,
    )
    metaflag.run(values.dump)


if __name__ == "__main__":
    main()
