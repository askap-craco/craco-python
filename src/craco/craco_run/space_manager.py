# this module contains all functions related to space management
# i.e., file deleting, information collecting etc.

from craco.datadirs import SchedDir, ScanDir, RunDir

import os
import gzip
import glob

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

class HeadNodeManager:
    def __init__(self, sbid):
        self.sbid = sbid
        self.scheddir = SchedDir(sbid)

        self._get_workdir()

    def _get_workdir(self):
        sched_headdir = self.scheddir.sched_head_dir
        self.workdir = os.path.join(sched_headdir, "SEARCHMETA")
        os.makedirs(self.workdir, exist_ok=True)

    def run_scans_check(self):
        scans = self.scheddir.scans
        logger.info(f"check status for {len(scans)} scans...")
        for scan in scans:
            scanmanager = ScanManager(self.sbid, scan)
            scanmanager.run_check()

    def run_scans_delete(self):
        scans = self.scheddir.scans
        logger.info(f"Deleting {len(scans)} scans...")
        for scan in scans:
            scanmanager = ScanManager(self.sbid, scan)
            scanmanager.run_delete()

### functions to check log file...
def _load_logfile_realtime(scandir):
    logfile = f"{scandir.scan_head_dir}/run.log"
    if os.path.exists(logfile):
        with open(logfile) as fp:
            log = fp.readlines()
        return log[0]
    logfile = f"{scandir.scan_head_dir}/run.log.gz"
    if os.path.exists(logfile):
        with gzip.open(logfile, "rb") as fp:
            log = fp.readlines()#.decode("utf-8")
        return log[0].decode("utf-8")
    return None

def _load_logfile_offline(scandir, runname="results"):
    logfiles = glob.glob(f"{scandir.scan_head_dir}/{runname}/SB*.log")
    if len(logfiles) == 0: 
        return None
    logfile = sorted(logfiles)[-1]
    if os.path.exists(logfile):
        with open(logfile) as fp:
            log = fp.readlines()
        return log[0]
    return None

###############


class ScanManager:
    def __init__(self, sbid, scan, runname="results"):
        logger.info(f"Initializing ScanManager for scan {scan}...")
        self.sbid = sbid
        self.scan = scan
        self.scandir = ScanDir(sbid, scan)
        self.rundir = RunDir(self.sbid, self.scan, runname)

        self._get_workdir()

    def _get_workdir(self):
        sched_headdir = self.scandir.scheddir.sched_head_dir
        self.workdir = os.path.join(sched_headdir, "SEARCHMETA", self.scan)
        os.makedirs(self.workdir, exist_ok=True)

    def _check_pcb(self, beam=None):
        if beam is None:
            for ibeam in range(36):
                pcbpath = self.rundir.beam_pcb(ibeam)
                if pcbpath is None: return False
            return True
        pcbpath = self.rundir.beam_pcb(beam)
        if pcbpath is None: return False
        return True
    
    ### organize information about the scan
    ### what we need is (1) command line file; (2) summary file
    def _get_cmdline(self):
        # try realtime format
        cmdline = _load_logfile_realtime(self.scandir)
        if cmdline is not None: return cmdline
        # try offline format
        cmdline = _load_logfile_offline(self.scandir)
        if cmdline is not None: return cmdline
        return None
    
    def _get_rankfile(self):
        rank_file = f"{self.scandir.scan_head_dir}/beam_only.rank"
        if os.path.exists(rank_file): return rank_file
        rank_file = f"{self.scandir.scan_head_dir}/mpipipeline.rank"
        if os.path.exists(rank_file): return rank_file
        return None

    def _get_scansummary(self):
        summary_file = f"{self.scandir.scan_head_dir}/scan_summary.json"
        if os.path.exists(summary_file): return summary_file
        return None

    ### run the check, and copy related file
    def _run_prepare(self):
        cmdline = self._get_cmdline()
        rankfile = self._get_rankfile()
        summaryfile = self._get_scansummary()
        pcbcheck = self._check_pcb()
        
        if cmdline is None:
            logger.info(f"Scan {self.scan} has missing command line file...")
            return False
        with open(f"{self.workdir}/cmdline.txt", "w") as fp:
            fp.write(cmdline)
        
        if rankfile is not None:
            os.system(f"cp {rankfile} {self.workdir}/")

        if summaryfile is not None:
            os.system(f"cp {summaryfile} {self.workdir}/")

        if not pcbcheck:
            logger.info(f"Scan {self.scan} has missing PCB files.")
            return True # if no pcb found, we can delete the scan

        return True

    def run_check(self):
        preparestatus = self._run_prepare()
        if preparestatus:
            logger.info(f"SBID {self.sbid} - Scan {self.scan} is ready for deletion.")
            os.system(f"touch {self.workdir}/READY")

    def run_delete(self, safe=True):
        if os.path.exists(f"{self.workdir}/READY"):
            if safe:
                allfiles = glob.glob(f"{self.workdir}/*")
                if len(allfiles) == 1:
                    logger.info(f"will not delete this scan as only READY file is available")
                    return
            logger.info(f"Deleting SBID {self.sbid} scan {self.scan}...")
            # we will keep the structure... but not the files
            os.system(f"rm -rf {self.scandir.scan_head_dir}/*")

if __name__ == "__main__":
    for sbid in range(70000, 84500):
        try:
            headmanager = HeadNodeManager(sbid)
            headmanager.run_scans_check()
            headmanager.run_scans_delete()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt detected, exiting...")
            break
        except Exception as error:
            logger.info(f"Failed to check scan for sbid {sbid}...")
            with open("failed_sbid.txt", "a") as fp:
                fp.write(f"{sbid} - {error}\n")

        # note -  50000 to 65000 done...


    # sbid = 60000
    # headmanager = HeadNodeManager(sbid)
    # headmanager.run_scans_check()
    # headmanager.run_scans_delete()