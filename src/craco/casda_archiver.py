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
import glob
import json
import subprocess
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import sqlite3
import xml.etree.ElementTree as ET
from enum import IntEnum
from typing import Optional, Union

from astropy.io import fits
from astropy.time import Time
from craft import uvfits

import numpy as np

from aces.askapdata.schedblock import SB, SchedulingBlock

from craco.fixuvfits import fix
from craco.datadirs import SchedDir, ScanDir, format_sbid
from craco.tools import cracocal2casatab

from craco.craco_run import auto_sched

from clink import api as clink
from configparser import ConfigParser

import psycopg2
from psycopg2.extras import RealDictCursor
_psycopg2_UniqueViolation = psycopg2.errors.UniqueViolation

### this is function to load tables from database
try:
    from sqlalchemy import create_engine
except ImportError:
    create_engine = None


import logging
logging.basicConfig(
    level = logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def metadata_dict2xml(metadata, indent=2):
    items = [" " * indent + f"<{k}>{v}</{k}>" for k, v in metadata.items()]
    return "<metadata>\n" + "\n".join(items) + "\n</metadata>"

def execute_fixuvfits(uvfitspath):
    try: 
        logger.info(f"fixing {uvfitspath}...")
        fix(uvfitspath)
        return 0
    except Exception as error:
        logger.info(f"cannot fix {uvfitspath} due to - {error}")
        return -1

class UvfitsCasdaMetadata:
    def __init__(self, uvfitspath):
        self.uvfitspath = os.path.abspath(uvfitspath)
        self._fix_uvfits() # fix uvfits file...
        self.uvsource = uvfits.open(uvfitspath)
        self.hdulist = fits.open(uvfitspath)

        ### load basic info
        self._load_pointing_config()
        self._load_observation_config()
        self._load_owner_config()
        self._load_freq_config()

        ### load calibration files
        self._load_calfile()

    def _fix_uvfits(self,):
        canwrite = os.access(self.uvfitspath, os.W_OK)
        logger.info(f"checking uvfits permission with OS.ACCESS - {canwrite}")
        if not canwrite:
            cmd = f"chmod +w {self.uvfitspath}"
            logger.info(f"executing {cmd} to add write permission...")
            os.system(cmd)
        fixstatus = execute_fixuvfits(self.uvfitspath)
        if not canwrite:
            cmd = f"chmod -w {self.uvfitspath}"
            logger.info(f"removing write permission...")
            os.system(cmd)

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
    def casdacasatabfname(self,):
        return f"cracoCal.{self.field}.SB{self.sbid}.beam{self.beam}.B0"
    
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
        xmlfname = self.casdafname.replace(".uvfits", ".craco_metadata.xml")
        logger.info(f"dumping metadata info to {xmlfname}")
        with open(f"{folder}/{xmlfname}", "w") as fp:
            fp.write(casdametaxml)

    def prepare_casda_upload(self, casacaltab=True):
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
        if not casacaltab:
            os.system(f"cp {self.calnpy} {calfolder}")
            os.system(f"cp {self.freqnpy} {calfolder}")
        else:
            ### here we convert casa numpy file to casatable
            cracocal2casatab.run_convert(
                tabpath = f"{calfolder}/{self.casdacasatabfname}",
                calpath = self.calnpy,
                freqpath = self.freqnpy,
                overwrite = False
            )

class ScanCasdaMetadata:

    CASDA_ARCHIVE_TS_SOCKET = "/data/craco/craco/tmpdir/queues/casdaupload"
    TMPDIR = "/data/craco/craco/tmpdir"
    # CASDA_ARCHIVE_TS_SOCKET = "/data/craco/craco/wan342/tmpdir/queues/casdaupload"
    # TMPDIR = "/data/craco/craco/wan342/tmpdir"

    def __init__(self, sbid, scan, tstart):
        """
        here we use craco argument definition;
        sbid contains SB0; scan is two digit value;
        tstart is literally the scan
        """
        logger.info(f"running casda preparation for {sbid} {scan}/{tstart}")
        self.scheddir = SchedDir(sbid=sbid)
        self.scandir = ScanDir(sbid=sbid, scan=f"{scan}/{tstart}")
        self.sbid = int(format_sbid(sbid, padding=False, prefix=False))
        self.cracoscan = tstart # this is cracoscan - timestamp
        self.archivefolder = f"/data/craco/craco/archive/SB{self.sbid}"

    def run_scan_casda_prepare(self):
        for uvfitspath in self.scandir.uvfits_paths:
            logging.info(f"looking into {uvfitspath}")
            try:
                ucm = UvfitsCasdaMetadata(uvfitspath=uvfitspath)
                ucm.prepare_casda_upload()
            except Exception as error:
                logger.error(f"cannot prepare for casda upload for {uvfitspath}... Error: {error}...")

    # def start_casda_rsync(self, target="setonix:/scratch/ja3/zwan4817/askapbuffer"):
    def start_casda_rsync(self, target="ozstartrans:/fred/oz002/zwang/forVM/askapbuffer"):
        """
        function to start rsync job to askapbuffer
        """
        cmd = f"rsync -av -P -L --include='***/{self.cracoscan}/***' --include='***/cal/***' --include='*/' --exclude='*' {self.archivefolder} {target}"
        ### run the command with tsp
        environment = {
            "TS_SOCKET": self.CASDA_ARCHIVE_TS_SOCKET,
            "TMPDIR": self.TMPDIR,
        }
        ecopy = os.environ.copy()
        ecopy.update(environment)
        subprocess.run(
            [f"tsp {cmd}"], shell=True, capture_output=True,
            text=True, env=ecopy,
        )
        logger.info(f"Queued casda uploading job - with command - {cmd}")

# NEW: clink integration
# ==============================================================================
# CLINK Event System Integration & Metadata Helpers
# ==============================================================================

def parse_sbid(sbid: Union[int, str]) -> int:
    """
    Extract clean integer SBID regardless of whether input is int, string, or contains 'SB'/'SB0' prefix.

    Examples:
      parse_sbid(82418)       -> 82418
      parse_sbid("82418")     -> 82418
      parse_sbid("SB82418")   -> 82418
      parse_sbid("SB082418")  -> 82418
    """
    if isinstance(sbid, int):
        return sbid
    s = str(sbid).strip()
    if s.upper().startswith("SB"):
        s = s[2:]
    return int(s)


def parse_metadata_xml(xml_path: str) -> dict:
    """Parse a CRACO XML metadata file into a dict."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    meta = {}
    for child in root:
        tag = child.tag
        val = child.text
        if val is not None:
            try:
                if "." in val:
                    val = float(val)
                else:
                    val = int(val)
            except ValueError:
                logger.debug(f"Failed to cast metadata '{tag}' value '{val}' to numeric. Leaving as string.")
        meta[tag] = val
    return meta


def get_calibration_files(archive_folder: str) -> dict:
    """List calibration tables/files under archive_folder/cal."""
    cal_dir = os.path.join(archive_folder, "cal")
    if not os.path.exists(cal_dir):
        return {"cal_folder": cal_dir, "files": []}
    files = [f for f in sorted(os.listdir(cal_dir)) if not f.startswith(".")]
    return {"cal_folder": cal_dir, "files": files}


def build_ready_for_copy_payload(sbid: Union[int, str], archive_folder: Optional[str] = None, include_file_size: bool = False) -> dict:
    """Construct CLINK ready_for_copy event payload matching cpmanager schema."""
    sbid_int = parse_sbid(sbid)
    sbid_str = str(sbid_int)

    if archive_folder is None:
        cand_standard = f"/data/craco/craco/archive/SB{sbid_int}"
        cand_padded = f"/data/craco/craco/archive/SB{sbid_int:06d}"
        if os.path.exists(cand_standard):
            archive_folder = cand_standard
        elif os.path.exists(cand_padded):
            archive_folder = cand_padded
        else:
            archive_folder = cand_standard

    scans_data = []
    project = None
    fieldname = None
    sample_obs_params = {}

    if os.path.exists(archive_folder):
        for entry in sorted(os.listdir(archive_folder)):
            scan_dir = os.path.join(archive_folder, entry)
            if os.path.isdir(scan_dir) and entry != "cal":
                scan_files = []
                xml_files = sorted(glob.glob(os.path.join(scan_dir, "*.craco_metadata.xml")))
                for xml_file in xml_files:
                    meta = parse_metadata_xml(xml_file)
                    if project is None and "project" in meta:
                        project = str(meta["project"])
                    if fieldname is None and "fieldname" in meta:
                        fieldname = str(meta["fieldname"])
                    if not sample_obs_params:
                        sample_obs_params = {k: str(v) for k, v in meta.items() if v is not None}

                    file_info = {
                        "filename": meta.get("filename", ""),
                        "metadata_file": os.path.basename(xml_file),
                        "beam": meta.get("beam"),
                        "scanstart": meta.get("scanstart"),
                        "scanend": meta.get("scanend"),
                        "ra": meta.get("ra"),
                        "dec": meta.get("dec"),
                        "polarisations": meta.get("polarisations"),
                        "numchan": meta.get("numchan"),
                        "centrefreq": meta.get("centrefreq"),
                        "chanwidth": meta.get("chanwidth"),
                        "timeSteps": meta.get("timeSteps"),
                        "inttime": meta.get("inttime")
                    }
                    if include_file_size:
                        data_file_path = os.path.join(scan_dir, meta.get("filename", ""))
                        if os.path.exists(data_file_path) and os.path.isfile(data_file_path):
                            file_info["size_bytes"] = os.path.getsize(data_file_path)
                        if os.path.exists(xml_file) and os.path.isfile(xml_file):
                            file_info["metadata_size_bytes"] = os.path.getsize(xml_file)
                    scan_files.append(file_info)

                scans_data.append({
                    "scanid": entry,
                    "files": scan_files
                })

    cal_info = get_calibration_files(archive_folder)

    obs_parameters = {
        "common.cp.processing_priority": "STANDARD",
        "sbid": sbid_str,
        "project": project or "UNKNOWN",
        "fieldname": fieldname or "UNKNOWN"
    }
    for k, v in sample_obs_params.items():
        obs_parameters[k] = str(v)

    payload = {
        "schedulingBlock": {
            "id": sbid_str,
            "alias": fieldname or "",
            "owner": project or "UNKNOWN",
            "state": "OBSERVED",
            "obsParameters": obs_parameters
        },
        "craco": {
            "sbid": sbid_str,
            "project": project or "UNKNOWN",
            "fieldname": fieldname or "UNKNOWN",
            "archive_folder": archive_folder,
            "calibration": cal_info,
            "scans": scans_data
        }
    }
    return payload


def setup_clink_environment(config_path: Optional[str] = None):
    """Load CLINK transport settings into os.environ."""
    if config_path and os.path.exists(config_path):
        logger.info(f"Loading CLINK config from {config_path}")
        with open(config_path, "r") as f:
            if config_path.endswith(".json"):
                cfg = json.load(f)
            else:
                cfg = {}
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        cfg[k.strip()] = v.strip().strip('"').strip("'")
        for k, v in cfg.items():
            os.environ[k] = str(v)


# NEW: clink integration
class ClinkPublisher:
    """Publishes CLINK events from SKADI."""

    def __init__(self, participant_name: str = "au.csiro.atnf.askap.craco", config_path: Optional[str] = None):
        setup_clink_environment(config_path)
        from clink import api as clink
        self.clink = clink
        self.participant = clink.Participant(participant_name)

    def emit_ready_for_copy(
        self,
        sbid: Union[int, str],
        event_type: str = "au.csiro.atnf.askap.craco.ready_for_copy",
        archive_folder: Optional[str] = None,
        subject: Optional[str] = None,
        test: bool = False,
        include_file_size: bool = False,
    ) -> bool:
        """Emit ready_for_copy CLINK event for an SBID or specific archive folder."""
        if not self.participant and not test:
            logger.error("CLINK participant not initialized. Cannot emit ready_for_copy event.")
            return False

        sbid_int = parse_sbid(sbid)

        payload = build_ready_for_copy_payload(
            sbid_int, 
            archive_folder=archive_folder,
            include_file_size=include_file_size
        )

        if subject is None:
            folder_path = payload.get("craco", {}).get("archive_folder", archive_folder)
            subject_urn = f"urn:askap:craco:::archive-folder/{folder_path}"
        else:
            subject_urn = subject

        if test:
            logger.info(f"TEST MODE: Would emit CLINK ready_for_copy event for SBID {sbid_int} (Type: {event_type})")
            print("--- CLINK EVENT PAYLOAD ---")
            print(f"Subject URN: {subject_urn}")
            print(f"Event Type : {event_type}")
            print("Payload    :")
            print(json.dumps(payload, indent=2))
            print("---------------------------")
            return True

        logger.info(f"Emitting CLINK ready_for_copy event for SBID {sbid_int} (Type: {event_type})...")
        try:
            self.participant.emit_event(
                subject=subject_urn,
                type=event_type,
                data=payload
            )
            logger.info(f"Successfully emitted CLINK event {event_type} for SBID {sbid_int}")

            try:
                am = ArchiveManager()
                am.update_archive_status(sbid=sbid_int, scan="SB_ALL", setonix_status=ArchiveStatus.READY_FOR_COPY_SENT)
            except Exception as e:
                logger.debug(f"Database status update skipped: {e}")
            return True
        except Exception as error:
            logger.error(f"Failed to emit CLINK event for SBID {sbid_int}: {error}")
            return False


# NEW: clink integration
class ClinkListener:
    """Listener daemon consuming datamanager CLINK events."""

    def __init__(self, participant_name: str = "au.csiro.atnf.askap.craco", config_path: Optional[str] = None):
        setup_clink_environment(config_path)
        self.clink = clink
        self.participant = clink.Participant(participant_name)
        self._register_handlers()

    def _extract_sbid(self, event) -> Optional[int]:
        """Extract SBID integer from event subject URN or event payload."""
        # 1. Check data payload fields
        if hasattr(event, "data") and isinstance(event.data, dict):
            sb_id = (
                event.data.get("sbid")
                or event.data.get("schedulingBlock", {}).get("id")
                or event.data.get("craco", {}).get("sbid")
            )
            if sb_id:
                try:
                    return int(sb_id)
                except (ValueError, TypeError) as e:
                    logger.error(f"Failed to parse integer from payload sbid field '{sb_id}'.")
                    raise ValueError(f"Failed to parse integer from payload sbid field '{sb_id}'. {e}")

        # 2. Check subject URN resource ID
        if hasattr(event, "subject_urn") and event.subject_urn and getattr(event.subject_urn, "resource", None):
            res_id = str(event.subject_urn.resource.id)
            try:
                return int(res_id)
            except ValueError:
                match = re.search(r"(\d+)", res_id)
                if match:
                    return int(match.group(1))
                else:
                    logger.error(f"Failed to parse integer from URN resource ID '{res_id}'.")
                    raise ValueError(f"Failed to parse integer from URN resource ID '{res_id}'.")

        # 3. Fallback: search raw subject string for SBID digits
        if hasattr(event, "subject") and event.subject:
            match = re.search(r"(\d+)", str(event.subject))
            if match:
                return int(match.group(1))

        return None

    def _register_handlers(self):
        """Register event handlers for copy and purge events."""
        am = ArchiveManager()

        @self.participant.on_event("au.csiro.atnf.askap.datamanager.copy.added_to_queue", name="craco.on_copy_queued", suppress_exceptions=True)
        def on_copy_queued(event, **kwargs):
            sbid = self._extract_sbid(event)
            logger.info(f"Received CLINK event: copy.added_to_queue for SBID {sbid}")
            if sbid:
                try:
                    am.update_archive_status(sbid=sbid, scan="SB_ALL", setonix_status=ArchiveStatus.COPY_QUEUED)
                except Exception as e:
                    logger.debug(f"DB update note: {e}")

        @self.participant.on_event("au.csiro.atnf.askap.datamanager.copy.started", name="craco.on_copy_started", suppress_exceptions=True)
        def on_copy_started(event, **kwargs):
            sbid = self._extract_sbid(event)
            logger.info(f"Received CLINK event: copy.started for SBID {sbid}")
            if sbid:
                try:
                    am.update_archive_status(sbid=sbid, scan="SB_ALL", setonix_status=ArchiveStatus.COPY_EXECUTING)
                except Exception as e:
                    logger.debug(f"DB update note: {e}")

        @self.participant.on_event("au.csiro.atnf.askap.datamanager.copy.completed", name="craco.on_copy_completed", suppress_exceptions=True)
        def on_copy_completed(event, **kwargs):
            sbid = self._extract_sbid(event)
            logger.info(f"Received CLINK event: copy.completed for SBID {sbid}")
            if sbid:
                try:
                    am.update_archive_status(sbid=sbid, scan="SB_ALL", setonix_status=ArchiveStatus.COPY_FINISHED)
                except Exception as e:
                    logger.debug(f"DB update note: {e}")

        @self.participant.on_event("au.csiro.atnf.askap.cpmanager.ready_for_purge", name="craco.on_ready_for_purge", suppress_exceptions=True)
        def on_ready_for_purge(event, **kwargs):
            sbid = self._extract_sbid(event)
            logger.info(f"Received CLINK event: ready_for_purge for SBID {sbid}")
            if sbid:
                try:
                    am.update_archive_status(sbid=sbid, scan="SB_ALL", setonix_status=ArchiveStatus.READY_FOR_PURGE)
                except Exception as e:
                    logger.debug(f"DB update note: {e}")

        @self.participant.on_event("au.csiro.atnf.askap.datamanager.purge.completed", name="craco.on_purge_completed", suppress_exceptions=True)
        @self.participant.on_event("au.csiro.atnf.askap.datamanager.purge.deleted", name="craco.on_purge_deleted", suppress_exceptions=True)
        def on_purge_completed(event, **kwargs):
            sbid = self._extract_sbid(event)
            logger.info(f"Received CLINK event: purge completed for SBID {sbid}")
            if sbid:
                try:
                    am.update_archive_status(sbid=sbid, scan="SB_ALL", setonix_status=ArchiveStatus.PURGED)
                except Exception as e:
                    logger.debug(f"DB update note: {e}")

    def start_listening(self):
        """Start blocking consumer loop."""
        if not self.participant:
            logger.error("CLINK participant not initialized. Cannot start listening.")
            return
        logger.info("Starting CLINK listener daemon for CRACO...")
        self.participant.consume()


# ==============================================================================
# Archiving Database & Status Tracking
# ==============================================================================

# NEW: clink integration
#### now we need to move everything to normal postgresql database

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

### this function to get connection details...
def get_psql_connect(section="dbreader"):
    if psycopg2 is None:
        raise ImportError("psycopg2 is required for PostgreSQL connection.")
    config = load_config(section=section)
    return psycopg2.connect(**config)

def get_psql_engine(section="dbreader"):
    if create_engine is None:
        raise ImportError("sqlalchemy is required for get_psql_engine.")
    c = load_config(section=section)
    engine_str = "postgresql+psycopg2://"
    engine_str += f"""{c["user"]}:{c["password"]}@{c["host"]}:{c["port"]}/{c["database"]}"""
    return create_engine(engine_str)

class ArchiveStatus(IntEnum):
    """Archiving lifecycle status codes."""
    DEFAULT = 0
    QUEUED = 1
    EXECUTING = 2
    FINISHED = 3
    READY_FOR_COPY_SENT = 10  # ready_for_copy emitted from Skadi
    COPY_QUEUED = 11          # Datamanager queued copy job
    COPY_EXECUTING = 12       # Copy job running
    COPY_FINISHED = 13        # Copy job completed
    READY_FOR_PURGE = 20      # SBID flagged ready for purge
    PURGED = 30               # Purge completed
    ERRORED = -1

class SkadiStatus(IntEnum):
    DEFAULT = 0
    READY = 1
    ERRORED = -1

class ArchiveManager:
    """Manages data archiving records for Acacia and Setonix archiving processes."""

    def __init__(self, db_path: str = "archive_status.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize the database and create the table if it doesn't exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS archive_records (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    sbid            INTEGER NOT NULL,
                    scan            TEXT NOT NULL,
                    acacia_status   INTEGER NOT NULL DEFAULT 0,
                    setonix_status  INTEGER NOT NULL DEFAULT 0,
                    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(sbid, scan)
                )
            """)
            conn.commit()

    # ------------------------------------------------------------------ #
    #  Write operations                                                    #
    # ------------------------------------------------------------------ #

    def insert_record(
        self,
        sbid: int,
        scan: str,
        uvfits_count: int = 0,
        skadi_status: SkadiStatus = SkadiStatus.DEFAULT,
        acacia_status: ArchiveStatus = ArchiveStatus.DEFAULT,
        setonix_status: ArchiveStatus = ArchiveStatus.DEFAULT,
    ) -> int:
        """
        Insert a new archive record.

        Returns the row id of the newly inserted record.
        Raises ValueError if the (sbid, scan) pair already exists.
        """
        try:
            with get_psql_connect(section="dbwriter") as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO archives (sbid, scan, uvfits_count, skadi_status, acacia_status, setonix_status)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (sbid, scan, uvfits_count, int(skadi_status), int(acacia_status), int(setonix_status)),
                )
                conn.commit()
                return cur.lastrowid
        except psycopg2.errors.UniqueViolation:
            raise ValueError(f"Record with SBID={sbid} and scan='{scan}' already exists.")

    def update_archive_status(
        self,
        sbid: int,
        scan: str,
        uvfits_count: Optional[int] = None,
        skadi_status: Optional[SkadiStatus] = None,
        acacia_status: Optional[ArchiveStatus] = None,
        setonix_status: Optional[ArchiveStatus] = None,
    ) -> bool:
        """
        Update acacia_status and/or setonix_status for a given (sbid, scan) pair.

        Pass only the field(s) you want to change; omitted fields are left untouched.
        Returns True if a row was updated, False if no matching record was found.
        Raises ValueError if neither status field is provided.
        """
        if acacia_status is None and setonix_status is None and skadi_status is None and uvfits_count is None:
            raise ValueError("At least one of acacia_status, setonix_status, skadi_status or uvfits_count must be provided.")

        if scan == "SB_ALL":
            logger.info(f"updating all scans for SBID {sbid}...")
            update_status = self.update_archive_status_sbid(
                sbid=sbid,
                skadi_status=skadi_status,
                acacia_status=acacia_status,
                setonix_status=setonix_status,
            )
            return update_status

        fields, values = [], []
        if acacia_status is not None:
            fields.append("acacia_status = %s")
            values.append(int(acacia_status))
        if setonix_status is not None:
            fields.append("setonix_status = %s")
            values.append(int(setonix_status))
        if skadi_status is not None:
            fields.append("skadi_status = %s")
            values.append(int(skadi_status))
        if uvfits_count is not None:
            fields.append("uvfits_count = %s")
            values.append(uvfits_count)

        fields.append("updated_at = CURRENT_TIMESTAMP")
        values.extend([sbid, scan])

        with get_psql_connect(section="dbwriter") as conn:
            cur = conn.cursor()
            cur.execute(
                f"UPDATE archives SET {', '.join(fields)} WHERE sbid = %s AND scan = %s",
                values,
            )
            conn.commit()
        return cur.rowcount > 0

    def update_archive_status_sbid(
        self, sbid: int,
        skadi_status: Optional[SkadiStatus] = None,
        acacia_status: Optional[ArchiveStatus] = None,
        setonix_status: Optional[ArchiveStatus] = None,
    ):
        if acacia_status is None and setonix_status is None and skadi_status is None:
            raise ValueError("At least one of acacia_status, setonix_status, or skadi_status must be provided.")

        fields, values = [], []
        if acacia_status is not None:
            fields.append("acacia_status = %s")
            values.append(int(acacia_status))
        if setonix_status is not None:
            fields.append("setonix_status = %s")
            values.append(int(setonix_status))
        if skadi_status is not None:
            logger.warning(f"you are trying to update the skadi status for one sbid... please use with caution...")
            fields.append("skadi_status = %s")
            values.append(int(skadi_status))

        fields.append("updated_at = CURRENT_TIMESTAMP")
        values.append(sbid)

        with get_psql_connect(section="dbwriter") as conn:
            cur = conn.cursor()
            cur.execute(
                f"UPDATE archives SET {', '.join(fields)} WHERE sbid = %s",
                values,
            )
            conn.commit()
        logger.info(f"{cur.rowcount} row(s) updated...")
        return cur.rowcount > 0

    # ------------------------------------------------------------------ #
    #  Read operations                                                     #
    # ------------------------------------------------------------------ #

    def get_record(self, sbid: int, scan: str) -> Optional[dict]:
        """Fetch a single record by (sbid, scan). Returns None if not found."""
        with get_psql_connect(section="dbreader") as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                "SELECT * FROM archives WHERE sbid = %s AND scan = %s",
                (sbid, scan),
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def get_all_records(self) -> list[dict]:
        """Return all records in the database."""
        with self._get_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute("SELECT * FROM archives ORDER BY id")
            rows = cur.fetchall()
            return [dict(r) for r in rows]

    def get_records_by_query(self, query: str) -> list[dict]:
        """Return all records in the database."""
        with get_psql_connect(section="dbreader") as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(query)
            rows = cur.fetchall()
            return [dict(r) for r in rows]

    def get_records_by_status(
        self,
        skadi_status: Optional[SkadiStatus] = None,
        acacia_status: Optional[ArchiveStatus] = None,
        setonix_status: Optional[ArchiveStatus] = None,
    ) -> list[dict]:
        """Filter records by one or both status fields."""
        conditions, values = [], []
        if skadi_status is not None:
            conditions.append("skadi_status = %s")
            values.append(int(skadi_status))
        if acacia_status is not None:
            conditions.append("acacia_status = %s")
            values.append(int(acacia_status))
        if setonix_status is not None:
            conditions.append("setonix_status = %s")
            values.append(int(setonix_status))

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        with self._get_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                f"SELECT * FROM archives {where} ORDER BY id", values
            )
            rows = cur.fetchall()
            return [dict(r) for r in rows]

    def update_observation_status_sbid(self, sbid):
        if auto_sched is None:
            raise ImportError("auto_sched module could not be loaded due to missing dependencies.")
        auto_sched.push_sbid_observation(sbid=sbid)

    # check skadi_status
    def _check_skadi_status(self, sbid, scan):
        scandir = ScanDir(sbid=sbid, scan=scan)
        try: uvfitscount = scandir.uvfits_count
        except:
            logger.info(f"cannot get uvfits count for sbid={sbid}, scan={scan}...")
            uvfitscount = 0
        if uvfitscount == 36:
            self.update_archive_status(sbid=sbid, scan=scan, skadi_status=SkadiStatus.READY, uvfits_count=uvfitscount)
        else:
            self.update_archive_status(sbid=sbid, scan=scan, skadi_status=SkadiStatus.ERRORED, uvfits_count=uvfitscount)
        return

    ### functions to run automatically every few minutes ###
    ########################################################
    def regular_db_update(self, setonix=True, acacia=False):
        """
        this function is used to update the database regularly including
        (1) check skadi_status for not ready sbid/scan
        (2) check sbid status for observation with no archiving happending
        """
        ### update skadi_status....
        logger.info(f"checking skadi_status for all records with skadi_status=0...")
        records = self.get_records_by_query("SELECT * FROM archives WHERE skadi_status = 0")
        # ^note - put <= 0 if you want to include errored scan as well...
        for record in records:
            sbid = record["sbid"]
            scan = record["scan"]
            logger.info(f"checking skadi_status for sbid={sbid}, scan={scan}...")
            self._check_skadi_status(sbid=sbid, scan=scan)

        ### update observation tables...
        logger.info(f"checking sbid status for all records with acacia_status=0 or setonix_status=0...")
        logger.info(f"setonix - {setonix}; acacia - {acacia}")
        query = "SELECT * FROM archives WHERE "
        querycond = []
        if setonix: querycond.append("setonix_status = 0")
        if acacia: querycond.append("acacia_status = 0")
        query += " OR ".join(querycond)
        records = self.get_records_by_query(query)
        logger.info(f"{len(records)} records found...")
        updatesbids = set([record["sbid"] for record in records])
        logger.info(f"updating {len(updatesbids)} sbids...")
        for sbid in updatesbids:
            logger.info(f"updating sbid={sbid}...")
            self.update_observation_status_sbid(sbid=sbid)

    def regular_get_sbid_to_run(self, setonix=True, acacia=False):
        """
        get list of sbid to run archive jobs...
        """
        logger.info(f"checking sbid status for all records with acacia_status=0 or setonix_status=0...")
        logger.info(f"setonix - {setonix}; acacia - {acacia}")
        #### time to build the query...
        query = "SELECT a.sbid, a.scan FROM archives a JOIN observation o ON a.sbid = o.sbid WHERE "
        querycond = []
        if setonix: querycond.append("a.setonix_status = 0")
        if acacia: querycond.append("a.acacia_status = 0")
        query += "({}) ".format(" OR ".join(querycond))
        query += "AND a.skadi_status = 1 "
        # add queries for observation
        query += "AND o.status > 3"
        records = self.get_records_by_query(query)
        logger.info(f"{len(records)} records found...")
        ### need to divide them into two categories... all sbids with sbids with some scans errored
        updatesbids = set([record["sbid"] for record in records])
        logger.info(f"found {len(updatesbids)} sbids to run archive jobs...")
        return list(updatesbids)


    def __repr__(self) -> str:
        count = len(self.get_all_records())
        return f"<ArchiveManager db='{self.db_path}' records={count}>"

class SBIDManager:
    """Manages SBID information records."""

    def __init__(self, db_path: str = "archive.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize the sbid_records table if it doesn't exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sbid_records (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    sbid        INTEGER NOT NULL UNIQUE,
                    acacia_archive_loc       TEXT NOT NULL,
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    # ------------------------------------------------------------------ #
    #  Write operations                                                    #
    # ------------------------------------------------------------------ #

    def insert_record(self, sbid: int, acacia_archive_loc: str) -> int:
        """
        Insert a new SBID record.

        Returns the row id of the newly inserted record.
        Raises ValueError if the sbid already exists.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "INSERT INTO sbid_records (sbid, acacia_archive_loc) VALUES (?, ?)",
                    (sbid, acacia_archive_loc),
                )
                conn.commit()
                return cursor.lastrowid
        except sqlite3.IntegrityError as e:
            logger.info(f"Record with SBID={sbid} already exists. will not update the record")
            #raise sqlite3.IntegrityError(f"Record with SBID={sbid} already exists. {e}")
    def update_record(self, sbid: int, acacia_archive_loc: str) -> bool:
        """
        Update the acacia_archive_loc for a given sbid.

        Returns True if a row was updated, False if no matching record was found.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE sbid_records
                SET acacia_archive_loc = ?, updated_at = CURRENT_TIMESTAMP
                WHERE sbid = ?
                """,
                (acacia_archive_loc, sbid),
            )
            conn.commit()
            return cursor.rowcount > 0

    # ------------------------------------------------------------------ #
    #  Read operations (stub — fill in as needed)                         #
    # ------------------------------------------------------------------ #

    def get_record(self, sbid: int) -> Optional[dict]:
        """Fetch a single record by sbid. Returns None if not found."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM sbid_records WHERE sbid = ?",
                (sbid,),
            ).fetchone()
            return dict(row) if row else None

    def get_all_records(self) -> list[dict]:
        """Return all records in the database."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM sbid_records ORDER BY id"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_records_by_acacia_archive_loc(self, acacia_archive_loc: str) -> list[dict]:
        """Return all records with a given acacia_archive_loc."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM sbid_records WHERE acacia_archive_loc = ? ORDER BY id",
                (acacia_archive_loc,),
            ).fetchall()
            return [dict(r) for r in rows]

    def __repr__(self) -> str:
        count = len(self.get_all_records())
        return f"<SBIDManager db='{self.db_path}' records={count}>"

def main():
    """
    Stage uvfits data, calibration tables, and metadata for archiving and handle CLINK event publishing/listening.

    Examples:
      # 1. Prepare archive folder and emit CLINK ready_for_copy event for SBID 82418:
      python -m craco.casda_archiver --sbid 82418 --prepare --emit-clink

      # 2. Run listener daemon to record copy/purge lifecycle events:
      python -m craco.casda_archiver --listen

      # 3. Specify custom CLINK transport config file:
      python -m craco.casda_archiver --sbid 82418 --emit-clink --clink-config /path/to/clink_env.conf
    """
    description_text = (
        "Stage uvfits data, calibration tables, and metadata for archiving and handle CLINK event publishing/listening.\n\n"
        "Examples:\n"
        "  # Prepare archive folder and emit ready_for_copy event for SBID 82418:\n"
        "  python -m craco.casda_archiver --sbid 82418 --prepare --emit-clink\n\n"
        "  # Run listener daemon to record copy/purge lifecycle events:\n"
        "  python -m craco.casda_archiver --listen\n\n"
        "  # Emit event using custom transport config file:\n"
        "  python -m craco.casda_archiver --sbid 82418 --emit-clink --clink-config /path/to/clink_env.conf\n"
    )
    parser = ArgumentParser(
        description=description_text,
        formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument("--sbid", type=parse_sbid, help="SBID of the data to be archived (e.g., 82418 or SB82418)")
    parser.add_argument("--scan", default=None, type=str, help="scan id of the data to be")
    parser.add_argument("--tstart", default=None,type=str, help="scan start time of the data to be archived")
    parser.add_argument("--prepare", action="store_true", help="whether to run prepare, i.e., convert and link data to archive folder")
    parser.add_argument("--rsync", action="store_true", help="whether to start rsync job to upload data to given place")
    parser.add_argument("--target", type=str, default="setonix:/scratch/ja3/zwan4817/askapbuffer", help="the target for rsync upload")
    # NEW: clink integration
    parser.add_argument("--emit-clink", action="store_true", help="whether to emit a CLINK ready_for_copy event for the SBID")
    parser.add_argument("--listen", action="store_true", help="whether to run CLINK listener daemon to record Pawsey archiving/purge events")
    parser.add_argument("--clink-config", type=str, default=None, help="path to CLINK transport config file (JSON or KEY=VAL)")
    parser.add_argument("--event-type", type=str, default="au.csiro.atnf.askap.craco.ready_for_copy", help="CLINK event type string for ready_for_copy emission")
    parser.add_argument("--subject", type=str, default=None, help="Optional custom Subject URN string for ready_for_copy emission")
    parser.add_argument("--test", action="store_true", help="Test mode: print payload to stdout instead of sending to the broker")
    parser.add_argument("--include-size", action="store_true", help="Include file sizes in the CLINK event payload")
    args = parser.parse_args()

    # NEW: clink integration    
    if args.listen:
        logger.info("Starting CLINK listener mode...")
        listener = ClinkListener(config_path=args.clink_config)
        listener.start_listening()
        return

    if args.sbid is None and not args.listen:
        logger.error("Please specify --sbid or --listen")
        return

    if args.scan is None or args.tstart is None:
        logger.info(f"no scan/tstart provided, will run for all scans...")
        scheddir = SchedDir(sbid=args.sbid)
        scans = scheddir.scans
    else:
        scans = [f"{args.scan}/{args.tstart}"]

    for scan in scans:
        scm = ScanCasdaMetadata(sbid=args.sbid, scan=scan.split("/")[0], tstart=scan.split("/")[1])
        if args.prepare:
            scm.run_scan_casda_prepare()
        if args.rsync:
            scm.start_casda_rsync(target=args.target)

    if args.emit_clink or args.test:
        publisher = ClinkPublisher(config_path=args.clink_config)
        publisher.emit_ready_for_copy(
            sbid=args.sbid, 
            event_type=args.event_type, 
            subject=args.subject,
            test=args.test,
            include_file_size=args.include_size
        )


if __name__ == "__main__":
    main()

    ##### NEW: clink integration testing & example usage...
    # Example 1: Emit ready_for_copy event for SB82418
    # sbid = 82418
    # pub = ClinkPublisher()
    # pub.emit_ready_for_copy(sbid=sbid)

    # Example 2: Start long-running listener daemon
    # listener = ClinkListener()
    # listener.start_listening()

    ##### legacy testing...
    # uvfitspath = "/CRACO/DATA_01/craco/SB076946/scans/00/20250916164012/b18.uvfits"
    # UCM = UvfitsCasdaMetadata(uvfitspath=uvfitspath)
    # UCM.prepare_casda_upload()
    # import glob
    # sbid = "SB077974"
    # scan = "00"
    # # tstart = "20250916164012"
    # _scanpaths = glob.glob(f"/data/craco/craco/{sbid}/scans/{scan}/20*")
    # tstarts = [i.split("/")[-1] for i in _scanpaths]

    # for tstart in tstarts:
    #     scm = ScanCasdaMetadata(sbid=sbid, scan=scan, tstart=tstart)
    #     scm.run_scan_casda_prepare()
    #     # scm.start_casda_rsync()
    #     # break