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
import subprocess
import sqlite3
from enum import IntEnum
from typing import Optional

from astropy.io import fits
from astropy.time import Time
from craft import uvfits

import numpy as np

from aces.askapdata.schedblock import SB, SchedulingBlock

from craco.fixuvfits import fix
from craco.datadirs import SchedDir, ScanDir, format_sbid
from craco.tools import cracocal2casatab

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

### database related...
class ArchiveStatus(IntEnum):
    DEFAULT = 0
    QUEUED = 1
    EXECUTING = 2
    FINISHED = 3
    ERRORED = -1

class ArchiveManager:
    """Manages data archiving records for Acacia and Setonix archiving processes."""

    def __init__(self, db_path: str = "archive_status.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
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
        acacia_status: ArchiveStatus = ArchiveStatus.DEFAULT,
        setonix_status: ArchiveStatus = ArchiveStatus.DEFAULT,
    ) -> int:
        """
        Insert a new archive record.

        Returns the row id of the newly inserted record.
        Raises ValueError if the (sbid, scan) pair already exists.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO archive_records (sbid, scan, acacia_status, setonix_status)
                    VALUES (?, ?, ?, ?)
                    """,
                    (sbid, scan, int(acacia_status), int(setonix_status)),
                )
                conn.commit()
                return cursor.lastrowid
        except sqlite3.IntegrityError:
            raise ValueError(f"Record with SBID={sbid} and scan='{scan}' already exists.")

    def update_status(
        self,
        sbid: int,
        scan: str,
        acacia_status: Optional[ArchiveStatus] = None,
        setonix_status: Optional[ArchiveStatus] = None,
    ) -> bool:
        """
        Update acacia_status and/or setonix_status for a given (sbid, scan) pair.

        Pass only the field(s) you want to change; omitted fields are left untouched.
        Returns True if a row was updated, False if no matching record was found.
        Raises ValueError if neither status field is provided.
        """
        if acacia_status is None and setonix_status is None:
            raise ValueError("At least one of acacia_status or setonix_status must be provided.")

        fields, values = [], []
        if acacia_status is not None:
            fields.append("acacia_status = ?")
            values.append(int(acacia_status))
        if setonix_status is not None:
            fields.append("setonix_status = ?")
            values.append(int(setonix_status))

        fields.append("updated_at = CURRENT_TIMESTAMP")
        values.extend([sbid, scan])

        with self._get_connection() as conn:
            cursor = conn.execute(
                f"UPDATE archive_records SET {', '.join(fields)} WHERE sbid = ? AND scan = ?",
                values,
            )
            conn.commit()
            return cursor.rowcount > 0

    # ------------------------------------------------------------------ #
    #  Read operations                                                     #
    # ------------------------------------------------------------------ #

    def get_record(self, sbid: int, scan: str) -> Optional[dict]:
        """Fetch a single record by (sbid, scan). Returns None if not found."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM archive_records WHERE sbid = ? AND scan = ?",
                (sbid, scan),
            ).fetchone()
            return dict(row) if row else None

    def get_all_records(self) -> list[dict]:
        """Return all records in the database."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM archive_records ORDER BY id").fetchall()
            return [dict(r) for r in rows]

    def get_records_by_status(
        self,
        acacia_status: Optional[ArchiveStatus] = None,
        setonix_status: Optional[ArchiveStatus] = None,
    ) -> list[dict]:
        """Filter records by one or both status fields."""
        conditions, values = [], []
        if acacia_status is not None:
            conditions.append("acacia_status = ?")
            values.append(int(acacia_status))
        if setonix_status is not None:
            conditions.append("setonix_status = ?")
            values.append(int(setonix_status))

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        with self._get_connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM archive_records {where} ORDER BY id", values
            ).fetchall()
            return [dict(r) for r in rows]

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
        except sqlite3.IntegrityError:
            logger.info(f"Record with SBID={sbid} already exists. will not update the record")

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

    


if __name__ == "__main__":
    # uvfitspath = "/CRACO/DATA_01/craco/SB076946/scans/00/20250916164012/b18.uvfits"
    # UCM = UvfitsCasdaMetadata(uvfitspath=uvfitspath)
    # UCM.prepare_casda_upload()
    import glob
    sbid = "SB077974"
    scan = "00"
    # tstart = "20250916164012"
    _scanpaths = glob.glob(f"/data/craco/craco/{sbid}/scans/{scan}/20*")
    tstarts = [i.split("/")[-1] for i in _scanpaths]

    for tstart in tstarts:
        scm = ScanCasdaMetadata(sbid=sbid, scan=scan, tstart=tstart)
        scm.run_scan_casda_prepare()
        # scm.start_casda_rsync()
        # break