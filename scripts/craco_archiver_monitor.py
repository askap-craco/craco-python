#!/usr/bin/env python

import os

from craco.casda_archiver import ArchiveManager, ScanCasdaMetadata, ClinkPublisher
from craco.datadirs import SchedDir

from craco.mattermost_messager import MattermostPostManager

import logging
logging.basicConfig(
    level = logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def check_clink_credential():
    """
    Check if the clink credential is valid
    """
    if "CLINK_CREDENTIAL_PATH" in os.environ:
        clink_credential_path = os.environ["CLINK_CREDENTIAL_PATH"]
        if not os.path.exists(clink_credential_path):
            logger.error(f"CLINK_CREDENTIAL_PATH {clink_credential_path} does not exist.")
            return 
        return clink_credential_path
    logger.error("CLINK_CREDENTIAL_PATH is not set in the environment variables.")
    return
    
def update_database():
    """
    Update the database (e.g., check skadi status, update database to check status)
    """
    archive_manager = ArchiveManager()
    archive_manager.regular_db_update(setonix=True, acacia=False)

def find_sbids():
    """
    Find all SBIDs that needs to be archived
    """
    archive_manager = ArchiveManager()
    sbids = archive_manager.regular_get_sbid_to_run(setonix=True, acacia=False)
    return sbids


def run_sbid(sbid, clink_credential_path,dryrun=True, mattermost=False):
    """
    Run the archiver for a given SBID
    """
    scheddir = SchedDir(sbid=sbid)
    scans = scheddir.scans

    if mattermost:
        mm = MattermostPostManager()
        mm.post_message(f"Starting archiving preparation for SBID {sbid}")

    ### prepare for the archiving...
    for scan in scans:
        logger.info(f"Preparing scan {scan} for archiving...")
        try:
            scm = ScanCasdaMetadata(sbid=sbid, scan=scan.split("/")[0], tstart=scan.split("/")[1])
            if mattermost:
                mm.post_message(f"Preparing SBID {sbid} - scan {scan} for archiving...")
            scm.run_scan_casda_prepare()
        except Exception as e:
            logger.error(f"Error preparing scan {scan} for archiving: {e}")
            ### at this stage, there is not too many things you can do, so just log the error and continue...
            # return

    ### make clink for the sbid...
    publisher = ClinkPublisher(config_path=clink_credential_path)
    publisher.emit_ready_for_copy(
        sbid = sbid,
        event_type = "au.csiro.atnf.askap.craco.ready_for_copy",
        subject = None,
        test = dryrun,
        include_file_size = True,
    )

    if mattermost:
        mm.post_message(f"Clink event emitted for SBID {sbid}. Archiving preparation completed.")


def run(dryrun=True, mattermost=True):
    """
    Run the archiver monitor script
    """

    ### regularly update the database...
    update_database()

    ### get credential path for clink...
    clink_credential_path = check_clink_credential()
    if clink_credential_path is None:
        raise ValueError("CLINK_CREDENTIAL_PATH is not set or invalid. Please set the environment variable to a valid path.")

    ### find sbids that needs to be archived...
    sbids = find_sbids()
    logger.info(f"SBIDs to be archived: {sbids}")

    if len(sbids) == 0:
        logger.info("No SBIDs to be archived.")
        return
    else:
        if mattermost:
            mm = MattermostPostManager()
            mm.post_message(f"Found {len(sbids)} SBIDs to be archived: {sbids}")

    ### run the archiver for each sbid...
    for sbid in sbids:
        logger.info(f"Running archiver for SBID {sbid}...")
        run_sbid(sbid, clink_credential_path, dryrun=dryrun, mattermost=mattermost)

if __name__ == "__main__":
    run(dryrun=False, mattermost=True)
    # import argparse

    # parser = argparse.ArgumentParser(description="CRACO Archiver Monitor")
    # parser.add_argument("--dryrun", action="store_true", help="Run in dry run mode (no actual archiving will be performed)")
    # args = parser.parse_args()

    # run(dryrun=args.dryrun)
