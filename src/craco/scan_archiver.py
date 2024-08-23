import glob
import subprocess as S
from craco.datadirs import SchedDir, ScanDir, format_sbid
import tempfile, os, sys
import logging 
from logging.handlers import RotatingFileHandler
import argparse
import json
from craco.craco_run.auto_sched import SlackPostManager
from craco.mattermost_messager import MattermostPostManager
from craco.fixuvfits import fix

logname = "/CRACO/SOFTWARE/craco/craftop/logs/archive_scan.log"
log = logging.getLogger(__name__)
stdout_handler = logging.StreamHandler(sys.stdout)
file_handler = RotatingFileHandler(logname, maxBytes=10000000, backupCount=10000)
logging.basicConfig(handlers=[file_handler, stdout_handler],
                    format='[%(asctime)s] %(levelname)s: %(message)s',
                    level=logging.DEBUG)

class TrivialEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, bytes):
            d = o.decode('utf-8')
        else:
            d = super().default(o)
        return d
    

def fetch_rclone_cfg():
    cmd = ["rclone", "config", "file"]
    try:
        c = S.Popen(cmd, stdout=S.PIPE, stderr = S.PIPE)
        (out, err) = c.communicate()
    except FileNotFoundError as FE:
        raise RuntimeError(f"Unknown commands - {cmd}")
    
    if err.decode('utf-8') == '':
        return out.decode('utf-8').strip().split()[-1]

def check_write_permissions(destination):
    remote = destination[0]
    dest_path = destination[1]
    with tempfile.NamedTemporaryFile() as t:
        cmd = ['rclone', 'copy', '-P', t.name, f"{remote}:{dest_path}/"]
        c = S.Popen(cmd, stdout=S.PIPE, stderr = S.PIPE)
        (out, err) = c.communicate()
        if err.decode('utf-8') == '':
            cmd = ['rclone', 'delete', f"{remote}:{dest_path}/{os.path.basename(t.name)}"]
            cx = S.Popen(cmd, stdout=S.PIPE, stderr = S.PIPE)
            return 0
        else:
            raise Exception(f"Could not write a test file to the destination path due to the following error\n{err.decode('utf-8')}")

def execute(command_with_args):
    """
    Execute the given `command_with_args` using Popen

    Args:
        - command_with_args (list) : An array with the command to execute,
                                        and its arguments. Each argument is given
                                        as a new element in the list.
    """
    log.info("Invoking : %s", " ".join(command_with_args))
    try:
        with S.Popen(
                command_with_args,
                stdout=S.PIPE,
                stderr=S.PIPE) as proc:
            (out, err) = proc.communicate()

            #out = proc.stdout.read()
            #err = proc.stderr.read()

            log.debug(out)
            if err:
                log.warning(err.decode("utf-8").replace("\\n", "\n"))

            return {
                "code": proc.returncode,
                "out": out,
                "error": err
            }
    except Exception as generic_e:
        log.exception("Error running command. Reason: %s", generic_e)
        return {
            "code": -30,
            "out": "",
            "error": generic_e
        } 

def parse_scandir_env(path):
    parts = path.strip().split("/")
    if len(parts) > 0:
        for ip, part in enumerate(parts):
            if part.startswith("SB"):
                sbid = part
                scanid = parts[ip + 2]
                tstart = parts[ip + 3]
                
                if len(sbid) == 8 and len(scanid) == 2 and len(tstart) == 14:
                    return sbid, scanid, tstart

    raise RuntimeError(f"Could not parse sbid, scanid and tstart from {path}")


def run_with_tsp(destination_str):
    log.info(f"Queuing up archive scan")

    #ARCHIVE_TS_ONFINISH = "report_craco_archive"
    ARCHIVE_TS_SOCKET = "/data/craco/craco/tmpdir/queues/archive"
    TMPDIR = "/data/craco/craco/tmpdir"

    environment = {
        "TS_SOCKET": ARCHIVE_TS_SOCKET,
        "TMPDIR": TMPDIR,
    }
    ecopy = os.environ.copy()
    ecopy.update(environment)

    try:
        scan_dir = os.environ['SCAN_DIR']
        sbid, scanid, tstart = parse_scandir_env(scan_dir)
    except Exception as KE:
        log.critical(f"Could not fetch the scan directory from environment variables!!")
        log.critical(KE)
        return
    else:
        sbid, scanid, tstart = parse_scandir_env(scan_dir)
        cmd = f"""archive_scan -sbid {sbid} -scanid {scanid} -tstart {tstart} -dest {destination_str}"""

        S.run(
            [f"tsp {cmd}"], shell=True, capture_output=True,
            text=True, env=ecopy,
        )
        log.info(f"Queued archive scan job - with command - {cmd}")


class ScanArchiver:

    def __init__(self, sbid, scanid, tstart, destination_str):
        '''
        SBID: str, SBID of the observation to be copied, can accpet SB0xxxxx, SBxxxxx, xxxxx formats
        scanid: str, Scanid of the scan - example '00'
        tstart: str, Tstart of the scan - example '2024080212412'
        destination_str: str, remote:dest_path - example (acacia:GSPs/Pxxx/)
        '''
        self.destination_str = destination_str
        destination = (destination_str.split(":")[0], destination_str.split(":")[1])
        self.scan = ScanDir(sbid, f"{scanid}/{tstart}")
        check_write_permissions(destination)
        self.destination = destination
        self.dest_scan_path = os.path.join(self.destination_str, format_sbid(self.scan.scheddir.sbid), self.scan.scan)
        log.debug(f"Destination scan path is {self.dest_scan_path}")
        self.base_cmd = ["rclone"]
        self.record_name = os.path.join(self.scan.scan_head_dir, "scan_archiver_record.json")
        self.record = open(self.record_name, 'w')


    def execute_fixuvfits(self):
        log.debug("Running fixuvfits")
        for uvf in self.scan.uvfits_paths:
            log.debug(f"Fixing - {uvf}")
            if uvf:
                try:
                    fix(uvf)
                except Exception as e:
                    log.exception(f"Could not run fixuvfits on {uvf} because of: \n{e}")
                    pass
            

    def execute_copy_jobs(self, dry=False):
        self.jobs_launched = {}
        self.jobs_errored = {}
        self.jobs_finished = {}
        all_datadirs = [self.scan.scan_head_dir] + list(self.scan.scan_data_dirs)

        for jobid, datadir in enumerate(all_datadirs):
            node_name = datadir.strip().split("/")[2]
            dest_path = os.path.join(self.dest_scan_path, node_name)

            #options = ["copy", f"{datadir}", f"{self.destination[0]}:{dest_path}"]
            options = ["copy", f"{datadir}", f"{dest_path}"]
            if dry:
                options += ["--dry-run"]
            cmd = self.base_cmd + options
            try:
                result = execute(cmd)
                self.jobs_launched[jobid] = {'datadir':datadir, 'result':result}
                if result['code'] != 0:
                    log.info(f"Jobid {jobid} errored with {result['error']}")
                    self.jobs_errored[jobid] = {'datadir':datadir, 'result':result}
                else:
                    log.info(f"Jobid {jobid} finished without any errors")
                    self.jobs_finished[jobid] = {'datadir':datadir, 'result':result}
            except Exception as generic_e:
                log.info(f"Jobid {jobid} raised an exception {generic_e}")
                self.jobs_errored[jobid] = {'datadir':datadir, 'result':result}
        
        #stats_cmd = self.base_cmd + ["size", f"{self.destination[0]}:{self.dest_scan_path}"]
        stats_cmd = self.base_cmd + ["size", f"{self.dest_scan_path}"]
        stats = execute(stats_cmd)
        self.stats = {'dest_path': self.dest_scan_path, 'stats':stats }

    @property
    def num_launched(self):
        return len(self.jobs_launched)

    @property
    def num_finished(self):
        return len(self.jobs_finished)

    @property
    def num_errored(self):
        return len(self.jobs_errored)
    
    def dump_records(self):
        log.info(f"Dumping the records as a json file - {self.record_name}")
        self.output_dict = {'launched':self.jobs_launched, 'finished':self.jobs_finished, 'errored':self.jobs_errored, 'final_stats':self.stats}
        #log.debug(output_dict)
        json.dump(self.output_dict, self.record, sort_keys=True, indent=4, cls=TrivialEncoder)
        
    def close(self):
        log.info("Closing the json file")
        self.record.close()

    def compose_message(self):
        msg = f"CRACO archive scan summary:\nDestination: {self.dest_scan_path}\n"
        if len(self.output_dict['launched']) == 0:
            msg += f"Did not launch any copy jobs :(\n"
        else:
            num_launched = len(self.output_dict['launched'])
            msg += f"Launched {num_launched} copy jobs\n"
            num_finished = len(self.output_dict['finished'])
            msg += f"{num_finished} jobs finished without any errors\n"
            num_errored = len(self.output_dict['errored'])
            msg += f"{num_errored} jobs finished with errors\n"
            
            if num_errored > 0:
                error_msgs = [f"  {key['datadir']} --> {key['result']['error'].decode('utf-8')}" for key in self.jobs_errored]
                error_msg = "\n\n".join(error_msgs)
                msg += error_msg

        so = self.output_dict['final_stats']
        stats_msg = f"Destination stats:\n{so['stats']['out'].decode('utf-8')}\n{so['stats']['error'].decode('utf-8')}\n"
        msg += stats_msg
        log.debug(f"Composed message -\n{msg}")
        return msg

    def send_msg(self, msg):
        log.debug(f"Sending message: \n{msg}")
        try:
            sp = SlackPostManager(test=False, channel="C06FCTQ6078")
            sp.post_message(msg)

            mp = MattermostPostManager()
            mp.post_message(msg)
        except Exception as e:
            log.exception(f"Posting message didn't work because of: \n{e}")
            raise e

    def run(self, dry=False):
        self.execute_fixuvfits()
        self.execute_copy_jobs(dry=dry)
        self.dump_records()
        self.close()
        msg = self.compose_message()
        self.send_msg(msg)

def get_parser():
    a = argparse.ArgumentParser()
    a.add_argument("-sbid", type=str, help="SBID", required=True)
    a.add_argument("-scanid", type=str, help="scanid", required=True)
    a.add_argument("-tstart", type=str, help="tstart", required=True)
    a.add_argument("-dest", type=str, help="Destination string (hostname:/path/to/dest) - acacia:GSPs/AS400/", required=True)
    a.add_argument("-dry", action='store_true', help="Do a dry run only (def:False)", default=False)

    args = a.parse_args()
    return args

def main():
    args = get_parser()
    sa = ScanArchiver(args.sbid, args.scanid, args.tstart, args.dest)
    sa.run(dry = args.dry)

if __name__ == '__main__':
    main()