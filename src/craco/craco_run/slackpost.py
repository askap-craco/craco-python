from slack_sdk import WebClient
from configparser import ConfigParser

from craco.datadirs import DataDirs, SchedDir, ScanDir, RunDir, CandDir

from craco import craco_cand
from craft import craco_plan

import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import subprocess

import logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:
    from astropy.coordinates import SkyCoord
    from astropy import units
except:
    print("cannot import astropy package...")

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

### for fixuvfits
def fixuvfits(fitsfile):
    env = os.environ.copy()
    cmd = f"fixuvfits {fitsfile}"
    print(f"executing - {cmd}")
    complete_process = subprocess.run(
        [cmd], shell=True, capture_output=True,
        text=True, env=env,
    )
    log.info(complete_process.stdout)

class SlackPostManager:
    def __init__(self, test=True, channel=None):
        self.__load_webclient(test=test, channel=channel)

    def __load_webclient(self, test=True, channel=None):
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
        elif isinstance(msg_blocks, dict):
            msg_blocks = [msg_blocks]
        elif isinstance(msg_blocks, list):
            msg_blocks = msg_blocks

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

    def upload_file(self, files, comment="", thread_ts=None, mention_team=False, title=None):
        if mention_team: comment += f" {self.mention_msg}"
        if isinstance(files, str):
            try:
                if thread_ts is None: # thread_ts should be a string!!!
                    postresponse = self.client.files_upload_v2(
                        channel=self.channel,
                        initial_comment=comment,
                        title=title,
                        file=files
                    )
                else:
                    postresponse = self.client.files_upload_v2(
                        channel=self.channel,
                        initial_comment=comment,
                        thread_ts=thread_ts,
                        title=title,
                        file=files
                    )
            except Exception as error:
                log.error(f"error uploading file - {error}")
                return None
        if isinstance(files, list):
            if isinstance(title, list): 
                files_upload = [
                    dict(file=f, title=t) for f,t in zip(files, title)
                ]
            else:
                files_upload = [dict(file=f) for f in files]
                
            try:
                postresponse = self.client.files_upload_v2(
                    channel = self.channel,
                    initial_comment=comment,
                    file_uploads=files_upload,
                    thread_ts = thread_ts
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


class RealTimeCandAlarm:
    def __init__(self, snippetfolder, channel="C06C6D3V03S"):
        self.canddir = CandDir(**self._extract_info_from_path(snippetfolder))
        self.candrow = self._load_candidate(self.canddir.cand_info).iloc[0]
        self.candall = self._load_candidate(self.canddir.rundir.beam_unique_cand(self.canddir.beam))
        self.snippetfolder = snippetfolder
        ### work directory ###
        self.workdir = f"{snippetfolder}/post"
        os.makedirs(self.workdir, exist_ok=True)
        os.system(f"rm {self.workdir}/*")

        ### load other attributes...
        self._load_cand_prop()
        self._load_file_info()

        ### initiate slackbot
        self.slackbot = SlackPostManager(channel=channel)

    def _extract_info_from_path(self, candpath):
        pathsplit = candpath.split("/")
        self.sbid = int(pathsplit[4][2:])
        self.beam = int(pathsplit[8][4:])
        self.iblk = int(pathsplit[10][4:])
        self.scan = pathsplit[6]
        self.tstart = pathsplit[7]
        return dict(
            sbid = self.sbid,
            beam = self.beam,
            iblk = self.iblk,
            scan = f"{self.scan}/{self.tstart}",
        )

    def _load_candidate(self, candpath):
        if candpath.endswith(".csv"): return pd.read_csv(candpath, index_col = 0)
        
        dtype = np.dtype(
            [
                ('SNR',np.float32), # physical snr
                ('lpix', np.float32), # l pixel of the detection
                ('mpix', np.float32), # m pixel of the detection
                ('boxcwidth', np.float32), # boxcar width, 0 for 1 to 7 for 8
                ('time', np.float32), # sample of the detection in a given block
                ('dm', np.float32), # dispersion measure in hardware
                ('iblk', np.float32), # index of the block of the detection, by default each block is 256 samples
                ('rawsn', np.float32), # raw snr
                ('totalsample', np.float32), # sample index over the whole observation
                ('obstimesec', np.float32), # time difference between the detection and the start of the observation
                ('mjd', np.float64), # detection time in mjd
                ('dmpccm', np.float32), # physical dispersion measure
                ('ra', np.float64), 
                ('dec', np.float64),
                ("ibeam", int), # beam index
                ("latencyms", np.float32) # detection latency
            ]
        )

        cand_np = np.loadtxt(candpath, dtype=dtype, ndmin=1)
        return pd.DataFrame(cand_np)

    def _load_cand_prop(self):
        self.snr = self.candrow["SNR"]
        self.lpix, self.mpix = int(self.candrow["lpix"]), int(self.candrow["mpix"])
        self.boxcwidth = int(self.candrow["boxcwidth"])
        self.totalsample = int(self.candrow["totalsample"])
        self.dm = self.candrow["dmpccm"]
        self.mjd = self.candrow["mjd"]
        self.ra, self.dec = self.candrow["ra"], self.candrow["dec"]
        self.coord = SkyCoord(f"{self.ra}d {self.dec}d")
        self.galcoord = self.coord.galactic
        self.gl, self.gb = self.galcoord.l.deg, self.galcoord.b.deg

        self.candprop = dict(
            ra_deg=self.ra, dec_deg=self.dec, dm_pccm3 = self.dm,
            total_sample = self.totalsample, boxc_width = self.boxcwidth,
            lpix = self.lpix, mpix = self.mpix
        )

    def _load_file_info(self):
        self.icspath = self.canddir.scandir.beam_ics_path(self.canddir.beam)
        self.pcbpath = self.canddir.scandir.beam_pcb_path(self.canddir.beam)
        self.uvfitspath = self.canddir.cand_snippet_uvfits_path
        self.calpath = self.canddir.scheddir.beam_cal_path(self.canddir.beam)

    def _find_close_cand_type(self, timewindow=8):
        filtdf = self.candall[
            (self.candall["snr"] >= self.snr - 0.1) &
            (abs(self.totalsample - self.candall["total_sample"]) <= timewindow)
        ]
        matches = filtdf["MATCH_name"].unique()
        strmatches = [i for i in matches if isinstance(i, str)]
        return strmatches

    ### run function
    def run_flag_sidelobe(self, ):
        close_known_type = self._find_close_cand_type(timewindow=8)
        if len(close_known_type) != 0:
            self.sidelobe_flag = ",".join(close_known_type)
        else:
            self.sidelobe_flag = None

    ########### now time to work out uvfits etc...
    def _plot_cand_filt(self, filpath, padding=75, ):
        v, taxis, faxis = craco_cand.load_filterbank_with_ftaxis(
            filpath, self.totalsample - padding, padding * 2
        )
        v = craco_cand.normalise_filterbank(v)

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(1, 1, 1)

        ax.imshow(
            np.squeeze(v.T), aspect="auto", origin="lower",
            extent = [taxis[0], taxis[-1], faxis[0], faxis[-1]],
            interpolation="none", cmap="coolwarm"
        )
        ax.axvline(x=self.totalsample, color="k", ls="--")
        ax.set_xlabel("sample")
        ax.set_ylabel("frequency")

        return fig, ax

    def run_simple_filterbanks(self,):
        ### ics and pcb filterbank
        try:
            fig, _ = self._plot_cand_filt(filpath = self.icspath)
            fig.savefig(f"{self.workdir}/ics.png", bbox_inches="tight")
        except Exception as error:
            print(error)
            pass

        try:
            fig, _ = self._plot_cand_filt(filpath = self.pcbpath)
            fig.savefig(f"{self.workdir}/pcb.png", bbox_inches="tight")
        except:
            pass

        plt.close("all")

    def run_plots(self, padding=75, zoom_r=10):
        fixuvfits(self.uvfitspath)
        cand = craco_cand.Cand(
            uvfits=self.uvfitspath, calfile=self.calpath, 
            pcbpath=self.pcbpath, **self.candprop
        )
        cand.extract_data(padding=padding)
        
        ### normal data...
        cand.process_data(zoom_r=zoom_r)
        fig, _ = cand.plot_filtb(dm=0)
        fig.savefig(f"{self.workdir}/filtb_DM0.png", bbox_inches="tight")
        fig, _ = cand.plot_filtb(dm=cand.dm_pccm3, keepnan=True)
        fig.savefig(f"{self.workdir}/filtb_DMcand.png", bbox_inches="tight")
        fig = cand.plot_diagnostic_images()
        fig.savefig(f"{self.workdir}/normal_image.png", bbox_inches="tight")

        ### larger fov
        lplan = craco_plan.PipelinePlan(
            cand.canduvfits.datauvsource, "--ndm 2 --npix 512 --fov 2.2d",
        )
        cand.process_data(plan=lplan, zoom_r=zoom_r)
        fig = cand.plot_diagnostic_images(plan=lplan)
        fig.savefig(f"{self.workdir}/double_image.png", bbox_inches="tight")

        plt.close("all")

    ### send alert
    def _format_gui_link(self):
        url = f"""http://localhost:8024/candidate?sbid={self.sbid}&beam={self.beam}&scan={self.scan}&tstart={self.tstart}&runname=results"""
        url += f"""&dm={self.dm}&boxcwidth={self.boxcwidth}&lpix={self.lpix}&mpix={self.mpix}"""
        url += f"""&totalsample={self.totalsample}&ra={self.ra}&dec={self.dec}"""
        return url
    
    def _format_nice_candidate_block(self, ):
        divider = {"type": "divider"}
        ### header about the warning
        try:
            header_msg = f":sunny: *New CRACO Trigger in SB{self.sbid} BEAM{self.beam:0>2}*\n"
            header_msg += f"*<{self._format_gui_link()}|Click the link to access the GUI>*"
            try:
                if self.sidelobe_flag is not None:
                    header_msg += f"\nPotential Sidelobe From *{self.sidelobe_flag}*"
            except:
                header_msg += "\n**Unable to run sidelobe filter**"

            warn_header = {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": header_msg
                }
            }

            warn_body = {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*coord*\n{self.coord.to_string('hmsdms')}"},
                    {"type": "mrkdwn", "text": f"*galcoord*\n{self.gl:.4f}d, {self.gb:.4f}d"},
                    {"type": "mrkdwn", "text": f"*SNR*\n{self.snr:.1f}"},
                    {"type": "mrkdwn", "text": f"*MJD*\n{self.mjd:.8f}"},
                    {"type": "mrkdwn", "text": f"*DM*\n{self.dm:.1f} pc cm^-3"},
                    {"type": "mrkdwn", "text": f"*width*\n{self.boxcwidth} sample"},
                ]
            }

            return [divider, warn_header, warn_body, divider]
        except Exception as error:
            header_msg = f":zap: Unable to load info for cand {self.snippetfolder}"
            warn_header = {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": header_msg
                }
            }
            error_msg = f"ERR INFO - {error}"
            error_header = {
                "type": "context",
                "elements": [
                    dict(type="mrkdwn", text=error_msg)
                ]
            } 
            return [divider, warn_header, error_header, divider]


    def send_alarm(self, main_ts=None):
        main_msg = self._format_nice_candidate_block()
        main_res = self.slackbot.post_message(main_msg, thread_ts=main_ts)
        if main_ts is None: 
            main_ts = self.slackbot.get_thread_ts_from_response(main_res)

        return main_ts

    def combine_image_to_list(self,):
        fnames = ["ics.png", "pcb.png", "filtb_DM0.png", "filtb_DMcand.png", "normal_image.png", "double_image.png"]
        files = [f"{self.workdir}/{fname}" for fname in fnames if os.path.exists(f"{self.workdir}/{fname}")]
        title = [fname for fname in fnames if os.path.exists(f"{self.workdir}/{fname}")]
        return files, title
    
    def combine_simple_filtb_to_list(self, ):
        fnames = ["ics.png", "pcb.png"]
        files = [f"{self.workdir}/{fname}" for fname in fnames if os.path.exists(f"{self.workdir}/{fname}")]
        title = [fname for fname in fnames if os.path.exists(f"{self.workdir}/{fname}")]
        return files, title

    
    def send_image_thread(self, main_ts=None):
        files, title = self.combine_image_to_list()
        if len(files) > 0: self.slackbot.upload_file(files=files, title=title, thread_ts=main_ts)
        else: self.slackbot.post_message("no image found...", thread_ts=main_ts)

    def send_simple_filterbank_thread(self, main_ts=None):
        files, title = self.combine_simple_filtb_to_list()
        if len(files) > 0: self.slackbot.upload_file(files=files, title=title, thread_ts=main_ts)
        else: self.slackbot.post_message("no ics/pcb filterbank image found...", thread_ts=main_ts)

    def run_all(self,):
        try: self.run_flag_sidelobe()
        except: pass

        main_ts = self.send_alarm(main_ts=None)

        try: self.run_simple_filterbanks()
        except: pass
        try: self.run_plots()
        except: pass

        self.send_image_thread(main_ts = main_ts)

class RealTimeScanAlarm:
    def __init__(self, outdir, channel="C06C6D3V03S", postlimit=10):
        self.snippets = glob.glob(f"/CRACO/DATA_??/craco/{outdir}/beam??/candidates/*")
        self.slackbot = SlackPostManager(channel=channel)
        self._parse_info_from_outdir(outdir)

        self._get_post_method(postlimit=postlimit)
        self._get_all_alerts()

    def _parse_info_from_outdir(self, outdir):
        "SB069387/scans/00/20241216001642"
        outdirsplit = outdir.split("/")
        self.sbid = int(outdirsplit[0][2:])
        self.scan = outdirsplit[2]
        self.tstart = outdirsplit[3]
        self.scandir = ScanDir(sbid=self.sbid, scan=f"{self.scan}/{self.tstart}")

    def _get_post_method(self, postlimit=10):
        if len(self.snippets) >= postlimit:
            self.postmethod = "scan"
        elif len(self.snippets) == 0:
            self.postmethod = "none"
        else:
            self.postmethod = "cand"

    def _get_all_alerts(self):
        candalerts = []
        for snippetpath in self.snippets:
            candalarm = RealTimeCandAlarm(snippetfolder=snippetpath, channel=self.slackbot.channel)
            # candalarm.run_flag_sidelobe()
            candalerts.append(candalarm)
        self.candalerts = candalerts

        ### get information from it...
        trigger_beams = []
        for candalarm in self.candalerts:
            trigger_beams.append(candalarm.beam)
        self.scan_trigger_beams = sorted(set(trigger_beams))

    def _format_candidate_dataframe(self,):
        rows = [candalarm.candrow.to_frame().T for candalarm in self.candalerts]
        self.scanalerts = pd.concat(rows)
        self.scanalerts.to_csv(f"{self.scandir.scan_head_dir}/all_triggers.csv")

    def _cand_post(self):
        for candalarm in self.candalerts:
            candalarm.run_all()
    
    ### for whole scan post...
    def _format_nice_scan_block(self,):
        header_msg = f":rain_cloud: *{len(self.snippets)} New Triggers in SB{self.sbid} {self.scan}/{self.tstart}*\n"
        beam_list_str = ",".join(str(i) for i in self.scan_trigger_beams)
        header_msg += f"Trigger(s) from {len(self.scan_trigger_beams)} beam(s) - {beam_list_str}"
        warn_header = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": header_msg
            }
        }

        divider = {"type": "divider"}

        return [divider, warn_header, divider]

    def _scan_post(self):
        main_msg = self._format_nice_scan_block()
        main_res = self.slackbot.post_message(main_msg, thread_ts=None)
        main_ts = self.slackbot.get_thread_ts_from_response(main_res)

        for candalarm in self.candalerts:
            candalarm.run_flag_sidelobe()
            candalarm.send_alarm(main_ts=main_ts)
            candalarm.run_simple_filterbanks()
            candalarm.send_simple_filterbank_thread(main_ts=main_ts)
            time.sleep(5)

    def run_main(self):
        if self.postmethod == "none": return
        if self.postmethod == "scan":
            self._scan_post()

        if self.postmethod == "cand":
            self._cand_post()

### localiser alarm...
class LocalAlarm:
    def __init__(self, workdir, channel="C06C6D3V03S"):
        self.slackbot = SlackPostManager(channel=channel)
        self.workdir = workdir

    def _check_file_exists(self, path):
        if os.path.exists(path): return True
        return False

    def post_main_alarm(self):
        corrfpath = f"{self.workdir}/coord_correct.txt"
        if self._check_file_exists(corrfpath):
            with open(corrfpath, "r") as fp:
                msg = fp.read()
            msg = msg.strip("\n")
        else:
            msg = f"Error... No coordinate file found under {self.workdir}"

        msg_blocks = [
            dict(
                type="section",
                text=dict(type="mrkdwn", text=f"*[LOCALISER]* Localisation Report\nWorkdir - {self.workdir}")
            ),
            dict(
                type="rich_text",
                elements=[
                    dict(
                        type="rich_text_preformatted",
                        elements=[dict(type="text", text=msg)]
                    )
                ]
            ),
        ]

        response = self.slackbot.post_message(msg_blocks)
        self.main_ts = self.slackbot.get_thread_ts_from_response(response)

    def post_images(self,):
        ### post burst image...
        burstpng = f"{self.workdir}/burstfield.png"
        if self._check_file_exists(burstpng):
            self.slackbot.upload_file(burstpng, "Burst Field Image", thread_ts=self.main_ts)
        else:
            self.slackbot.post_message("cannot find burst field image...", thread_ts=self.main_ts)

        ### post field RACS comparison
        fieldracspng = f"{self.workdir}/field_racs.png"
        if self._check_file_exists(fieldracspng):
            self.slackbot.upload_file(fieldracspng, "Field-RACS comparison", thread_ts=self.main_ts)
        else:
            self.slackbot.post_message("cannot find field racs comparison image...", thread_ts=self.main_ts)

        bootstrap_fieldracs_png = f"{self.workdir}/bootstrap.field_racs.png"
        if self._check_file_exists(bootstrap_fieldracs_png):
            self.slackbot.upload_file(bootstrap_fieldracs_png, "Field-RACS Bootstrap Histogram", thread_ts=self.main_ts)
        else:
            self.slackbot.post_message("cannot find field racs bootstrap image...", thread_ts=self.main_ts)

        ### post RACS ref comparison
        racsrefpng = f"{self.workdir}/racs_ref.png"
        if self._check_file_exists(racsrefpng):
            self.slackbot.upload_file(racsrefpng, "RACS-Ref comparison", thread_ts=self.main_ts)
        else:
            self.slackbot.post_message("cannot find racs ref comparison image...", thread_ts=self.main_ts)

        bootstrap_racsref_png = f"{self.workdir}/bootstrap.racs_ref.png"
        if self._check_file_exists(bootstrap_racsref_png):
            self.slackbot.upload_file(bootstrap_racsref_png, "RACS-Ref Bootstrap Histogram", thread_ts=self.main_ts)
        else:
            self.slackbot.post_message("cannot find racs ref bootstrap image...", thread_ts=self.main_ts)

    def postalarm(self):
        self.post_main_alarm()
        self.post_images()


class TabAlarm:
    def __init__(self, workdir, channel="C06C6D3V03S"):
        self.slackbot = SlackPostManager(channel=channel)
        self.workdir = workdir

    def post_status(self, ):
        filfiles = glob.glob(f"{self.workdir}/*.fil")
        if len(filfiles) == 0: filemsg = "no filterbank files found!"
        else: filemsg = "\n".join(filfiles)
        msg_blocks = [
            dict(
                type="section",
                text=dict(type="mrkdwn", text=f"*[Tab]* Tied-Array Beam Filterbank Job\n_work directory_: {self.workdir}\n{len(filfiles)} filterbank files found...")
            ),
            dict(
                type="rich_text",
                elements=[
                    dict(
                        type="rich_text_preformatted",
                        elements=[dict(type="text", text=filemsg)]
                    )
                ]
            ),
        ]
        self.slackbot.post_message(msg_blocks)



        

    