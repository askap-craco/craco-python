from slack_sdk import WebClient
from configparser import ConfigParser

from craco.datadirs import DataDirs, SchedDir, ScanDir, RunDir

from craco import craco_cand
from craft import craco_plan

import os
import numpy as np
import glob

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


class CandAlarm:
    def __init__(self, bestcand_dict, outdir, channel="C06C6D3V03S", plotdir=None):
        self.outdir = outdir
        self.plotdir = plotdir
        self._get_info_from_outdir()

        self.bestcand_dict = bestcand_dict
        self._get_galactic_info()

        self.slackbot = SlackPostManager(channel=channel)

    def _get_info_from_outdir(self,):
        outdir_split = self.outdir.split("/")
        self.sbid = int(outdir_split[4][2:])
        self.scan = outdir_split[-2]
        self.tstart = outdir_split[-1]

        try: self.scandir = ScanDir(sbid=self.sbid, scan=f"{self.scan}/{self.tstart}")
        except: self.scandir = None

        if self.plotdir is None:
            if self.scandir is None: self.plotdir = "."
            else: self.plotdir = self.scandir.scheddir.sched_head_dir + \
                f"/cand/{self.scan}/{self.tstart}/{self.total_sample}_DM{self.dm_pccm:.0}"

        if not os.path.exists(self.plotdir):
            os.makedirs(self.plotdir)

    def _get_galactic_info(self, ):
        self.coord = SkyCoord(self.ra, self.dec, unit=units.degree)
        self.galcoord = self.coord.galactic

        self.gl = self.galcoord.l.deg
        self.gb = self.galcoord.b.deg

    @property
    def beam(self):
        return self.bestcand_dict.get("ibeam")
    
    @property
    def dm_pccm(self):
        return self.bestcand_dict.get("dm_pccm3")
    
    @property
    def boxcwidth(self):
        return self.bestcand_dict.get("boxc_width")
    
    @property
    def lpix(self):
        return self.bestcand_dict.get("lpix")

    @property
    def mpix(self):
        return self.bestcand_dict.get("mpix")
    
    @property
    def total_sample(self):
        return self.bestcand_dict.get("total_sample")

    @property
    def ra(self):
        return self.bestcand_dict.get("ra_deg")

    @property
    def dec(self):
        return self.bestcand_dict.get("dec_deg")
    
    @property
    def mjd(self):
        return self.bestcand_dict.get("mjd")
    
    @property
    def snr(self):
        return self.bestcand_dict.get("snr")
    
    ### functions for posting messages...
    def _format_gui_link(self):
        url = f"""http://localhost:8024/candidate?sbid={self.sbid}&beam={self.beam}&scan={self.scan}&tstart={self.tstart}&runname=results"""
        url += f"""&dm={self.dm_pccm}&boxcwidth={self.boxcwidth}&lpix={self.lpix}&mpix={self.mpix}"""
        url += f"""&totalsample={self.total_sample}&ra={self.ra}&dec={self.dec}"""
        return url

    def _format_nice_candidate_block(self, ):
        ### header about the warning
        header_msg = f"*New CRACO Trigger in SB{self.sbid} BEAM{self.beam:0>2}*\n"
        header_msg += f"*<{self._format_gui_link()}|Click the link to access the GUI>*"

        warn_header = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": header_msg
            }
        }

        divider = {"type": "divider"}

        warn_body = {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*coord*\n{self.ra:.4f}d, {self.dec:.4f}d"},
                {"type": "mrkdwn", "text": f"*galcoord*\n{self.gl:.4f}d, {self.gb:.4f}d"},
                {"type": "mrkdwn", "text": f"*SNR*\n{self.snr:.1f}"},
                {"type": "mrkdwn", "text": f"*MJD*\n{self.mjd:.8f}"},
                {"type": "mrkdwn", "text": f"*DM*\n{self.dm_pccm:.1f}pc cm^-3"},
                {"type": "mrkdwn", "text": f"*width*\n{self.boxcwidth} sample"},
            ]
        }

        return [divider, warn_header, warn_body, divider]
    
    ### source plot related...
    def make_cand_plot(self, padding=75, zoom_r=10,):
        self.uvpath = self.scandir.beam_uvfits_path(self.beam)
        self.calpath = self.scandir.scheddir.beam_cal_path(self.beam)

        candrow = dict(
            ra_deg=self.ra, dec_deg=self.dec, dm_pccm3=self.dm_pccm,
            total_sample=self.total_sample, boxc_width=self.boxcwidth,
            lpix=self.lpix, mpix=self.mpix
        )
        
        cand = craco_cand.Cand(uvfits=self.uvpath, calfile=self.calpath, **candrow)
        cand.extract_data(padding=padding)
        cand.process_data(zoom_r=zoom_r)

        ### get filterbank, image
        fig, _ = cand.plot_filtb(dm=0)
        fig.savefig(f"{self.plotdir}/filtb_DM0.png", bbox_inches="tight")
        fig, _ = cand.plot_filtb(dm=cand.dm_pccm3, keepnan=True)
        fig.savefig(f"{self.plotdir}/filtb_DMcand.png", bbox_inches="tight")
        fig, _ = cand.plot_dmt(dmfact=30, ndm=30)
        fig.savefig(f"{self.plotdir}/butterfly_plot.png", bbox_inches="tight")

        ### diagnose plot
        fig = cand.plot_diagnostic_images()
        fig.savefig(f"{self.plotdir}/normal_image.png", bbox_inches="tight")

        ### save filterbank, images npy file, and header
        filtb = cand.filtb.copy()
        filtb = filtb.filled(0.)
        np.save(f"{self.plotdir}/filtb.npy", filtb)
        imgcand = cand.imgcube[cand.image_start_index:cand.image_end_index + 1]
        np.save(f"{self.plotdir}/imgcube.npy", imgcand)

        with open(f"{self.plotdir}/wcs.txt", "w") as fp:
            fp.write(cand.canduvfits.plan.wcs.__str__())

        lplan = craco_plan.PipelinePlan(
            cand.canduvfits.datauvsource, "--ndm 2 --npix 512 --fov 2.2d",
        )
        lplan = craco_plan.PipelinePlan(
            cand.canduvfits.datauvsource, "--ndm 2 --npix 512 --fov 2.2d",
        )
        cand.process_data(plan=lplan, zoom_r=10)

        fig = cand.plot_diagnostic_images()
        fig.savefig(f"{self.plotdir}/double_image.png", bbox_inches="tight")



    def sendalarm(self):
        main_msg = self._format_nice_candidate_block()
        main_res = self.slackbot.post_message(main_msg)
        main_ts = self.slackbot.get_thread_ts_from_response(main_res)

        fullinfo = "*Full pipeline candidate info* " + self.bestcand_dict.__str__()
        fullinfo += "\n*Full outdir* " + self.outdir
        _ = self.slackbot.post_message(fullinfo, thread_ts=main_ts)

        return main_ts

    def run_cand_plot(self, main_ts=None):
        self.make_cand_plot()
        _ = self.slackbot.upload_file(f"{self.plotdir}/filtb_DM0.png", "zero-DM filterbank", thread_ts=main_ts)
        _ = self.slackbot.upload_file(f"{self.plotdir}/filtb_DMcand.png", "pipeline-DM filterbank", thread_ts=main_ts)
        _ = self.slackbot.upload_file(f"{self.plotdir}/butterfly_plot.png", "zero-DM filterbank", thread_ts=main_ts)
        _ = self.slackbot.upload_file(f"{self.plotdir}/normal_image.png", "256-pix image", thread_ts=main_ts)
        _ = self.slackbot.upload_file(f"{self.plotdir}/double_image.png", "512-pix image", thread_ts=main_ts)
        _ = self.slackbot.upload_file(f"{self.plotdir}/filtb.npy", "tied-array beam filterbank data", thread_ts=main_ts)
        _ = self.slackbot.upload_file(f"{self.plotdir}/imgcube.npy", "synthesized images data", thread_ts=main_ts)
        _ = self.slackbot.upload_file(f"{self.plotdir}/wcs.txt", "WCS information", thread_ts=main_ts)

    def postall(self):
        main_ts = self.sendalarm()
        self.run_cand_plot(main_ts=main_ts)

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
            self.slackbot.upload_file(fieldracspng, "RACS-Ref comparison", thread_ts=self.main_ts)
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



        

    