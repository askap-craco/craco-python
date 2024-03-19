#!/usr/bin/env python
import sys
import json
from operator import itemgetter
import requests
from xml.etree.ElementTree import parse
from threading import (Timer, Condition)

import epics

import pprint

import time
import datetime


def get_beast_config(file_name="alarms.xml"):
    """
    Parse an cs-studio beast xml config file into a `dict`
    :param file_name: the BEAST config xml file name

    :return: `dict` with PV names as keys and attributes as values in a nested
              `dict`
    """
    tree = parse(file_name)
    root = tree.getroot()
    pvs = {}
    keys = ("delay", "description", "enabled")
    action_keys = ("details", "delays", "title")
    for pv in root.iter("pv"):
        name = pv.get("name")
        d = {}
        for child in pv:

            if child.tag in keys:
                d[child.tag] = child.text.strip()
            for elem in child:
                if elem.tag in action_keys:
                    txt = elem.text.strip()
                    if txt.startswith("mailto"):
                        txt = txt.split("body=")[-1]
                    d["_".join((child.tag, elem.tag))] = txt
        if "enabled" in d and d["enabled"].strip() == "false":
            continue
        pvs[name] = d
    return pvs


class AlarmStack(object):
    def __init__(self):
        self.condition = Condition()
        self.items = []

    def put(self, item):
        with self.condition:
            self.items.append(item)
            self.condition.notify()

    def get(self, length=20, blocking=True):
        """
        Get `length` items of the stack
        :param length: number of items to fetch
        :param blocking: block on empty stack
        :return:
        """
        with self.condition:
            while blocking and len(self.items) == 0:
                self.condition.wait()
            if len(self.items) > length:
                items = self.items[:length]
                self.items = self.items[length:]
            else:
                items = self.items[:]
                self.items = []
        return items


def alarm_timer(queue, value, severity, status, **kwargs):
    out = {"value": value, "severity": severity, "status": status}
    out.update(kwargs)
    queue.put(out)


class Alarm(object):
    def __init__(self, name, queue, **kwargs):
        self.pv = None
        self._timer = None
        self._severity = None
        self.queue = queue
        self.config = kwargs
        self.name = name
        self._setup_pv(name)

    def _setup_pv(self, name):
        self.pv = epics.PV(name, auto_monitor=epics.dbr.DBE_ALARM)
        self.severity = self.pv.severity
        self.pv.add_callback(self._callback)
        print("set up", name, self.severity)

    def _cancel_timer(self):
        if self._timer and self._timer.is_alive():
            self._timer.cancel()
            print("Cancelled timer", self.name, file=sys.stderr)
        self._timer = None

    def _callback(self, char_value=None, severity=None, status=None,
                  **kw):
        if severity == 0:
            self._cancel_timer()
            return
        else:
            if severity < self.severity:
                self._cancel_timer()
            delay = float(self.config["delay"])
            print("Started timer", self.name, severity, file=sys.stderr)
            self._timer = Timer(delay, alarm_timer,
                                [self.queue,
                                 char_value, severity, status],
                                self.config)
            self._timer.daemon = True
            self._timer.start()


SEVERITY = ("NO ALARM", "MINOR", "MAJOR", "INVALID")

STATUS = (
        "No Alarm",
        "Read",
        "Write",
        "Hihi",
        "High",
        "Lolo",
        "Low",
        "State",
        "Cos",
        "Comm",
        "Timeout",
        "HWLimit",
        "Calc",
        "Scan",
        "Link",
        "Soft",
        "Bad Sub",
        "Undefined",
        "Disable",
        "Simm",
        "Read Access",
        "Write Access"
        )
COLOURS = ("grey", "warning", "danger", "#ff1d8e")


class SlackAlert(object):
    def __init__(self, webhook):
        self._url = webhook

    def new_message(self):
        return {"username": "TelescopeAlarm",
                "text": "Alarm notification @channel",
                "attachments": [],
                "link_names": 1}
    
    @staticmethod
    def markup(alarm):
        status = alarm.get("status", 0)
        severity = alarm.get("severity", 1)
        value = alarm.get("value", " - ")
        description = alarm.get("description", "")
        detail = alarm.get("automated_action_details", "")
        detail = detail.format(SEVERITY[severity], value)
        guide = alarm.get("guidance_details", "")
        out = {"fallback": detail, "color": COLOURS[severity],
               "text": "{0}".format(detail), "title": description,
               "fields": [{"title": "Severity", "value": SEVERITY[severity],
                           "short": "true"},
                          {"title": "Status", "value": STATUS[status],
                           "short": "true"}],
               "footer": guide
               }
        return out

    def publish(self, alarms):
        message = self.new_message()
        for alarm in sorted(alarms, key=itemgetter('severity'), reverse=True):
            message["attachments"].append(self.markup(alarm))
        jmessage = json.dumps(message)
        r = requests.post(self._url, jmessage,
                          headers={'content-type': 'application/json'})

        pprint.pprint(message)

if __name__ == "__main__":
    alarm_config = get_beast_config()

    hook = "TBD"
    #hook = "https://hooks.slack.com/services/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"
    #hook = "https://hooks.slack.com/services/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"
    #hook = "https://hooks.slack.com/services/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"
    
# RMS This was the old one which looked like it was being sent to the main CRAFT channel
    hook = "https://hooks.slack.com/services/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"
    

# from SBSTATEMONITOR this one looks like it was being sent to coschedblock
    hook="https://hooks.slack.com/services/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"
    
    sa = SlackAlert(hook)
    stack = AlarmStack()
    substart = datetime.datetime.now()
    for k, v in list(alarm_config.items()):
        alarm = Alarm(k, stack, **v)

    subend = datetime.datetime.now()
    sub_interval = subend - substart

    jmessage = {
        'text':'beast2py started monitoring {:d} alarm points. It took {} to subscribe to them'.format(len(list(alarm_config.items())), str(sub_interval))
        }
    print(jmessage)
    r = requests.post(hook, json.dumps(jmessage),
                      headers={'content-type': 'application/json'})

    print(r)


    while True:
        alarms = stack.get()
        sa.publish(alarms)
        # slack API calls are throttle to one a second
        time.sleep(1.5)
    #sa.publish(alarms.values())
