import requests
import json
import os
import sys

def get_webhook_env_var():
    try:
        webhook = os.environ['MATTERMOST_CRACO_HOOK']
    except KeyError as KE:
        webhook = None
    return webhook


class MattermostPostManager:

    def __init__(self, webhook = None):
        if webhook:
            self.webhook = webhook
        else:
            self.webhook = get_webhook_env_var()

        if self.webhook is None:
            raise ValueError(f"Webhook not specified and could not fetch from env")
        
    def post_message(self, msg:str, header:str = None):
        '''
        msg = str
        header (optional) = str
        '''
        if not header:
            header = {'Content-Type':'application/json'}

        data = {'text': msg}

        resp = requests.post(self.webhook, data = json.dumps(data), headers = header)
        return resp