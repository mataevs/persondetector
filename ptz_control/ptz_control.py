'''
Script that controls a PTZ camera (Samsung SNP-3120V) - pan, tilt, zoom
'''

import requests
import sys

from requests.auth import HTTPDigestAuth
from requests.auth import HTTPBasicAuth

# the IP of the PTZ camera
ip = '172.16.5.217'

# user and password required by the PTZ camera (default values for Samsung SNP-3120V)
username = 'admin'
password = '4321'

# base URL for requesting MJPEG video (for Samsung SNP-3120V)
base_ptz_url = 'http://' + ip + '/cgi-bin/ptz.cgi'


def setPTZ(pan, tilt, zoom):
    try:
        if pan > 359:
            pan -= 359
        reqParams = {'move': 'absmove', 'pan': pan, 'tilt': tilt, 'zoom': zoom}
        r = requests.get(base_ptz_url, params=reqParams, auth=HTTPDigestAuth(username, password))

        # if things go wrong
        if r.status_code != 200:
            error_description = "[ERROR] Connection did not return OK."
            if r.status_code == 401:
                error_description += " Reason: Authentication failed (401)."
            elif r.status_code == 403:
                error_description += " Reason: Forbidden (403)."
            elif r.status_code == 404:
                error_description += " Reason: Not found (404)."
            elif r.status_code == 405:
                error_description += " Reason: Not allowed (405)."
            else:
                error_description += " Reason: (" + str(r.status_code) + ")."
            print error_description
            sys.exit()
        else:
            if r.content == "OK":
                print "Moved to ", pan, tilt, zoom
            else:
                print "Command error"

    # hint for checking the camera's IP
    except requests.ConnectionError, err:
        print "[ERROR] Unable to connect to PTZ camera. Reason: " + str(err)
        sys.exit()


def getPTZ():
    try:
        reqParams = {'query': 'ptz'}
        r = requests.get(base_ptz_url, params=reqParams, auth=HTTPDigestAuth(username, password))

        # if things go wrong
        if r.status_code != 200:
            error_description = "[ERROR] Connection did not return OK."
            if r.status_code == 401:
                error_description += " Reason: Authentication failed (401)."
            elif r.status_code == 403:
                error_description += " Reason: Forbidden (403)."
            elif r.status_code == 404:
                error_description += " Reason: Not found (404)."
            elif r.status_code == 405:
                error_description += " Reason: Not allowed (405)."
            else:
                error_description += " Reason: (" + str(r.status_code) + ")."
            print error_description
            sys.exit()
        else:
            if r.content != "NG":
                pan = tilt = zoom = -1
                for line in r.iter_lines():
                    if 'pan' in line:
                        pan = int(line.split(':')[1])
                    elif 'tilt' in line:
                        tilt = int(line.split(':')[1])
                    elif 'zoom' in line:
                        zoom = int(line.split(':')[1])
                return pan, tilt, zoom
            else:
                print "Command error"
                return getPTZ()  # retry

    # hint for checking the camera's IP
    except requests.ConnectionError, err:
        print "[ERROR] Unable to connect to PTZ camera. Reason: " + str(err)
        sys.exit()
