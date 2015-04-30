'''
Script that gets frames from a PTZ camera (Samsung SNP-3120V)
'''

import requests
from requests.auth import HTTPDigestAuth
from requests.auth import HTTPBasicAuth

import itertools
import sys

from StringIO import StringIO
from imageUtil import *
from PIL import Image

# the IP of the PTZ camera
ip = '192.168.0.100'

# various parameters for the request sent to the camera's server; 'msubmenu=jpg' is required, 
# the rest can be altered according to specifications (resolution: 1 (640x480), 3 (320x240); frate: 1 - 25)
reqParams = {'msubmenu': 'mjpg', 'profile': '1', 'resolution': '1', 'frate': '10', 'compression': '1'}

# user and password required by the PTZ camera (default values for Samsung SNP-3120V)
username = 'admin'
password = '4321'

# base URL for requesting MJPEG video (for Samsung SNP-3120V)
base_ptz_url = 'http://' + ip + '/cgi-bin/video.cgi'


def getFrame(enableViewer=False):
    try:
        r = requests.get(base_ptz_url, params=reqParams, auth=HTTPDigestAuth(username, password), stream=True,
                         hooks=dict(response=response_hook))

        content = ""
        i = 0
        corrupted = 0
        iteration = 0
        for chunk in r.iter_chunks():
            if "--Samsung" or "Content" in chunk:
                header = chunk[chunk.find("--Samsung"):chunk.find("\n\r")]
                if header:
                    # print "Header chunk:\n" + chunk
                    toAdd = chunk[:chunk.find("--Samsung")].strip('\n\r')
                    if toAdd:
                        #print "\n\n\nTo add after header:\n" + toAdd
                        #print "\nAt:\n" + content
                        content += toAdd  # add whatever is before the header
                    #print "\n\n\nTo add before header:\n" + chunk[:chunk.find("--Samsung")]
                    #print "\nAt:\n" + content

                    if iteration > 0:
                        try:
                            img = Image.open(StringIO(content))
                            if enableViewer:
                                yield img, image_to_base64(img)
                            else:
                                yield image_to_base64(img)
                        except IOError, e:
                            print "Caught corrupted frame [" + str(corrupted) + "]"
                            corrupted += 1
                            pass

                    iteration += 1
                    content = b""
                    toAdd = chunk[chunk.find("--Samsung") + 68:].strip('\n\r')
                    if toAdd:
                        content = toAdd  # add the rest of the chunk to the content (next frame content)
                else:

                    toAdd = chunk
                    if toAdd:
                        content += toAdd  # chunk without header
            else:
                toAdd = chunk
                if toAdd:
                    content += chunk  # chunk without header

    # hint for checking the camera's IP
    except requests.ConnectionError, err:
        print "[ERROR] Unable to connect to PTZ camera. Reason: " + str(err)
        sys.exit()


def response_hook(response, *args, **kwargs):
    response.iter_chunks = lambda amt=None: iter_chunks(response.raw._fp, amt=amt)
    return response


def iter_chunks(response, amt=None):
    """
    A copy-paste version of httplib.HTTPConnection._read_chunked() that
    yields chunks served by the server.
    """
    if response.chunked:
        while True:
            line = response.fp.readline().strip()
            arr = line.split(';', 1)
            try:
                chunk_size = int(arr[0], 16)
            except ValueError:
                response.close()
                raise httplib.IncompleteRead(chunk_size)
            if chunk_size == 0:
                break
            value = response._safe_read(chunk_size)
            yield value
            # we read the whole chunk, get another
            response._safe_read(2)  # toss the CRLF at the end of the chunk

        # read and discard trailer up to the CRLF terminator
        ### note: we shouldn't have any trailers!
        while True:
            line = response.fp.readline()
            if not line:
                # a vanishingly small number of sites EOF without
                # sending the trailer
                break
            if line == '\r\n':
                break

        # we read everything; close the "file"
        response.close()
    else:
        # Non-chunked response. If amt is None, then just drop back to
        # response.read()
        if amt is None:
            yield response.read()
        else:
            # Yield chunks as read from the HTTP connection
            while True:
                ret = response.read(amt)
                if not ret:
                    break
                yield ret
