import base64
import time as time_

from PIL import Image

'''
def base64_to_image(data, width, height):
    """ Convert a base-64 image to an Image object
        from the PIL library. """
    decoded_string = base64.b64decode(data)
    return decoded_string
    #return Image.frombuffer("RGB", (width, height), decoded_string)
'''

def millis():
    return int(round(time_.time() * 1000))

def image_to_base64(image):
    """ Convert an image from PIL format to base64 for
        further processing down the pipeline. """
    
    return {
        "created_at" : millis(),
        "context" : "",
        "sensor_type" : "ptz",
        "sensor_id" : "samsumg_ptz",
        "sensor_position" : [0, 0, 0],
        "type" : "image_rgb",
        "image_rgb": { "encoder_name" : "jpg", "image" : base64.b64encode(image.tostring()), "width" : image.size[0], "height": image.size[1] }
    }
