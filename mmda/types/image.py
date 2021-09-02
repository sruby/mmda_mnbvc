"""


Monkey patch the PIL.Image methods to add base64 conversion

"""

import base64
from io import BytesIO
from pdf2image import convert_from_path as _convert_from_path

from PIL import Image


def tobase64(self):
    # Ref: https://stackoverflow.com/a/31826470
    buffered = BytesIO()
    self.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode("utf-8")


def frombase64(img_str):
    # Use the same naming style as the original Image methods
    buffered = BytesIO(base64.b64decode(img_str))
    img = Image.open(buffered)
    return img

load_pdf_images_from_path = _convert_from_path 
Image.Image.tobase64 = tobase64 # This is the method applied to individual Image classes
Image.Image.to_json = tobase64 # Use the same API as the others
Image.frombase64 = frombase64 # This is bind to the module, used for loading the images