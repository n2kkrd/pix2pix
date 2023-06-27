import numpy as np

from util import lab_to_rgb
import mxnet.ndarray as nd


def lab_parts_to_rgb(image, brightness, ctx):
    if image is None:
        raise ValueError("Image can not be None")
    if brightness is None:
        raise ValueError("brightness can not be None")
    if isinstance(image, np.ndarray):
        image = np.squeeze(image, axis=0)
    else:
        image = np.squeeze(image.asnumpy(), axis=0)
    if isinstance(brightness, np.ndarray):
        brightness = np.squeeze(brightness, axis=0)
    else:
        brightness = np.squeeze(brightness.asnumpy(), axis=0)
    a_chan, b_chan = np.split(image, 2, 0)
    lab = np.stack([((brightness + 1) / 2) * 100, a_chan * 110, b_chan * 110], axis=3)
    rgb = lab_to_rgb(nd.array(np.squeeze(lab, axis=0)).as_in_context(ctx), ctx=ctx)
    return np.ndarray.astype(rgb.asnumpy() * 255, dtype='uint8')
