# Customary Imports:
import numpy as np
import tensorflow as tf
from tensorflow.image import resize
import utils.downsampling_utils

##################################################################################################################################
'''
DOWNSAMPLING UTILS:
'''
##################################################################################################################################
def downsample(x, down_ratio=(2,1), min_shape=(128,128), method = 'bicubic'):
    x = (x/np.iinfo(x.dtype).max).astype(np.float32)
    if x.shape[0] < min_shape[0]:
        x = resize(x, [min_shape[0], x.shape[1]], method, antialias=True)
    if x.shape[1] < min_shape[1]:
        x = resize(x, [x.shape[1], min_shape[1]], method, antialias=True)
    down = x[::down_ratio[0], ::down_ratio[1], :]
    x = (x, down)
    return x

def downsample_zerofill(x, down_ratio=None, sparsity=None, min_shape=(128,128), method='bicubic'):
    x = (x/np.iinfo(x.dtype).max).astype(np.float32)
    if x.shape[0] < min_shape[0]:
        x = resize(x, [min_shape[0], x.shape[1]], method, antialias=True)
    if x.shape[1] < min_shape[1]:
        x = resize(x, [x.shape[1], min_shape[1]], method, antialias=True)
    if down_ratio is not None and sparsity is None:
        mask = np.zeros(x.shape, dtype = np.float32)
        mask[::down_ratio[0], ::down_ratio[1], :] = 1
        down = x * mask
        #mask[mask == 0] = 0.5
        x = np.dstack((x, down, mask))
    elif down_ratio is None and sparsity is not None:
        mask = np.random.random(x.shape).astype(np.float32)
        mask[mask > sparsity] = 1
        mask[mask <= sparsity] = 0
        down = x * mask
        #mask[mask == 0] = 0.5
        x = np.dstack((x, down, mask))
    else:
        print('ERROR: Please, input either a downsampling ratio or sparsity.')
    return x