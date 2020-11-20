# Customary Imports:
import tensorflow as tf
import utils.model_utils

##################################################################################################################################
'''
MODEL UTILS:
'''
##################################################################################################################################
# Custom Metrics:
def normalize(tensor, clip=False):
    # Normalizes Tensor from 0-1
    if clip:
        out = tf.clip_by_value(tensor, 0, 1)
    else:
        out = tf.math.divide_no_nan(tf.math.subtract(tensor, tf.math.reduce_min(tensor)),
                                    tf.math.subtract(tf.math.reduce_max(tensor), tf.math.reduce_min(tensor)))
    return out
def PSNR(y_true, y_pred):
    max_pixel = 1.0
    y_true_norm = tf.map_fn(normalize, y_true)
    y_pred_norm = tf.map_fn(normalize, y_pred)
    PSNR = tf.image.psnr(y_true_norm, y_pred_norm, max_pixel)
    return PSNR

def SSIM(y_true, y_pred):
    max_pixel = 1.0
    y_true_norm = tf.map_fn(normalize, y_true)
    y_pred_norm = tf.map_fn(normalize, y_pred)
    SSIM = tf.image.ssim(y_true_norm,y_pred_norm,max_pixel,filter_size=11,
                         filter_sigma=1.5,k1=0.01,k2=0.03)
    return SSIM

def MS_SSIM(y_true, y_pred):
    max_pixel = 1.0
    y_true_norm = tf.map_fn(normalize, y_true)
    y_pred_norm = tf.map_fn(normalize, y_pred)
    MS_SSIM = tf.image.ssim_multiscale(y_true_norm,
                                       y_pred_norm,
                                       max_pixel)
    return MS_SSIM
