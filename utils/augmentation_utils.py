# Customary Imports:
import numpy as np
import tensorflow as tf
from tensorflow.image import random_crop, random_brightness, random_contrast, \
                             random_flip_left_right, random_flip_up_down, resize
import utils.augmentation_utils

##################################################################################################################################
'''
AUGMENTATION UTILS:
'''
##################################################################################################################################
# General Augmentation Functions:
def add_rand_gaussian_noise(x, mean_val = 0.0, std_lower = 0.001, std_upper = 0.005, prob = 0.1, seed = None):
    '''
    This function introduces additive Gaussian Noise with a given mean and std, at
    a certain given probability.
    '''
    rand_var = tf.random.uniform(shape=(), seed=seed)
    if rand_var < prob:
        std = tf.random.uniform(shape = (), minval=std_lower, 
                                maxval=std_upper, seed=seed)
        noise = tf.random.normal(shape=x.shape, mean=mean_val, 
                                 stddev=std, dtype=tf.float32, seed=seed)
        x = tf.math.add(x, noise)
        x = tf.clip_by_value(x, 0, 1)
    return x
def add_rand_bright_shift(x, max_shift = 0.12, prob = 0.1, seed=None):
    '''
    Equivalent to adjust_brightness() using a delta randomly
    picked in the interval [-max_delta, max_delta) with a
    given probability that this function is performed on an image.
    The pixels lower than 0 are clipped to 0 and the pixels higher 
    than 1 are clipped to 1.
    '''
    rand_var = tf.random.uniform(shape=(), seed=seed)
    if rand_var < prob:
        x = tf.image.random_brightness(image=x, 
                                       max_delta=max_shift, 
                                       seed=seed)
        x = tf.clip_by_value(x, 0, 1)
    return x
def add_rand_contrast(x, lower = 0.2, upper = 1.8, prob = 0.1, seed=None):
    '''
    For each channel, this Op computes the mean of the image pixels in the channel 
    and then adjusts each component x of each pixel to (x - mean) * contrast_factor + mean
    with a given probability that this function is performed on an image. The pixels lower
    than 0 are clipped to 0 and the pixels higher than 1 are clipped to 1.
    '''
    rand_var = tf.random.uniform(shape=(),seed=seed)
    if rand_var < prob:
        x = tf.image.random_contrast(image=x, lower=lower, 
                                     upper=upper, seed=seed)
        x = tf.clip_by_value(x, 0, 1)
    return x

# Augmentation Pipeline Functions:
def augment(x, prob = 1/3, max_shift = 0.10, lower_con_factor = 0.2, upper_con_factor = 1.8,
            mean = 0.0, std_lower = 0.003, std_upper = 0.015, seed=None):
    if tf.random.uniform(()) > 0.5:
        x = tf.image.random_flip_left_right(x)
    if tf.random.uniform(()) > 0.5:
        x = tf.image.random_flip_up_down(x)
    x = tf.concat([add_rand_bright_shift(tf.slice(x, [0,0,0], [-1,-1,2]), max_shift, prob, seed=seed), 
                   tf.slice(x, [0,0,2], [-1,-1,-1])], axis = 2)
    x = tf.concat([add_rand_contrast(tf.slice(x, [0,0,0], [-1,-1,2]), lower_con_factor, upper_con_factor, prob, seed=seed), 
                   tf.slice(x, [0,0,2], [-1,-1,-1])], axis = 2)
    x = tf.concat([add_rand_gaussian_noise(tf.slice(x, [0,0,0], [-1,-1,2]), mean, std_lower, std_upper, prob//5, seed=seed), 
                   tf.slice(x, [0,0,2], [-1,-1,-1])], axis = 2)
    return x
def keras_augment(x, fill_mode='constant', interpolation_order=0, 
                  max_width_shift=0.1, max_height_shift=0.1, deg=None,seed=None):
    x = tf.keras.preprocessing.image.random_shift(x, max_width_shift, max_height_shift, row_axis=0, col_axis=1, 
                                                  channel_axis=2, fill_mode=fill_mode, interpolation_order=0)
    if deg is not None:
        x = tf.keras.preprocessing.image.random_rotation(x, deg, row_axis=0, col_axis=1, channel_axis=2, 
                                                         fill_mode=fill_mode, interpolation_order=interpolation_order)
        x = tf.keras.preprocessing.image.random_shear(x, deg, row_axis=0, col_axis=1, channel_axis=2, 
                                                      fill_mode=fill_mode, interpolation_order=interpolation_order)
    return x