# Customary Imports:
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Add, Input
from tensorflow.keras.layers import BatchNormalization, UpSampling2D, Concatenate, Conv2DTranspose, DepthwiseConv2D

###################################################################################################
'''
MODEL DEFINITION:
Modified UNet/UNet++/SqueezeNet/MobileNet
'''
def UNet(img, filters = 32, kernel_size = 3, padding = 'same',
         activation = 'relu', kernel_initializer = 'glorot_normal', prob = 0.0):
    shortcut1_1 = img
    [out, shortcut1_2] = DownBlock(img, filters, kernel_size, padding, activation, kernel_initializer, prob)
    residual = shortcut1_2
    shortcut1_2 = Conv2D_BatchNorm(shortcut1_2, filters//4, kernel_size=1, strides=1, padding=padding,
                                   activation=activation, kernel_initializer=kernel_initializer, prob=prob)
    shortcut1_2 = Conv2D_BatchNorm(shortcut1_2, filters//4, kernel_size=3, strides=1, padding=padding,
                                   activation=activation, kernel_initializer=kernel_initializer, prob=prob)
    shortcut1_2 = Conv2D_BatchNorm(shortcut1_2, filters//4, kernel_size=7, strides=1, padding=padding,
                                   activation=activation, kernel_initializer=kernel_initializer, prob=prob)
    shortcut1_2 = Conv2D_BatchNorm(shortcut1_2, filters//4, kernel_size=3, strides=1, padding=padding,
                                   activation=activation, kernel_initializer=kernel_initializer, prob=prob)
    shortcut1_2 = Conv2D_BatchNorm(shortcut1_2, filters, kernel_size=1, strides=1, padding=padding,
                                   activation=activation, kernel_initializer=kernel_initializer, prob=prob)
    shortcut1_2 = Add()([residual, shortcut1_2])
    
    [out, shortcut2_1] = DownBlock(out, filters*2, kernel_size, padding, activation, kernel_initializer, prob)
    residual = shortcut2_1
    shortcut2_1 = Conv2D_BatchNorm(shortcut2_1, (filters*2)//4, kernel_size=1, strides=1, padding=padding,
                                   activation=activation, kernel_initializer=kernel_initializer, prob=prob)
    shortcut2_1 = Conv2D_BatchNorm(shortcut2_1, (filters*2)//4, kernel_size=3, strides=1, padding=padding,
                                   activation=activation, kernel_initializer=kernel_initializer, prob=prob)
    shortcut2_1 = Conv2D_BatchNorm(shortcut2_1, (filters*2)//4, kernel_size=5, strides=1, padding=padding,
                                   activation=activation, kernel_initializer=kernel_initializer, prob=prob)
    shortcut2_1 = Conv2D_BatchNorm(shortcut2_1, (filters*2)//4, kernel_size=3, strides=1, padding=padding,
                                   activation=activation, kernel_initializer=kernel_initializer, prob=prob)
    shortcut2_1 = Conv2D_BatchNorm(shortcut2_1, filters*2, kernel_size=1, strides=1, padding=padding,
                                   activation=activation, kernel_initializer=kernel_initializer, prob=prob)
    shortcut2_1 = Add()([residual, shortcut2_1])
    
    [out, shortcut3_1] = DownBlock(out, filters*2*2, kernel_size, padding, activation, kernel_initializer, prob)
    residual = shortcut3_1
    shortcut3_1 = Conv2D_BatchNorm(shortcut3_1, (filters*2*2)//4, kernel_size=1, strides=1, padding=padding,
                                   activation=activation, kernel_initializer=kernel_initializer, prob=prob)
    shortcut3_1 = Conv2D_BatchNorm(shortcut3_1, (filters*2*2)//4, kernel_size=3, strides=1, padding=padding,
                                   activation=activation, kernel_initializer=kernel_initializer, prob=prob)
    shortcut3_1 = Conv2D_BatchNorm(shortcut3_1, filters*2*2, kernel_size=1, strides=1, padding=padding,
                                   activation=activation, kernel_initializer=kernel_initializer, prob=prob)
    shortcut3_1 = Add()([residual, shortcut3_1])
        
    [out, shortcut4_1] = DownBlock(out, filters*2*2*2, kernel_size, padding, activation, kernel_initializer, prob)
    shortcut4_1 = Conv2D_BatchNorm(shortcut4_1, filters*2*2*2, kernel_size=3, strides=1, padding=padding,
                                   activation=activation, kernel_initializer=kernel_initializer, prob=prob)

    out = BridgeBlock(out, filters*2*2*2*2, kernel_size, padding, activation, kernel_initializer, prob)

    out = Concatenate()([out, shortcut4_1])
    out = UpBlock(out, filters*2*2*2, kernel_size, padding, activation, kernel_initializer, prob)
    out = Concatenate()([out, shortcut3_1])
    out = UpBlock(out, filters*2*2, kernel_size, padding, activation, kernel_initializer, prob)
    out = Concatenate()([out, shortcut2_1])
    out = UpBlock(out, filters*2, kernel_size, padding, activation, kernel_initializer, prob)
    out = Concatenate()([out, shortcut1_2])


    out = Conv2D_BatchNorm(out, filters, kernel_size, strides=1, padding=padding,
                           activation=activation, kernel_initializer=kernel_initializer, prob=prob)

    out = Conv2D_BatchNorm(out, filters, kernel_size, strides=1, padding=padding,
                           activation=activation, kernel_initializer=kernel_initializer, prob=prob)

    out = Conv2D(filters=1, kernel_size=1,
                 strides=1, padding=padding,
                 activation='linear',
                 kernel_initializer=kernel_initializer)(out)
    out = Add()([out, shortcut1_1])
    out = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True,
                             scale=True, beta_initializer='zeros', gamma_initializer='ones',
                             moving_mean_initializer='zeros', moving_variance_initializer='ones',
                             beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                             gamma_constraint=None)(out)

    return out

def DownBlock(img, filters, kernel_size, padding, activation, kernel_initializer, prob=0.0):
    #print('DOWN_in: '+str(img.shape))
    out = Conv2D_BatchNorm(img, filters, kernel_size, strides=1, padding=padding,
                           activation=activation, kernel_initializer = kernel_initializer, prob=prob)
    out = Conv2D_BatchNorm(out, filters, kernel_size, strides=1, padding=padding,
                           activation=activation, kernel_initializer = kernel_initializer, prob=prob)
    shortcut = out
    out = MaxPooling2D(pool_size = (2,2))(out)
    #print('DOWN_out: '+str(out.shape))
    return [out, shortcut]

def BridgeBlock(img, filters, kernel_size, padding, activation, kernel_initializer, prob=0.0):
    #print('UP_in: '+str(img.shape))
    #print(filters)
    out = Conv2D_BatchNorm(img, filters, kernel_size, strides = 1, padding=padding,
                           activation=activation, kernel_initializer = kernel_initializer, prob=prob)
    out = Conv2D_BatchNorm(out, filters//2, kernel_size, strides = 1, padding=padding,
                           activation=activation, kernel_initializer = kernel_initializer, prob=prob)
    out = UpSampling2D(size = (2,2), interpolation='bilinear')(out)
    #print('UP_out: '+str(out.shape))
    return out

def UpBlock(img, filters, kernel_size, padding, activation, kernel_initializer, prob=0.0):
    #print('UP_in: '+str(img.shape))
    #print(filters)
    out = Conv2D_BatchNorm(img, filters, kernel_size, strides = 1, padding=padding,
                           activation=activation, kernel_initializer = kernel_initializer, prob=prob)
    out = Conv2D_BatchNorm(out, filters, kernel_size, strides = 1, padding=padding,
                           activation=activation, kernel_initializer = kernel_initializer, prob=prob)
    out = UpSampling2D(size = (2,2), interpolation='bilinear')(out)
    #print('UP_out: '+str(out.shape))
    return out

###################################################################################################
'''
MODEL FUNCTIONS:
'''
def Conv2D_BatchNorm(img, filters, kernel_size=3, strides=1, padding='same',
                     activation='linear', kernel_initializer='glorot_normal', prob=0.0,
                     dilation_rate=1):
    if kernel_size>1:
      if filters//img.shape[-1] == 0:
        out = Conv2D(filters=filters, kernel_size=1,
              strides=strides, padding=padding,
              activation=activation,
              kernel_initializer=kernel_initializer, 
              dilation_rate=dilation_rate)(img)
      else:
        out = img
      out = DepthwiseConv2D(depth_multiplier=filters//out.shape[-1], 
                            kernel_size=kernel_size,
                            strides=strides, padding=padding,
                            activation=activation,
                            kernel_initializer=kernel_initializer, 
                            dilation_rate=dilation_rate)(out)
      out = tf.keras.layers.SpatialDropout2D(prob)(out)
      out = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True,
                               scale=True, beta_initializer='zeros', gamma_initializer='ones',
                               moving_mean_initializer='zeros', moving_variance_initializer='ones',
                               beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                               gamma_constraint=None)(out)
      out = Conv2D(filters=filters, kernel_size=1,
                   strides=strides, padding=padding,
                   activation=activation,
                   kernel_initializer=kernel_initializer, 
                   dilation_rate=dilation_rate)(out)
    else:
      out = Conv2D(filters=filters, kernel_size=kernel_size,
                   strides=strides, padding=padding,
                   activation=activation,
                   kernel_initializer=kernel_initializer, 
                   dilation_rate=dilation_rate)(img)
    out = tf.keras.layers.SpatialDropout2D(prob)(out)
    out = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True,
                             scale=True, beta_initializer='zeros', gamma_initializer='ones',
                             moving_mean_initializer='zeros', moving_variance_initializer='ones',
                             beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                             gamma_constraint=None)(out)
    return out

###################################################################################################
'''
FUNCTION TO INSTANTIATE MODEL:
'''
def getModel(input_shape, filters, kernel_size, padding='same', activation='relu',
             kernel_initializer='glorot_normal', prob=0.0):
    model_inputs = Input(shape=input_shape, name='img')
    model_outputs = UNet(model_inputs, filters=filters, kernel_size=kernel_size, padding=padding,
                         activation=activation, kernel_initializer=kernel_initializer, prob=prob)
    model = Model(model_inputs, model_outputs, name='UNet_Model')
    return model
getModel.__name__ = 'UNet_Model'
