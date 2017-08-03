'''
Created on Aug 3, 2017

@author: reischt1
'''
import tensorflow as tf



def weight_variable(shape, name=None):
    """ Initializes the variable with random numbers (not 0) """
    initial = tf.truncated_normal(shape, stddev=0.1)
    if name:
        return tf.Variable(initial, name=name)
    else:
        return tf.Variable(initial)

def bias_variable(shape, name=None):
    """ Initializes the bias variable with 0.1 everywhere """
    initial = tf.constant(0.1, shape=shape)
    if name:
        return tf.Variable(initial, name=name)
    else:
        return tf.Variable(initial)

def conv3d(x, W):
    """ Short definition of the convolution function for 3d """
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')

def define_model(Fz_xyz, keep_prob, parameters, logfile):
    """ Defines the model, that then will be trained or evaluated by the other functions.
    Input: Placeholder for the Fz values, dicitionary containing the parameters, handle for logfile.
    Output: outputlayer for the training. 
    """

    DBShape = parameters['DBShape']
    outChannels = parameters['outChannels']
    
    logfile.write('now define conv1 \n')
    #1st conv layer: Convolve the input (Fz_xyz) with 16 different filters, don't do maxpooling!  
    with tf.name_scope('conv1'):
        w_conv1 = weight_variable([4,4,4,DBShape[-1],16], 'wcv1')
        b_conv1 = bias_variable([16], 'bcv1')
        convLayer_1 = tf.nn.tanh(tf.add(conv3d(Fz_xyz, w_conv1),b_conv1))
        tf.summary.histogram("weights", w_conv1)
        tf.summary.histogram("biases", b_conv1)
        tf.summary.histogram("activations", convLayer_1)

    logfile.write('defining conv2 \n')
    #2nd conv layer: Convolve the result from layer 1 with 32 different filters, don't do maxpooling!
    with tf.name_scope('conv2'):
        w_conv2 = weight_variable([4, 4, 4, 16, 32], 'wcv2')
        b_conv2 =  bias_variable([32], 'bcv2')
        convLayer_2 = tf.nn.tanh(tf.add(conv3d(convLayer_1, w_conv2),b_conv2))
        tf.summary.histogram("weights", w_conv2)
        tf.summary.histogram("biases", b_conv2)
        tf.summary.histogram("activations", convLayer_2)
                
    logfile.write('all conv layers defined, defining fc layer \n')
    # fc 1   
    with tf.name_scope('fc'):
        w_fc1 = weight_variable([DBShape[-2]*32, 64], 'wfc1')
        b_fc1 = bias_variable(DBShape[:2]+[64,], 'bfc1')
        convLayer_2_flat = tf.reshape(convLayer_2, [-1,]+DBShape[:2]+ [DBShape[-2]*32])
        fcLayer_1 = tf.nn.relu(tf.tensordot(convLayer_2_flat, w_fc1,axes=[[3],[0]])+b_fc1)  #Is this correct? the result of the tensordot and the b_fc1 don't have the same dimensions
        tf.summary.histogram("weights", w_fc1)
        tf.summary.histogram("biases", b_fc1)
        tf.summary.histogram("activations", fcLayer_1)
    
    # Dropout
    with tf.name_scope('dropout'):
        fcLayer_1_dropout = tf.nn.dropout(fcLayer_1, keep_prob)
    
    # Readout Layer
    with tf.name_scope('outputLayer'):
        w_out = weight_variable([64, outChannels], 'wout')
        b_out = bias_variable(DBShape[:2]+[outChannels], 'bout')
        outputLayer = tf.nn.relu(tf.add(tf.tensordot(fcLayer_1_dropout, w_out, axes=[[3],[0]]), b_out))
        tf.summary.histogram("weights", w_out)
        tf.summary.histogram("biases", b_out)
        tf.summary.histogram("activations", outputLayer)
        tf.summary.image('predictions', outputLayer, parameters['numberTBImages'])

    return outputLayer
