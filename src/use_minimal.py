import tensorflow as tf
# import readAFMDATAfile as readAFM
import readAFMHDF5 as readAFM
import numpy as np
import time
import argparse
import h5py


parser = argparse.ArgumentParser()
parser.add_argument("name", help="Name given to all output files.")
parser.add_argument("-i", "--input_file", default="parameters.in", help="the path to the input file (default: %(default)s)")
# parser.add_argument("-o", "--output", default="output/", help="produced files and folders will be saved here (default: %(default)s)")
args=parser.parse_args()

# parser.add_argument("-v", "--verbosity", action="count", default=0)

print('Parsed name argument {} of type {}.'.format(args.name, type(args.name)))
print('Parsing parameters from {}'.format(args.input_file))

parameters = {'restorePath': "../save/CNN_minimal_TR1_00.ckpt",                                                     # Typically: "./save/CNN_minimal_TR1_{}.ckpt"
              'savePath': None,         # Typically: 'savePath': "./save/CNN_minimal_TR1_{}.ckpt".format(args.name)
              'DBPath': '../AFMDB_version_01.hdf5',
              'viewPath': '../scratch/viewfile_{}.hdf5'.format(args.name),
              'logPath': '../scratch/out_minimal_{}.log'.format(args.name),
              'testbatchSize': 50}          


# Here smt like parameters.update(parsedParameters)


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



if __name__=='__main__':

    logfile=open(parameters['logPath'], 'w', 0)


    logfile.write('define the first two placeholders \n')
    solution = tf.placeholder(tf.float32, [None, 81, 81, 5])
    Fz_xyz = tf.placeholder(tf.float32, [None, 81, 81, 41, 1])

    convVars_1 = {'weights': weight_variable([4,4,4,1,16], 'wcv1'),
                    'biases': bias_variable([16], 'bcv1')}

    convVars_2 = {'weights': weight_variable([4, 4, 4, 16, 32], 'wcv2'),
                    'biases': bias_variable([32], 'bcv2')}

    fcVars_1 = {'weights': weight_variable([41*32, 64], 'wfc1'),
                    'biases': bias_variable([81, 81, 64], 'bfc1')}

    outputVars = {'weights': weight_variable([64, 5], 'wout'),
                    'biases': bias_variable([81, 81, 5], 'bout')}


    logfile.write('now define conv1 \n')
    #1st conv layer: Convolve the input (Fz_xyz) with 16 different filters, don't do maxpooling!  
    convLayer_1 = tf.nn.tanh(tf.add(conv3d(Fz_xyz, convVars_1['weights']),convVars_1['biases']))

    logfile.write('defining conv2 \n')
    #2nd conv layer: Convolve the result from layer 1 with 32 different filters, don't do maxpooling!
    convLayer_2 = tf.nn.tanh(tf.add(conv3d(convLayer_1, convVars_2['weights']),convVars_2['biases']))
        
    logfile.write('all conv layers defined, defining fc layer \n')
    # fc 1   
    convLayer_2_flat = tf.reshape(convLayer_2, [-1, 81, 81, 41*32])
    fcLayer_1 = tf.nn.relu(tf.tensordot(convLayer_2_flat, fcVars_1['weights'],axes=[[3],[0]])+fcVars_1['biases'])  #Is this correct? the result of the tensordot and the b_fc1 don't have the same dimensions
    
    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    fcLayer_1_dropout = tf.nn.dropout(fcLayer_1, keep_prob)
    
    # Readout Layer
    outputLayer = tf.nn.relu(tf.add(tf.tensordot(fcLayer_1_dropout, outputVars['weights'], axes=[[3],[0]]), outputVars['biases']))


#     set up evaluation system
#     cost = tf.reduce_mean(tf.abs(tf.subtract(prediction, solution)))
    cost = tf.reduce_sum(tf.square(tf.subtract(outputLayer, solution)))      
    accuracy = cost/parameters['testbatchSize']
#     accuracy = tf.reduce_mean(tf.cast(tf.abs(tf.subtract(prediction, solution)), tf.float32))
    
    # Crate saver
    saver = tf.train.Saver()
    
    # Init op
    init_op = tf.global_variables_initializer()
    
    # Start session
    logfile.write('it worked so far, now start session \n')
    
    with tf.Session() as sess:
        init_op.run()
    
        print("b_conv1, as initialized: ")
        print(sess.run(convVars_1['biases']))
                
#         Pack this into a function!
        if parameters['restorePath']:
            saver.restore(sess, parameters['restorePath'])
            logfile.write("Model restored. \n")
            print("Model restored. See here b_conv1 restored:")
            print(sess.run(convVars_1['biases']))
            logfile.write('Variables initialized successfully \n')
        
        AFMdata = readAFM.AFMdata(parameters['DBPath'])
        
        testbatch = AFMdata.batch_test(parameters['testbatchSize'])
        testaccuracy=accuracy.eval(feed_dict={Fz_xyz: testbatch['forces'], solution: testbatch['solutions'], keep_prob: 1.0})
        logfile.write("test accuracy %g \n"%testaccuracy)
        
        # Save two np.arrays to be able to view it later.
        viewfile = h5py.File(parameters['viewPath'], 'w')
        viewfile.attrs['testaccuracy']=testaccuracy
        viewfile.create_dataset('predictions', data=outputLayer.eval(feed_dict={Fz_xyz: testbatch['forces'], keep_prob: 1.0}))
        viewfile.create_dataset('solutions', data=testbatch['solutions'])
        viewfile.create_dataset('AtomPosition', data=testbatch['atomPosition'])
        viewfile.create_dataset('fzvals', data=testbatch['forces'])
        viewfile.close()
        
        logfile.write('finished! \n')
        logfile.close()
        print 'Finished!'