import tensorflow as tf
# import readAFMDATAfile as readAFM
import readAFMHDF5 as readAFM
import numpy as np
import time
import argparse
import h5py
import utils


parser = argparse.ArgumentParser()
parser.add_argument("name", help="Name given to all output files.")
parser.add_argument("-i", "--input_file", default="parameters.in", help="the path to the input file (default: %(default)s)")
# parser.add_argument("-o", "--output", default="output/", help="produced files and folders will be saved here (default: %(default)s)")
args=parser.parse_args()

# parser.add_argument("-v", "--verbosity", action="count", default=0)

print('Parsed name argument {} of type {}.'.format(args.name, type(args.name)))
print('Parsing parameters from {}'.format(args.input_file))

parsedParameters = utils.parseInputFile(args.input_file)

# These are the default Parameters!
parameters = {'train': True,
              'restorePath': None,                                                # Typically: "./save/CNN_minimal_TR1_{}.ckpt"
              'savePath': None,         # Typically: 'savePath': "./save/CNN_minimal_TR1_{}.ckpt".format(args.name)
#               'savePath': "../save{}/CNN_minimal_TR1.ckpt".format(args.name),         # Typically: 'savePath': "./save/CNN_minimal_TR1_{}.ckpt".format(args.name)
              'DBPath': '../AFMDB_version_01.hdf5',
              'DBShape': [81,81,41,1],
              'outChannels': 1,
              'viewPath': './viewfile_{}.hdf5'.format(args.name),
              'logPath': './out_minimal_{}.log'.format(args.name),
              'trainstepsNumber': 1,
              'trainbatchSize':1,
              'testbatchSize': 1,
              'logdir': './save{}/'.format(args.name),
              'infoString': 'minimalAFM_{}'.format(args.name),
              'LearningRate': 0.001,
              'useRuntimeSolution': False,
              'RuntimeSol.method': 'xymap_collapsed', 
              'RuntimeSol.COMposition': [0.,0.,0.], 
              'RuntimeSol.sigmabase': 1.0, 
              'RuntimeSol.amplificationFactor': 1.0
              }

parameters.update(parsedParameters)

LOGDIR = parameters['logdir']
DBShape = parameters['DBShape']
outChannels = parameters['outChannels']

# Output has to have the same xy dimensions as input (DBShape)



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

def make_viewfile(parameters, testaccuracy, predictions, labels, atomPosition):
    viewfile = h5py.File(parameters['viewPath'], 'w')
    viewfile.attrs['testaccuracy']=testaccuracy
    viewfile.attrs['infoString']=parameters['infoString']
    viewfile.create_dataset('predictions', data=predictions)
    viewfile.create_dataset('solutions', data=labels)
    viewfile.create_dataset('AtomPosition', data=atomPosition)
    viewfile.close()

def define_model(Fz_xyz, logfile):

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
        tf.summary.image('predictions', outputLayer, 5)

    return outputLayer

def train_model(Fz_xyz, solution, keep_prob, logfile):
    # Define model:
    outputLayer = define_model(Fz_xyz, logfile)
    
#     set up evaluation system
#     cost = tf.reduce_mean(tf.abs(tf.subtract(prediction, solution)))
    with tf.name_scope('cost'):
        cost = tf.reduce_sum(tf.square(tf.subtract(outputLayer, solution)))/float(parameters['trainbatchSize'])
        tf.summary.scalar('cost', cost)

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_sum(tf.square(tf.subtract(outputLayer, solution)))/float(parameters['testbatchSize'])
        tf.summary.scalar('accuracy',accuracy)
#     accuracy = tf.reduce_mean(tf.cast(tf.abs(tf.subtract(prediction, solution)), tf.float32))
    
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(parameters['LearningRate']).minimize(cost)
    
    # Crate saver
    saver = tf.train.Saver()
    
    # Init op
    init_op = tf.global_variables_initializer()
    
    # Start session
    logfile.write('it worked so far, now start session \n')

    summ = tf.summary.merge_all()

    with tf.Session() as sess:
        init_op.run()
    
#         print("b_conv1, as initialized: ")
#         print(sess.run(b_conv1))
                
#         Pack this into a function!
        if parameters['restorePath']:
            saver.restore(sess, parameters['restorePath'])
            logfile.write("Model restored. \n")
            print("Model restored. See here b_conv1 restored:")
#             print(sess.run(b_conv1))
            logfile.write('Variables initialized successfully \n')
        
        AFMdata = readAFM.AFMdata(parameters['DBPath'], shape=parameters['DBShape'])
    #     AFMdata = readAFM.AFMdata('/tmp/reischt1/AFMDB_version_01.hdf5')
        writer = tf.summary.FileWriter(LOGDIR+parameters['infoString'])
        writer.add_graph(sess.graph)        
        
        # Do stochastic training:
        for i in range(parameters['trainstepsNumber']):
            logfile.write('Starting Run #%i \n'%(i))
            timestart=time.time()
            if parameters['useRuntimeSolution']:
                batch = AFMdata.batch_runtimeSolution(parameters['trainbatchSize'], 
                                                      outputChannels=parameters['outChannels'], 
                                                      method=parameters['RuntimeSol.method'],
                                                      COMposition=parameters['RuntimeSol.COMposition'],
                                                      sigmabase=parameters['RuntimeSol.sigmabase'],
                                                      amplificationFactor=parameters['RuntimeSol.amplificationFactor'])            
            else:
                batch = AFMdata.batch(parameters['trainbatchSize'], outputChannels=parameters['outChannels'])

                
            logfile.write('read batch successfully \n')
    
            if i%100 == 0:
                [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={Fz_xyz:batch['forces'], solution: batch['solutions'], keep_prob: 1.0})
                logfile.write("step %d, training accuracy %g \n"%(i, train_accuracy))
                writer.add_summary(s, i)
                if parameters['savePath']:
                    save_path=saver.save(sess, parameters['savePath'], i)
                    logfile.write("Model saved in file: %s \n" % save_path)

            train_step.run(feed_dict={Fz_xyz: batch['forces'], solution: batch['solutions'], keep_prob: 0.6})
            timeend=time.time()
            logfile.write('ran train step in %f seconds \n' % (timeend-timestart))

        
        
        if parameters['useRuntimeSolution']:
            testbatch = AFMdata.batch_runtimeSolution(parameters['trainbatchSize'], 
                                                      outputChannels=parameters['outChannels'], 
                                                      method=parameters['RuntimeSol.method'],
                                                      COMposition=parameters['RuntimeSol.COMposition'],
                                                      sigmabase=parameters['RuntimeSol.sigmabase'],
                                                      amplificationFactor=parameters['RuntimeSol.amplificationFactor'],
                                                      returnAtomPositions=True)            
        else:
            testbatch = AFMdata.batch(parameters['testbatchSize'], outputChannels=parameters['outChannels'], returnAtomPositions=True)
        
        
        testaccuracy=accuracy.eval(feed_dict={Fz_xyz: testbatch['forces'], solution: testbatch['solutions'], keep_prob: 1.0})
        logfile.write("test accuracy %g \n"%testaccuracy)
        
        make_viewfile(parameters, testaccuracy, outputLayer.eval(feed_dict={Fz_xyz: testbatch['forces'], keep_prob: 1.0}), testbatch['solutions'], testbatch['atomPosition'])
        
    return 0

def eval_model(Fz_xyz, solution, keep_prob, logfile):
    # Define model:
    outputLayer = define_model(Fz_xyz, logfile)
    
#     set up evaluation system
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_sum(tf.square(tf.subtract(outputLayer, solution)))/float(parameters['testbatchSize'])
        tf.summary.scalar('accuracy',accuracy)
#     accuracy = tf.reduce_mean(tf.cast(tf.abs(tf.subtract(prediction, solution)), tf.float32))

    # Crate saver
    saver = tf.train.Saver()
    
    # Init op
    init_op = tf.global_variables_initializer()
    
    # Start session
    logfile.write('it worked so far, now start session \n')

    summ = tf.summary.merge_all()

    with tf.Session() as sess:
        init_op.run()
    
        if parameters['restorePath']:
            saver.restore(sess, parameters['restorePath'])
            logfile.write("Model restored. \n")
            print("Model restored.")
            logfile.write('Variables initialized successfully \n')
        
        AFMdata = readAFM.AFMdata(parameters['DBPath'], shape=parameters['DBShape'])
    #     AFMdata = readAFM.AFMdata('/tmp/reischt1/AFMDB_version_01.hdf5')
        writer = tf.summary.FileWriter(LOGDIR+parameters['infoString'])
        writer.add_graph(sess.graph)
        
        if parameters['useRuntimeSolution']:
            testbatch = AFMdata.batch_runtimeSolution(parameters['trainbatchSize'], 
                                                      outputChannels=parameters['outChannels'], 
                                                      method=parameters['RuntimeSol.method'],
                                                      COMposition=parameters['RuntimeSol.COMposition'],
                                                      sigmabase=parameters['RuntimeSol.sigmabase'],
                                                      amplificationFactor=parameters['RuntimeSol.amplificationFactor'],
                                                      returnAtomPositions=True)            
        else:
            testbatch = AFMdata.batch(parameters['testbatchSize'], outputChannels=parameters['outChannels'], returnAtomPositions=True)
        
        [testaccuracy, s] = sess.run([accuracy, summ],feed_dict={Fz_xyz: testbatch['forces'], solution: testbatch['solutions'], keep_prob: 1.0})     

        logfile.write("test accuracy %g \n"%testaccuracy)
        writer.add_summary(s)
        
        # Save two np.arrays to be able to view it later.
        # make_viewfile(parameters, testaccuracy, outputLayer.eval(feed_dict={Fz_xyz: testbatch['forces'], keep_prob: 1.0}), testbatch['solutions'], testbatch['atomPosition'])
        
    return 0



if __name__=='__main__':

    logfile=open(parameters['logPath'], 'w', 0)

    logfile.write('define the  placeholders \n')
    Fz_xyz = tf.placeholder(tf.float32, [None,]+DBShape)
    tf.summary.image('Fzinput_0', Fz_xyz[:,:,:,0,:], 5)
    tf.summary.image('Fzinput_half', Fz_xyz[:,:,:,int(Fz_xyz.shape[2]/2),:], 5)
    tf.summary.image('Fzinput_last', Fz_xyz[:,:,:,-1,:], 5)
    print(int(Fz_xyz.shape[2]/2))
    
    solution = tf.placeholder(tf.float32, [None,]+DBShape[:2]+[outChannels])
    tf.summary.image('solutions', solution, 5)

    keep_prob = tf.placeholder(tf.float32)
    
    print(parameters['train'])

    if parameters['train']:
        print('Start training:')
        train_model(Fz_xyz, solution, keep_prob, logfile)
    else:
        print('Start evaluation:')
        eval_model(Fz_xyz, solution, keep_prob, logfile)

        
    logfile.write('finished! \n')
    logfile.close()
    print 'Finished!'
