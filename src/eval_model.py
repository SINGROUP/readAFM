'''
Created on Aug 3, 2017

@author: reischt1
'''

import readAFMHDF5 as readAFM
import time
from utils import * 
import tensorflow as tf

def eval_model(model_function, Fz_xyz, solution, keep_prob, posxyz, parameters, logfile):
    LOGDIR = parameters['logdir']       
    
    # Define model:
    outputLayer = model_function(Fz_xyz, keep_prob, parameters, logfile)
    
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
        writer = tf.summary.FileWriter(LOGDIR)
        writer.add_graph(sess.graph)
        
        if parameters['useRuntimeSolution']:
            testbatch = AFMdata.batch_runtimeSolution(parameters['testbatchSize'], 
                                                      outputChannels=parameters['outChannels'], 
                                                      method=parameters['RuntimeSol.method'],
                                                      COMposition=parameters['RuntimeSol.COMposition'],
                                                      sigmabasexy=parameters['RuntimeSol.sigmabasexy'],
                                                      sigmabasez=parameters['RuntimeSol.sigmabasez'],
                                                      amplificationFactor=parameters['RuntimeSol.amplificationFactor'],
                                                      returnAtomPositions=True,
                                                      orientationsOnly=False,
                                                      rootGroup='/')            
        else:
            testbatch = AFMdata.batch(parameters['testbatchSize'], outputChannels=parameters['outChannels'], returnAtomPositions=True)
        

        [testaccuracy, s] = sess.run([accuracy, summ],feed_dict={Fz_xyz: testbatch['forces'], solution: testbatch['solutions'], keep_prob: 1.0, posxyz: [map(str, bla) for bla in testbatch['atomPosition']]})     

        logfile.write("test accuracy %g \n"%testaccuracy)
        writer.add_summary(s)
        
        # Save two np.arrays to be able to view it later.
        # make_viewfile(parameters, testaccuracy, outputLayer.eval(feed_dict={Fz_xyz: testbatch['forces'], keep_prob: 1.0}), testbatch['solutions'], testbatch['atomPosition'])
        
    return 0
