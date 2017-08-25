'''
Created on Aug 3, 2017

@author: reischt1
'''

import tensorflow as tf
import readAFMHDF5 as readAFM
import time
from utils import *


def train_model(model_function, Fz_xyz, solution, keep_prob, posxyz, parameters, logfile):
    """Takes model function, and trains it.
    
    What kind of database, solutions, etc. to use can be specified in the parameters-dictionary.
    For some reason tensorflow wants the placeholders to be defined on the topmost level, so they need to be passed to this function, although they will be filled and used only within this function.
    
    Args:
        model_function: Function that defines the model, see model.py, should be model_function(Fz_xyz, keep_prob, parameters, logfile), returns tensor outputlayer (batchsize, xdim, ydim, outChannels)
        Fz_xyz: Placeholder for the force values
        solution: placeholder for the solutions
        keep_prob: placeholder for the keep probability of the dropout layer
        posxyz: placeholder for the xyz positions, to be stored as text for tensorboard
        parameters: dict containing the parameters
        logfile: handle for the logfile
        
    Returns:
        If finished without error =0
    
    """

    LOGDIR = parameters['logdir']       
    # Define model:
    outputLayer = model_function(Fz_xyz, keep_prob, parameters, logfile)
    
#     set up evaluation system
    with tf.name_scope('cost'):
        # cost = tf.reduce_sum(tf.square(tf.subtract(outputLayer, solution)))/float(parameters['trainbatchSize']) 
        cost = (1.- parameters['costWeight'])*(1./parameters['RuntimeSol.amplificationFactor'])*tf.reduce_sum(tf.multiply(tf.square(tf.subtract(outputLayer, solution)), solution))/float(parameters['trainbatchSize']) + parameters['costWeight']*tf.reduce_sum(tf.square(tf.subtract(outputLayer, solution)))/float(parameters['trainbatchSize'])
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
        writer = tf.summary.FileWriter(LOGDIR)
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
                                                      sigmabasexy=parameters['RuntimeSol.sigmabasexy'],
                                                      sigmabasez=parameters['RuntimeSol.sigmabasez'],
                                                      amplificationFactor=parameters['RuntimeSol.amplificationFactor'],
                                                      orientationsOnly=True,
                                                      rootGroup='/train/')            
            else:
                batch = AFMdata.batch(parameters['trainbatchSize'], outputChannels=parameters['outChannels'])

                
            logfile.write('read batch successfully \n')
    
            if i%parameters['logEvery'] == 0:
                testbatch = AFMdata.batch_runtimeSolution(parameters['testbatchSize'], 
                                      outputChannels=parameters['outChannels'], 
                                      method=parameters['RuntimeSol.method'],
                                      COMposition=parameters['RuntimeSol.COMposition'],
                                      sigmabasexy=parameters['RuntimeSol.sigmabasexy'],
                                      sigmabasez=parameters['RuntimeSol.sigmabasez'],
                                      amplificationFactor=parameters['RuntimeSol.amplificationFactor'],
                                      orientationsOnly=True,
                                      rootGroup='/validation/',
                                      returnAtomPositions=True)   
                [train_accuracy, s] = sess.run([accuracy, summ], 
                                               feed_dict={Fz_xyz:testbatch['forces'], 
                                                          solution: testbatch['solutions'], 
                                                          keep_prob: 1.0, 
                                                          posxyz: [map(str, bla) for bla in testbatch['atomPosition']]})
                logfile.write("step %d, training accuracy %g \n"%(i, train_accuracy))
                writer.add_summary(s, i)
            if i%parameters['saveEvery']==0 and parameters['saveName']:
                save_path=saver.save(sess, LOGDIR+parameters['saveName'], i)
                logfile.write("Model saved in file: %s \n" % save_path)

            train_step.run(feed_dict={Fz_xyz: batch['forces'], solution: batch['solutions'], keep_prob: 0.6})
            timeend=time.time()
            logfile.write('ran train step in %f seconds \n' % (timeend-timestart))

        
        
        if parameters['useRuntimeSolution']:
            testbatch = AFMdata.batch_runtimeSolution(parameters['testbatchSize'], 
                                                      outputChannels=parameters['outChannels'], 
                                                      method=parameters['RuntimeSol.method'],
                                                      COMposition=parameters['RuntimeSol.COMposition'],
                                                      sigmabasexy=parameters['RuntimeSol.sigmabasexy'],
                                                      sigmabasez=parameters['RuntimeSol.sigmabasez'],
                                                      amplificationFactor=parameters['RuntimeSol.amplificationFactor'],
                                                      orientationsOnly=True,
                                                      rootGroup='/validation/',
                                                      returnAtomPositions=True)            
        else:
            testbatch = AFMdata.batch(parameters['testbatchSize'], outputChannels=parameters['outChannels'], returnAtomPositions=True)
        
        
        [testaccuracy, s] = sess.run([accuracy, summ], feed_dict={Fz_xyz: testbatch['forces'], solution: testbatch['solutions'], keep_prob: 1.0, posxyz: [map(str, bla) for bla in testbatch['atomPosition']]})
        logfile.write("test accuracy %g \n"%testaccuracy)
        
        make_viewfile(parameters, testaccuracy, outputLayer.eval(feed_dict={Fz_xyz: testbatch['forces'], keep_prob: 1.0}), testbatch['solutions'], testbatch['atomPosition'])
        
    return 0