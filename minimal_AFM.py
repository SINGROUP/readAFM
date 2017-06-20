import tensorflow as tf
import readAFMDATAfile as readAFM
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("name")
args=parser.parse_args()

print('Parsed name argument {} of type {}.'.format(args.name, type(args.name)))

def weight_variable(shape):
  """ Initializes the variable with random numbers (not 0) """
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  """ Initializes the bias variable with 0.1 everywhere """
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv3d(x, W):
  """ Short definition of the convolution function for 3d """
  return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')


# Import data -- TO DO!!! For this make a class with training and validation data, this is still just a dummy!!! Maybe ask Jay to program this?


if __name__=='__main__':

  logfile=open('out_minimal_{}.log'.format(args.name), 'w', 0)
  logfile.write('Start with importing the data \n')
  # AFMdata=readAFM.afmmolecule('dsgdb9nsd_000001.afmdata')
  inputData=np.array((None,81,81,41,1))
  solutionData = np.array((None,81,81,5))

  logfile.write('define the first two placeholders \n')
  # Shape x: batch, x, y, z, Fz
  Fz_xyz = tf.placeholder(tf.float32, [None, 81, 81, 41, 1])
  # Standard output var -> one hot 5d (or so) vector for the atoms e.g. 0=O, 1=C, 2=F, ..., probably wrong dimensions!!! eithern [None, 5] for every pixel or [None, 41, 41, 5]
  solution = tf.placeholder(tf.float32, [None, 81, 81, 5])


  logfile.write('now define conv1 \n')
  #1st conv layer: Convolve the input (Fz_xyz) with 16 different filters, don't do maxpooling!
  W_conv1 = weight_variable([4,4,4,1,16])
  b_conv1 = bias_variable([16])

  h_conv1 = tf.nn.relu(tf.add(conv3d(Fz_xyz, W_conv1),b_conv1))
  #  h_pool1 = max_pool_2x2(h_conv1)


  logfile.write('defining conv2 \n')
  #2nd conv layer: Convolve the result from layer 1 with 32 different filters, don't do maxpooling!
  W_conv2 = weight_variable([4, 4, 4, 16, 32])
  b_conv2 = bias_variable([32])

  h_conv2 = tf.nn.relu(tf.add(conv3d(h_conv1, W_conv2),b_conv2))
  # h_pool2 = max_pool_2x2(h_conv2)

  logfile.write('all conv layers defined, defining fc layer \n')

  #fully connected layer -- das muss anders aussehen
  W_fc1 = weight_variable([41*32, 64])
  b_fc1 = bias_variable([81, 81, 64])

  h_conv2_flat = tf.reshape(h_conv2, [-1, 81, 81, 41*32])
  h_fc1 = tf.nn.relu(tf.tensordot(h_conv2_flat, W_fc1,axes=[[3],[0]])+b_fc1)  #Is this correct? the result of the tensordot and the b_fc1 don't have the same dimensions

  # Dropout
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
 
  # Readout Layer
  W_fc2 = weight_variable([64,5])
  b_fc2 = bias_variable([81,81,5])

  y_conv = tf.add(tf.tensordot(h_fc1_drop, W_fc2, axes=[[3],[0]]), b_fc2)  # May be wrong? the result of the tensordot and the b_fc1 don't have the same dimensions

  # set up evaluation system
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=solution, logits=y_conv))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv,2), tf.argmax(solution,2))  # Das stimmt so aber eh nicht!!!
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # Crate saver
  saver = tf.train.Saver()
  
  # Init op
  init_op = tf.global_variables_initializer()

  # Start session
  logfile.write('it worked so far, now start session \n')
  sess = tf.InteractiveSession()

  sess.run(init_op)

  print("b_conv1, as initialized: ")
  print(sess.run(b_conv1))

  saver.restore(sess, "/scratch/work/reischt1/calculations/minimal_06_implement_start_from_checkpoint/save/CNN_minimal_TR1.ckpt")
  logfile.write("Model restored. \n")
  print("Model restored. See here b_conv1 restored:")
  print(sess.run(b_conv1))
  # logfile.write('Variables initialized successfully \n')




  # Do stochastic training:
  for i in range(2000):
    try:
      logfile.write('Starting Run #%i \n'%(i))
      timestart=time.time()
      batch = readAFM.AFMdata('/tmp/reischt1/outputxyz').batch(50)
      logfile.write('read batch successfully \n')

      if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={Fz_xyz:batch[0], solution: batch[1], keep_prob: 1.0})
        logfile.write("step %d, training accuracy %g \n"%(i, train_accuracy))
        save_path=saver.save(sess, "/scratch/work/reischt1/calculations/minimal_06_implement_start_from_checkpoint/save/CNN_minimal_TR1_{}.ckpt".format(args.name))
        logfile.write("Model saved in file: %s \n" % save_path)

      train_step.run(feed_dict={Fz_xyz: batch[0], solution: batch[1], keep_prob: 0.5})
      timeend=time.time()
      logfile.write('ran train step in %f seconds \n' % (timeend-timestart))
    except IndexError:
      print 'Index Error for this File'
  
  testbatch = readAFM.AFMdata('/tmp/reischt1/outputxyz').batch(50)
  logfile.write("test accuracy %g \n"%accuracy.eval(feed_dict={Fz_xyz: testbatch[0], solution: testbatch[1], keep_prob: 1.0}))


  logfile.write('finished! \n')
  logfile.close()
  print 'Finished!'
                















