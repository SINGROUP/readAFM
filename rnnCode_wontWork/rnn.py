from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import argparse
import readAFMDATAfile as readAFM

'''
To find likelihood of finding atoms at certain points from MechAFM grid using recurrent neural networks
'''

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


# Training Parameters
learning_rate = 0.0001
#training_iters = 100000
batch_size = 2
#display_step = 10

# Network Parameters
input_size = 32 # This is the number of output channels after two convolutions
step_size = 41 # timesteps. Equal to the number of points in the Z direction
hidden_size = 50 # hidden layer num of features
num_classes = 5 # Each point in XY will only generate one value for each of the 5 types of atoms. This is compared with the value at the same point in the solution matrix


logfile=open('out_minimal_{}.log'.format(args.name), 'w', 0)

## Input to the graph
#logfile.write("define the input and expected output placeholders\n")
#x = tf.placeholder("float", [None, step_size, input_size])
#y = tf.placeholder("float", [None, num_classes])

logfile.write('define the first two placeholders \n')
# Shape x: batch, x, y, z, Fz
Fz_xyz = tf.placeholder(tf.float32, [None, 81, 81, 41, 1])
# Standard output var -> one hot 5d (or so) vector for the atoms e.g. 0=O, 1=C, 2=F, ..., probably wrong dimensions!!! eithern [None,
solution = tf.placeholder(tf.float32, [None, 81, 81, 5])


# Define weights and biases for the neuron after evaluation from LSTM. Do not confuse with the weights and biases used for the 2 conv layers
logfile.write("define the weights and biases for the result of recursive function \n")
weights = {
    'out': tf.Variable(tf.random_normal([hidden_size, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}


logfile.write('define conv1 \n')
#1st conv layer: Convolve the input (Fz_xyz) with 16 different filters, don't do maxpooling!
W_conv1 = weight_variable([10, 10, 10, 1, 16])
b_conv1 = bias_variable([16])

h_conv1 = tf.nn.relu(tf.add(conv3d(Fz_xyz, W_conv1),b_conv1))
#  h_pool1 = max_pool_2x2(h_conv1)



logfile.write('define conv2 \n')
#2nd conv layer: Convolve the result from layer 1 with 32 different filters, don't do maxpooling!
W_conv2 = weight_variable([10, 10, 10, 16, 32])
b_conv2 = bias_variable([32])

h_conv2 = tf.nn.relu(tf.add(conv3d(h_conv1, W_conv2),b_conv2))
# h_pool2 = max_pool_2x2(h_conv2)

#Define LSTM cell to be used
lstm_cell = rnn.BasicLSTMCell(hidden_size, forget_bias=1.0)

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, step_size, input_size)
    # Required shape: 'step_size' tensors list of shape (batch_size, input_size)

    # Unstack to get a list of 'step_size' tensors of shape (batch_size, input_size)
    x = tf.unstack(x, step_size, 1)

    # Define a lstm cell with tensorflow
    #lstm_cell = rnn.BasicLSTMCell(hidden_size, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

#x = tf.placeholder(tf.float32, [None, step_size, input_size])
#y = tf.placeholder(tf.float32, [None, num_classes])

#accuracy_individual = tf.Variable(tf.zeros(81*81), dtype=tf.float32)
#accuracy_individual = []
cost = 0
#k = 0

print("Definitions are done. Now going to the loop")

for i in range(81):
    for j in range(81):
        print("Still stuck in the loop. Might take a while. Run number " + str(i) + ", " + str(j))
        x = h_conv2[i, j, :, :]
        y = solution[i, j, :]
        pred = RNN(x, weights, biases)
        #Defining loss as a simple difference operation between predicted and expected output
        difference = tf.abs(tf.subtract(pred, y))
        cost += tf.reduce_mean(difference)
        #accuracy_individual.append(tf.reduce_mean(np.divide(difference, y)))
        #k += 1


## Define loss and optimizer
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

# Define the optimizer being used
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
# correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
#accuracy = tf.reduce_mean(accuracy_individual, dtype=np.float32)

# Initializing the variables
init = tf.global_variables_initializer()

logfile.write('Moving on to the session run, now')

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    batch = readAFM.AFMdata("$WRKDIR/varmad1/outputxyz").batch(batch_size)
    logfile.write('read batch success')
    sess.run(optimizer, feed_dict={Fz_xyz: batch[0], solution: batch[1]})
    # Calculate batch accuracy
    #acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
    # Calculate batch loss
    loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
    print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "blah")
    print("Optimisation finished")

#with tf.Session() as sess:
#    sess.run(init)
#    step = 1
#    # Keep training until reach max iterations
#    while step * batch_size < training_iters:
#        batch_x, batch_y = mnist.train.next_batch(batch_size)
#        # Reshape data to get 28 seq of 28 elements
#        batch_x = batch_x.reshape((batch_size, step_size, input_size))
#        # Run optimization op (backprop)
#        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
#        if step % display_step == 0:
#            # Calculate batch accuracy
#            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
#            # Calculate batch loss
#            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
#            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
#                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
#                  "{:.5f}".format(acc))
#        step += 1
#    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    #test_len = 128
    #test_data = mnist.test.images[:test_len].reshape((-1, step_size, input_size))
    #test_label = mnist.test.labels[:test_len]
    #print("Testing Accuracy:", \
    #    sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

