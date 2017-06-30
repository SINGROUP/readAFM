# commands to execute with matplotlib turned on

# make this a class or function after resolving the matplotlib issues


import numpy as np
import matplotlib.pyplot as plt
import glob
import __main__


batchsize = 50                                 # set manually!

def compare(solfile='default', predfile='default', atom = 0, batch = 0):
    """ Compare the solution (in solfile) with the predicted xy-map. Batchindex and atomidex specify what part of the solution should be viewed. """

    if solfile == 'default':
        solfile  = glob.glob('./view_solution_*.npy')[0]
    if predfile == 'default':
        predfile = glob.glob('./view_prediction_*.npy')[0]

    print('Opening solution from: %s' % solfile)
    view_sol=open(solfile,'r')
    print('Opening solution from: %s ' % predfile)
    view_pred=open(predfile,'r')

    solution = np.load(view_sol)
    prediction = np.load(view_pred)

    fig = plt.figure(figsize=(6, 3.2))

    ax = fig.add_subplot(111)
    ax.set_title('Color map of prediction - solution')
    plt.imshow(prediction[batch,:,:,atom]-solution[batch,:,:,atom])
    ax.set_aspect('equal')

    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')
    plt.show()

def view(infile, atom = 0, batch = 0):
    """ View a map given in 'infile'. Batchindex and atomidex specify what part of the solution should be viewed. """

    print('Opening solution from: %s' % infile)
    view_in=open(infile,'r')

    viewmap = np.load(view_in)

    fig = plt.figure(figsize=(6, 3.2))

    ax = fig.add_subplot(111)
    ax.set_title('Color map of {}'.format(infile))
    plt.imshow(viewmap[batch,:,:,atom])
    ax.set_aspect('equal')

    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')
    plt.show()
    
def multiple(infile, x = 1, y = 1, batch = 0, interval=1):
    """ View a map given in 'infile'. Batchindex and atomidex specify what part of the solution should be viewed. """

    print('Opening solution from: %s' % infile)
    view_in=open(infile,'r')

    viewmap = np.load(view_in)

    fig = plt.figure(figsize=(6, 3.2))
  
    for i in range(x*y):
        subplotnumber = 100*x + 10*y + i + 1
        
        ax = fig.add_subplot(subplotnumber)
        ax.set_title('Color map of {}'.format(infile))
        plt.imshow(viewmap[batch,:,:,i*interval])
        plt.colorbar(orientation='vertical')
    ax.set_aspect('equal')
    
#     cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
#     cax.get_xaxis().set_visible(False)
#     cax.get_yaxis().set_visible(False)
#     cax.patch.set_alpha(0)
#     cax.set_frame_on(False)
#     plt.colorbar(orientation='vertical')
    plt.show()

if __name__ == __main__:
    view('./testsolution.npy')