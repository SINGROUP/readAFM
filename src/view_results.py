# commands to execute with matplotlib turned on

# make this a class or function after resolving the matplotlib issues


import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py

def make_plot(array):
    fig = plt.figure(figsize=(6, 3.2))

    ax = fig.add_subplot(111)
    ax.set_title('Color map')
    plt.imshow(array)
    ax.set_aspect('equal')

    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')
    plt.show()

def make_multiple(array, x=1, y=1, interval=1):
    fig = plt.figure(figsize=(6, 3.2))
  
    for i in range(x*y):
        subplotnumber = 100*x + 10*y + i + 1
        
        ax = fig.add_subplot(subplotnumber)
        ax.set_title('Color map')
        plt.imshow(array[:,:,i*interval])
        plt.colorbar(orientation='vertical')
    ax.set_aspect('equal')
    
#     cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
#     cax.get_xaxis().set_visible(False)
#     cax.get_yaxis().set_visible(False)
#     cax.patch.set_alpha(0)
#     cax.set_frame_on(False)
#     plt.colorbar(orientation='vertical')
    plt.show()

def view_hdf5(filename, name='fzvals'):
    f = h5py.File(filename, 'r')
    print 'print geht'
    for molecule in f.keys():
        for orientation in f[molecule]:
            if name in f[molecule+'/'+orientation].keys():
#                 print f[molecule+'/'+orientation+'/'+name].shape
                make_plot(f[molecule+'/'+orientation+'/'+name][:,:,0])

def view_2(array1, array2):
    fig = plt.figure(figsize=(6, 3.2))

    ax = fig.add_subplot(121)
    plt.imshow(array1)
    ax.set_aspect('equal')
    plt.colorbar(orientation='vertical')
    
    ax2 = fig.add_subplot(122)
    plt.imshow(array2)
    ax2.set_aspect('equal')
    plt.colorbar(orientation='vertical')

    plt.show()

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

if __name__ == '__main__':
#     view_hdf5('/m/home/home5/58/reischt1/unix/Downloads/db.hdf5')
#     view_hdf5('../AFMDB_version_01.hdf5')
#     f = h5py.File('../AFMDB_version_01.hdf5', 'r')
#     print(f.keys())
#     f = h5py.File('/m/home/home5/58/reischt1/unix/Downloads/db.hdf5','r')
#     print f['molecule000038/orientation1/solution'].shape, f['molecule000038/orientation1/fzvals'].shape
#     view_2(f['molecule000719/orientation1/solution'][:,:,0], f['molecule000719/orientation1/fzvals'][:,:,0,0])
#     make_plot(f['molecule1/orientation1/fzvals'][:,:,40]-f['molecule1600/orientation1/fzvals'][:,:,40])
#     make_plot(f['molecule58/orientation1/fzvals'][10,:,:])
#     print(np.amax(f['molecule58/orientation1/fzvals'][:,:,20]),np.argmax(f['molecule58/orientation1/fzvals'][:,:,20]))
#     make_plot(f['molecule100/orientation1/fzvals'][:,:,0])
#     make_plot(f['molecule1001/orientation1/fzvals'][:,:,0])
#     make_plot(f['molecule1600/orientation1/fzvals'][:,:,0])
#     make_plot(f['molecule25/orientation1/fzvals'][:,:,0])
#     make_plot(f['molecule600/orientation1/fzvals'][:,:,0])
#     make_plot(f['molecule980/orientation1/fzvals'][:,:,0])
    f=h5py.File('/l/reischt1/toyDB_v06.hdf5', 'r')
    print f.keys()
    view_2(f['molecule100/orientation1/fzvals'][:,:,40], f['molecule100/orientation1/solution'][:,:,0])