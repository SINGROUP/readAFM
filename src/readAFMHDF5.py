import numpy as np
import random
import h5py

class AFMdata:
    """ Class for opening HDF5 file. """
    def __init__(self, FileName):
        """ Opens hdf5 file for reading. """
        self.f = h5py.File(FileName, "r")

    def batch(self, batchsize):
        batch_Fz=np.zeros((batchsize,81,81,41,1))   # Maybe I can solve this somehow differently by not hardcoding the dimensions? For now I want to hardcode the dimensions, since the NN is also not flexible concerning them.
        batch_solutions=np.zeros((batchsize,81,81,5))
        for i in range(0,batchsize):
            randommolecule=self.f[random.choice(self.f.keys())]  # Choose a random molecule
            randomorientation=randommolecule[random.choice(randommolecule.keys())]   # Choose a random Orientation
            print 'Looking at file ' + randomorientation.name

            batch_Fz[i]=randomorientation['fzvals'][...]
            batch_solutions[i]=randomorientation['solution'][...]
        return {'forces': batch_Fz, 'solutions': batch_solutions}



if __name__=='__main__':
    print 'Hallo Main'
    datafile = AFMdata('AFMDB_version_01.hdf5')
    
