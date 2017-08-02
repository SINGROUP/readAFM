import numpy as np
import random
import h5py
from math import exp
import numpy as np
import time

# Si = I, Al = A, Na = D, Cl = L

def atomSignal(evalvect, meanvect, atomNameString, sigmabasexy=1.0, sigmabasez=1.0, amplification=1.0):
    """ This is not a Normal Distribution!!! It's a gauss-like distribution, 
    but we normalize with the relative atom size instead of 1/sqrt(2*pi*sigma**2), 
    this is s.t. the different elements give different 'signals'. 
    Sigmabase is multiplied with the relative covalent ratius, with C corresponding to 1.
    I think it can be interpreted as a length in A.
    """
    covalentRadii = {'H' : 31, 'C' : 76, 'O' : 66, 'N' : 71, 'F' : 57, 'I' : 111, 'B' : 82, 'A' : 121, 'D' : 154, 'P' : 106, 'S' : 105, 'L' : 102}
    sigmabasexy = sigmabasexy * covalentRadii[atomNameString]/76
    sigmabasez = sigmabasez * covalentRadii[atomNameString]/76
    normalisation = amplification*(covalentRadii[atomNameString])/76.
    return normalisation*exp(-((evalvect[0]-meanvect[0])**2+(evalvect[1]-meanvect[1])**2)/sigmabasexy**2)*exp(-((evalvect[2]-meanvect[2])**2)/sigmabasez**2)

class AFMdata:
    """ Class for opening HDF5 file. """
    def __init__(self, FileName, shape=(81, 81, 41, 1)):
        """ Opens hdf5 file FileName for reading. Shape has to contain the Shape of the DB, in the form (x,y,z,inChannels). """
        self.f = h5py.File(FileName, "r+")
        self.shape = tuple(shape)
 

    def solution_xymap_projection(self, datasetString, COMposition=[0.,0.,0.], sigmabasexy=1.0, sigmabasez=1.0, amplificationFactor=1.0):
        """Returns solution to train. Project the atom positions on the xy-plane with Amplitudes decaying like a Gaussian with the radius as variance. and write it on the correct level of the np-array.
        The last index of the array corresponds to the atom type:
        0 = C
        1 = H
        2 = 0
        3 = N
        4 = F
    
        """
    
        atomNameString=self.f[datasetString].attrs['atomNameString']
        atomPosition=self.f[datasetString+'/atomPosition']
        AtomDict = {'C': 0, 'H': 1, 'O': 2, 'N': 3, 'F': 4, 'S': 5, 'I' : 6, 'B' : 7, 'A' : 8, 'D' : 9, 'P' : 10, 'L' : 11}
        #print raw
        projected_array = np.zeros((self.f[datasetString].attrs['divxyz'][0], self.f[datasetString].attrs['divxyz'][1], 12))   # x, y, AtomNumber as in the dict
        masses = {'H' : 1.008, 'C' : 12.011, 'O' : 15.9994, 'N' : 14.0067, 'S' : 32.065, 'F' : 18.9984, 'I' : 28.0855, 'B' : 10.811, 'A' : 26.9815, 'D' : 22.9898, 'P' : 30.9738, 'L' : 35.453}
        covalentRadii = {'H' : 31, 'C' : 76, 'O' : 66, 'N' : 71, 'F' : 57, 'I' : 111, 'B' : 82, 'A' : 121, 'D' : 154, 'P' : 106, 'S' : 105, 'L' : 102}
        # Calculate Center Of Mass:
        COM = np.array(COMposition)
        totalMass=0.0
        for i in range(len(atomNameString)):
            atomVector = atomPosition[i,:]
            COM += atomVector*masses[atomNameString[i]]
            totalMass+=masses[atomNameString[i]]
        COM = COM/totalMass
    
        widthX=self.f[datasetString].attrs['widthxyz'][0]
        widthY=self.f[datasetString].attrs['widthxyz'][1]
        
        max_Zposition = 0.0
        indexOf_max_Zposition_in_atomNameString = 0
    
        for i in range(len(atomNameString)):
            if atomPosition[i, 2] > max_Zposition:
                max_Zposition = atomPosition[i, 2]
                indexOf_max_Zposition_in_atomNameString = i
    
        matrixPositionZIndex = int(round((atomPosition[indexOf_max_Zposition_in_atomNameString,2]-COM[2]+10)/self.f[datasetString].attrs['dxyz'][2]))
    

    def batch_runtimeSolution(self, 
                              batchsize, 
                              outputChannels=1, 
                              method='xymap_collapsed', 
                              COMposition=[0.,0.,0.], 
                              sigmabasexy=1.0,
                              sigmabasez=1.0, 
                              amplificationFactor=1.0, 
                              returnAtomPositions=False,
                              verbose=True, 
                              orientationsOnly=False,
                              rootGroup='/'):
        """ To use if the DB contains no solutions or if one wants to skip the 'add_labels' step. 
        Output channels has to match the method.
        Methods are: xymap_collapsed, xymap_projection, singleAtom
        """
        batch_Fz=np.zeros((batchsize,)+self.shape)   # Maybe I can solve this somehow differently by not hardcoding the dimensions? For now I want to hardcode the dimensions, since the NN is also not flexible concerning them.
        batch_solutions=np.zeros((batchsize,)+self.shape[:-2]+(outputChannels,))
        if returnAtomPositions:
            batch_atomPositions=[]
            
        # make sample keylist here with random.sample and then iterate through it

        randomkeys = []
        if orientationsOnly:
            print(self.f)
            randomkeys = random.sample(list(self.f[rootGroup].keys()), batchsize)
            randomkeys = [self.f[rootGroup][i].name for i in randomkeys]

        else:
            for i in range(batchsize):
                randommolecule=self.f[random.choice(self.f.keys())]  # Choose a random molecule
                randomorientation=randommolecule[random.choice(randommolecule.keys())]   # Choose a random Orientation
                randomkeys.append(randomorientation.name)
            
        for i in range(batchsize):
            
            randomorientation=self.f[randomkeys[i]]
            if verbose:
                print('Looking at file ' + randomorientation.name)

            batch_Fz[i]=randomorientation['fzvals'][...].reshape(self.shape)
            if method=='xymap_collapsed':
                batch_solutions[i]=self.solution_xymap_collapsed(randomorientation.name, COMposition, sigmabasexy, sigmabasez, amplificationFactor)[...]
            elif method=='xymap_projection':
                batch_solutions[i]=self.solution_xymap_projection(randomorientation.name, COMposition, sigmabasexy, sigmabasez, amplificationFactor)[...]
            elif method=='singleAtom':
                batch_solutions[i]=self.solution_singleAtom(randomorientation.name, sigmabasexy, sigmabasez, amplificationFactor)[...]
                
            if returnAtomPositions:
                batch_atomPositions.append([randomorientation.attrs['atomNameString'], randomorientation['atomPosition'][...]])
                
        if returnAtomPositions:
            return {'forces': batch_Fz, 'solutions': batch_solutions, 'atomPosition': batch_atomPositions}
        else:
            return {'forces': batch_Fz, 'solutions': batch_solutions}
    
    def batch(self, batchsize, outputChannels=1, returnAtomPositions=False, verbose=True, orientationsOnly=False):
        """ Returns (training)batches as dictionaries with 'forces' and 'solutions' with arrays of shape
        forces: (batchsize,)+shape
        solutions: (batchsize,)+shape[:-2]+(outputChannels,)"""
        
        batch_Fz=np.zeros((batchsize,)+self.shape)
        batch_solutions=np.zeros((batchsize,)+self.shape[:2]+(outputChannels,))
        
        if returnAtomPositions:
            batch_atomPositions=[]
        
        for i in range(0,batchsize):
            randommolecule=self.f[random.choice(list(self.f.keys()))]  # Choose a random molecule
            if orientationsOnly:
                randomorientation = randommolecule  # There is no molecule-level
            else:
                randomorientation=randommolecule[random.choice(list(randommolecule.keys()))]   # Choose a random Orientation
            if verbose:
                print('Looking at file ' + randomorientation.name)

            batch_Fz[i]=randomorientation['fzvals'][...].reshape(self.shape)
            if outputChannels == 1:
                batch_solutions[i]=np.sum(randomorientation['solution'][...], axis=-1, keepdims=True).reshape((self.shape[:-2]+(outputChannels,)))
            else:
                batch_solutions[i]=randomorientation['solution'][...].reshape((self.shape[:-2]+(outputChannels,)))

            if returnAtomPositions:
                batch_atomPositions.append(randomorientation['atomPosition'][...])

        if returnAtomPositions:
            return {'forces': batch_Fz, 'solutions': batch_solutions, 'atomPosition': map(str,batch_atomPositions)}
        else:
            return {'forces': batch_Fz, 'solutions': batch_solutions}    
        
    def add_labels(self, method='xymap_collapsed', COMposition=[0.,0.,0.], sigmabasexy=1.0, sigmabasez=1.0, amplificationFactor=1.0):
        
        for molstr in self.f.keys():
            timestart=time.time()
            molecule = self.f[molstr]
            print(molstr)
            for ortnstr in molecule.keys():
                orientation=molecule[ortnstr]
                
                if method=='xymap_collapsed':
                    orientation.create_dataset('solution', data=self.solution_xymap_collapsed(orientation.name, COMposition, sigmabasexy, sigmabasez, amplificationFactor)[...])
                elif method=='xymap_projection':
                    orientation.create_dataset('solution', data=self.solution_xymap_projection(orientation.name, COMposition, sigmabasexy, sigmabasez, amplificationFactor)[...])
                elif method=='singleAtom':
                    orientation.create_dataset('solution', data=self.solution_singleAtom(orientation.name, sigmabasexy, sigmabasez, amplificationFactor)[...])

                print(ortnstr, orientation.name)
                
            timeend=time.time()
            print("This molecule took %f seconds to label."%(timeend-timestart))
            
            
    def change_labels(self, method='xymap_collapsed', COMposition=[0.,0.,0.], sigmabasexy=1.0, sigmabasez=1.0, amplificationFactor=1.0):
        """Options for method: 'xymap_collapsed', 'xymap_projection', 'singleAtom'
        """
        for molstr in self.f.keys():
            timestart=time.time()
            molecule = self.f[molstr]
            print(molstr)
            for ortnstr in molecule.keys():
                orientation=molecule[ortnstr]
                
                if method=='xymap_collapsed':
                    orientation['solution'][...]=self.solution_xymap_collapsed(orientation.name, COMposition, sigmabasexy, sigmabasez, amplificationFactor)[...]
                elif method=='xymap_projection':
                    orientation['solution'][...]=self.solution_xymap_projection(orientation.name, COMposition, sigmabasexy, sigmabasez, amplificationFactor)[...]
                elif method=='singleAtom':
                    orientation['solution'][...]=self.solution_singleAtom(orientation.name, sigmabasexy, sigmabasez, amplificationFactor)[...]

                print(ortnstr, orientation.name)
                
            timeend=time.time()
            print("This molecule took %f seconds to relabel."%(timeend-timestart))
                

        
if __name__=='__main__':
    print('Hallo Main')
    datafile = AFMdata('/l/reischt1/toyDB_v15_merged.hdf5', shape=(41,41,41,1))
#     print(datafile.solution_xymap_collapsed('molecule1/orientation1'))
    testbatch = datafile.batch_runtimeSolution(20, orientationsOnly=True, rootGroup='/train', returnAtomPositions=True)
    print testbatch['atomPosition']
