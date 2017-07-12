import numpy as np
import random
import h5py
from math import exp
from numpy import zeros, shape


class AFMdata:
    """ Class for opening HDF5 file. """
    def __init__(self, FileName, shape=(81, 81, 41, 1)):
        """ Opens hdf5 file FileName for reading. Shape has to contain the Shape of the DB, in the form (x,y,z,inChannels). """
        self.f = h5py.File(FileName, "r+")
        self.shape = tuple(shape)

    def batch(self, batchsize, outputChannels=1):
        """ Returns (training)batches as dictionaries with 'forces' and 'solutions' with arrays of shape
        forces: (batchsize,)+shape
        solutions: (batchsize,)+shape[:-2]+(outputChannels,)"""
        
        batch_Fz=np.zeros((batchsize,)+self.shape)
        batch_solutions=np.zeros((batchsize,)+self.shape[:2]+(outputChannels,))
        
        for i in range(0,batchsize):
            randommolecule=self.f[random.choice(self.f.keys())]  # Choose a random molecule
            randomorientation=randommolecule[random.choice(randommolecule.keys())]   # Choose a random Orientation
            print 'Looking at file ' + randomorientation.name

            batch_Fz[i]=randomorientation['fzvals'][...].reshape(self.shape)
            if outputChannels == 1:
                batch_solutions[i]=np.sum(randomorientation['solution'][...], axis=-1, keepdims=True).reshape((self.shape[:-2]+(outputChannels,)))
            else:
                batch_solutions[i]=randomorientation['solution'][...].reshape((self.shape[:-2]+(outputChannels,)))

        return {'forces': batch_Fz, 'solutions': batch_solutions}
   

    def solution_xymap_projection(self, datasetString, COMposition=[0,0,0]):
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
        AtomDict = {'C': 0, 'H': 1, 'O': 2, 'N': 3, 'F': 4}
        #print raw
        projected_array = np.zeros((self.f[datasetString].attrs['divxyz'][0], self.f[datasetString].attrs['divxyz'][1], 5))   # x, y, AtomNumber as in the dict
        masses = {'H' : 1.008, 'C' : 12.011, 'O' : 15.9994, 'N' : 14.0067, 'S' : 32.065, 'F' : 18.9984}
        covalentRadii = {'H' : 31, 'C' : 76, 'O' : 66, 'N' : 71, 'F' : 57}
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
    
        def atomSignal(evalvect, meanvect, atomNameString):
            """ This is not a Normal Distribution!!! It's a gauss-like distribution, but we normalize with the relative atom size instead of 1/sqrt(2*pi*sigma**2), this is s.t. the different elements give different 'signals'. """
            # Lets try it without Normalisation
    #             return 1/sqrt(2*pi*sigma**2)*exp(-((x-xmen)**2+(y-ymean)**2+(z-zmean)**2)/sigma**2)
            sigma = 4.0*(covalentRadii[atomNameString[i]])/76   # This i should not be here, am I right???
            normalisation = 1.0*(covalentRadii[atomNameString[i]])/76.
    #             normalisation = 1.0
            return normalisation*exp(-((evalvect[0]-meanvect[0])**2+(evalvect[1]-meanvect[1])**2+(evalvect[2]-meanvect[2])**2)/sigma**2)
    
        for i in range(len(atomNameString)):
            xPos = atomPosition[i,0]-COM[0]+(widthX/2.)
            yPos = atomPosition[i,1]-COM[1]+(widthY/2.)
            zPos = atomPosition[i,2]-COM[2]+10         # Here 10 is just an arbitrary number used to define a new Z grid such that all the values of zPos are positive. The new centre of mass will be at (widthX/2, widthY/2, 10). The projected area matrix will be evaluated at a height of index number 140 as the mechafm tip is 4 angstroms above the COM of the molecule and 10*10+4*10 is 140. This is to ensure that this grid is above the topmost atom, as the AFM results are read from above 
            xPosInt = int(round(xPos/self.f[datasetString].attrs['dxyz'][0]))       # Attention: The center of the xy grid is the center of mass of the molecule.
            yPosInt = int(round(yPos/self.f[datasetString].attrs['dxyz'][1]))       # There is some kind of bug here!!! Idk what, but I just catch it. Maybe have a look at it, bc maybe the whole calculation is wrong.
            zPosInt = int(round(zPos/self.f[datasetString].attrs['dxyz'][2]))
    #           print atomPosition[i,0], COM[0], xPos, xPosInt, yPosInt, atomNameString[i], raw[4][1]
            
            selectedAtomGridIndex = AtomDict[atomNameString[i]]
    
            #76 is the cov radius of carbon, 1 is an arbitrary value for the variance to allow enough spreading
            #print variance
    #             covarianceMatrix = [[variance,0,0],[0,variance,0],[0,0,variance]]
    #             gaussianDistribution = multivariate_normal(mean=[xPosInt, yPosInt, zPosInt], cov=covarianceMatrix)
            
            #find gaussianDistribution.pdf() at each point on the XY matrix at height 140 and add to the matrix of that type of atom
            for yIndexIter in range(self.f[datasetString].attrs['divxyz'][1]):
                for xIndexIter in range(self.f[datasetString].attrs['divxyz'][0]):
                    projected_array[xIndexIter, yIndexIter, selectedAtomGridIndex] += atomSignal([xIndexIter, yIndexIter, matrixPositionZIndex], [xPosInt, yPosInt, zPosInt], atomNameString)
    
        projected_array = projected_array * 10
        
        return projected_array
    
        
    def solution_xymap_collapsed(self, dataSetString):
        """Gives a version of the xymap solution collapsed to only one map, asking the question 'atom or not?' instead of 'What kind of atom?' """
        return np.sum(self.solution_xymap_projection(dataSetString),axis=-1, keepdims=True)

    def solution_toyDB(self, orientationGroupString):
        """ Adapted for the special case of the toyDB, that does not shift to the COM. """
        atomPos = self.f[orientationGroupString+'/atomPosition'][0,:]
        
        def atomSignal(evalvect, meanvect, atomNameString):
            """ This is not a Normal Distribution!!! It's a gauss-like distribution, 
            but we normalize with the relative atom size instead of 1/sqrt(2*pi*sigma**2), 
            this is s.t. the different elements give different 'signals'. """
            covalentRadii = {'H' : 31, 'C' : 76, 'O' : 66, 'N' : 71, 'F' : 57}  
            sigma = 1.0*(covalentRadii[atomNameString])/76
            normalisation = 1.0*(covalentRadii[atomNameString])/76.
            return normalisation*exp(-((evalvect[0]-meanvect[0])**2+(evalvect[1]-meanvect[1])**2+(evalvect[2]-meanvect[2])**2)/sigma**2)
        
        solutionArray = np.zeros((41,41,1))
        for xindex in range(41):
            for yindex in range(41):
                solutionArray[xindex, yindex, 0]+=atomSignal([float(xindex)*0.2,float(yindex)*0.2,0.0*0.2], atomPos, 'C')
                
        return solutionArray

    def batch_fresh_solution(self, batchsize, outputChannels=1):
        """ To use if the DB contains no solutions or if one wants to skip the 'add_labels' step. """
        batch_Fz=np.zeros((batchsize,)+self.shape)   # Maybe I can solve this somehow differently by not hardcoding the dimensions? For now I want to hardcode the dimensions, since the NN is also not flexible concerning them.
        batch_solutions=np.zeros((batchsize,)+self.shape[:-2]+(outputChannels,))
        for i in range(0,batchsize):
            randommolecule=self.f[random.choice(self.f.keys())]  # Choose a random molecule
            randomorientation=randommolecule[random.choice(randommolecule.keys())]   # Choose a random Orientation
            print 'Looking at file ' + randomorientation.name

            batch_Fz[i]=randomorientation['fzvals'][...].reshape(self.shape)
            batch_solutions[i]=self.solution_xymap_collapsed(randomorientation.name)[...]
        return {'forces': batch_Fz, 'solutions': batch_solutions}
    
    def batch_test(self, batchsize, outputChannels=1):
        """ Returns a batch that includes the AtomPositions, so that a more comprehensible viewfile can be made. """
        
        batch_Fz=np.zeros((batchsize,)+self.shape)
        batch_solutions=np.zeros((batchsize,)+self.shape[:2]+(outputChannels,))
        batch_atomPositions=[]
        
        for i in range(0,batchsize):
            randommolecule=self.f[random.choice(self.f.keys())]  # Choose a random molecule
            randomorientation=randommolecule[random.choice(randommolecule.keys())]   # Choose a random Orientation
            print 'Looking at file ' + randomorientation.name

            batch_Fz[i]=randomorientation['fzvals'][...].reshape(self.shape)
            batch_atomPositions.append(randomorientation['atomPosition'][...])
            
            if outputChannels == 1:
                batch_solutions[i]=np.sum(randomorientation['solution'][...], axis=-1, keepdims=True).reshape((self.shape[:-2]+(outputChannels,)))
            else:
                batch_solutions[i]=randomorientation['solution'][...].reshape((self.shape[:-2]+(outputChannels,)))        
        return {'forces': batch_Fz, 'solutions': batch_solutions, 'atomPosition': np.array(batch_atomPositions)}
    
    
if __name__=='__main__':
    print 'Hallo Main'
    datafile = AFMdata('/l/reischt1/AFMDB_version_01.hdf5', shape=(81,81,41,1))
    print(datafile.batch_test(10, outputChannels=5))
    