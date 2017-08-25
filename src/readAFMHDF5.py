import numpy as np
import random
import h5py
from math import exp
import time

def atomSignal(evalvect, meanvect, atomNameString, sigmabasexy=1.0, sigmabasez=1.0, amplification=1.0):
    """ Calculates the signal at evalvect of the atom sitting at meanvect.
    
    This is not a Normal Distribution!!! It's a gaussian-like distribution, 
    but we normalize with the relative atom size instead of 1/sqrt(2*pi*sigma**2), 
    this is s.t. the different elements give different 'signals'. 
    Sigmabase is multiplied with the relative covalent ratius, with C corresponding to 1.
    I think it can be interpreted as a length in A.
    
    Args: 
        evalvect: vector (np.array or list with at least 3 entries) where to evaluate the atomSignals
        meanvect: vector (format as evalvect) for the position of the atom
        sigmabasexy: 'standarddeviation' of the gaussian in the xy-plane
        sigmabasez: 'standarrdeviation' of the gaussina in z-direction 
        amplification: Factor to amplify the gaussian by. Basically the strength of the 'signal' of a C atom at its position. 
        
    Returns:
        float, 'Signal' value, to be summed up at every position for every atom. 
    """
    # covalentRadii = {'H' : 31, 'C' : 76, 'O' : 66, 'N' : 71, 'F' : 57, 'Si' : 111, 'B' : 82, 'Al' : 121, 'Na' : 154, 'P' : 106, 'S' : 105, 'Cl' : 102}  
    covalentRadii = {'H' : 31, 'C' : 76, 'O' : 66, 'N' : 71, 'F' : 57, 'I' : 111, 'B' : 82, 'A' : 121, 'D' : 154, 'P' : 106, 'S' : 105, 'L' : 102}
    [sigmabasexy, sigmabasez] = map(lambda x: x*((float(covalentRadii[atomNameString]))/76.),[sigmabasexy, sigmabasez])
    normalisation = amplification*(covalentRadii[atomNameString])/76.
    return normalisation*exp(-((evalvect[0]-meanvect[0])**2+(evalvect[1]-meanvect[1])**2)/sigmabasexy**2)*exp(-((evalvect[2]-meanvect[2])**2)/sigmabasez**2)

class AFMdata:
    """ Class for opening the HDF5 file containing the DB. 
    """
    
    def __init__(self, FileName, shape=(41, 41, 41, 1)):
        """ Opens hdf5 file FileName for reading.
        
        Args:
            FileName: string with the path to the hdf5 file containing the database.
            shape: has to contain the Shape of the DB, in the form (x,y,z,inChannels). 
        """
        
        self.f = h5py.File(FileName, "r+")
        self.shape = tuple(shape)
 

    def solution_xymap_projection(self, datasetString, COMposition=[0.,0.,0.], sigmabasexy=1.0, sigmabasez=1.0, amplificationFactor=1.0):
        """Returns solution to train. 
        
        Project the atom positions on the xy-plane with Amplitudes decaying like a Gaussian, specified in atomSignal. The output array has several output channels corresponding to the different elements. The last index of the array determines the element:
        0 = C
        1 = H
        2 = 0
        3 = N
        4 = F
        
        Args:
            datasetString: path to the dataset in the hdf5 file
            COMposition: Center Of Mass position, some DBs have the COM centered at [.01, .01, .0]
            sigmabasexy: 'standarddeviation' of the gaussian in the xy-plane
            sigmabasez: 'standarrdeviation' of the gaussina in z-direction 
            amplification: Factor to amplify the gaussian by. Basically the strength of the 'signal' of a C atom at its position. 
            
        Returns:
            array with the same xy-dimensions as the db-shape, with the atom signals broken down to the atoms they result from.
        """
    
        atomNameString=self.f[datasetString].attrs['atomNameString']
        atomPosition=self.f[datasetString+'/atomPosition']
        # AtomDict =  {'C': 0, 'H': 1, 'O': 2, 'N': 3, 'F': 4, 'S': 5, 'Si' : 6, 'B' : 7, 'Al' : 8, 'Na' : 9, 'P' : 10, 'Cl' : 11}
        AtomDict = {'C': 0, 'H': 1, 'O': 2, 'N': 3, 'F': 4, 'S': 5, 'I' : 6, 'B' : 7, 'A' : 8, 'D' : 9, 'P' : 10, 'L' : 11}
        #print raw
        projected_array = np.zeros((self.f[datasetString].attrs['divxyz'][0], self.f[datasetString].attrs['divxyz'][1], 12))   # x, y, AtomNumber as in the dict
        # masses = {'H' : 1.008, 'C' : 12.011, 'O' : 15.9994, 'N' : 14.0067, 'S' : 32.065, 'F' : 18.9984, 'Si' : 28.0855, 'B' : 10.811, 'Al' : 26.9815, 'Na' : 22.9898, 'P' : 30.9738, 'Cl' : 35.453}
        masses = {'H' : 1.008, 'C' : 12.011, 'O' : 15.9994, 'N' : 14.0067, 'S' : 32.065, 
                  'F' : 18.9984, 'I' : 28.0855, 'B' : 10.811, 'A' : 26.9815, 'D' : 22.9898, 
                  'P' : 30.9738, 'L' : 35.453}
#         covalentRadii = {'H' : 31, 'C' : 76, 'O' : 66, 'N' : 71, 'F' : 57}
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
                    projected_array[xIndexIter, yIndexIter, selectedAtomGridIndex] += atomSignal([xIndexIter, yIndexIter, matrixPositionZIndex], 
                                                                                                 [xPosInt, yPosInt, zPosInt], 
                                                                                                 atomNameString[i], 
                                                                                                 sigmabasexy=sigmabasexy,
                                                                                                 sigmabasez=sigmabasez, 
                                                                                                 amplification=amplificationFactor)
        
        return projected_array
    
        
    def solution_xymap_collapsed(self, dataSetString, COMposition=[0.,0.,0.], sigmabasexy=1.0, sigmabasez=1.0, amplificationFactor=1.0):
        """ Gives a version of the xymap solution collapsed to only one map, asking the question 'atom or not?' instead of 'What kind of atom?' 
        
        Args:
            datasetString: path to the dataset in the hdf5 file
            COMposition: Center Of Mass position, some DBs have the COM centered at [.01, .01, .0]
            sigmabasexy: 'standarddeviation' of the gaussian in the xy-plane
            sigmabasez: 'standarrdeviation' of the gaussina in z-direction 
            amplification: Factor to amplify the gaussian by. Basically the strength of the 'signal' of a C atom at its position. 
            
        Returns:
            array with the same xy-dimensions as the db-shape, with the atom signals collapsed down (=summed up) to one layer
        
        """
        
        return np.sum(self.solution_xymap_projection(dataSetString, COMposition, sigmabasexy, sigmabasez, amplificationFactor),axis=-1, keepdims=True)

    def solution_singleAtom(self, orientationGroupString, sigmabasexy=1.0, sigmabasez=1.0, amplificationFactor=1.0):
        """ Adapted for the special case of the toyDB containing a single atom, where the COM is not shifted. 
        
        Args:
            orientationGroupString: path to the group in the hdf5 file
            sigmabasexy: 'standarddeviation' of the gaussian in the xy-plane
            sigmabasez: 'standarrdeviation' of the gaussina in z-direction 
            amplification: Factor to amplify the gaussian by. Basically the strength of the 'signal' of a C atom at its position. 
            
        Returns:
            array with the same xy-dimensions as the db-shape, containing the atom signals.

        """
        atomPos = self.f[orientationGroupString+'/atomPosition'][0,:]
                
        solutionArray = np.zeros((41,41,1))
        for xindex in range(41):
            for yindex in range(41):
                solutionArray[xindex, yindex, 0]+=atomSignal([float(xindex)*0.2,float(yindex)*0.2,0.0*0.2], 
                                                             atomPos, 
                                                             atomNameString='C', 
                                                             sigmabasexy=sigmabasexy, 
                                                             sigmabasez=sigmabasez, 
                                                             amplification=amplificationFactor)
                     
        return solutionArray

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

        Args:
            batchsize: batchsize, how many sets of fz-data + label should be returned
            outputChannels: Number of output channels, should match the labelling method -> 'xymap_collapsed" and 'singleAtom' 1, 'xymap_projection' 5 or 12
            method: method on how to calculate the solution (label), options are: xymap_collapsed, xymap_projection, singleAtom
            COMposition: Center Of Mass position, some DBs have the COM centered at [.01, .01, .0]
            sigmabasexy: 'standarddeviation' of the gaussian in the xy-plane
            sigmabasez: 'standarrdeviation' of the gaussina in z-direction 
            amplification: Factor to amplify the gaussian by. Basically the strength of the 'signal' of a C atom at its position. 
            returnAtomPositions: If True, the AtomPositons are included in the return dictionary
            verbose: If True, the name of the randomly selected molecules/orientations will be printed to stdout
            orientationsOnly: Set False if the .hdf5 has the structure /rootGroup/moleculeXXXX/orientationXXXXX, set true if there is only one level /rootGroup/molXXXXortnXXXX
            rootGroup: Which is the group to select randomly from? usually /train or /validation    
            
        Returns: a dictionary containing 
            'forces': a np.ndarray of shape (batchsize,)+self.shape
            'solutions': a np.ndarray of shape (batchsize,)+self.shape[:-2]+(outputChannels,)
            'atomPosition': only if returnAtomPositions == True, a list containing the atom positions in the same order as they appear in the atomNameString
            
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
        """ Returns (training)batches as dictionaries with 'forces' and 'solutions'
        
        Use only with labelled databases, i.e. if the method 'add_labels' has been used on it before.

        Args:
            batchsize: batchsize, how many sets of fz-data + label should be returned
            outputChannels: Number of output channels, should match the labelling method -> 'xymap_collapsed" and 'singleAtom' 1, 'xymap_projection' 5 or 12
            returnAtomPositions: If True, the AtomPositons are included in the return dictionary
            verbose: If True, the name of the randomly selected molecules/orientations will be printed to stdout
            orientationsOnly: Set False if the .hdf5 has the structure /rootGroup/moleculeXXXX/orientationXXXXX, set true if there is only one level /rootGroup/molXXXXortnXXXX
            
        Returns: a dictionary containing 
            'forces': a np.ndarray of shape (batchsize,)+self.shape
            'solutions': a np.ndarray of shape (batchsize,)+self.shape[:-2]+(outputChannels,)
            'atomPosition': only if returnAtomPositions == True, a list containing the atom positions in the same order as they appear in the atomNameString
        """
        
#             rootGroup: Which is the group to select randomly from? usually /train or /validation    

        
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
        

    def validationbatch_runtimeSolution(self, 
                              batchsize, 
                              outputChannels=1, 
                              method='xymap_collapsed', 
                              COMposition=[0.,0.,0.], 
                              sigmabasexy=1.0,
                              sigmabasez=1.0, 
                              amplificationFactor=1.0, 
                              returnAtomPositions=False,
                              verbose=True, 
                              orientationsOnly=True,
                              rootGroup='/validation'):
        """ To use if the DB contains no solutions or if one wants to skip the 'add_labels' step. Does not draw randomly. 

        To use if one does not want randomly selected orientations. Is useful to be able to compare certain molecules analyzed with different neural nets.
        Output channels has to match the method.
        Methods are: xymap_collapsed, xymap_projection, singleAtom

        Args:
            batchsize: batchsize, how many sets of fz-data + label should be returned
            outputChannels: Number of output channels, should match the labelling method -> 'xymap_collapsed" and 'singleAtom' 1, 'xymap_projection' 5 or 12
            method: method on how to calculate the solution (label), options are: xymap_collapsed, xymap_projection, singleAtom
            COMposition: Center Of Mass position, some DBs have the COM centered at [.01, .01, .0]
            sigmabasexy: 'standarddeviation' of the gaussian in the xy-plane
            sigmabasez: 'standarrdeviation' of the gaussina in z-direction 
            amplification: Factor to amplify the gaussian by. Basically the strength of the 'signal' of a C atom at its position. 
            returnAtomPositions: If True, the AtomPositons are included in the return dictionary
            verbose: If True, the name of the randomly selected molecules/orientations will be printed to stdout
            orientationsOnly: Set False if the .hdf5 has the structure /rootGroup/moleculeXXXX/orientationXXXXX, set true if there is only one level /rootGroup/molXXXXortnXXXX
            rootGroup: Which is the group to select randomly from? usually /train or /validation    
            
        Returns: a dictionary containing 
            'forces': a np.ndarray of shape (batchsize,)+self.shape
            'solutions': a np.ndarray of shape (batchsize,)+self.shape[:-2]+(outputChannels,)
            'atomPosition': only if returnAtomPositions == True, a list containing the atomNameString and the atom positions in the same order as they appear in the atomNameString
            
        """
        batch_Fz=np.zeros((batchsize,)+self.shape)   # Maybe I can solve this somehow differently by not hardcoding the dimensions? For now I want to hardcode the dimensions, since the NN is also not flexible concerning them.
        batch_solutions=np.zeros((batchsize,)+self.shape[:-2]+(outputChannels,))
        if returnAtomPositions:
            batch_atomPositions=[]
            
        # make sample keylist here with random.sample and then iterate through it

        keys = []
        if orientationsOnly:
            keys = list(self.f[rootGroup].keys())[:batchsize]
            keys = [self.f[rootGroup][i].name for i in keys]

        else:
            raise KeyError('Validation is only possible with orientations only')
        
        for i in range(batchsize):
            
            orientation=self.f[keys[i]]
            if verbose:
                print('Looking at file ' + orientation.name)

            batch_Fz[i]=orientation['fzvals'][...].reshape(self.shape)
            if method=='xymap_collapsed':
                batch_solutions[i]=self.solution_xymap_collapsed(orientation.name, COMposition, sigmabasexy, sigmabasez, amplificationFactor)[...]
            elif method=='xymap_projection':
                batch_solutions[i]=self.solution_xymap_projection(orientation.name, COMposition, sigmabasexy, sigmabasez, amplificationFactor)[...]
            elif method=='singleAtom':
                batch_solutions[i]=self.solution_singleAtom(orientation.name, sigmabasexy, sigmabasez, amplificationFactor)[...]
            else:
                raise IOError('No such method as {}'.format(method))
                
            if returnAtomPositions:
                batch_atomPositions.append([orientation.attrs['atomNameString'], orientation['atomPosition'][...]])
                
        if returnAtomPositions:
            return {'forces': batch_Fz, 'solutions': batch_solutions, 'atomPosition': batch_atomPositions}
        else:
            return {'forces': batch_Fz, 'solutions': batch_solutions}        
        
        
    def add_labels(self, method='xymap_collapsed', COMposition=[0.,0.,0.], sigmabasexy=1.0, sigmabasez=1.0, amplificationFactor=1.0):
        """ Annotates a .hdf5 database with solutions. Use if no previous labels exist, if there are existing labels, use change_labels. 
        
        Args:
            method: method on how to calculate the solution (label), options are: xymap_collapsed, xymap_projection, singleAtom
            COMposition: Center Of Mass position, some DBs have the COM centered at [.01, .01, .0]
            sigmabasexy: 'standarddeviation' of the gaussian in the xy-plane
            sigmabasez: 'standarrdeviation' of the gaussina in z-direction 
            amplificationFactor: Factor to amplify the gaussian by. Basically the strength of the 'signal' of a C atom at its position. 
        Returns:
            No return.
        """
        
        
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
        """Annotates a .hdf5 database with solutions. Use if add_labels has been used before to change them.
        
        Options for method: 'xymap_collapsed', 'xymap_projection', 'singleAtom'
        
        Args:
            method: method on how to calculate the solution (label), options are: xymap_collapsed, xymap_projection, singleAtom
            COMposition: Center Of Mass position, some DBs have the COM centered at [.01, .01, .0]
            sigmabasexy: 'standarddeviation' of the gaussian in the xy-plane
            sigmabasez: 'standarrdeviation' of the gaussina in z-direction 
            amplificationFactor: Factor to amplify the gaussian by. Basically the strength of the 'signal' of a C atom at its position. 
        Returns:
            No return.
        
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
    datafile = AFMdata('/scratch/work/reischt1/flatMolecules_v18.hdf5', shape=(41,41,41,1))
#     print(datafile.solution_xymap_collapsed('molecule1/orientation1'))
    testbatch = datafile.batch_runtimeSolution(20, orientationsOnly=True, rootGroup='/train', returnAtomPositions=True)
    print testbatch['atomPosition']
    
    
    
