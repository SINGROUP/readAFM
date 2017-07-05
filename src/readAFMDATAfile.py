#Disclaimer: This was built for python 2.7, it does not work with python3 !!!


import struct
import glob
import numpy as np
import random
from math import sqrt, exp, pi
import h5py


class afmmolecule:

    """Take inputfile name and return data..."""

    def __init__(self, iFN):
        self.inputFileName=iFN
        self.afmdataFile = open(self.inputFileName, "rb")
        self.numAtoms = 0        
        asd = self.afmdataFile.read(4)
        self.totalNumOrientations = struct.unpack('i', asd)[0]  #total number of orientations
        self.afmdataFile.seek(0)

    def F_orientation_zxy(self, orientationNumber, zIndex, xIndex, yIndex):
        """ Returns Fz for a given tip position.
        
        orientationNumber = 1      modify to get desired orientation index. Starts from 0
        zIndex = 23         modify to get desired z index. Starts from 0 
        xIndex = 12         modify to get desired x index. Starts from 0 
        yIndex = 3          modify to get desired y index. Starts from 0  """

        fz = 0.0            #value of force in z direction will be stored here
        tipPosition = [0.0, 0.0, 0.0]           #position of tip will be stored here in the order z then x then y
        atomNameString = ""         #All the atoms in this molecule will be stored here. For example "CHHHH"
        atomPosition = np.zeros((30,3))         #30 being max number of atoms in molecule. 3 standing for x, y, z positions
        widthZ = 0.0
        widthX = 0.0
        widthY = 0.0
        dz = 0.0
        dx = 0.0
        dy = 0.0
        divZ = 0
        divX = 0
        divY = 0

        asd = self.afmdataFile.read(4)
        self.totalNumOrientations = struct.unpack('i', asd)[0]  #total number of orientations
        # print 'Total number of Orientations: ', self.totalNumOrientations

        asd = self.afmdataFile.read(4)
        self.numAtoms = struct.unpack('i', asd)[0]  #total number of atoms
        # print 'Number of Atoms: ', self.numAtoms


        self.afmdataFile.seek(-4, 1)         #To go back to the beginning of the file. Ignoring the first line, which specifies the number of orientations
        for i in range(0, orientationNumber):
            self.afmdataFile.seek(4 + 13*self.numAtoms, 1)            #To get past the xyz data
            self.afmdataFile.seek(24, 1)            #To get past the data about x,y,z width and dx, dy, dz
            asd = self.afmdataFile.read(4)
            tempdivz = struct.unpack('i', asd)[0]
            asd = self.afmdataFile.read(4)
            tempdivx = struct.unpack('i', asd)[0]
            asd = self.afmdataFile.read(4)
            tempdivy = struct.unpack('i', asd)[0]
            self.afmdataFile.seek(4*tempdivx*tempdivy*tempdivz, 1)

        #This loop reads the positions of the atoms to the np-array atomPosition
        self.afmdataFile.seek(4, 1)          #To get past the number of atoms which we already know
        for j in range(0, self.numAtoms):
            asd = self.afmdataFile.read(1)
            atomNameString += struct.unpack('c', asd)[0]              
            asd = self.afmdataFile.read(4)
            atomPosition[j, 0] = struct.unpack('f', asd)[0]            #X position
            asd = self.afmdataFile.read(4)
            atomPosition[j, 1] = struct.unpack('f', asd)[0]            #Y position
            asd = self.afmdataFile.read(4)
            atomPosition[j, 2] = struct.unpack('f', asd)[0]            #Z position


        asd = self.afmdataFile.read(4)
        widthZ = struct.unpack('f', asd)[0]
        asd = self.afmdataFile.read(4)
        widthX = struct.unpack('f', asd)[0]
        asd = self.afmdataFile.read(4)
        widthY = struct.unpack('f', asd)[0]
        asd = self.afmdataFile.read(4)
        dz = struct.unpack('f', asd)[0]
        asd = self.afmdataFile.read(4)
        dx = struct.unpack('f', asd)[0]
        asd = self.afmdataFile.read(4)
        dy = struct.unpack('f', asd)[0]
        asd = self.afmdataFile.read(4)
        divZ = struct.unpack('i', asd)[0]
        asd = self.afmdataFile.read(4)
        divX = struct.unpack('i', asd)[0]
        asd = self.afmdataFile.read(4)
        divY = struct.unpack('i', asd)[0]

        self.afmdataFile.seek(4*zIndex*divX*divY, 1)
        self.afmdataFile.seek(4*xIndex*divY, 1)
        self.afmdataFile.seek(4*yIndex, 1)
        asd = self.afmdataFile.read(4)
        fz = struct.unpack('f', asd)[0]
        tipPosition[0] = zIndex*dz
        tipPosition[1] = xIndex*dx
        tipPosition[2] = yIndex*dy

        # rewind filestream:
        self.afmdataFile.seek(0)

        return [[tipPosition,fz], atomNameString, atomPosition]
        


    def F_orientation(self, orientationNumber=1):
        """ Gives complete Fz-array for the specified orientation of the molecule.  """

        fz=0.0   #fz will be stored here
        atomNameString = ""         #All the atoms in this molecule will be stored here. For example "CHHHH"
        atomPosition = np.zeros((30,3))         #30 being max number of atoms in molecule. 3 standing for x, y, z positions
        widthZ = 0.0
        widthX = 0.0
        widthY = 0.0
        dz = 0.0
        dx = 0.0
        dy = 0.0
        divZ = 0
        divX = 0
        divY = 0

        asd = self.afmdataFile.read(4)
        self.totalNumOrientations = struct.unpack('i', asd)[0]  #total number of orientations
        # print 'Total number of Orientations: ', self.totalNumOrientations

        asd = self.afmdataFile.read(4)
        self.numAtoms = struct.unpack('i', asd)[0]  #total number of atoms
        # print 'Number of Atoms: ', self.numAtoms


        self.afmdataFile.seek(-4, 1)         #To go back to the beginning of the file. Ignoring the first line, which specifies the number of orientations
        for i in range(0, orientationNumber):
            self.afmdataFile.seek(4 + 13*self.numAtoms, 1)            #To get past the xyz data
            self.afmdataFile.seek(24, 1)            #To get past the data about x,y,z width and dx, dy, dz
            asd = self.afmdataFile.read(4)
            tempdivz = struct.unpack('i', asd)[0]
            asd = self.afmdataFile.read(4)
            tempdivx = struct.unpack('i', asd)[0]
            asd = self.afmdataFile.read(4)
            tempdivy = struct.unpack('i', asd)[0]
            self.afmdataFile.seek(4*tempdivx*tempdivy*tempdivz, 1)

        #This loop reads the positions of the atoms to the np-array atomPosition
        self.afmdataFile.seek(4, 1)          #To get past the number of atoms which we already know
        for j in range(0, self.numAtoms):
            asd = self.afmdataFile.read(1)
            atomNameString += struct.unpack('c', asd)[0]              
            asd = self.afmdataFile.read(4)
            atomPosition[j, 0] = struct.unpack('f', asd)[0]            #X position
            asd = self.afmdataFile.read(4)
            atomPosition[j, 1] = struct.unpack('f', asd)[0]            #Y position
            asd = self.afmdataFile.read(4)
            atomPosition[j, 2] = struct.unpack('f', asd)[0]            #Z position


        asd = self.afmdataFile.read(4)
        widthZ = struct.unpack('f', asd)[0]
        asd = self.afmdataFile.read(4)
        widthX = struct.unpack('f', asd)[0]
        asd = self.afmdataFile.read(4)
        widthY = struct.unpack('f', asd)[0]
        asd = self.afmdataFile.read(4)
        dz = struct.unpack('f', asd)[0]
        asd = self.afmdataFile.read(4)
        dx = struct.unpack('f', asd)[0]
        asd = self.afmdataFile.read(4)
        dy = struct.unpack('f', asd)[0]
        asd = self.afmdataFile.read(4)
        divZ = struct.unpack('i', asd)[0]
        asd = self.afmdataFile.read(4)
        divX = struct.unpack('i', asd)[0]
        asd = self.afmdataFile.read(4)
        divY = struct.unpack('i', asd)[0]

        fzarray = np.zeros((divX,divY,divZ,1))

        for h in range(0,divZ):                     #Jay, please check if this is right!!!
            for i in range(0,divX):
                for j in range(0,divY):
                    asd = self.afmdataFile.read(4)
                    fzarray[i,j,h,0] = struct.unpack('f',asd)[0]    # ATTENTION: XYZ convention!!!
                    
#                    self.afmdataFile.seek(4*zIndex*divX*divY, 1)
#                    self.afmdataFile.seek(4*xIndex*divY, 1)
#                    self.afmdataFile.seek(4*yIndex, 1)
#                    asd = self.afmdataFile.read(4)
#                    fz = struct.unpack('f', asd)[0]
#                    tipPosition[0] = zIndex*dz
#                    tipPosition[1] = xIndex*dx
#                    tipPosition[2] = yIndex*dy

        # rewind filestream:
        self.afmdataFile.seek(0)
        return {'fzarray': fzarray, 
                'atomNameString': atomNameString, 
                'atomPosition': atomPosition, 
                'widths': [widthX, widthY, widthZ], 
                'stepwidths': [dx,dy,dz],
                'divs': [divX,divY,divZ]}
#         return [fzarray, atomNameString, atomPosition, [widthX,widthY,widthZ],[dx,dy,dz],[divX,divY,divZ]]


    def solution_xymap_projection(self, orientationNumber):
        """Returns solution to train. Project the atom positions on the xy-plane with Amplitudes decaying like a Gaussian with the radius as variance. and write it on the correct level of the np-array.
        The last index of the array corresponds to the atom type:
        0 = C
        1 = H
        2 = 0
        3 = N
        4 = F
    
        """
        rawdata=self.F_orientation(orientationNumber)
        atomNameString=rawdata['atomNameString']
        atomPosition=rawdata['atomPosition']
        AtomDict = {'C': 0, 'H': 1, 'O': 2, 'N': 3, 'F': 4}
        #print raw
        projected_array = np.zeros((rawdata['divs'][0], rawdata['divs'][1], 5))   # x, y, AtomNumber as in the dict
        masses = {'H' : 1.008, 'C' : 12.011, 'O' : 15.9994, 'N' : 14.0067, 'S' : 32.065, 'F' : 18.9984}
        covalentRadii = {'H' : 31, 'C' : 76, 'O' : 66, 'N' : 71, 'F' : 57}
        # Calculate Center Of Mass:
        COM = np.zeros((3))
        totalMass=0.0
        for i in range(len(atomNameString)):
            atomVector = atomPosition[i,:]
            COM += atomVector*masses[atomNameString[i]]
            totalMass+=masses[atomNameString[i]]
        COM = COM/totalMass

        widthX=rawdata['widths'][0]
        widthY=rawdata['widths'][1]
        
        max_Zposition = 0.0
        indexOf_max_Zposition_in_atomNameString = 0

        for i in range(len(atomNameString)):
            if atomPosition[i, 2] > max_Zposition:
                max_Zposition = atomPosition[i, 2]
                indexOf_max_Zposition_in_atomNameString = i

        matrixPositionZIndex = int(round((atomPosition[indexOf_max_Zposition_in_atomNameString,2]-COM[2]+10)/rawdata['stepwidths'][2]))

        def atomSignal(evalvect, meanvect, atomNameString):
            """ This is not a Normal Distribution!!! It's a gauss-like distribution, but we normalize with the relative atom size instead of 1/sqrt(2*pi*sigma**2), this is s.t. the different elements give different 'signals'. """
            # Lets try it without Normalisation
#             return 1/sqrt(2*pi*sigma**2)*exp(-((x-xmen)**2+(y-ymean)**2+(z-zmean)**2)/sigma**2)
            sigma = 5.0*(covalentRadii[atomNameString[i]])/76
            normalisation = 1.0*(covalentRadii[atomNameString[i]])/76.
#             normalisation = 1.0
            return normalisation*exp(-((evalvect[0]-meanvect[0])**2+(evalvect[1]-meanvect[1])**2+(evalvect[2]-meanvect[2])**2)/sigma**2)

        for i in range(len(atomNameString)):
            xPos = atomPosition[i,0]-COM[0]+(widthX/2.)
            yPos = atomPosition[i,1]-COM[1]+(widthY/2.)
            zPos = atomPosition[i,2]-COM[2]+10         # Here 10 is just an arbitrary number used to define a new Z grid such that all the values of zPos are positive. The new centre of mass will be at (widthX/2, widthY/2, 10). The projected area matrix will be evaluated at a height of index number 140 as the mechafm tip is 4 angstroms above the COM of the molecule and 10*10+4*10 is 140. This is to ensure that this grid is above the topmost atom, as the AFM results are read from above 
            xPosInt = int(round(xPos/rawdata['stepwidths'][0]))       # Attention: The center of the xy grid is the center of mass of the molecule.
            yPosInt = int(round(yPos/rawdata['stepwidths'][1]))       # There is some kind of bug here!!! Idk what, but I just catch it. Maybe have a look at it, bc maybe the whole calculation is wrong.
            zPosInt = int(round(zPos/rawdata['stepwidths'][2]))
#           print atomPosition[i,0], COM[0], xPos, xPosInt, yPosInt, atomNameString[i], raw[4][1]
            
            selectedAtomGridIndex = AtomDict[atomNameString[i]]

            #76 is the cov radius of carbon, 1 is an arbitrary value for the variance to allow enough spreading
            #print variance
#             covarianceMatrix = [[variance,0,0],[0,variance,0],[0,0,variance]]
#             gaussianDistribution = multivariate_normal(mean=[xPosInt, yPosInt, zPosInt], cov=covarianceMatrix)
            
            #find gaussianDistribution.pdf() at each point on the XY matrix at height 140 and add to the matrix of that type of atom
            for yIndexIter in range(rawdata['divs'][1]):
                for xIndexIter in range(rawdata['divs'][0]):
                    projected_array[xIndexIter, yIndexIter, selectedAtomGridIndex] += atomSignal([xIndexIter, yIndexIter, matrixPositionZIndex], [xPosInt, yPosInt, zPosInt], atomNameString)

            
            #print(COM[0], COM[1], widthX, widthY, i, xPos, yPos, xPosInt, yPosInt, zPosInt)
            #projected_array[xPosInt, yPosInt, AtomDict[atomNameString[i]]] = 1
        
        #print projected_array[9, 41, 3]
        #print projected_array[10, 41, 3]
        #print projected_array[72, 39, 0]

        #maxValues = np.zeros(5)
        #for i in range(5):
        #    maxValues[i] = np.amax(projected_array[:, :, i])
        #    projected_array[:, :, i] = np.divide(projected_array[:, :, i], maxValues[i])
        
        projected_array = projected_array * 10
        
        return projected_array
    
    def solution_xymap_withoutH(self, orientationNumber):
        """ Manipulating hydrogens atom radius s.t. it is a lot smaller than all other atoms. """
        rawdata=self.F_orientation(orientationNumber)
        atomNameString=rawdata['atomNameString']
        atomPosition=rawdata['atomPosition']
        AtomDict = {'C': 0, 'H': 1, 'O': 2, 'N': 3, 'F': 4}
        #print raw
        projected_array = np.zeros((rawdata['divs'][0], rawdata['divs'][1], 5))   # x, y, AtomNumber as in the dict
        masses = {'H' : 1.008, 'C' : 12.011, 'O' : 15.9994, 'N' : 14.0067, 'S' : 32.065, 'F' : 18.9984}
        covalentRadii = {'H' : 0.1, 'C' : 76, 'O' : 66, 'N' : 71, 'F' : 57}  # Use 10 instead of 31 for H
        # Calculate Center Of Mass:
        COM = np.zeros((3))
        totalMass=0.0
        for i in range(len(atomNameString)):
            atomVector = atomPosition[i,:]
            COM += atomVector*masses[atomNameString[i]]
            totalMass+=masses[atomNameString[i]]
        COM = COM/totalMass

        widthX=rawdata['widths'][0]
        widthY=rawdata['widths'][1]
        
        max_Zposition = 0.0
        indexOf_max_Zposition_in_atomNameString = 0

        for i in range(len(atomNameString)):
            if atomPosition[i, 2] > max_Zposition:
                max_Zposition = atomPosition[i, 2]
                indexOf_max_Zposition_in_atomNameString = i

        matrixPositionZIndex = int(round((atomPosition[indexOf_max_Zposition_in_atomNameString,2]-COM[2]+10)/rawdata['stepwidths'][2]))

        def atomSignal(evalvect, meanvect, atomNameString):
            """ This is not a Normal Distribution!!! It's a gauss-like distribution, but we normalize with the relative atom size instead of 1/sqrt(2*pi*sigma**2), this is s.t. the different elements give different 'signals'. """
            # Lets try it without Normalisation
#             return 1/sqrt(2*pi*sigma**2)*exp(-((x-xmen)**2+(y-ymean)**2+(z-zmean)**2)/sigma**2)
            sigma = 5.0*(covalentRadii[atomNameString[i]])/76
            normalisation = 1.0*(covalentRadii[atomNameString[i]])/76.
#             normalisation = 1.0
            return normalisation*exp(-((evalvect[0]-meanvect[0])**2+(evalvect[1]-meanvect[1])**2+(evalvect[2]-meanvect[2])**2)/sigma**2)

        for i in range(len(atomNameString)):
            xPos = atomPosition[i,0]-COM[0]+(widthX/2.)
            yPos = atomPosition[i,1]-COM[1]+(widthY/2.)
            zPos = atomPosition[i,2]-COM[2]+10         # Here 10 is just an arbitrary number used to define a new Z grid such that all the values of zPos are positive. The new centre of mass will be at (widthX/2, widthY/2, 10). The projected area matrix will be evaluated at a height of index number 140 as the mechafm tip is 4 angstroms above the COM of the molecule and 10*10+4*10 is 140. This is to ensure that this grid is above the topmost atom, as the AFM results are read from above 
            xPosInt = int(round(xPos/rawdata['stepwidths'][0]))       # Attention: The center of the xy grid is the center of mass of the molecule.
            yPosInt = int(round(yPos/rawdata['stepwidths'][1]))       # There is some kind of bug here!!! Idk what, but I just catch it. Maybe have a look at it, bc maybe the whole calculation is wrong.
            zPosInt = int(round(zPos/rawdata['stepwidths'][2]))
#           print atomPosition[i,0], COM[0], xPos, xPosInt, yPosInt, atomNameString[i], raw[4][1]
            
            selectedAtomGridIndex = AtomDict[atomNameString[i]]

            #76 is the cov radius of carbon, 1 is an arbitrary value for the variance to allow enough spreading
            #print variance
#             covarianceMatrix = [[variance,0,0],[0,variance,0],[0,0,variance]]
#             gaussianDistribution = multivariate_normal(mean=[xPosInt, yPosInt, zPosInt], cov=covarianceMatrix)
            
            #find gaussianDistribution.pdf() at each point on the XY matrix at height 140 and add to the matrix of that type of atom
            for yIndexIter in range(rawdata['divs'][1]):
                for xIndexIter in range(rawdata['divs'][0]):
                    projected_array[xIndexIter, yIndexIter, selectedAtomGridIndex] += atomSignal([xIndexIter, yIndexIter, matrixPositionZIndex], [xPosInt, yPosInt, zPosInt], atomNameString)

            
            #print(COM[0], COM[1], widthX, widthY, i, xPos, yPos, xPosInt, yPosInt, zPosInt)
            #projected_array[xPosInt, yPosInt, AtomDict[atomNameString[i]]] = 1
        
        #print projected_array[9, 41, 3]
        #print projected_array[10, 41, 3]
        #print projected_array[72, 39, 0]

        #maxValues = np.zeros(5)
        #for i in range(5):
        #    maxValues[i] = np.amax(projected_array[:, :, i])
        #    projected_array[:, :, i] = np.divide(projected_array[:, :, i], maxValues[i])
        
        projected_array = projected_array * 10
        
        return projected_array

    def solution_xymap_naive_projection(self, orientationNumber):
        """Returns solution to train. Project the atom positions on the xy-plane with Amplitudes decaying like a Gaussian with the radius as variance. and write it on the correct level of the np-array.
        The last index of the array corresponds to the atom type:
        0 = C
        1 = H
        2 = 0
        3 = N
        4 = F
    
        This is not written very transparently. It is just solution_xymap_projection but the gaussians are not decaying in z-Direction.
    
        """
        rawdata=self.F_orientation(orientationNumber)
        atomNameString=rawdata['atomNameString']
        atomPosition=rawdata['atomPosition']
        AtomDict = {'C': 0, 'H': 1, 'O': 2, 'N': 3, 'F': 4}
        #print raw
        projected_array = np.zeros((rawdata['divs'][0], rawdata['divs'][1], 5))   # x, y, AtomNumber as in the dict
        masses = {'H' : 1.008, 'C' : 12.011, 'O' : 15.9994, 'N' : 14.0067, 'S' : 32.065, 'F' : 18.9984}
        covalentRadii = {'H' : 31, 'C' : 76, 'O' : 66, 'N' : 71, 'F' : 57}
        # Calculate Center Of Mass:
        COM = np.zeros((3))
        totalMass=0.0
        for i in range(len(atomNameString)):
            atomVector = atomPosition[i,:]
            COM += atomVector*masses[atomNameString[i]]
            totalMass+=masses[atomNameString[i]]
        COM = COM/totalMass

        widthX=rawdata['widths'][0]
        widthY=rawdata['widths'][1]
        
        max_Zposition = 0.0
        indexOf_max_Zposition_in_atomNameString = 0

        for i in range(len(atomNameString)):
            if atomPosition[i, 2] > max_Zposition:
                max_Zposition = atomPosition[i, 2]
                indexOf_max_Zposition_in_atomNameString = i

        matrixPositionZIndex = int(round((atomPosition[indexOf_max_Zposition_in_atomNameString,2]-COM[2]+10)/rawdata['stepwidths'][2]))

        def atomSignal(evalvect, meanvect, atomNameString):
            """ This is not a Normal Distribution!!! It's a gauss-like distribution, but we normalize with the relative atom size instead of 1/sqrt(2*pi*sigma**2), this is s.t. the different elements give different 'signals'. """
            # Lets try it without Normalisation
#             return 1/sqrt(2*pi*sigma**2)*exp(-((x-xmen)**2+(y-ymean)**2+(z-zmean)**2)/sigma**2)
            sigma = 5.0*(covalentRadii[atomNameString[i]])/76
            normalisation = 1.0*(covalentRadii[atomNameString[i]])/76.
#             normalisation = 1.0
            return normalisation*exp(-((evalvect[0]-meanvect[0])**2+(evalvect[1]-meanvect[1])**2)/sigma**2)

        for i in range(len(atomNameString)):
            xPos = atomPosition[i,0]-COM[0]+(widthX/2.)
            yPos = atomPosition[i,1]-COM[1]+(widthY/2.)
            zPos = atomPosition[i,2]-COM[2]+10         # Here 10 is just an arbitrary number used to define a new Z grid such that all the values of zPos are positive. The new centre of mass will be at (widthX/2, widthY/2, 10). The projected area matrix will be evaluated at a height of index number 140 as the mechafm tip is 4 angstroms above the COM of the molecule and 10*10+4*10 is 140. This is to ensure that this grid is above the topmost atom, as the AFM results are read from above 
            xPosInt = int(round(xPos/rawdata['stepwidths'][0]))       # Attention: The center of the xy grid is the center of mass of the molecule.
            yPosInt = int(round(yPos/rawdata['stepwidths'][1]))       # There is some kind of bug here!!! Idk what, but I just catch it. Maybe have a look at it, bc maybe the whole calculation is wrong.
            zPosInt = int(round(zPos/rawdata['stepwidths'][2]))
#           print atomPosition[i,0], COM[0], xPos, xPosInt, yPosInt, atomNameString[i], raw[4][1]
            
            selectedAtomGridIndex = AtomDict[atomNameString[i]]

            #76 is the cov radius of carbon, 1 is an arbitrary value for the variance to allow enough spreading
            #print variance
#             covarianceMatrix = [[variance,0,0],[0,variance,0],[0,0,variance]]
#             gaussianDistribution = multivariate_normal(mean=[xPosInt, yPosInt, zPosInt], cov=covarianceMatrix)
            
            #find gaussianDistribution.pdf() at each point on the XY matrix at height 140 and add to the matrix of that type of atom
            for yIndexIter in range(rawdata['divs'][1]):
                for xIndexIter in range(rawdata['divs'][0]):
                    projected_array[xIndexIter, yIndexIter, selectedAtomGridIndex] += atomSignal([xIndexIter, yIndexIter, matrixPositionZIndex], [xPosInt, yPosInt, zPosInt], atomNameString)

            
            #print(COM[0], COM[1], widthX, widthY, i, xPos, yPos, xPosInt, yPosInt, zPosInt)
            #projected_array[xPosInt, yPosInt, AtomDict[atomNameString[i]]] = 1
        
        #print projected_array[9, 41, 3]
        #print projected_array[10, 41, 3]
        #print projected_array[72, 39, 0]

        #maxValues = np.zeros(5)
        #for i in range(5):
        #    maxValues[i] = np.amax(projected_array[:, :, i])
        #    projected_array[:, :, i] = np.divide(projected_array[:, :, i], maxValues[i])
        
        projected_array = projected_array * 10
        
        return projected_array

    
    def solution_xymap_collapsed(self, orientationNumber):
        """Gives a version of the xymap solution collapsed to only one map, asking the question 'atom or not?' instead of 'What kind of atom?' """
        collapsed_array = np.sum(self.solution_xymap_projection(orientationNumber),axis=-1, keepdims=True)
#         collapsed_array.shape = (81,81,1)
        return collapsed_array
    
    def solution_xymap_naive_collapsed(self, orientationNumber):
        """ Collapses the naive solution to only one map. """
        return np.sum(self.solution_xymap_naive_projection(orientationNumber), axis = -1, keepdims = True)
            
            
class AFMdata:
    """ Takes database folder and gives training batches. """
    def __init__(self, FolderName):
        self.datafiles = glob.glob(FolderName+'/*')

    def batch(self, batchsize):
        batch_Fz=np.zeros((batchsize,81,81,41,1))   # Maybe I can solve this somehow differently by not hardcoding the dimensions?
        batch_solutions=np.zeros((batchsize,81,81,5))
        for i in range(0,batchsize):
            randommolecule=afmmolecule(random.choice(self.datafiles))  # Choose a random molecule
            orientation=random.randint(0,randommolecule.totalNumOrientations-1)   # Choose a random Orientation
            print 'Looking at file ', randommolecule.inputFileName, ' at orientation ', orientation
            batch_Fz[i]=randommolecule.F_orientation(orientation)['fzarray']
            batch_solutions[i]=randommolecule.solution_xymap_projection(orientation)
        return {'forces': batch_Fz, 'solutions': batch_solutions}



if __name__=='__main__':
    print 'Hallo Main'
    
    datafile = afmmolecule('../outputxyz/dsgdb9nsd_000485.afmdata')
    out = h5py.File('../scratch/solutions000485.hdf5', 'w')
    
    xyz = datafile.F_orientation(0)['atomPosition']
    np.savetxt('../scratch/positions000485.xyz', xyz, delimiter=' ')
    
    testsols = np.zeros((10,81,81,5))
    collsols = np.zeros((10,81,81,1))
    wohsols =  np.zeros((10,81,81,5))
    fz = np.zeros((10,81,81,41))
    naivesols = np.zeros((10,81,81,5))
    naivecoll = np.zeros((10,81,81,1))
    for i in range(10):
        testsols[i,:,:,:] = datafile.solution_xymap_projection(i+17)
        collsols[i,:,:,:] = datafile.solution_xymap_collapsed(i+17)
        wohsols[i,:,:,:] = datafile.solution_xymap_withoutH(i+17)
        fz[i,:,:,:] = datafile.F_orientation(i+17)['fzarray'][:,:,:,0]
        naivesols[i,:,:,:] = datafile.solution_xymap_naive_projection(i+17)
        naivecoll[i,:,:,:] = datafile.solution_xymap_naive_collapsed(i+17)
        
    out.create_dataset('/testsolution', data=testsols)
    out.create_dataset('/testsolutioncollapsed', data=collsols)
    out.create_dataset('/fzarray', data=fz)
    out.create_dataset('/testsolutionwoH', data=wohsols)
    out.create_dataset('/testsolutionnaive', data=naivesols)
    out.create_dataset('/testsolutionnaivecollapsed', data=naivecoll)
    
    out.close()