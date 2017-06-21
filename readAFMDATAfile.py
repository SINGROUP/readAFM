

# Disclaimer: This was built for python 2.7, it does not work with python3 !!!


import struct
import glob
import argparse
import numpy as np
import random


# parser = argparse.ArgumentParser(description="Take a .AFMDATA file and parse its contents")
# parser.add_argument("-i", "--input_file", default="input.afmdata", help="the path to the input .AFMDATA file (default: %(default)s)")
# args = parser.parse_args()

# inputFileName = args.input_file

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
        """orientationNumber = 1      modify to get desired orientation index. Starts from 0
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
        """ Gives data back for the specified orientation of the molecule.  """

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
                    fzarray[i,j,h,0] = struct.unpack('f',asd)[0]
                    
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

        return [fzarray, atomNameString, atomPosition, [widthX,widthY,widthZ],[dx,dy,dz],[divX,divY,divZ]]



    def solution_xymap_dummy(self,orientation):
        """Dummy solution to train. xy-map with one hot vecors correspoding to the position of the atoms."""
        solution=np.zeros((41,40,5))
        solution[20,20,0]=1
        solution[30,34,0]=1
        solution[10,10,1]=1
        solution[20,25,2]=1
        return solution


    def solution_xymap_projection(self, orientationNumber):
        """Returns solution to train. Here xy-map with one hot vecors correspoding to the position of the atoms. Project the atom positions on the xy-plane and write it on the correct level of the np-array.
        The last index of the array corresponds to the atom type:
        0 = C
        1 = H
        2 = 0
        3 = N
        4 = F
    
        """
        raw=self.F_orientation(orientationNumber)
        atomNameString=raw[1]
        atomPosition=raw[2]
        AtomDict = {'C': 0, 'H': 1, 'O': 2, 'N': 3, 'F': 4}
        
        projected_array = np.zeros((raw[5][0],raw[5][1],5))   # x, y, AtomNumber as in the dict
        masses = {'H' : 1.008, 'C' : 12.011, 'O' : 15.9994, 'N' : 14.0067, 'S' : 32.065, 'F' : 18.9984}
        # Calculate Center Of Mass:
        COM = np.zeros((3))
        totalMass=0.0
        for i in range(len(atomNameString)):
            atomVector = atomPosition[i,:]
            # xPos = atomPosition[i,0]
            # yPos = atomPosition[i,1]
            # zPos = atomPosition[i,2]
            COM += atomVector*masses[atomNameString[i]]
            totalMass+=masses[atomNameString[i]]
        COM = COM/totalMass

        widthX=raw[3][0]
        widthY=raw[3][1]
        for i in range(len(atomNameString)):
            xPos = atomPosition[i,0]-COM[0]+(widthX/2.)
            yPos = atomPosition[i,1]-COM[1]+(widthY/2.)
            xPosInt = int(round(xPos/raw[4][0]))       # Attention: The center of the xy grid is the center of mass of the molecule.
            yPosInt = int(round(yPos/raw[4][1]))       # There is some kind of bug here!!! Idk what, but I just catch it. Maybe have a look at it, bc maybe the whole calculation is wrong.
#            print atomPosition[i,0], COM[0], xPos, xPosInt, yPosInt, atomNameString[i], raw[4][1]

            print(COM[0], COM[1], widthX, widthY, i, xPos, yPos, xPosInt, yPosInt)

            projected_array[xPosInt, yPosInt, AtomDict[atomNameString[i]]] = 1
        return projected_array


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
            batch_Fz[i]=randommolecule.F_orientation(orientation)[0]
            batch_solutions[i]=randommolecule.solution_xymap_projection(orientation)
        return [batch_Fz, batch_solutions]



if __name__=='__main__':
    print 'Hallo Main'
    datafile = afmmolecule('./outputxyz/dsgdb9nsd_000485.afmdata')
    datafile.solution_xymap_projection(10)
