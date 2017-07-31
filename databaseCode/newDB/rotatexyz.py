import numpy
import os
import subprocess
import math

# Rotates input xyz file and generates 100 output xyz files at random orientations along with input.scan for each of them to give as input to mechAFM

def rotate(inFileName = "input.xyz", output_folder = "randomRotateOutput/"):            # These are the default file and folder names. IGNORE
    
    if not output_folder.endswith('/'):
        output_folder += '/'
    if not os.path.exists(output_folder):                                               # Makes the output folder if not already made. This code is unnecessary when the function is called from dostuff.py as that script makes the directory already
        os.makedirs(output_folder)
    
    outFileName = output_folder + 'out'                                                 # The rotated orientations are the inputs to MechAFM. The files are named out0.scan, out0.xyz; out1.scan, out1.xyz and so on...
    inFile = open(inFileName, "r")
    atomLimit = 1000 # Maximum number of atoms allowed in program

    alpha = 0.0 # Angle of rotation about Z axis
    beta = 0.0 # Angle of rotation about Y axis
    gamma = 0.0 # Angle of rotation about X axis
    
    atomName = [None] * atomLimit
    xIn = [None] * atomLimit
    yIn = [None] * atomLimit
    zIn = [None] * atomLimit
    xOut = [None] * atomLimit
    yOut = [None] * atomLimit
    zOut = [None] * atomLimit
    
    fileContents = inFile.readlines()
    numberOfAtoms = int(fileContents[0])

    for atomIndex in range(0, numberOfAtoms):
        lineContents = fileContents[atomIndex+2].split()
        atomName[atomIndex] = lineContents[0]
        xIn[atomIndex] = float(lineContents[1])
        yIn[atomIndex] = float(lineContents[2])
        zIn[atomIndex] = float(lineContents[3])

    for rotateIndex in range(0, 100):                                                   # The number of random rotations that will be performed. Edit this number (100 by default) to desired number
        outFile = open(outFileName + "%s" % rotateIndex + ".xyz", "w+")
        outFile.write(str(numberOfAtoms) + "\n")
        outFile.write(" " + "\n")
        alpha = numpy.random.random_sample()*(numpy.pi)
        beta = numpy.random.random_sample()*(numpy.pi)
        gamma = numpy.random.random_sample()*(numpy.pi)
        xMax = 0.0          # These are values of extremities of the produced molecule
        xMin = 0.0          # Helps in defining the area of scan in MechAFM
        yMax = 0.0          #
        yMin = 0.0          #
        zMax = 0.0          #
        
        for atomIndex in range(0, numberOfAtoms):
            # below is the code to rotate the input XYZ file randomly
            xOut = xIn[atomIndex]*numpy.cos(alpha)*numpy.cos(beta) + yIn[atomIndex]*(numpy.cos(alpha)*numpy.sin(beta)*numpy.sin(gamma)-numpy.sin(alpha)*numpy.cos(gamma)) + zIn[atomIndex]*(numpy.cos(alpha)*numpy.sin(beta)*numpy.cos(gamma)+numpy.sin(alpha)*numpy.sin(gamma))
            yOut = xIn[atomIndex]*numpy.sin(alpha)*numpy.cos(beta) + yIn[atomIndex]*(numpy.sin(alpha)*numpy.sin(beta)*numpy.sin(gamma)+numpy.cos(alpha)*numpy.cos(gamma)) + zIn[atomIndex]*(numpy.sin(alpha)*numpy.sin(beta)*numpy.cos(gamma)-numpy.cos(alpha)*numpy.sin(gamma))
            zOut = - xIn[atomIndex]*numpy.sin(beta) + yIn[atomIndex]*numpy.cos(beta)*numpy.sin(gamma) + zIn[atomIndex]*numpy.cos(beta)*numpy.cos(gamma)
            outString = atomName[atomIndex] + "\t" + str(xOut) + "\t" + str(yOut) + "\t" + str(zOut)
            outFile.write(outString + "\n")
            if atomIndex == 0 :
                xMax = xOut
                xMin = xOut
                yMax = yOut
                yMin = yOut
                zMax = zOut
            else :                                                                      # To find max and min values of X, Y and Z in the rotated XYZ file
                if xOut > xMax :
                    xMax = xOut
                elif xOut < xMin :
                    xMin = xOut
                if yOut > yMax :
                    yMax = yOut
                elif yOut < yMin :
                    yMin = yOut
                if zOut > zMax :
                    zMax = zOut
        #xWidth = xMax - xMin + 1.5
        #yWidth = yMax - yMin + 1.5
        #if xWidth > yWidth :
        #    deltaPosition = xWidth/40.0 # We want at most 40 points for each scan on the XY plane
        #else :
        #    deltaPosition = yWidth/40.0
        
        outScanFile = open(outFileName + "%s" % rotateIndex + ".scan", "w+")
        
        roundedZMax = float(math.ceil(1000*(zMax)))/1000                                # The lowest Z will be 6 Armstrongs above the highest atom. The highest will be 10.05 armstrongs above the highest atom. This will ensure we have 41 divisions of Z when each step is 0.1 armstrongs
                                                                                        # Rounding off to ensure that number of divisions will be 41
        
        # below are the contents for the input scan file
        scanOut = '''xyzfile    out%s
paramfile    parameters.dat
tipatom      T
dummyatom    X

units        kcal/mol

minterm      f
etol         0.001
ftol         0.001
dt           0.001
maxsteps     50000

minimiser    FIRE
integrator   midpoint

coulomb      off

rigidgrid    off

flexible     off

area         8.0 8.0

zhigh        %s
zlow         %s
dx           0.2
dy           0.2
dz           0.1

bufsize      10000
gzip         off
statistics   on''' % (str(rotateIndex) + ".xyz", roundedZMax + 10.05, roundedZMax + 6.0)
        
        outScanFile.write(scanOut)
        
        if not os.path.exists(outFileName + "%s" % rotateIndex):            # Makes directory in the same folder containing mechAFM output if it doesn't exist already
            os.makedirs(outFileName + "%s" % rotateIndex)
