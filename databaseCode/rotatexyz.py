import numpy
import os
import subprocess
# from time import sleep

# Rotates input xyz file and generates 100 output xyz files at random orientations along with input.scan for each of them to give as input to mechAFM

def rotate(inFileName = "input.xyz", output_folder = "randomRotateOutput/"):
    if not output_folder.endswith('/'):
        output_folder += '/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    outFileName = output_folder + 'out'
    inFile = open(inFileName, "r")
    atomLimit = 100 # Maximum number of atoms allowed in program

    alpha = 0.0 # Angle of rotation about Z axis
    beta = 0.0 # Angle of rotation about Y axis
    gamma = 0.0 # Angle of rotation about X axis

    fileContents = inFile.readlines()
    numberOfAtoms = int(fileContents[0])
    atomName = [None] * atomLimit
    xIn = [None] * atomLimit
    yIn = [None] * atomLimit
    zIn = [None] * atomLimit
    xOut = [None] * atomLimit
    yOut = [None] * atomLimit
    zOut = [None] * atomLimit

    for atomIndex in range(0, numberOfAtoms):
        lineContents = fileContents[atomIndex+2].split()
        atomName[atomIndex] = lineContents[0]
        xIn[atomIndex] = float(lineContents[1])
        yIn[atomIndex] = float(lineContents[2])
        zIn[atomIndex] = float(lineContents[3])

    for rotateIndex in range(0, 100):
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
            else :
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
        xWidth = xMax - xMin + 1.5
        yWidth = yMax - yMin + 1.5
        if xWidth > yWidth :
            deltaPosition = xWidth/40.0 # We want at most 40 points for each scan on the XY plane
        else :
            deltaPosition = yWidth/40.0
        outScanFile = open(outFileName + "%s" % rotateIndex + ".scan", "w+")

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

area         %s %s

zhigh        8.0
zlow         4.0
dx           %s
dy           %s
dz           0.1

bufsize      10000
gzip         off
statistics   on''' % (str(rotateIndex) + ".xyz", 8.0, 8.0, 0.1, 0.1)
        outScanFile.write(scanOut)
        # print scanOut
        if not os.path.exists(outFileName + "%s" % rotateIndex):            # Makes directory in the same folder containing mechAFM output
            os.makedirs(outFileName + "%s" % rotateIndex)
        # print "../bin/mechafm-omp %s %s" % (outFileName + "%s" % rotateIndex + ".scan", outFileName + "%s" % rotateIndex)
        # # sleep(1)
        # callString = "../bin/mechafm-omp " + os.getcwd() + '/' + outFileName + str(rotateIndex) + ".scan " + os.getcwd() + '/' + outFileName + str(rotateIndex)
        # print callString
        # # subprocess.call(callString.split())           # Assuming mechAFM is in parent directory
        # os.system(callString)
