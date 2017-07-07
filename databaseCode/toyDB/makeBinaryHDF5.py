import struct
import glob
import argparse
import h5py
import numpy as np

parser = argparse.ArgumentParser(description="Take a scan file after MechAFM runs on it, compile all the mechAFM outputs and output a binary file with XYZ and MechAFM outputs")
parser.add_argument("-i", "--input_file", default="input.scan", help="the path to the input .SCAN file (default: %(default)s)")
parser.add_argument("-o", "--input_folder", default="input.scan", help="the path to the input folder containing all the files (default: %(default)s)")
args = parser.parse_args()

inputFolderName = args.input_folder
if not inputFolderName.endswith("/"):
        inputFolderName += "/"

inputFileName = args.input_file
scanFile = open(inputFileName, "r")
xyzFile = open(inputFileName[:-5]+ ".xyz", "r")
f = h5py.File(inputFileName[:-5] + ".hdf5" , "a")


widthX = 0.0
widthY = 0.0
widthZ = 0.0
zhigh = 0.0
zlow = 0.0
dx = 0.0
dy = 0.0
dz = 0.0
divX = 0
divY = 0
divZ = 0

moleculenumberstr = inputFileName[:-5].split("/")[-1]
f.create_group('molecule' + moleculenumberstr)
orientationstring = 'molecule' + moleculenumberstr + "/orientation1"                #only 1 orientation each


scanFileContents = scanFile.readlines()
scanFileContentsLength = len(scanFileContents)
for i in range(0, scanFileContentsLength):
    scanFileLine = scanFileContents[i].split()
    scanFileLineLength = len(scanFileLine)
    if not scanFileLineLength == 0:
        scanFileLineKeyword = scanFileLine[0]
        if scanFileLineKeyword == "area":
            widthX = float(scanFileLine[1])
            widthY = float(scanFileLine[2])
        elif scanFileLineKeyword == "zhigh":
            zhigh = float(scanFileLine[1])
        elif scanFileLineKeyword == "zlow":
            zlow = float(scanFileLine[1])
        elif scanFileLineKeyword == "dx":
            dx = float(scanFileLine[1])
        elif scanFileLineKeyword == "dy":
            dy = float(scanFileLine[1])
        elif scanFileLineKeyword == "dz":
            dz = float(scanFileLine[1])
widthZ = zhigh - zlow
divX = int(widthX/dx) + 1
divY = int(widthY/dy) + 1
divZ = int(widthZ/dz) + 1           # 1 is added because the first grid point is at 0,0,0

atomNameString = ""

xyzFileContents = xyzFile.readlines()
xyzFileNumAtoms = int(xyzFileContents[0])
#outFile.write(struct.pack('i', xyzFileNumAtoms))
atomPosition = np.zeros((xyzFileNumAtoms, 3), np.float32)

for j in range(0, xyzFileNumAtoms):
    xyzFileLine = xyzFileContents[j+2].split()
    atomNameString += xyzFileLine[0]
    atomPosition[j, 0] = float(xyzFileLine[1])
    atomPosition[j, 1] = float(xyzFileLine[2])
    atomPosition[j, 2] = float(xyzFileLine[3])
    # outFile.write(struct.pack('c', xyzFileLine[0]))
    #outFile.write(struct.pack('f', float(xyzFileLine[1])))
    #outFile.write(struct.pack('f', float(xyzFileLine[2])))
    #outFile.write(struct.pack('f', float(xyzFileLine[3])))

widthxyz = np.zeros(3, np.float32)
dxyz = np.zeros(3, np.float32)
divxyz = np.zeros(3, np.int32)
widthxyz[0] = widthX
widthxyz[1] = widthY
widthxyz[2] = widthZ
divxyz[0] = divX
divxyz[1] = divY
divxyz[2] = divZ
dxyz[0] = dx
dxyz[1] = dy
dxyz[2] = dz

fzarray = np.zeros((divZ, divX, divY), np.float32)

#outFile.write(struct.pack('f', widthZ))
#outFile.write(struct.pack('f', widthX))
#outFile.write(struct.pack('f', widthY))
#outFile.write(struct.pack('f', dz))
#outFile.write(struct.pack('f', dx))
#outFile.write(struct.pack('f', dy))
#outFile.write(struct.pack('i', divZ))
#outFile.write(struct.pack('i', divX))
#outFile.write(struct.pack('i', divY))
## firstDATFile = True
pathsToDATFiles = glob.glob(inputFolderName + "/*.dat")
pathsToDATFiles.sort(reverse=True)
zIndex = 0

pathsToDATFiles = pathsToDATFiles[:-1]                  # To make sure that parameters.dat is avoided while reading mechAFM outputs

for datFilePath in pathsToDATFiles:
    datFile = open(datFilePath, "r")
    datFileContents = datFile.readlines()
    datFileContentsLength = len(datFileContents)
    # datFileContents.sort(key = lambda x: (int(x.split()[1]), int(x.split()[2])))
    # if firstDATFile:
    #     outFile.write(struct.pack('i', datFileContentsLength))
    #     firstDATFile = False
    xIndex = 0
    yIndex = 0

    for i in range(0, datFileContentsLength):
        xIndex = int(datFileContents[i].split()[1])
        yIndex = int(datFileContents[i].split()[2])
        fzarray[xIndex, yIndex, zIndex] = float(datFileContents[i].split()[8])

        #outFile.write(struct.pack('f', float(datFileContents[i].split()[8])))          #Dumps f Z
    zIndex += 1
    datFile.close()

dsetfz = f.create_dataset(orientationstring+'/fzvals', fzarray.shape)
dsetfz[...] = fzarray
#dsetsol = f.create_dataset(orientationstring+'/solution', solutionarray.shape)
#dsetsol[...] = solutionarray
dsetpos = f.create_dataset(orientationstring+'/atomPosition', atomPosition.shape)
dsetpos[...] = atomPosition
f[orientationstring].attrs['atomNameString'] = atomNameString # This might be redundant, but it is saved along with the atom positio
f[orientationstring].attrs['widthxyz'] = widthxyz
f[orientationstring].attrs['dxyz'] = dxyz
f[orientationstring].attrs['divxyz'] = divxyz


scanFile.close()
xyzFile.close()
f.close()

# import struct
# import glob
# import argparse
#
# parser = argparse.ArgumentParser(description="Take the folder containing all rotations of a particular molecule as input, compile all the mechafm output .DAT files into one single .OUT file and all the .XYZ files into one .OUT file and put it into the same folder. Uses mechafm output folder, .SCAN, .XYZ for each orientation")
# parser.add_argument("-i", "--input_folder", default="input.xyz", help="the path to the input file (default: %(default)s)")
# args = parser.parse_args()
#
# if args.input_folder.endswith('/'):
#     inFolder = args.input_folder
# else:
#     inFolder = args.input_folder + '/'
#
# outputFileName = inFolder.split('/')[-2]            #Output file has same name as the input folder
#
# for xyzFilePath in glob.glob(inFolder+"*.xyz"):
#     xyzFile = open(xyzFilePath, "r")
#     scanFile = open(xyzFilePath[:-4]+".scan", "r")
#     outFile = open(inFolder+outputFileName+'_'+xyzFilePath[-5:]+".out", "wb+")           #OUT file containing all the mecchafm outputs
#     countDAT = 0
#     outFile.write(struct.pack('i', countDAT))
#     firstDATFile = True            #Assuming size of each DAT file for a particular orientation is the same, only size of first file is written into the file
#     mechOutputFolder = xyzFilePath[:-4]+'/'         #MechAFM ouput folder has the same name as that of xyz files. This folder contains all the DAT files
#     for datFilePath in glob.glob(mechOutputFolder+"*.dat"):
#         countDAT += 1
#         datFile = open(datFilePath, "r")
#         datFileContents = datFile.readlines()
#         numLinesDAT = len(datFileContents)
#         datFileContents.sort(key = lambda x: (int(x.split()[1]), int(x.split()[2])))
#         if firstDATFile:
#             outFile.write(struct.pack('i', numLinesDAT))
#             firstDATFile = False
#         for i in range(0, numLinesDAT):
#             outFile.write(struct.pack('f', float(datFileContents[i].split()[8])))          #Dumps f Z
#     if firstXYZFile:
#         firstXYZFile = False
#         outFile.seek(0)            #Goes to beginning of file to edit countDAT
#         # pickle.dump(countDAT, outFile)
#         outFile.write(struct.pack('i', countDAT))
#         outFile.seek(0, 2)          #Goes to end of file to continue adding data
# outFile.close()
