import h5py
import numpy as np
import glob

f = h5py.File("db.hdf5", "a")                                                                   # Name of single compiled output binary file

pathsToHDF5Files = glob.glob("./*.hdf5")                                                        # Edit this to the path of the folder with all the HDF5 files

for hdf5FilePath in pathsToHDF5Files:
    g = h5py.File(hdf5FilePath, "r")                                                            # Iterate through all the HDF5 files and compile info
    moleculenumberstr = hdf5filePath.split("/")[-1][:-5][-6:]                                   # [:-5] ensures that .hdf5 is not selected. [-6:] ensures that only the numbers from 'dsgdb9nsd_000001' are selected
    f.create_group('molecule'+moleculenumberstr)
    for i in range(100):                                                                        # Iterate through ech of the orientations
        orientationstring = 'molecule' + moleculenumberstr + '/orientation%d'%(i)
        fzarray = g[orientationstring + '/fzvals']
        atomPosition = g[orientationstring + '/atomPosition']

        atomNameString = g[orientationstring].attrs["atomNameString"]
        widthxyz = g[orientationstring].attrs["widthxyz"]
        dxyz = g[orientationstring].attrs["dxyz"]
        divxyz = g[orientationstring].attrs["divxyz"]
        zLowzHigh = g[orientationstring].attrs["zLowzHigh"]

        dsetfz = f.create_dataset(orientationstring+'/fzvals', fzarray.shape)
        dsetfz[...] = fzarray
        #dsetsol = f.create_dataset(orientationstring+'/solution', solutionarray.shape)
        #dsetsol[...] = solutionarray
        dsetpos = f.create_dataset(orientationstring+'/atomPosition', atomPosition.shape)
        dsetpos[...] = atomPosition

        f[orientationstring].attrs['atomNameString'] = atomNameString # This might be redundant, but it is saved along with the atom positi
        f[orientationstring].attrs['widthxyz'] = widthxyz
        f[orientationstring].attrs['dxyz'] = dxyz
        f[orientationstring].attrs['divxyz'] = divxyz
        f[orientationstring].attrs['zlowzHigh'] = zLowzHigh


#for i in range(1, 1001):
#    g = h5py.File("%d.hdf5"%(i,i), "r")
#    f.create_group('molecule' + str(i))
#    orientationstring = 'molecule' + str(i) + "/orientation1"
#    fzarray = g[orientationstring + '/fzvals']
#    atomPosition = g[orientationstring + '/atomPosition']
#
#    atomNameString = g[orientationstring].attrs["atomNameString"]
#    widthxyz = g[orientationstring].attrs["widthxyz"]
#    dxyz = g[orientationstring].attrs["dxyz"]
#    divxyz = g[orientationstring].attrs["divxyz"]
#
#    dsetfz = f.create_dataset(orientationstring+'/fzvals', fzarray.shape)
#    dsetfz[...] = fzarray
#    #dsetsol = f.create_dataset(orientationstring+'/solution', solutionarray.shape)
#    #dsetsol[...] = solutionarray
#    dsetpos = f.create_dataset(orientationstring+'/atomPosition', atomPosition.shape)
#    dsetpos[...] = atomPosition
#
#    f[orientationstring].attrs['atomNameString'] = atomNameString # This might be redundant, but it is saved along with the atom positi
#    f[orientationstring].attrs['widthxyz'] = widthxyz
#    f[orientationstring].attrs['dxyz'] = dxyz
#    f[orientationstring].attrs['divxyz'] = divxyz
