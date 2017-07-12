import numpy as np
import h5py

f = h5py.File("db.hdf5", "a")

for i in range(1681):
    g = h5py.File("%d/%d.hdf5"%(i,i), "r")
    f.create_group('molecule' + str(i))
    orientationstring = 'molecule' + str(i) + "/orientation1"
    fzarray = g[orientationstring + '/fzvals']
    atomPosition = g[orientationstring + '/atomPosition']
    
    atomNameString = g[orientationstring].attrs["atomNameString"]
    widthxyz = g[orientationstring].attrs["widthxyz"]
    dxyz = g[orientationstring].attrs["dxyz"]
    divxyz = g[orientationstring].attrs["divxyz"]

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

