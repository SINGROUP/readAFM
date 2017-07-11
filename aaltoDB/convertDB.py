import sys
sys.path.append('/u/58/reischt1/unix/ml_projects/readAFM/src')
import numpy as np
import h5py
import readAFMDATAfile as rAFM
import glob
import time
import readAFMHDF5 as rHDF5


def convert_afmdata_to_hdf5(pathToMolecules, pathToHDF5):
#     f = h5py.File("/l/reischt1/AFMDB_version_02.hdf5", "w")    
#     moleculefiles = glob.glob('/l/reischt1/outputxyz_80pnts/*')
    
    f = h5py.File(pathToHDF5)
    moleculefiles = glob.glob(pathToMolecules)
    
    for moleculefile in moleculefiles:
        time_start = time.time()
        moleculenumberstr = moleculefile.split('_')[2].split('.')[0]
        molstring = 'molecule'+ moleculenumberstr
    
        print('Starting molecule '+molstring)
    
        # Create group for the molecule
        f.create_group(molstring)
    
        # Open the molecule-class
        mol = rAFM.afmmolecule(moleculefile)
        molattrs = mol.return_totalNumbers()
    
        # Give the molecules it's most imortant properties, at the time of writing this, this includes totalNumOrientations, numAtoms, atomNameString
        for attr in molattrs.keys(): 
            f[molstring].attrs[attr] = molattrs[attr]
    
        for orientation in range(molattrs['totalNumOrientations']):
    
            orientationstring = molstring+'/orientation'+str(orientation)
    
            print('Doing Orientation: '+orientationstring)
    
            # Read data from file
            fzdata=mol.F_orientation(orientation)
            fzarray=fzdata[0]
            atomNameString=fzdata[1]
            atomPosition = fzdata[2]
            widthxyz = fzdata[3]
            dxyz = fzdata[4]
            divxyz = fzdata[5]
            solutionarray = mol.solution_xymap_projection(orientation)
    
            #Write to HDF5-file:
            #First write the np.arrays:
            dsetfz = f.create_dataset(orientationstring+'/fzvals', fzarray.shape)
            dsetfz[...] = fzarray
            dsetsol = f.create_dataset(orientationstring+'/solution', solutionarray.shape)
            dsetsol[...] = solutionarray
            dsetpos = f.create_dataset(orientationstring+'/atomPosition', atomPosition.shape)
            dsetpos[...] = atomPosition
    
            # Write stuff into the attrs of the orientation:
            f[orientationstring].attrs['atomNameString'] = fzdata[1] # This might be redundant, but it is saved along with the atom positions, that's why it is important (the order of it!).
            f[orientationstring].attrs['widthxyz'] = fzdata[3]
            f[orientationstring].attrs['dxyz'] = fzdata[4]
            f[orientationstring].attrs['divxyz'] = fzdata[5]
    
        time_end = time.time()
        print("This molecule took %f seconds to convert." % (time_end-time_start))
    
    f.close()

def change_labels(PathToHDF5):
    f = rHDF5.AFMdata(PathToHDF5)
    for molstr in f.f.keys():
        timestart=time.time()
        molecule = f.f[molstr]
        print(molstr)
        for ortnstr in molecule.keys():
            orientation=molecule[ortnstr]
            orientation['solution'][...]=f.solution_toyDB(orientation.name)[...]
#             del(orientation['solution'])
#             orientation.create_dataset('solution', data=f.solution_xymap_collapsed(orientation.name))
            print(ortnstr, orientation.name)
        timeend=time.time()
        print("This molecule took %f seconds to relabel."%(timeend-timestart))
            
def add_labels(PathToHDF5):
    f = rHDF5.AFMdata(PathToHDF5)
    for molstr in f.f.keys():
        molecule = f.f[molstr]
        for ortnstr in molecule.keys():
            orientation=molecule[ortnstr]
            orientation.create_dataset('solution',data=f.solution_xymap_collapsed(orientation.name)[...])
            print('Completeded %s bzw %s' %(ortnstr, orientation.name))
        



if __name__ == "__main__":
#     convert_afmdata_to_hdf5('/l/reischt1/outputxyz_80pnts/', '/l/reischt1/AFMDB_version_02.hdf5')
#     change_labels('/l/reischt1/AFMDB_version_02.hdf5')
    add_labels('/l/reischt1/toyDB_v07.hdf5')