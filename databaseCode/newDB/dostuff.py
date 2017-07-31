import numpy
import os
import argparse
import rotatexyz

parser = argparse.ArgumentParser(description="Take an XYZ file as input, rotate it in various directions and output a bunch of XYZ files into separate folders")
parser.add_argument("-i", "--input_file", default="input.xyz", help="the path to the input file (default: %(default)s)")
parser.add_argument("-o", "--output", default="output/", help="produced files and folders will be saved here (default: %(default)s)")
args = parser.parse_args()


#The below content will be copied into a file called parameters.dat which will be part of the MechAFM input for each of the XYZ files
parametersContent = '''# Parameters for a system from a paper

#      name  |  epsilon (kcal/mol)  |  sigma (A)  |  mass (amu)  |  charge (e)
atom    C         0.07000              3.55000       12.01100       0.00000
atom    H         0.03350              2.42000        1.00800       0.00000
atom    O         0.11080              2.98504       15.99940       0.00000
atom    Cl        0.32433              3.41652       35.45300       0.00000
atom    Al        0.65000              3.56500       26.98154       0.00000
atom    P         0.58500              3.83244       30.97376       0.00000
atom    Na        0.02939              2.52000       22.98977       0.00000
atom    Si        0.61000              3.31000       28.08550       0.00000
atom    N         0.19200              3.31988       14.00670       0.00000
atom    S         0.43560              3.63599       32.06500       0.00000
atom    F         0.11080              2.90789       18.99840       0.00000
atom    B         0.10500              3.63000       10.81000       0.00000
atom    X         0.07000              3.55000       12.01100       0.02100
atom    T         0.19200              3.15000       15.99900      -0.02100

# Boron parameters guessed from Baowan & Hill, IET Micro & Nano Letters 2:46 (2007)
# Carbon, oxygen and hydrogen parameters from original CHARMM force field

# Pair style to overwrite and default LJ-mixing
#            atom1  |  atom2  |  pair_style  |  parameters (eps,sig for LJ; De,a,re for Morse)
# pair_ovwrt     C        T         morse          1 2 3
pair_ovwrt     X        T         lj          20.000   3.5500

# Tip harmonic constraint
#     force constant (kcal/mol)  | distance (A)
harm          0.72000                 0.00

# Additional parameters for making the molecules flexible

# We need to know the topology, so list the possible bonds and their expected length
#           atom1  |  atom2  |  exp. length (A)
# topobond      C         C         1.430
# topobond      C         H         1.095
# topobond      C         B         1.534

# bonds are assumed harmonic and in their equilibrium position (in the xyz file)
#               force constant (kcal/mol)
bond              25.000

# angles are assumed harmonic and in their equilibrium position (in the xyz file)
#               force constant (kcal/mol)
angle             0.2500

# dihedrals are assumed harmonic and in their equilibrium position (in the xyz file)
#               force constant (kcal/mol)
dihedral          0.2500

# substrate support using a 10-4 wall potential
#               epsilon (kcal/mol) |  sigma (A) | lambda (A) | r_cut (A) | lateral constant (kcal/mol)
substrate         0.100                 3.0         3.0          7.5        0.01'''

# To make an output directory if it does not exist already
if args.output.endswith('/'):
    output_folder = args.output
else:
    output_folder = args.output + '/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

inFileName = args.input_file
fileName = inFileName.split('/')[-1]                                            # Stores the name of the input XYZ file in fileName ignoring the rest of the path

if not os.path.exists(output_folder + fileName[:-4] + '/'):                     # [:-4] is so that the input file name excludes the ".xyz" part so as to make a folder where each orientation will reside
    os.makedirs(output_folder + fileName[:-4] + '/')

parameterFile = open(output_folder + fileName[:-4] + '/parameters.dat', "w+")   # Create a parameters.dat file within the subfolder
parameterFile.write(parametersContent)
rotatexyz.rotate(inFileName, output_folder + fileName[:-4] + '/')               # Call the function which rotates input XYZ file into different orientations in 3D
