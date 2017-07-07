import numpy
import os
import argparse
import makeXYZ

parser = argparse.ArgumentParser(description="Take an XYZ file as input, rotate it in various directions and output a bunch of XYZ files into separate folders")
parser.add_argument("-o", "--output", default="output/", help="produced files and folders will be saved here (default: %(default)s)")
args = parser.parse_args()

#parametersContent = '''# Parameters for a system from a paper
#
##      name  |  epsilon (kcal/mol)  |  sigma (A)  |  mass (amu)  |  charge (e)
#atom    C         0.07000              3.55000       12.01100       0.00000
#atom    H         0.03350              2.42000        1.00800       0.00000
#atom    O         0.11080              2.98504       15.99940       0.00000
#atom    N         0.19200              3.31988       14.00670       0.00000
#atom    S         0.43560              3.63599       32.06500       0.00000
#atom    F         0.11080              2.90789       18.99840       0.00000
#atom    B         0.10500              3.63000       10.81000       0.00000
#atom    X        20.00000              3.55000       12.01100       0.02100
#atom    T         0.19200              3.15000       15.99900      -0.02100
#
## Boron parameters guessed from Baowan & Hill, IET Micro & Nano Letters 2:46 (2007)
## Carbon, oxygen and hydrogen parameters from original CHARMM force field
#
## Pair style to overwrite and default LJ-mixing
##            atom1  |  atom2  |  pair_style  |  parameters (eps,sig for LJ; De,a,re for Morse)
## pair_ovwrt     C        T         morse          1 2 3
## pair_ovwrt     H        T         lj             1 2
#
## Tip harmonic constraint
##     force constant (kcal/mol)  | distance (A)
#harm          0.72000                 0.00
#
## Additional parameters for making the molecules flexible
#
## We need to know the topology, so list the possible bonds and their expected length
##           atom1  |  atom2  |  exp. length (A)
## topobond      C         C         1.430
## topobond      C         H         1.095
## topobond      C         B         1.534
#
## bonds are assumed harmonic and in their equilibrium position (in the xyz file)
##               force constant (kcal/mol)
#bond              25.000
#
## angles are assumed harmonic and in their equilibrium position (in the xyz file)
##               force constant (kcal/mol)
#angle             0.2500
#
## dihedrals are assumed harmonic and in their equilibrium position (in the xyz file)
##               force constant (kcal/mol)
#dihedral          0.2500
#
## substrate support using a 10-4 wall potential
##               epsilon (kcal/mol) |  sigma (A) | lambda (A) | r_cut (A) | lateral constant (kcal/mol)
#substrate         0.100                 3.0         3.0          7.5        0.01'''
#

if args.output.endswith('/'):
    output_folder = args.output
else:
    output_folder = args.output + '/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# if args.input.endswith('/'):
#     input_folder = args.input
# else:
#     input_folder = args.input + '/'

#parameterFile = open(output_folder + 'parameters.dat', "w+")
#parameterFile.write(parametersContent)
makeXYZ.makeIt(output_folder)
