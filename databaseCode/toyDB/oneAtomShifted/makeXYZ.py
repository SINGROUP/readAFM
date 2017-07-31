import numpy
import os
import subprocess
# from time import sleep

# Rotates input xyz file and generates 100 output xyz files at random orientations along with input.scan for each of them to give as input to mechAFM

def makeIt(output_folder = "randomRotateOutput/"):
    fileNumber = 0
    for x in range(41):
        for y in range(41):

            xyzOut = '''1

C %s %s 0.0''' % (x*0.2, y*0.2)

            scanOut = '''xyzfile    %s
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
center       %s %s

zhigh        10.0
zlow         6.0
dx           %s
dy           %s
dz           0.1

bufsize      10000
gzip         off
statistics   on''' % (str(fileNumber) + ".xyz", x*0.2 + 0.01, y*0.2 + 0.01, 0.2, 0.2)

            parametersContent = '''# Parameters for a system from a paper

#      name  |  epsilon (kcal/mol)  |  sigma (A)  |  mass (amu)  |  charge (e)
atom    C         0.07000              3.55000       12.01100       0.00000
atom    H         0.03350              2.42000        1.00800       0.00000
atom    O         0.11080              2.98504       15.99940       0.00000
atom    N         0.19200              3.31988       14.00670       0.00000
atom    S         0.43560              3.63599       32.06500       0.00000
atom    F         0.11080              2.90789       18.99840       0.00000
atom    B         0.10500              3.63000       10.81000       0.00000
atom    X        20.00000              3.55000       12.01100       0.02100
atom    T         0.19200              3.15000       15.99900      -0.02100

# Boron parameters guessed from Baowan & Hill, IET Micro & Nano Letters 2:46 (2007)
# Carbon, oxygen and hydrogen parameters from original CHARMM force field

# Pair style to overwrite and default LJ-mixing
#            atom1  |  atom2  |  pair_style  |  parameters (eps,sig for LJ; De,a,re for Morse)
# pair_ovwrt     C        T         morse          1 2 3
# pair_ovwrt     H        T         lj             1 2

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



            os.makedirs(output_folder + str(fileNumber))
            
            xyzFile = open(output_folder + str(fileNumber) + "/"  + str(fileNumber) + ".xyz", "w+")
            xyzFile.write(xyzOut)

            scanFile = open(output_folder + str(fileNumber) + "/"  + str(fileNumber) + ".scan", "w+")
            scanFile.write(scanOut)
            
            paraFile = open(output_folder + str(fileNumber) + "/"  + "parameters.dat", "w+")
            paraFile.write(parametersContent)

            xyzFile.close()
            scanFile.close()
            paraFile.close()
            print("done with file number " + str(fileNumber))
            fileNumber += 1
