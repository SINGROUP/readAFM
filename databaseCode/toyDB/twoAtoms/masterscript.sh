#!/bin/bash

OUTPUTXYZ=$WRKDIR/varmad1/MechAFM/myCode/toyDB/twoAtoms/output
cd $OUTPUTXYZ
for i in *; do
    cp $OUTPUTXYZ/../template.job $OUTPUTXYZ/$i
    cd $OUTPUTXYZ/$i
    sbatch ./template.job
    cd ..
done
