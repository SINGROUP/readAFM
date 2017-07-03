#!/bin/bash

INPUTXYZFOLDER=$WRKDIR/MechAFM/myCode/FINAL/inputXYZ
OUTPUTXYZ=$WRKDIR/MechAFM/myCode/FINAL/outputxyz
DOSTUFF=$WRKDIR/MechAFM/myCode/dostuff.py
cd $INPUTXYZFOLDER
for i in *.xyz; do
    INPUTXYZ=$INPUTXYZFOLDER/$i
    python $DOSTUFF -i $INPUTXYZ -o $OUTPUTXYZ
    cp $INPUTXYZFOLDER/template.job $OUTPUTXYZ/${i%.xyz}
    cd $OUTPUTXYZ/${i%.xyz}
    sbatch ./template.job
    cd ..
done
